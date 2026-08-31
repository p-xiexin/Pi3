"""Sliding-window geometry, stable image queries, and verified track factors."""

import math
from dataclasses import dataclass

import torch

from .geometric_verification import verify_packet
from .keyframes import select_keyframes_eq4_window
from .scale_metric import (
    MIN_SCALE_POINTS,
    SCALE_SAMPLE_POINTS,
    estimate_chunk_scale,
)
from .tracks import sample_map


PI3_VALID_CONFIDENCE = 0.1
PI3_MASK_CONFIDENCE = 0.3


@dataclass(frozen=True)
class WindowState:
    """One decoded window with the masks consumed by tracking and keyframing."""

    frame_ids: tuple
    images: torch.Tensor
    track_images: torch.Tensor
    points: torch.Tensor
    confidence: torch.Tensor
    dense_confidence: torch.Tensor
    track_valid_mask: torch.Tensor
    poses: torch.Tensor
    metric_scale: torch.Tensor
    K: torch.Tensor


class FrameStore:
    """Persist dense frame geometry and stable keyframe query identities."""

    def __init__(self, dataset):
        self.paths = dataset.paths
        self.K = dataset.K
        self.valid_mask = dataset.valid_mask
        self.keyframes = set()
        self.dense = {}
        self.anchors = {}
        self.track_ids = {}
        self.next_track_id = 0

    def add_dense(self, frame_id, image, points, confidence, scale=1.0):
        """Cache one RGB image, depth map, and confidence map for each frame."""
        frame_id = int(frame_id)
        if frame_id not in self.dense:
            scale = float(scale)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("cached Pi3 depth scale must be finite and positive")
            stored = tuple(
                value.detach().cpu()
                for value in (image, points[..., 2], confidence)
            )
            self.dense[frame_id] = (*stored, scale)

    def add_keyframe(self, frame_id, image, points, confidence, keys, queries, anchors, weights):
        """Store one keyframe and assign every image query a stable track ID."""
        frame_id = int(frame_id)
        self.keyframes.add(frame_id)
        self.add_dense(frame_id, image, points, confidence)
        if frame_id not in self.anchors:
            self.anchors[frame_id] = tuple(
                value.detach().cpu() for value in (keys, queries, anchors, weights)
            )
            count = int(keys.numel())
            self.track_ids[frame_id] = torch.arange(
                self.next_track_id,
                self.next_track_id + count,
                dtype=torch.long,
            )
            self.next_track_id += count
        else:
            cached_keys = self.anchors[frame_id][0]
            if not torch.equal(cached_keys, keys.detach().cpu()):
                raise RuntimeError(
                    f"keyframe {frame_id} image query identity changed"
                )
            if frame_id not in self.track_ids:
                raise RuntimeError(
                    f"keyframe {frame_id} has no stable track ID registry"
                )
        return self.track_ids[frame_id].to(keys.device)


class WindowTracker:
    """Convert either tracks model into the same persistent graph factors."""

    def __init__(self, geometry_model, tracks_model, features, frames, dataset, config):
        self.geometry_model = geometry_model.eval()
        self.tracks_model = tracks_model
        self.features = features
        self.frames = frames
        self.dataset = dataset
        self.device = torch.device(config["device"])
        self.pi3_mask_confidence_threshold = float(
            config.get("pi3_mask_confidence_threshold", PI3_MASK_CONFIDENCE)
        )
        self.keyframe_projection_threshold = float(
            config.get("keyframe_projection_threshold", 0.7)
        )
        self.keyframe_confidence_threshold = float(
            config.get("keyframe_confidence_threshold", PI3_VALID_CONFIDENCE)
        )
        self.keyframe_max_interval = int(config.get("keyframe_max_interval", 5))
        self.processed_pairs = set()

    def _chunk_scale(self, frame_ids, depth, confidence):
        """Align one raw Pi3 chunk to cached depths from its overlap frames."""
        if all(int(frame_id) in self.frames.dense for frame_id in frame_ids):
            return depth.new_tensor(1.0)
        overlap = [
            (local, int(frame_id))
            for local, frame_id in enumerate(frame_ids)
            if int(frame_id) in self.frames.dense
        ]
        if not overlap:
            return depth.new_tensor(1.0)
        local_ids = torch.tensor(
            [local for local, _ in overlap], device=depth.device, dtype=torch.long
        )
        overlap_ids = [frame_id for _, frame_id in overlap]
        reference_depth = torch.stack(
            [self.frames.dense[frame_id][1].to(depth) for frame_id in overlap_ids]
        )
        reference_confidence = torch.stack(
            [
                self.frames.dense[frame_id][2].to(confidence)
                for frame_id in overlap_ids
            ]
        )
        reference_scale = depth.new_tensor(
            [self.frames.dense[frame_id][3] for frame_id in overlap_ids]
        )
        return estimate_chunk_scale(
            depth[local_ids],
            reference_depth,
            reference_scale,
            current_confidence=confidence[local_ids],
            reference_confidence=reference_confidence,
            sample_points=SCALE_SAMPLE_POINTS,
            minimum_points=MIN_SCALE_POINTS,
        )

    def _valid_tracks(self, output, valid_mask):
        """Apply the graph observation checks to tracks from one reference frame."""
        height, width = valid_mask.shape[-2:]
        tracks, scores = output["tracks"], output["confidence"]
        valid = (
            torch.isfinite(tracks).all(-1) & torch.isfinite(scores) & (scores > 0)
            & (tracks[..., 0] >= 0) & (tracks[..., 0] <= width - 1)
            & (tracks[..., 1] >= 0) & (tracks[..., 1] <= height - 1)
        )
        pixels = tracks.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).round().long()
        pixels[..., 0].clamp_(0, width - 1)
        pixels[..., 1].clamp_(0, height - 1)
        local_frames = torch.arange(tracks.shape[0], device=self.device)[:, None]
        return valid & valid_mask[local_frames, pixels[..., 1], pixels[..., 0]]

    def _reference_tracks(
        self, frame_ids, images, points, confidence, valid_mask, reference, cache
    ):
        """Extract one keyframe query set and cache its window-wide matches."""
        if reference not in cache:
            frame_id = int(frame_ids[reference])
            keys, queries, anchors, weights = self._queries(
                frame_id,
                images[reference],
                points[reference],
                confidence[reference],
                valid_mask[reference],
            )
            if not keys.numel():
                raise RuntimeError(f"keyframe {frame_id} has no valid image queries")
            output = self.tracks_model.track(reference, queries)
            cache[reference] = {
                "keys": keys,
                "queries": queries,
                "anchors": anchors,
                "weights": weights,
                "output": output,
                "valid": self._valid_tracks(output, valid_mask),
            }
        return cache[reference]

    def _queries(self, frame_id, image, points, confidence, valid_mask):
        """Return every selected image query with a stable per-keyframe identity."""
        if frame_id in self.frames.anchors:
            return tuple(value.to(self.device) for value in self.frames.anchors[frame_id])
        queries = self.features.extract(image, valid_mask)
        anchors = sample_map(points, queries)[0]
        weights = torch.ones(queries.shape[0], device=self.device, dtype=queries.dtype)
        # Query identity is fixed once the selected frontend rows are created.
        # Sliding and loop packets then reuse these row identities unchanged.
        keys = torch.arange(queries.shape[0], device=self.device, dtype=torch.long)
        return keys, queries, anchors, weights

    def _factors(
        self,
        output,
        frame_ids,
        reference,
        keys,
        track_ids,
        queries,
        anchors,
        anchor_weights,
        track_valid,
    ):
        """Convert one raw reference track result into image observations."""
        tracks, scores = output["tracks"], output["confidence"]
        valid = track_valid.clone()
        ref_id = int(frame_ids[reference])
        valid[reference] = True
        target_valid = valid.clone()
        target_valid[reference] = False
        keep = target_valid.any(0)
        if not keep.any():
            return None
        keys, track_ids, queries = keys[keep], track_ids[keep], queries[keep]
        anchors, anchor_weights = anchors[keep], anchor_weights[keep]
        tracks, scores = tracks[:, keep], scores[:, keep]
        valid = valid[:, keep]
        point_index = torch.arange(keys.numel(), device=self.device)
        obs_frames = [torch.full_like(point_index, ref_id)]
        obs_points = [point_index]
        obs_uv = [queries]
        obs_weights = [anchor_weights]
        for target, target_id in enumerate(frame_ids):
            if target == reference:
                continue
            mask = valid[target]
            if not mask.any():
                continue
            obs_frames.append(
                torch.full(
                    (int(mask.sum()),),
                    int(target_id),
                    device=self.device,
                    dtype=torch.long,
                )
            )
            obs_points.append(point_index[mask])
            obs_uv.append(tracks[target, mask])
            obs_weights.append(scores[target, mask])
        return {
            "keys": keys, "track_ids": track_ids, "anchors": anchors,
            "obs_frames": torch.cat(obs_frames), "obs_points": torch.cat(obs_points),
            "obs_uv": torch.cat(obs_uv), "obs_weights": torch.cat(obs_weights),
        }

    def _keep_fresh_observations(self, packet):
        """Drop repeated reference-target measurements after geometry verification."""
        fresh_parts = []
        for part in packet["parts"]:
            reference = int(part["reference"])
            obs_frames = part["obs_frames"]
            fresh = torch.tensor(
                [
                    int(frame_id) == reference
                    or (reference, int(frame_id)) not in self.processed_pairs
                    for frame_id in obs_frames.tolist()
                ],
                device=obs_frames.device,
                dtype=torch.bool,
            )
            point_count = int(part["track_ids"].numel())
            target = fresh & (obs_frames != reference)
            support = torch.bincount(
                part["obs_points"][target], minlength=point_count
            )
            point_keep = support > 0
            if not bool(point_keep.any()):
                continue
            observation_keep = fresh & point_keep[part["obs_points"]]
            point_map = torch.full(
                (point_count,), -1, device=obs_frames.device, dtype=torch.long
            )
            point_map[point_keep] = torch.arange(
                int(point_keep.sum()), device=obs_frames.device
            )
            output = dict(part)
            for name in ("keys", "track_ids", "anchors", "queries", "anchor_weights"):
                value = output.get(name)
                if torch.is_tensor(value) and value.shape[:1] == point_keep.shape:
                    output[name] = value[point_keep.to(value.device)]
            for name in ("obs_frames", "obs_uv", "obs_weights"):
                output[name] = output[name][observation_keep.to(output[name].device)]
            kept_points = part["obs_points"][observation_keep]
            output["obs_points"] = point_map[kept_points]
            fresh_parts.append(output)
        packet["parts"] = fresh_parts
        return packet

    def _reference_indices(self, frame_ids, references):
        """Resolve explicit loop references into local window indices."""
        if references is None:
            return None
        local = {frame_id: index for index, frame_id in enumerate(frame_ids)}
        missing = [int(frame_id) for frame_id in references if int(frame_id) not in local]
        if missing:
            raise ValueError(f"reference frames are absent from window: {missing}")
        unregistered = [
            int(frame_id)
            for frame_id in references
            if int(frame_id) not in self.frames.track_ids
        ]
        if unregistered:
            raise RuntimeError(
                "loop references have no stable track ID registry: "
                f"{unregistered}"
            )
        return list(dict.fromkeys(local[int(frame_id)] for frame_id in references))

    def _infer_window(self, frame_ids):
        """Decode Pi3 geometry and prepare the selected tracks frontend once."""
        images = self.dataset.read(frame_ids).to(self.device)
        model_images = images
        if self.frames.valid_mask is not None:
            model_images = images * self.frames.valid_mask.to(self.device)[None, None]
        geometry, tracks_state = self.geometry_model.infer_window(
            model_images.unsqueeze(0)
        )
        points = geometry["local_points"].squeeze(0)
        confidence = geometry["conf"].squeeze(0).sigmoid().squeeze(-1)
        image_valid_mask = torch.ones_like(confidence, dtype=torch.bool)
        if self.frames.valid_mask is not None:
            image_valid_mask &= self.frames.valid_mask.to(self.device)[None]
        pi3_valid = (
            torch.isfinite(points).all(-1)
            & torch.isfinite(confidence)
            & (points[..., 2] > 0)
            & (confidence > PI3_VALID_CONFIDENCE)
            & image_valid_mask
        )
        valid_counts = pi3_valid.flatten(1).sum(-1)
        if (valid_counts == 0).any():
            local = int(torch.nonzero(valid_counts == 0, as_tuple=False)[0])
            raise RuntimeError(f"frame {frame_ids[local]} has no valid Pi3 geometry")

        track_valid_mask = (
            torch.isfinite(confidence)
            & (confidence > self.pi3_mask_confidence_threshold)
            & image_valid_mask
        )
        track_images = images * track_valid_mask[:, None]
        dense_confidence = torch.where(track_valid_mask, confidence, 0)
        if not torch.equal(track_valid_mask, image_valid_mask):
            tracks_state = None
        self.tracks_model.prepare_window(track_images.unsqueeze(0), tracks_state)

        poses = geometry["camera_poses"].squeeze(0)
        poses = torch.linalg.inv(poses[0])[None] @ poses
        K = self.frames.K.to(self.device).expand(len(frame_ids), -1, -1).clone()
        scale = self._chunk_scale(frame_ids, points[..., 2], dense_confidence)
        for local, frame_id in enumerate(frame_ids):
            self.frames.add_dense(
                frame_id,
                images[local],
                points[local],
                dense_confidence[local],
                scale=scale,
            )
        return WindowState(
            frame_ids=frame_ids,
            images=images,
            track_images=track_images,
            points=points,
            confidence=confidence,
            dense_confidence=dense_confidence,
            track_valid_mask=track_valid_mask,
            poses=poses,
            metric_scale=scale,
            K=K,
        )

    def _keyframe_indices(self, window, explicit_keyframes):
        """Select sliding keyframes or reuse the explicit loop references."""
        if explicit_keyframes is not None:
            return explicit_keyframes
        return select_keyframes_eq4_window(
            window.frame_ids,
            window.points,
            window.confidence,
            window.poses,
            window.K,
            window.track_valid_mask,
            existing_keyframes=self.frames.keyframes,
            projection_threshold=self.keyframe_projection_threshold,
            confidence_threshold=self.keyframe_confidence_threshold,
            maximum_interval=self.keyframe_max_interval,
        ).tolist()

    def _build_packet(self, window, keyframes, is_loop):
        """Track selected references and assemble an unverified factor packet."""
        track_cache = {}
        parts = []
        visualization = []
        for reference in keyframes:
            frame_id = window.frame_ids[reference]
            tracked = self._reference_tracks(
                window.frame_ids,
                window.track_images,
                window.points,
                window.confidence,
                window.track_valid_mask,
                reference,
                track_cache,
            )
            if not is_loop:
                visualization.append({
                    "frontend": getattr(self.tracks_model, "name", "unknown"),
                    "reference": frame_id,
                    "frame_ids": torch.tensor(
                        window.frame_ids, device=self.device, dtype=torch.long
                    ),
                    "query_points": tracked["queries"],
                    "raw_tracks": tracked["output"]["tracks"],
                    "frontend_valid": tracked["valid"],
                    "visualization_confidence": tracked["output"].get(
                        "visualization_confidence",
                        tracked["output"]["confidence"],
                    ),
                    "visualization_score": tracked["output"].get(
                        "visualization_score"
                    ),
                    "visualization_confidence_label": tracked["output"].get(
                        "visualization_confidence_label", "confidence"
                    ),
                })
            track_ids = self.frames.add_keyframe(
                frame_id,
                window.images[reference],
                window.points[reference],
                window.dense_confidence[reference],
                tracked["keys"],
                tracked["queries"],
                tracked["anchors"],
                tracked["weights"],
            )
            already_processed = all(
                (frame_id, target) in self.processed_pairs
                for target in window.frame_ids
                if target != frame_id
            )
            if already_processed:
                continue
            part = self._factors(
                tracked["output"],
                window.frame_ids,
                reference,
                tracked["keys"],
                track_ids,
                tracked["queries"],
                tracked["anchors"],
                tracked["weights"],
                tracked["valid"],
            )
            if part is not None:
                part["reference"] = frame_id
                parts.append(part)
        return {
            "kind": "loop" if is_loop else "sliding",
            "frontend": getattr(self.tracks_model, "name", "unknown"),
            "frame_ids": torch.tensor(window.frame_ids, device=self.device),
            "keyframes": torch.tensor(
                [window.frame_ids[reference] for reference in keyframes],
                device=self.device,
                dtype=torch.long,
            ),
            "poses": torch.linalg.inv(window.poses),
            "pi3_T_WCs": window.poses,
            "metric_scale": window.metric_scale,
            "K": window.K,
            "parts": parts,
            "edges": [],
            "visualization": visualization,
        }

    def _verify_and_commit(self, packet, keyframes, minimum_edge_views, is_loop):
        """Verify candidate geometry and commit only accepted pair identities."""
        if not packet["parts"]:
            if is_loop:
                return packet
            raise RuntimeError("tracks frontend produced no candidate factors")
        pairs = None
        if is_loop:
            pairs = [
                (reference, target)
                for reference in keyframes
                for target in range(len(packet["frame_ids"]))
                if target != reference
            ]
        try:
            packet = verify_packet(
                packet,
                minimum_track_observations=minimum_edge_views,
                pairs=pairs,
                initialize=not is_loop,
            )
        except RuntimeError as error:
            if not is_loop:
                raise
            packet["parts"] = []
            packet["edges"] = []
            packet["geometry_error"] = str(error)
        else:
            packet = self._keep_fresh_observations(packet)
        for part in packet["parts"]:
            reference_id = int(part["reference"])
            self.processed_pairs.update(
                (reference_id, int(target_id))
                for target_id in part["obs_frames"].unique().tolist()
                if int(target_id) != reference_id
            )
        return packet

    @torch.no_grad()
    def track(self, frame_ids, references=None, minimum_edge_views=3):
        """Infer one window and return a factor packet without optimizing state."""
        if int(minimum_edge_views) < 2:
            raise ValueError("minimum_edge_views must be at least two")
        frame_ids = tuple(map(int, frame_ids))
        explicit_keyframes = self._reference_indices(frame_ids, references)
        window = self._infer_window(frame_ids)
        keyframes = self._keyframe_indices(window, explicit_keyframes)
        packet = self._build_packet(window, keyframes, references is not None)
        return self._verify_and_commit(
            packet, keyframes, int(minimum_edge_views), references is not None
        )


__all__ = [
    "FrameStore",
    "PI3_MASK_CONFIDENCE",
    "PI3_VALID_CONFIDENCE",
    "WindowState",
    "WindowTracker",
]
