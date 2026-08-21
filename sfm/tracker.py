"""Sliding-window geometry, shared SIFT queries, and tracks factor extraction."""

import torch

from .tracks import sample_map


PI3_VALID_CONFIDENCE = 0.1
KEYFRAME_PROJECTION_RATIO = 0.5
KEYFRAME_MATCH_COVERAGE_RATIO = 0.5


class FrameStore:
    """Persist keyframe geometry and SIFT anchors across overlapping windows."""

    def __init__(self, dataset):
        self.paths = dataset.paths
        self.K = dataset.K
        self.valid_mask = dataset.valid_mask
        self.keyframes = set()
        self.dense = {}
        self.anchors = {}

    def add_keyframe(self, frame_id, image, points, confidence, keys, queries, anchors, weights):
        """Store immutable CPU copies of dense geometry and sparse anchor metadata."""
        self.keyframes.add(int(frame_id))
        if frame_id not in self.dense:
            self.dense[int(frame_id)] = (
                image.detach().cpu(), points.detach().cpu(), confidence.detach().cpu()
            )
        if frame_id not in self.anchors:
            self.anchors[int(frame_id)] = tuple(
                value.detach().cpu() for value in (keys, queries, anchors, weights)
            )


class WindowTracker:
    """Convert either tracks model into the same persistent graph factors."""

    def __init__(self, geometry_model, tracks_model, features, frames, dataset, config):
        self.geometry_model = geometry_model.eval()
        self.tracks_model = tracks_model
        self.features = features
        self.frames = frames
        self.dataset = dataset
        self.device = torch.device(config["device"])
        self.processed_pairs = set()

    def _keyframes_projection_coverage(self, frame_ids, points, poses, K, valid_mask):
        """Select a new keyframe when existing views cover too little valid geometry."""
        height, width = valid_mask.shape[-2:]
        selected = [i for i, frame_id in enumerate(frame_ids) if frame_id in self.frames.keyframes]
        if not selected:
            selected = [0]
        for target in range(len(frame_ids)):
            if target in selected:
                continue
            target_valid = valid_mask[target]
            threshold = KEYFRAME_PROJECTION_RATIO * target_valid.sum()
            relative = torch.linalg.inv(poses[selected]) @ poses[target]
            X = points[target].reshape(-1, 3)
            Xr = torch.einsum("kij,mj->kmi", relative[:, :3, :3], X) + relative[:, None, :3, 3]
            projected = torch.einsum("kij,kmj->kmi", K[selected], Xr)
            uv = projected[..., :2] / projected[..., 2:3].clamp_min(1.0e-8)
            pixels = uv.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).round().long()
            pixels[..., 0].clamp_(0, width - 1)
            pixels[..., 1].clamp_(0, height - 1)
            reference_ids = torch.arange(len(selected), device=self.device)[:, None]
            projected_valid = valid_mask[selected][
                reference_ids, pixels[..., 1], pixels[..., 0]
            ]
            valid = (
                target_valid.reshape(1, -1)
                & projected_valid
                & (Xr[..., 2] > 0)
                & (uv[..., 0] >= 0) & (uv[..., 0] <= width - 1)
                & (uv[..., 1] >= 0) & (uv[..., 1] <= height - 1)
            )
            if valid.sum(dim=-1).max() < threshold:
                selected.append(target)
        return sorted(selected)

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
                raise RuntimeError(f"keyframe {frame_id} has no valid SIFT queries")
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

    def _keyframes_mast3r_fusion(
        self, frame_ids, images, points, confidence, valid_mask, track_cache
    ):
        """Select keyframes from match coverage against the latest keyframe."""
        existing = [
            local for local, frame_id in enumerate(frame_ids)
            if frame_id in self.frames.keyframes
        ]
        reference = existing[-1] if existing else 0
        selected = [reference]
        for target in range(reference + 1, len(frame_ids)):
            tracked = self._reference_tracks(
                frame_ids, images, points, confidence, valid_mask, reference, track_cache
            )
            required = KEYFRAME_MATCH_COVERAGE_RATIO * tracked["keys"].numel()
            if tracked["valid"][target].sum() < required:
                selected.append(target)
                reference = target
        return selected

    def _queries(self, frame_id, image, points, confidence, valid_mask):
        """Return stable feature IDs, pixels, 3D anchors, and confidence for a keyframe."""
        if frame_id in self.frames.anchors:
            return tuple(value.to(self.device) for value in self.frames.anchors[frame_id])
        queries = self.features.extract(image, valid_mask)
        anchors = sample_map(points, queries)[0]
        weights = sample_map(confidence[..., None], queries)[0, :, 0]
        valid = (
            (weights > PI3_VALID_CONFIDENCE) & torch.isfinite(anchors).all(-1)
            & (anchors[:, 2] > 0)
        )
        # The SIFT row index is retained as the feature identity. Reusing it in
        # later windows makes (reference frame, feature ID) a stable graph key.
        keys = torch.arange(queries.shape[0], device=self.device, dtype=torch.long)[valid]
        return keys, queries[valid], anchors[valid], weights[valid]

    def _factors(
        self,
        output,
        frame_ids,
        reference,
        keys,
        queries,
        anchors,
        anchor_weights,
        track_valid,
    ):
        """Convert one tracks result into observations and weighted pose-graph edges."""
        tracks, scores = output["tracks"], output["confidence"]
        valid = track_valid.clone()
        ref_id = int(frame_ids[reference])
        fresh = torch.tensor(
            [
                local == reference
                or (ref_id, int(frame_id)) not in self.processed_pairs
                for local, frame_id in enumerate(frame_ids)
            ],
            device=self.device, dtype=torch.bool,
        )
        valid &= fresh[:, None]
        # Anchors were validated when first created and remain the identity of
        # a persistent point even if Pi3 confidence changes in a later window.
        valid[reference] = True
        # A point may enter the persistent graph from a partial window, while
        # pose-graph edges count only tracks observed in at least three views.
        complete = valid.sum(0) >= 3
        target_valid = valid.clone()
        target_valid[reference] = False
        keep = target_valid.any(0)
        if not keep.any():
            return None
        keys, queries = keys[keep], queries[keep]
        anchors, anchor_weights = anchors[keep], anchor_weights[keep]
        tracks, scores = tracks[:, keep], scores[:, keep]
        valid, complete = valid[:, keep], complete[keep]
        point_index = torch.arange(keys.numel(), device=self.device)
        obs_frames = [torch.full_like(point_index, ref_id)]
        obs_points = [point_index]
        obs_uv = [queries]
        obs_weights = [anchor_weights]
        edge_source, edge_target, edge_weight = [], [], []
        for target, target_id in enumerate(frame_ids):
            if target == reference:
                continue
            self.processed_pairs.add((ref_id, int(target_id)))
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
            obs_weights.append(anchor_weights[mask] * scores[target, mask])
            graph_mask = mask & complete
            if graph_mask.any():
                edge_source.append(ref_id)
                edge_target.append(int(target_id))
                edge_weight.append(int(graph_mask.sum()))
        return {
            "keys": keys, "anchors": anchors,
            "obs_frames": torch.cat(obs_frames), "obs_points": torch.cat(obs_points),
            "obs_uv": torch.cat(obs_uv), "obs_weights": torch.cat(obs_weights),
            "edge_source": edge_source, "edge_target": edge_target, "edge_weight": edge_weight,
        }

    @torch.no_grad()
    def track(self, frame_ids):
        """Infer one window and return a factor packet without optimizing state."""
        images = self.dataset.read(frame_ids).to(self.device)
        model_images = images
        if self.frames.valid_mask is not None:
            model_images = images * self.frames.valid_mask.to(self.device)[None, None]
        # Geometry is decoded once. The selected tracks frontend then consumes
        # either the shared Glob3R state or its own VGGSfM feature pyramid.
        geometry, tracks_state = self.geometry_model.infer_window(model_images.unsqueeze(0))
        self.tracks_model.prepare_window(model_images.unsqueeze(0), tracks_state)
        points = geometry["local_points"].squeeze(0)
        confidence = geometry["conf"].squeeze(0).sigmoid().squeeze(-1)
        valid_mask = (
            torch.isfinite(points).all(-1)
            & torch.isfinite(confidence)
            & (points[..., 2] > 0)
            & (confidence > PI3_VALID_CONFIDENCE)
        )
        if self.frames.valid_mask is not None:
            valid_mask &= self.frames.valid_mask.to(self.device)[None]
        valid_counts = valid_mask.flatten(1).sum(-1)
        if (valid_counts == 0).any():
            local = int(torch.nonzero(valid_counts == 0, as_tuple=False)[0])
            raise RuntimeError(f"frame {frame_ids[local]} has no valid Pi3 geometry")
        confidence = torch.where(valid_mask, confidence, 0)
        poses = geometry["camera_poses"].squeeze(0)
        poses = torch.linalg.inv(poses[0])[None] @ poses
        K = self.frames.K.to(self.device).expand(len(frame_ids), -1, -1).clone()
        track_cache = {}
        # Switch the keyframe policy by commenting one call and enabling the other.
        # keyframes = self._keyframes_projection_coverage(frame_ids, points, poses, K, valid_mask)
        keyframes = self._keyframes_mast3r_fusion(
            frame_ids, images, points, confidence, valid_mask, track_cache
        )
        parts = []
        for reference in keyframes:
            frame_id = int(frame_ids[reference])
            tracked = self._reference_tracks(
                frame_ids, images, points, confidence, valid_mask, reference, track_cache
            )
            self.frames.add_keyframe(
                frame_id, images[reference], points[reference], confidence[reference],
                tracked["keys"], tracked["queries"], tracked["anchors"], tracked["weights"],
            )
            already_processed = all(
                (frame_id, int(target)) in self.processed_pairs
                for target in frame_ids
                if target != frame_id
            )
            if already_processed:
                continue
            part = self._factors(
                tracked["output"], frame_ids, reference,
                tracked["keys"], tracked["queries"], tracked["anchors"], tracked["weights"],
                tracked["valid"],
            )
            if part is not None:
                part["reference"] = frame_id
                parts.append(part)
        # Packets contain measurements only. FactorGraph owns persistent state
        # and the caller decides when to create a local or global graph view.
        packet = {
            "frame_ids": torch.tensor(frame_ids, device=self.device),
            "keyframes": torch.tensor(
                [int(frame_ids[reference]) for reference in keyframes],
                device=self.device,
                dtype=torch.long,
            ),
            "poses": torch.linalg.inv(poses), "K": K, "parts": parts, "edges": [],
        }
        local = {int(frame_id): i for i, frame_id in enumerate(frame_ids)}
        for part in parts:
            edges = zip(
                part["edge_source"], part["edge_target"], part["edge_weight"]
            )
            for source, target, weight in edges:
                relative = packet["poses"][local[target]] @ torch.linalg.inv(
                    packet["poses"][local[source]]
                )
                packet["edges"].append((source, target, relative, weight))
        return packet


__all__ = [
    "FrameStore",
    "KEYFRAME_MATCH_COVERAGE_RATIO",
    "KEYFRAME_PROJECTION_RATIO",
    "PI3_VALID_CONFIDENCE",
    "WindowTracker",
]
