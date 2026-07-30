"""Track association and pose-graph construction for local Glob3R SfM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch

from pi3.models.glob3r.geometry import sample_map_at_pixels


@dataclass
class PoseGraph:
    source: list[int] = field(default_factory=list)
    target: list[int] = field(default_factory=list)
    target_from_source: list[torch.Tensor] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)


@dataclass
class TrackObservation:
    track: int
    frame: int
    pixel: torch.Tensor
    confidence: torch.Tensor
    predicted_depth: torch.Tensor


@dataclass
class Tracks:
    """Multi-view track observations consumed by motion averaging and BA."""

    observations: torch.Tensor
    camera_indices: torch.Tensor
    point_indices: torch.Tensor
    confidence: torch.Tensor
    predicted_depth: torch.Tensor
    intrinsics: torch.Tensor


def _confidence_probability(
    confidence: Optional[torch.Tensor], shape, device, dtype
) -> torch.Tensor:
    if confidence is None:
        return torch.ones(shape, device=device, dtype=dtype)
    confidence = confidence.squeeze(-1)
    # Pi3/Pi3X confidence heads are trained as logits. Appendix C.1's
    # threshold is applied to the resulting confidence probability C_i.
    return confidence.sigmoid()


class _TrackAccumulator:
    def __init__(self, config, intrinsics: torch.Tensor):
        self.config = config
        self.intrinsics = intrinsics
        self.device, self.dtype = intrinsics.device, intrinsics.dtype
        self.samples: Dict[int, torch.Tensor] = {}
        self.track_ids: Dict[int, torch.Tensor] = {}
        self.observations: list[TrackObservation] = []
        self.seen: set[tuple[int, int]] = set()
        self.next_track = 0
        self.graph = PoseGraph()
        self.generator = torch.Generator(device=self.device).manual_seed(
            config.random_seed
        )

    def _sample_keyframe(self, frame: int, confidence: torch.Tensor) -> torch.Tensor:
        if frame in self.samples:
            return self.samples[frame]
        candidates = (
            confidence > self.config.depth_confidence_threshold
        ).nonzero(as_tuple=False)
        if candidates.numel() == 0:
            raise RuntimeError(
                f"keyframe {frame} has no pixels above depth confidence threshold"
            )
        count = min(self.config.tracking_points_per_keyframe, candidates.shape[0])
        order = torch.randperm(
            candidates.shape[0], generator=self.generator, device=self.device
        )[:count]
        yx = candidates[order]
        xy = yx[:, [1, 0]].to(self.dtype)
        self.samples[frame] = xy
        ids = torch.arange(
            self.next_track, self.next_track + count, device=self.device
        )
        self.next_track += count
        self.track_ids[frame] = ids
        return xy

    def _add_observation(
        self, track, frame, pixel, confidence, predicted_depth
    ) -> None:
        key = (int(track), int(frame))
        if key in self.seen:
            return
        self.seen.add(key)
        self.observations.append(
            TrackObservation(int(track), int(frame), pixel, confidence, predicted_depth)
        )

    def add_window(
        self,
        global_frames: Sequence[int],
        geometry: dict,
        matches: dict,
        metric_scale: torch.Tensor,
    ) -> None:
        local_points = geometry["local_points"][0] * metric_scale
        raw_confidence = geometry.get("conf")
        confidence = _confidence_probability(
            None if raw_confidence is None else raw_confidence[0],
            local_points.shape[:3],
            local_points.device,
            local_points.dtype,
        )
        camera_to_world = geometry["camera_poses"][0].clone()
        camera_to_world[:, :3, 3] *= metric_scale
        height, width = local_points.shape[1:3]

        for reference_local, matching in matches.items():
            reference_global = int(global_frames[reference_local])
            sample_xy = self._sample_keyframe(
                reference_global, confidence[reference_local]
            )
            track_ids = self.track_ids[reference_global]
            sample_x = sample_xy[:, 0].round().long().clamp(0, width - 1)
            sample_y = sample_xy[:, 1].round().long().clamp(0, height - 1)
            reference_confidence = confidence[reference_local, sample_y, sample_x]
            reference_depth = local_points[reference_local, sample_y, sample_x, 2]
            for track, pixel, conf, depth in zip(
                track_ids, sample_xy, reference_confidence, reference_depth
            ):
                self._add_observation(
                    track, reference_global, pixel, conf, depth
                )

            final_warp = (
                matching.warp_stages[-1]
                if matching.warp_stages
                else matching.coarse_warp
            )
            final_confidence = (
                matching.confidence_stages[-1]
                if matching.confidence_stages
                else matching.coarse_confidence
            )
            for target_offset, target_local in enumerate(matching.target_indices):
                target_global = int(global_frames[target_local])
                warp = final_warp[0, target_offset]
                warp_confidence = final_confidence[
                    0, target_offset, 0, sample_y, sample_x
                ]
                target_xy = warp[:, sample_y, sample_x].transpose(0, 1)
                target_depth = sample_map_at_pixels(
                    local_points[target_local, ..., 2][None, None],
                    target_xy[None, :, None],
                ).reshape(-1)
                combined_confidence = reference_confidence * warp_confidence
                valid = (
                    (warp_confidence >= self.config.warp_confidence_threshold)
                    & (target_xy[:, 0] >= 0)
                    & (target_xy[:, 0] <= width - 1)
                    & (target_xy[:, 1] >= 0)
                    & (target_xy[:, 1] <= height - 1)
                    & torch.isfinite(target_xy).all(dim=-1)
                    & (target_depth > 0)
                )
                for index in valid.nonzero(as_tuple=False).flatten():
                    self._add_observation(
                        track_ids[index],
                        target_global,
                        target_xy[index],
                        combined_confidence[index],
                        target_depth[index],
                    )
                valid_count = int(valid.sum())
                if valid_count:
                    relative = (
                        torch.linalg.inv(camera_to_world[target_local])
                        @ camera_to_world[reference_local]
                    )
                    self.graph.source.append(reference_global)
                    self.graph.target.append(target_global)
                    self.graph.target_from_source.append(relative)
                    self.graph.weight.append(float(valid_count))

    def build(self) -> Tracks:
        counts: Dict[int, int] = {}
        for observation in self.observations:
            counts[observation.track] = counts.get(observation.track, 0) + 1
        retained = sorted(track for track, count in counts.items() if count >= 2)
        remap = {track: index for index, track in enumerate(retained)}
        observations = [
            observation
            for observation in self.observations
            if observation.track in remap
        ]
        if not observations:
            raise RuntimeError("no multi-view tracks survived confidence filtering")
        return Tracks(
            observations=torch.stack(
                [observation.pixel for observation in observations]
            ),
            camera_indices=torch.tensor(
                [observation.frame for observation in observations],
                device=self.device,
            ),
            point_indices=torch.tensor(
                [remap[observation.track] for observation in observations],
                device=self.device,
            ),
            confidence=torch.stack(
                [observation.confidence for observation in observations]
            ).to(self.dtype),
            predicted_depth=torch.stack(
                [observation.predicted_depth for observation in observations]
            ).to(self.dtype),
            intrinsics=self.intrinsics,
        )


__all__ = [
    "PoseGraph",
    "TrackObservation",
    "Tracks",
    "_TrackAccumulator",
    "_confidence_probability",
]
