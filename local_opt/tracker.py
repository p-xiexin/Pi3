"""Glob3R matching post-processing, track construction, and motion averaging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .optimization import (
    maximum_spanning_tree_initialization,
    opt_pose_ray,
    robust_rotation_averaging,
)


@dataclass
class Tracks:
    """Sparse camera-point observations shared by paper Eqs. (5) and (6)."""

    observations: torch.Tensor
    camera_indices: torch.Tensor
    point_indices: torch.Tensor
    confidence: torch.Tensor
    predicted_depth: torch.Tensor
    intrinsics: torch.Tensor


@dataclass
class TrackerResult:
    """Tracks and the Eq. (5) initialization consumed by bundle adjustment."""

    world_to_camera: torch.Tensor
    points_3d: torch.Tensor
    tracks: Tracks
    objective: torch.Tensor


class Tracker:
    """Convert dense Eq. (2) matches into sparse tracks and solve Eq. (5)."""

    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config

    @torch.no_grad()
    def __call__(
        self,
        patch_tokens: torch.Tensor,
        encoder_features: list[torch.Tensor],
        window_images: torch.Tensor,
        Xs: torch.Tensor,
        Cs: torch.Tensor,
        T_CWs: torch.Tensor,
        ss: torch.Tensor,
        intrinsics: torch.Tensor,
        keyframes: np.ndarray,
        matching_callback: Optional[
            Callable[[torch.Tensor, Mapping[int, object]], None]
        ] = None,
    ) -> TrackerResult:
        """Build tracks and solve the unchanged Glob3R Eq. (5) objective."""

        device, dtype = Xs.device, Xs.dtype
        frame_count = Xs.shape[0]
        keyframe_indices = torch.arange(frame_count, device=device)[keyframes]
        (
            idx_i2j,
            valid_match_j,
            Qii,
            Qji,
            idx_j2i,
            valid_match_i,
        ) = self._match_keyframes(
            patch_tokens,
            encoder_features,
            window_images,
            Xs,
            keyframe_indices,
            matching_callback,
        )
        tracks = self._build_tracks(
            idx_i2j,
            valid_match_j,
            Qii,
            Qji,
            idx_j2i,
            valid_match_i,
            Xs,
            Cs,
            ss,
            intrinsics,
            keyframe_indices,
        )
        edge_source, edge_target, edge_transform, edge_weight = (
            self._build_pose_graph(
                idx_i2j,
                valid_match_j,
                Qii,
                Qji,
                idx_j2i,
                valid_match_i,
                Cs,
                T_CWs,
                ss,
                keyframe_indices,
            )
        )
        rotations, initial_centers, rays, points_3d = (
            self._initialize_motion(
                tracks,
                frame_count,
                edge_source,
                edge_target,
                edge_transform,
                edge_weight,
            )
        )

        camera_centers = initial_centers
        motion_objective = torch.full((), float("nan"), device=device, dtype=dtype)

        # Optional Glob3R Eq. (5) pose-ray optimization. Comment out this block
        # to pass the graph initialization directly to bundle adjustment.
        # motion = opt_pose_ray(
        #     rotations,
        #     rays,
        #     tracks.camera_indices,
        #     tracks.point_indices,
        #     tracks.confidence,
        #     initial_centers,
        #     points_3d,
        #     tracks.predicted_depth,
        #     iterations=self.config.translation_iterations,
        # )
        # camera_centers = motion.camera_centers
        # points_3d = motion.points_3d
        # motion_objective = motion.objective

        world_to_camera = torch.eye(4, device=device, dtype=dtype).repeat(
            frame_count, 1, 1
        )
        world_to_camera[:, :3, :3] = rotations
        world_to_camera[:, :3, 3] = -torch.einsum(
            "nij,nj->ni", rotations, camera_centers
        )
        return TrackerResult(
            world_to_camera=world_to_camera,
            points_3d=points_3d,
            tracks=tracks,
            objective=motion_objective,
        )

    def _match_keyframes(
        self,
        patch_tokens: torch.Tensor,
        encoder_features: list[torch.Tensor],
        window_images: torch.Tensor,
        Xs: torch.Tensor,
        keyframe_indices: torch.Tensor,
        matching_callback: Optional[
            Callable[[torch.Tensor, Mapping[int, object]], None]
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run Eq. (2) and organize dense forward and reciprocal matches."""

        device, dtype = Xs.device, Xs.dtype
        frame_count, height, width = Xs.shape[:3]
        keyframe_count = keyframe_indices.numel()
        pixel_count = height * width
        idx_i2j = torch.zeros(
            keyframe_count, frame_count, pixel_count, device=device, dtype=torch.long
        )
        valid_match_j = torch.zeros(
            keyframe_count,
            frame_count,
            pixel_count,
            1,
            device=device,
            dtype=torch.bool,
        )
        Qji = torch.zeros(
            keyframe_count,
            frame_count,
            pixel_count,
            1,
            device=device,
            dtype=dtype,
        )
        Qii = torch.zeros_like(Qji)
        matching_outputs: dict[int, object] = {}
        for reference_row, reference_index in enumerate(keyframe_indices):
            output = self.model.match_pair(
                patch_tokens,
                encoder_features,
                window_images,
                reference_index=int(reference_index),
            )
            if matching_callback is not None:
                matching_outputs[int(reference_index)] = output

            warp = output.warp_stages[-1] if output.warp_stages else output.coarse_warp
            confidence = (
                output.confidence_stages[-1]
                if output.confidence_stages
                else output.coarse_confidence
            )
            # [B, T, 2, H, W] -> [T, 2, H, W]
            warp = warp.squeeze(dim=0)
            # [B, T, 1, H, W] -> [T, 1, H, W]
            confidence = confidence.squeeze(dim=0)
            # [T, 1, H, W] -> [T, H, W]
            confidence = confidence.squeeze(dim=1)

            target_indices = torch.as_tensor(
                output.target_indices, device=device, dtype=torch.long
            )
            target_x = warp[:, 0].round().long().clamp(0, width - 1)
            target_y = warp[:, 1].round().long().clamp(0, height - 1)
            idx_i2j[reference_row, target_indices] = (
                target_y * width + target_x
            ).flatten(1)
            valid_match_j[reference_row, target_indices, :, 0] = (
                torch.isfinite(warp).all(dim=1)
                & (warp[:, 0] >= 0)
                & (warp[:, 0] <= width - 1)
                & (warp[:, 1] >= 0)
                & (warp[:, 1] <= height - 1)
            ).flatten(1)
            Qji[reference_row, target_indices, :, 0] = confidence.flatten(1)

            # Qji is predicted on the reference grid. Qii stores the same
            # confidence on the rounded target grid for symmetric filtering.
            Qii[reference_row].scatter_reduce_(
                1,
                idx_i2j[reference_row].unsqueeze(dim=-1),
                Qji[reference_row],
                reduce="amax",
                include_self=True,
            )

        if matching_callback is not None:
            # [B, N, 3, H, W] -> [N, 3, H, W]
            matching_callback(window_images.squeeze(dim=0), matching_outputs)

        # Reciprocal fields exist only when the target is itself a keyframe.
        idx_j2i = torch.zeros_like(idx_i2j)
        valid_match_i = torch.zeros_like(valid_match_j)
        keyframe_rows = torch.full(
            (frame_count,), -1, device=device, dtype=torch.long
        )
        keyframe_rows[keyframe_indices] = torch.arange(keyframe_count, device=device)
        reference_frames = keyframe_indices[:, None].expand(-1, frame_count)
        target_frames = torch.arange(frame_count, device=device).expand_as(
            reference_frames
        )
        reverse_rows = keyframe_rows[target_frames]
        has_reverse = (reference_frames != target_frames) & (reverse_rows >= 0)
        idx_j2i[has_reverse] = idx_i2j[
            reverse_rows[has_reverse], reference_frames[has_reverse]
        ]
        valid_match_i[has_reverse] = valid_match_j[
            reverse_rows[has_reverse], reference_frames[has_reverse]
        ]
        return idx_i2j, valid_match_j, Qii, Qji, idx_j2i, valid_match_i

    def _build_tracks(
        self,
        idx_i2j: torch.Tensor,
        valid_match_j: torch.Tensor,
        Qii: torch.Tensor,
        Qji: torch.Tensor,
        idx_j2i: torch.Tensor,
        valid_match_i: torch.Tensor,
        Xs: torch.Tensor,
        Cs: torch.Tensor,
        ss: torch.Tensor,
        intrinsics: torch.Tensor,
        keyframe_indices: torch.Tensor,
    ) -> Tracks:
        """Convert dense Eq. (2) predictions into multi-view observations."""

        device, dtype = Xs.device, Xs.dtype
        frame_count, height, width = Xs.shape[:3]
        keyframe_count = keyframe_indices.numel()
        pixel_count = height * width
        candidate_mask = (
            Cs[keyframe_indices].flatten(1) > self.config.depth_confidence_threshold
        )
        if not candidate_mask.any(dim=1).all():
            invalid_references = keyframe_indices[
                ~candidate_mask.any(dim=1)
            ].tolist()
            raise RuntimeError(
                f"keyframes {invalid_references} have no pixels above depth confidence threshold"
            )
        generator = torch.Generator(device=device).manual_seed(
            self.config.random_seed
        )
        sample_count = min(self.config.tracking_points_per_keyframe, pixel_count)
        sample_score = torch.rand(
            candidate_mask.shape, device=device, generator=generator
        ).masked_fill(~candidate_mask, -1)
        sampled_reference_pixels = sample_score.topk(sample_count, dim=1).indices
        sampled_valid = candidate_mask.gather(1, sampled_reference_pixels)
        sampled_reference_pixels = sampled_reference_pixels.masked_fill(
            ~sampled_valid, -1
        )

        Xs_scaled = Xs * ss
        sample_valid = sampled_reference_pixels >= 0
        sample_linear_index = sampled_reference_pixels.clamp_min(0)
        sample_x = sample_linear_index.remainder(width)
        sample_y = torch.div(sample_linear_index, width, rounding_mode="floor")
        sample_xy = torch.stack((sample_x, sample_y), dim=-1).to(dtype)
        track_ids = torch.arange(
            keyframe_count * sample_count, device=device
        ).reshape(keyframe_count, sample_count)
        keyframe_confidence = Cs[keyframe_indices[:, None], sample_y, sample_x]
        reference_depth = Xs_scaled[
            keyframe_indices[:, None], sample_y, sample_x, 2
        ]

        reference_frames = keyframe_indices[:, None].expand(-1, frame_count)
        target_frames = torch.arange(frame_count, device=device).expand_as(
            reference_frames
        )
        directed = reference_frames != target_frames
        target_frames = target_frames[directed]
        pair_reference = (
            torch.arange(keyframe_count, device=device)[:, None]
            .expand(keyframe_count, frame_count)[directed]
        )
        pair_sample_index = sample_linear_index[pair_reference]
        pair_idx_i2j = idx_i2j[directed]
        pair_valid_match_j = valid_match_j[directed].squeeze(dim=-1)
        pair_Qii = Qii[directed].squeeze(dim=-1)
        pair_Qji = Qji[directed].squeeze(dim=-1)
        target_point_maps = Xs[target_frames].flatten(1, 2)
        target_confidence_maps = Cs[target_frames].flatten(1)
        reference_confidence_maps = Cs[
            keyframe_indices[pair_reference]
        ].flatten(1)

        target_linear_index = pair_idx_i2j.gather(1, pair_sample_index)
        target_x = target_linear_index.remainder(width)
        target_y = torch.div(target_linear_index, width, rounding_mode="floor")
        target_xy = torch.stack((target_x, target_y), dim=-1).to(dtype)
        pair_valid = pair_valid_match_j.gather(1, pair_sample_index)
        target_points = target_point_maps.gather(
            1, target_linear_index.unsqueeze(dim=-1).expand(-1, -1, 3)
        )
        target_confidence = target_confidence_maps.gather(1, target_linear_index)
        target_match_confidence = pair_Qii.gather(1, target_linear_index)
        reference_confidence = reference_confidence_maps.gather(
            1, pair_sample_index
        )
        reference_match_confidence = pair_Qji.gather(1, pair_sample_index)
        match_confidence = torch.sqrt(
            target_match_confidence * reference_match_confidence
        )
        reverse_linear_index = idx_j2i[directed].gather(1, target_linear_index)
        reverse_valid = valid_match_i[directed].squeeze(dim=-1).gather(
            1, target_linear_index
        )
        keyframe_rows = torch.full(
            (frame_count,), -1, device=device, dtype=torch.long
        )
        keyframe_rows[keyframe_indices] = torch.arange(keyframe_count, device=device)
        has_reverse = keyframe_rows[target_frames] >= 0
        reciprocal = (~has_reverse[:, None]) | (
            reverse_valid & (reverse_linear_index == pair_sample_index)
        )
        target_depth = target_points[..., 2] * ss
        target_valid = (
            sample_valid[pair_reference]
            & pair_valid
            & reciprocal
            & (target_confidence > self.config.depth_confidence_threshold)
            & (reference_confidence > self.config.depth_confidence_threshold)
            & (match_confidence >= self.config.warp_confidence_threshold)
            & (target_depth > 0)
        )

        reference_camera = keyframe_indices[:, None].expand_as(track_ids)
        target_camera = target_frames[:, None].expand_as(target_linear_index)
        observation_track = torch.cat(
            (track_ids[sample_valid], track_ids[pair_reference][target_valid])
        )
        observations = torch.cat(
            (sample_xy[sample_valid], target_xy[target_valid])
        )
        observation_camera = torch.cat(
            (reference_camera[sample_valid], target_camera[target_valid])
        )
        tracking_confidence = torch.cat(
            (keyframe_confidence[sample_valid], match_confidence[target_valid])
        )
        predicted_depth = torch.cat(
            (reference_depth[sample_valid], target_depth[target_valid])
        )

        observation_count = torch.bincount(
            observation_track, minlength=keyframe_count * sample_count
        )
        retained_tracks = observation_count >= 2
        retained_observations = retained_tracks[observation_track]
        if not retained_observations.any():
            raise RuntimeError("no multi-view tracks survived confidence filtering")
        track_remap = retained_tracks.cumsum(dim=0) - 1
        return Tracks(
            observations=observations[retained_observations],
            camera_indices=observation_camera[retained_observations],
            point_indices=track_remap[
                observation_track[retained_observations]
            ],
            confidence=tracking_confidence[retained_observations],
            predicted_depth=predicted_depth[retained_observations],
            intrinsics=intrinsics,
        )

    def _build_pose_graph(
        self,
        idx_i2j: torch.Tensor,
        valid_match_j: torch.Tensor,
        Qii: torch.Tensor,
        Qji: torch.Tensor,
        idx_j2i: torch.Tensor,
        valid_match_i: torch.Tensor,
        Cs: torch.Tensor,
        T_CWs: torch.Tensor,
        ss: torch.Tensor,
        keyframe_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct the directed camera graph used to initialize Eq. (5)."""

        device, dtype = Cs.device, Cs.dtype
        frame_count = Cs.shape[0]
        keyframe_count = keyframe_indices.numel()
        reference_frames = keyframe_indices[:, None].expand(-1, frame_count)
        target_frames = torch.arange(frame_count, device=device).expand_as(
            reference_frames
        )
        directed = reference_frames != target_frames
        reference_frames = reference_frames[directed]
        target_frames = target_frames[directed]
        forward_index = idx_i2j[directed]
        forward_valid = valid_match_j[directed].squeeze(dim=-1)
        target_confidence = Cs[target_frames].flatten(1).gather(1, forward_index)
        reference_confidence = Cs[reference_frames].flatten(1)
        forward_confidence = torch.sqrt(
            Qii[directed].squeeze(dim=-1).gather(1, forward_index)
            * Qji[directed].squeeze(dim=-1)
        )
        reverse_index = idx_j2i[directed].gather(1, forward_index)
        reverse_valid = valid_match_i[directed].squeeze(dim=-1).gather(
            1, forward_index
        )
        reference_pixel = torch.arange(
            forward_index.shape[1], device=device
        ).expand_as(forward_index)
        keyframe_rows = torch.full(
            (frame_count,), -1, device=device, dtype=torch.long
        )
        keyframe_rows[keyframe_indices] = torch.arange(keyframe_count, device=device)
        has_reverse = keyframe_rows[target_frames] >= 0
        reciprocal = (~has_reverse[:, None]) | (
            reverse_valid & (reverse_index == reference_pixel)
        )
        edge_weight = (
            forward_valid
            & reciprocal
            & (target_confidence > self.config.depth_confidence_threshold)
            & (reference_confidence > self.config.depth_confidence_threshold)
            & (forward_confidence >= self.config.warp_confidence_threshold)
        ).sum(dim=1).to(dtype)
        valid_edge = edge_weight > 0
        edge_source = reference_frames[valid_edge]
        edge_target = target_frames[valid_edge]
        edge_weight = edge_weight[valid_edge]
        if edge_source.numel() == 0:
            raise RuntimeError("no valid pair-match edges were constructed")

        T_CWs = T_CWs.clone()
        T_CWs[:, :3, 3] *= ss
        edge_transform = (
            torch.linalg.inv(T_CWs[edge_target]) @ T_CWs[edge_source]
        ).to(dtype)
        return edge_source, edge_target, edge_transform, edge_weight

    def _initialize_motion(
        self,
        tracks: Tracks,
        frame_count: int,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        edge_transform: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize rotations, centers, rays, and sparse points for Eq. (5)."""

        device = tracks.observations.device
        dtype = tracks.observations.dtype
        initial_world_to_camera = maximum_spanning_tree_initialization(
            frame_count, edge_source, edge_target, edge_transform, edge_weight
        )
        rotations = robust_rotation_averaging(
            initial_world_to_camera,
            edge_source,
            edge_target,
            edge_transform,
            edge_weight,
            iterations=self.config.rotation_iterations,
        )
        initial_centers = -torch.einsum(
            "nij,nj->ni",
            initial_world_to_camera[:, :3, :3].transpose(-1, -2),
            initial_world_to_camera[:, :3, 3],
        )
        homogeneous = F.pad(tracks.observations, (0, 1), value=1.0)
        rays = torch.einsum(
            "oij,oj->oi",
            torch.linalg.inv(tracks.intrinsics[tracks.camera_indices]),
            homogeneous,
        )
        world_rays = torch.einsum(
            "oij,oj->oi",
            rotations[tracks.camera_indices].transpose(-1, -2),
            rays,
        )
        point_count = int(tracks.point_indices.max()) + 1
        points_3d = torch.zeros(point_count, 3, device=device, dtype=dtype)
        points_3d.index_add_(
            0,
            tracks.point_indices,
            initial_centers[tracks.camera_indices]
            + tracks.predicted_depth[:, None] * world_rays,
        )
        observation_count = torch.bincount(
            tracks.point_indices, minlength=point_count
        ).to(dtype)
        points_3d /= observation_count[:, None]
        return rotations, initial_centers, rays, points_3d


__all__ = ["Tracker", "TrackerResult", "Tracks"]
