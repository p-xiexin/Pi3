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
        keyframe_count = keyframes.size
        pair_count = keyframe_count * frame_count
        keyframe_indices = torch.arange(frame_count, device=device)[keyframes]
        height, width = Xs.shape[1:3]
        pixel_count = height * width

        # Eq. (2) predicts each selected reference against the complete window.
        # The dense correspondence tensors are tracker-local intermediates;
        # only the sparse observations assembled below become persistent Tracks.
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
        for reference_row, reference_index in enumerate(keyframes):
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

        # [K, N, M, ...] -> [K*N, M, ...]
        idx_i2j = idx_i2j.reshape(pair_count, pixel_count)
        valid_match_j = valid_match_j.reshape(pair_count, pixel_count, 1)
        Qii = Qii.reshape(pair_count, pixel_count, 1)
        Qji = Qji.reshape(pair_count, pixel_count, 1)

        # Symmetric fields are available only when the target is also a
        # keyframe with its own reference-to-window Eq. (2) prediction.
        idx_j2i = torch.zeros_like(idx_i2j)
        valid_match_i = torch.zeros_like(valid_match_j)
        reference_frames = keyframe_indices[:, None].expand(-1, frame_count)
        target_frames = torch.arange(frame_count, device=device).expand_as(
            reference_frames
        )
        keyframe_rows = torch.full(
            (frame_count,), -1, device=device, dtype=torch.long
        )
        keyframe_rows[keyframe_indices] = torch.arange(keyframe_count, device=device)
        reverse_rows = keyframe_rows[target_frames]
        has_reverse = (reference_frames != target_frames) & (reverse_rows >= 0)
        forward_pair = (
            torch.arange(pair_count, device=device)
            .reshape(keyframe_count, frame_count)[has_reverse]
        )
        reverse_pair = (
            reverse_rows[has_reverse] * frame_count + reference_frames[has_reverse]
        )
        idx_j2i[forward_pair] = idx_i2j[reverse_pair]
        valid_match_i[forward_pair] = valid_match_j[reverse_pair]

        candidate_mask = (
            Cs[keyframe_indices].flatten(1) > self.config.depth_confidence_threshold
        )
        if not candidate_mask.any(dim=1).all():
            invalid_references = keyframes[~candidate_mask.any(dim=1)]
            raise RuntimeError(
                f"keyframes {invalid_references} have no pixels above depth confidence threshold"
            )
        generator = torch.Generator(device=device).manual_seed(
            self.config.random_seed
        )
        sample_count = min(
            self.config.tracking_points_per_keyframe, height * width
        )
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

        directed = keyframe_indices[:, None] != target_frames
        target_frames = target_frames[directed]
        pair_reference = (
            torch.arange(keyframe_count, device=device)[:, None]
            .expand(keyframe_count, frame_count)[directed]
        )
        matching_pair = (
            torch.arange(pair_count, device=device)
            .reshape(keyframe_count, frame_count)[directed]
        )
        pair_sample_index = sample_linear_index[pair_reference]

        pair_idx_i2j = idx_i2j[matching_pair]
        pair_valid_match_j = valid_match_j[matching_pair].squeeze(dim=-1)
        pair_Qii = Qii[matching_pair].squeeze(dim=-1)
        pair_Qji = Qji[matching_pair].squeeze(dim=-1)
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
        reverse_linear_index = idx_j2i[matching_pair].gather(1, target_linear_index)
        reverse_valid = valid_match_i[matching_pair].squeeze(dim=-1).gather(
            1, target_linear_index
        )
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
        observations = observations[retained_observations]
        observation_camera = observation_camera[retained_observations]
        observation_point = track_remap[
            observation_track[retained_observations]
        ]
        tracking_confidence = tracking_confidence[retained_observations]
        predicted_depth = predicted_depth[retained_observations]
        tracks = Tracks(
            observations=observations,
            camera_indices=observation_camera,
            point_indices=observation_point,
            confidence=tracking_confidence,
            predicted_depth=predicted_depth,
            intrinsics=intrinsics,
        )

        reference_frames = reference_frames[directed]
        forward_index = idx_i2j[matching_pair]
        forward_valid = valid_match_j[matching_pair].squeeze(dim=-1)
        target_confidence = Cs[target_frames].flatten(1).gather(1, forward_index)
        reference_confidence = Cs[reference_frames].flatten(1)
        forward_confidence = torch.sqrt(
            Qii[matching_pair].squeeze(dim=-1).gather(1, forward_index)
            * Qji[matching_pair].squeeze(dim=-1)
        )
        reverse_index = idx_j2i[matching_pair].gather(1, forward_index)
        reverse_valid = valid_match_i[matching_pair].squeeze(dim=-1).gather(
            1, forward_index
        )
        reference_pixel = torch.arange(
            forward_index.shape[1], device=device
        ).expand_as(forward_index)
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
        motion = opt_pose_ray(
            rotations,
            rays,
            tracks.camera_indices,
            tracks.point_indices,
            tracks.confidence,
            initial_centers,
            tracks.predicted_depth,
            iterations=self.config.translation_iterations,
        )
        world_to_camera = torch.eye(4, device=device, dtype=dtype).repeat(
            frame_count, 1, 1
        )
        world_to_camera[:, :3, :3] = rotations
        world_to_camera[:, :3, 3] = -torch.einsum(
            "nij,nj->ni", rotations, motion.camera_centers
        )
        return TrackerResult(
            world_to_camera=world_to_camera,
            points_3d=motion.points_3d,
            tracks=tracks,
            objective=motion.objective,
        )


__all__ = ["Tracker", "TrackerResult", "Tracks"]
