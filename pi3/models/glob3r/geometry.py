"""Geometry supervision and optimization objectives from Glob3R.

These functions deliberately implement the mathematical objectives without
depending on a particular SfM solver.  A server-side training/evaluation setup
can pass their residuals to PyTorch, Ceres, or another optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def _pixel_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    homogeneous = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    return homogeneous.unsqueeze(0).expand(batch, -1, -1, -1)


def _inside_image(xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return (
        (xy[..., 0] >= 0)
        & (xy[..., 0] <= width - 1)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] <= height - 1)
    )


def sample_map_at_pixels(value: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``[B,C,H,W]`` maps at pixel-space coordinates."""

    height, width = value.shape[-2:]
    gx = 2.0 * xy[..., 0] / max(width - 1, 1) - 1.0
    gy = 2.0 * xy[..., 1] / max(height - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


@dataclass
class WarpSupervision:
    warp: torch.Tensor
    confidence: torch.Tensor
    mask: torch.Tensor
    projected_depth: torch.Tensor


def build_ground_truth_warp(
    reference_depth: torch.Tensor,
    target_depth: torch.Tensor,
    reference_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    target_from_reference: torch.Tensor,
    depth_threshold: float = 0.05,
) -> WarpSupervision:
    """Build dense reference-to-target supervision (Glob3R Eqs. (26)-(30)).

    Args:
        reference_depth: ``[B,H,W]``.
        target_depth: ``[B,T,H,W]``.
        reference_intrinsics: ``[B,3,3]``.
        target_intrinsics: ``[B,T,3,3]``.
        target_from_reference: ``[B,T,4,4]`` camera transforms.
    """

    batch, targets, height, width = target_depth.shape
    pixels = _pixel_grid(batch, height, width, reference_depth.device, reference_depth.dtype)
    pixels_flat = pixels.reshape(batch, -1, 3)
    depth_flat = reference_depth.reshape(batch, -1, 1)

    # Glob3R Eq. (26): x_tilde^(a->b) = K_b(R_(a->b)K_a^-1 x_tilde^a z^a + t_(a->b)).
    rays = torch.einsum("bij,bmj->bmi", torch.linalg.inv(reference_intrinsics), pixels_flat)
    points_reference = rays * depth_flat
    rotation = target_from_reference[..., :3, :3]
    translation = target_from_reference[..., :3, 3]
    points_target = torch.einsum("btij,bmj->btmi", rotation, points_reference) + translation[:, :, None]
    homogeneous_target = torch.einsum("btij,btmj->btmi", target_intrinsics, points_target)

    # Glob3R Eq. (27): project homogeneous coordinates and extract target depth.
    projected_depth = points_target[..., 2]
    projected_xy = homogeneous_target[..., :2] / homogeneous_target[..., 2:3].clamp_min(1e-8)
    projected_xy = projected_xy.reshape(batch, targets, height, width, 2)
    projected_depth = projected_depth.reshape(batch, targets, height, width)

    # Glob3R Eq. (28): W*_(a->b)(x^a) = x^(a->b).
    ground_truth_warp = projected_xy

    inside = _inside_image(projected_xy, height, width)
    sampled_target_depth = sample_map_at_pixels(
        target_depth.reshape(batch * targets, 1, height, width),
        projected_xy.reshape(batch * targets, height, width, 2),
    ).reshape(batch, targets, height, width)
    positive_geometry = (projected_depth > 0) & (sampled_target_depth > 0)
    relative_depth_error = (
        (sampled_target_depth - projected_depth).abs() / sampled_target_depth.clamp_min(1e-8)
    )

    # Glob3R Eq. (29): positive confidence is in-bounds, positive-depth, and depth-consistent.
    confidence = inside & positive_geometry & (relative_depth_error < depth_threshold)

    # Glob3R Eq. (30): supervise valid in-bound projections and explicit out-of-bound negatives.
    reference_valid = reference_depth[:, None] > 0
    mask = reference_valid & ((inside & positive_geometry) | (~inside))
    return WarpSupervision(ground_truth_warp, confidence, mask, projected_depth)


def relative_camera_transform(world_from_camera: torch.Tensor, reference_index: int = 0) -> torch.Tensor:
    """Return target-from-reference transforms for c2w input poses."""

    reference = world_from_camera[:, reference_index]
    target_indices = [i for i in range(world_from_camera.shape[1]) if i != reference_index]
    targets = world_from_camera[:, target_indices]
    return torch.linalg.inv(targets) @ reference[:, None]


# ================================================================
# Inference-only global SfM utilities (Glob3R Eqs. 4-6).
# These functions are kept for correspondence with the paper
# and are not used when training the matching head.
# ================================================================


def keyframe_projection_count(
    point_map: torch.Tensor,
    confidence: torch.Tensor,
    target_from_candidate: torch.Tensor,
    intrinsics: torch.Tensor,
    confidence_threshold: float,
) -> torch.Tensor:
    """Glob3R Eq. (4): maximum valid reprojected-pixel count over keyframes."""

    batch, keyframes = target_from_candidate.shape[:2]
    height, width = point_map.shape[-3:-1]
    points = point_map.reshape(batch, -1, 3)
    rotation = target_from_candidate[..., :3, :3]
    translation = target_from_candidate[..., :3, 3]
    projected_3d = torch.einsum("bkij,bmj->bkmi", rotation, points) + translation[:, :, None]
    homogeneous = torch.einsum("bkij,bkmj->bkmi", intrinsics, projected_3d)
    xy = homogeneous[..., :2] / homogeneous[..., 2:3].clamp_min(1e-8)
    valid = (
        _inside_image(xy, height, width)
        & (projected_3d[..., 2] > 0)
        & (confidence.reshape(batch, 1, -1) > confidence_threshold)
    )
    return valid.sum(dim=-1).max(dim=1).values


def robust_penalty(squared_residual: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber-style rho used by the robust objectives in Glob3R Eqs. (5)-(6)."""

    residual = squared_residual.clamp_min(0).sqrt()
    return torch.where(residual <= delta, 0.5 * squared_residual, delta * (residual - 0.5 * delta))


def motion_averaging_objective(
    camera_centers: torch.Tensor,
    points_3d: torch.Tensor,
    ray_depths: torch.Tensor,
    rotations: torch.Tensor,
    normalized_rays: torch.Tensor,
    observation_camera: torch.Tensor,
    observation_point: torch.Tensor,
    tracking_confidence: torch.Tensor,
    robust_delta: float = 1.0,
) -> torch.Tensor:
    """Glob3R Eq. (5): confidence-weighted multi-view ray-consistency objective."""

    centers = camera_centers[observation_camera]
    points = points_3d[observation_point]
    world_rays = torch.einsum(
        "oij,oj->oi", rotations[observation_camera].transpose(-1, -2), normalized_rays
    )
    predicted_points = centers + ray_depths[:, None] * world_rays
    squared = (points - predicted_points).square().sum(dim=-1)
    return (tracking_confidence * robust_penalty(squared, robust_delta)).sum()


def project_with_distortion(
    camera_points: torch.Tensor,
    intrinsics: torch.Tensor,
    distortion: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Projection pi(K, delta, T, X) appearing in Glob3R Eq. (6)."""

    xy = camera_points[..., :2] / camera_points[..., 2:3].clamp_min(1e-8)
    if distortion is not None:
        k1, k2, p1, p2 = distortion.unbind(dim=-1)
        x, y = xy.unbind(dim=-1)
        r2 = x.square() + y.square()
        radial = 1 + k1 * r2 + k2 * r2.square()
        xy = torch.stack(
            (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x.square()),
             y * radial + p1 * (r2 + 2 * y.square()) + 2 * p2 * x * y),
            dim=-1,
        )
    homogeneous = torch.einsum("...ij,...j->...i", intrinsics, F.pad(xy, (0, 1), value=1.0))
    return homogeneous[..., :2] / homogeneous[..., 2:3].clamp_min(1e-8)


def bundle_adjustment_objective(
    world_points: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    observations: torch.Tensor,
    observation_camera: torch.Tensor,
    observation_point: torch.Tensor,
    tracking_confidence: torch.Tensor,
    distortion: Optional[torch.Tensor] = None,
    robust_delta: float = 1.0,
) -> torch.Tensor:
    """Glob3R Eq. (6): confidence-weighted robust reprojection BA objective."""

    transform = world_to_camera[observation_camera]
    point = world_points[observation_point]
    camera_point = torch.einsum("oij,oj->oi", transform[..., :3, :3], point) + transform[..., :3, 3]
    dist = None if distortion is None else distortion[observation_camera]
    projected = project_with_distortion(camera_point, intrinsics[observation_camera], dist)
    squared = (projected - observations).square().sum(dim=-1)
    return (tracking_confidence * robust_penalty(squared, robust_delta)).sum()
