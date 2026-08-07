"""Single-scalar scale gauge used by Glob3R Eqs. (5) and (6)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BaselineScaleGauge:
    """Fix one baseline projection while leaving every 3D point free."""

    camera_index: int
    direction: torch.Tensor
    target: torch.Tensor


def camera_centers(T_CWs: torch.Tensor) -> torch.Tensor:
    """Recover world-frame camera centers from world-to-camera matrices."""

    Rs = T_CWs[..., :3, :3]
    ts = T_CWs[..., :3, 3]
    return -torch.einsum("...ji,...j->...i", Rs, ts)


def select_baseline_scale_gauge(
    centers: torch.Tensor,
    minimum_baseline: float = 1.0e-6,
) -> BaselineScaleGauge:
    """Choose the longest root-relative baseline as a stable scale anchor."""

    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 2:
        raise ValueError("scale gauge requires at least two [N,3] camera centers")
    offsets = centers[1:] - centers[0]
    lengths = offsets.norm(dim=-1)
    relative_index = int(lengths.argmax())
    camera_index = relative_index + 1
    target = lengths[relative_index]
    if not torch.isfinite(target) or float(target) <= minimum_baseline:
        raise RuntimeError(
            "cannot fix scale from camera baselines: all root-relative "
            f"baselines are <= {minimum_baseline:g}"
        )
    direction = offsets[relative_index] / target
    return BaselineScaleGauge(camera_index, direction, target)


def center_scale_residual_jacobian(
    centers: torch.Tensor,
    gauge: BaselineScaleGauge,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear scalar factor for additive world-frame camera centers."""

    baseline = centers[gauge.camera_index] - centers[0]
    residual = torch.dot(gauge.direction, baseline) - gauge.target
    return residual, gauge.direction


def pose_scale_residual_jacobian(
    T_CWs: torch.Tensor,
    gauge: BaselineScaleGauge,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scalar factor Jacobian for left SE(3) increments [translation, rotation]."""

    centers = camera_centers(T_CWs)
    residual, _ = center_scale_residual_jacobian(centers, gauge)
    rotation = T_CWs[gauge.camera_index, :3, :3]
    translation_jacobian = -(rotation @ gauge.direction)
    jacobian = torch.cat(
        (translation_jacobian, torch.zeros_like(translation_jacobian))
    )
    return residual, jacobian


__all__ = [
    "BaselineScaleGauge",
    "camera_centers",
    "center_scale_residual_jacobian",
    "pose_scale_residual_jacobian",
    "select_baseline_scale_gauge",
]
