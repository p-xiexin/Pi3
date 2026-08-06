"""Thin adapter from local SE(3) tensors to vendored DROID BA solvers."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from lietorch import SE3

from ..factor_graph import DroidFactorGraph
from . import geom as _geom
from .geom import projective_ops as _projective_ops


# The vendored files retain DROID-SLAM's absolute ``geom`` import. Install the
# local package alias only while importing them; their source remains unchanged.
_previous_geom = sys.modules.get("geom")
_previous_projective_ops = sys.modules.get("geom.projective_ops")
sys.modules["geom"] = _geom
sys.modules["geom.projective_ops"] = _projective_ops
try:
    from .geom.ba import BA, MoBA
finally:
    if _previous_geom is None:
        sys.modules.pop("geom", None)
    else:
        sys.modules["geom"] = _previous_geom
    if _previous_projective_ops is None:
        sys.modules.pop("geom.projective_ops", None)
    else:
        sys.modules["geom.projective_ops"] = _previous_projective_ops


@dataclass
class DroidBAResult:
    """Optimized DROID state in the local camera-pose convention."""

    T_WCs: torch.Tensor
    disps: torch.Tensor
    initial_error: torch.Tensor
    final_error: torch.Tensor


def _matrix_to_xyzw(rotation: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to LieTorch's xyzw quaternion convention."""

    m00 = rotation[..., 0, 0]
    m01 = rotation[..., 0, 1]
    m02 = rotation[..., 0, 2]
    m10 = rotation[..., 1, 0]
    m11 = rotation[..., 1, 1]
    m12 = rotation[..., 1, 2]
    m20 = rotation[..., 2, 0]
    m21 = rotation[..., 2, 1]
    m22 = rotation[..., 2, 2]
    magnitudes = torch.stack(
        (
            1 + m00 + m11 + m22,
            1 + m00 - m11 - m22,
            1 - m00 + m11 - m22,
            1 - m00 - m11 + m22,
        ),
        dim=-1,
    ).clamp_min(0).sqrt()
    candidates = torch.stack(
        (
            torch.stack((magnitudes[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, magnitudes[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, magnitudes[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, magnitudes[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    candidates = candidates / (2 * magnitudes.clamp_min(1.0e-8)[..., None])
    selector = F.one_hot(magnitudes.argmax(dim=-1), num_classes=4).to(
        rotation.dtype
    )
    quaternion_wxyz = (candidates * selector[..., None]).sum(dim=-2)
    return F.normalize(quaternion_wxyz[..., (1, 2, 3, 0)], dim=-1)


def _fixed_support_metrics(poses, disps, graph, support):
    coords, valid = _projective_ops.projective_transform(
        poses,
        disps,
        graph.intrinsics,
        graph.rs,
        graph.ts,
    )
    residual = support * graph.weight * (graph.target - coords).square()
    objective = 0.5 * (0.001 * residual).sum()
    rmse = torch.sqrt(
        residual.sum() / (support * graph.weight).sum().clamp_min(1.0e-8)
    )
    valid_ratio = (support * valid).sum() / support.sum().clamp_min(1)
    return objective, rmse, valid_ratio


@torch.no_grad()
def optimize_droid_ba(
    T_WCs: torch.Tensor,
    graph: DroidFactorGraph,
    solver: str = "moba",
    iterations: int = 12,
) -> DroidBAResult:
    """Run DROID BA/MoBA on an already constructed dense factor graph."""

    if solver not in {"ba", "moba"}:
        raise ValueError("solver must be 'ba' or 'moba'")

    T_CWs = torch.linalg.inv(T_WCs)
    pose_vectors = torch.cat(
        (T_CWs[:, :3, 3], _matrix_to_xyzw(T_CWs[:, :3, :3])), dim=-1
    ).unsqueeze(dim=0)
    poses = SE3.InitFromVec(pose_vectors)
    disps = graph.disps.clone()
    disps_init = disps.clone()
    source_indices = torch.unique(graph.rs)
    fixed_poses = 2 if solver == "ba" else 1
    low_height, low_width = graph.disps.shape[-2:]

    print(
        f"DROID {solver.upper()} input: "
        f"cameras={T_WCs.shape[0]}, edges={graph.rs.numel()}, "
        f"source_frames={source_indices.tolist()}, "
        f"resolution={low_height}x{low_width}, iterations={iterations}, "
        f"fixed_poses={fixed_poses}, depth_update={solver == 'ba'}"
    )
    with torch.cuda.device(T_WCs.device):
        _, initial_valid = _projective_ops.projective_transform(
            poses,
            disps,
            graph.intrinsics,
            graph.rs,
            graph.ts,
        )
        fixed_support = initial_valid * graph.weight.any(dim=-1, keepdim=True)
        initial_error, initial_rmse, initial_valid_ratio = _fixed_support_metrics(
            poses, disps, graph, fixed_support
        )
        print(
            f"DROID {solver.upper()} iteration 0: "
            f"fixed-support RMSE={float(initial_rmse):.6g}px, "
            f"valid={float(initial_valid_ratio):.2%}"
        )
        final_error = initial_error
        for iteration in range(iterations):
            if solver == "ba":
                poses, disps = BA(
                    graph.target,
                    graph.weight,
                    graph.damping,
                    poses,
                    disps,
                    graph.intrinsics,
                    graph.rs,
                    graph.ts,
                    fixedp=fixed_poses,
                )
            else:
                poses = MoBA(
                    graph.target,
                    graph.weight,
                    graph.damping,
                    poses,
                    disps,
                    graph.intrinsics,
                    graph.rs,
                    graph.ts,
                    fixedp=fixed_poses,
                )
            final_error, final_rmse, final_valid_ratio = _fixed_support_metrics(
                poses, disps, graph, fixed_support
            )
            print(
                f"DROID {solver.upper()} iteration {iteration + 1}: "
                f"fixed-support RMSE={float(final_rmse):.6g}px, "
                f"valid={float(final_valid_ratio):.2%}"
            )
            if solver == "ba":
                source_init = disps_init[:, source_indices]
                source_opt = disps[:, source_indices]
                valid = (source_init > 0) & (source_opt > 0)
                depth_ratios = (source_init / source_opt.clamp_min(1.0e-8))[valid]
                print(
                    "DROID BA depth ratio D_opt/D_init: "
                    f"min={float(depth_ratios.min()):.6g}, "
                    f"median={float(depth_ratios.median()):.6g}, "
                    f"max={float(depth_ratios.max()):.6g}, "
                    f"positive={float((source_opt > 0).float().mean()):.2%}"
                )

    print(
        f"DROID {solver.upper()} error: {float(initial_error):.6g} -> "
        f"{float(final_error):.6g}"
    )
    return DroidBAResult(
        T_WCs=torch.linalg.inv(poses.matrix().squeeze(dim=0)),
        disps=disps.squeeze(dim=0),
        initial_error=initial_error,
        final_error=final_error,
    )


__all__ = ["DroidBAResult", "optimize_droid_ba"]
