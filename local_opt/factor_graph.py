"""Build the dense directed factor graph consumed by DROID BA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .frame import Frames
from .matching import PairMatch

MIN_EDGE_COVISIBILITY = 0.05


@dataclass
class DroidFactorGraph:
    """DROID optimization tensors built from directed Glob3R matches."""

    rs: torch.Tensor          # [E]
    ts: torch.Tensor          # [E]
    match_indices: torch.Tensor  # [E], indices into the input PairMatch sequence
    covisibility: torch.Tensor   # [E]
    target: torch.Tensor      # [B, E, h, w, 2]
    weight: torch.Tensor      # [B, E, h, w, 2]
    disps: torch.Tensor       # [B, N, h, w]
    intrinsics: torch.Tensor  # [B, N, 4]
    damping: torch.Tensor     # [B, K, h, w]
    stride: int


@torch.no_grad()
def build_droid_factor_graph(
    frames: Frames,
    matches: list[PairMatch],
    stride: int = 8,
    eta: float = 0.01,
    depth_confidence_threshold: float = 0.1,
    warp_confidence_threshold: float = 0.6,
) -> DroidFactorGraph:
    """Convert Eq. (2) matches into the exact dense DROID factor tensors."""

    if not matches:
        raise RuntimeError("BA requires at least one directed match")

    device = frames.Xs_C.device
    dtype = frames.Xs_C.dtype
    height, width = frames.Xs_C.shape[1:3]
    low_height = len(range(0, height, stride))
    low_width = len(range(0, width, stride))
    ys_r = torch.arange(0, height, stride, device=device, dtype=dtype)
    xs_r = torch.arange(0, width, stride, device=device, dtype=dtype)
    ys_r, xs_r = torch.meshgrid(ys_r, xs_r, indexing="ij")
    reference_grid = torch.stack(
        (
            2 * xs_r / (width - 1) - 1,
            2 * ys_r / (height - 1) - 1,
        ),
        dim=-1,
    )

    rs = torch.tensor(
        [match.r for match in matches], device=device, dtype=torch.long
    )
    ts = torch.tensor(
        [match.t for match in matches], device=device, dtype=torch.long
    )
    match_indices = torch.arange(len(matches), device=device)
    W_r2t = torch.stack([match.W_r2t for match in matches])
    valid_r2t = torch.stack([match.valid_r2t for match in matches])
    Q_r2t = torch.stack([match.Q_r2t for match in matches])

    # DROID depths live at source pixels 0, s, 2s, ... . Sample those exact
    # reference positions from either coarse or refined Glob3R output.
    sampling_grid = reference_grid.unsqueeze(dim=0).expand(
        len(matches), -1, -1, -1
    )
    W_r2t = F.grid_sample(
        W_r2t.permute(0, 3, 1, 2),
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).permute(0, 2, 3, 1)
    Q_r2t = F.grid_sample(
        Q_r2t.unsqueeze(dim=1),
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(dim=1)
    valid_r2t = F.grid_sample(
        valid_r2t.to(dtype).unsqueeze(dim=1),
        sampling_grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(dim=1).bool()

    Ds_low = frames.Xs_C[..., 2][:, ::stride, ::stride]
    disps = torch.where(
        Ds_low > 0,
        Ds_low.reciprocal(),
        torch.zeros_like(Ds_low),
    ).unsqueeze(dim=0)
    Cs_r = frames.Cs[rs, ::stride, ::stride]
    normalized_W_r2t = torch.stack(
        (
            2 * W_r2t[..., 0] / (width - 1) - 1,
            2 * W_r2t[..., 1] / (height - 1) - 1,
        ),
        dim=-1,
    )
    # Preserve Eq. (2) subpixel coordinates when reading target confidence.
    Cs_t = F.grid_sample(
        frames.Cs[ts].unsqueeze(dim=1),
        normalized_W_r2t,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(dim=1)
    valid_r2t &= torch.isfinite(W_r2t).all(dim=-1)
    valid_r2t &= W_r2t[..., 0] >= 0
    valid_r2t &= W_r2t[..., 0] <= width - 1
    valid_r2t &= W_r2t[..., 1] >= 0
    valid_r2t &= W_r2t[..., 1] <= height - 1
    valid_r2t &= Ds_low[rs] > 0
    valid_r2t &= Cs_r > depth_confidence_threshold
    valid_r2t &= Cs_t > depth_confidence_threshold
    valid_r2t &= Q_r2t >= warp_confidence_threshold

    # A directed edge is useful only when enough of the reference grid is
    # jointly supported by geometry and the final Glob3R confidence mask.
    # Keeping very sparse edges introduces weakly constrained cameras into BA.
    edge_covisibility = valid_r2t.float().mean(dim=(1, 2))
    edge_mask = edge_covisibility >= MIN_EDGE_COVISIBILITY
    if not edge_mask.any():
        raise RuntimeError("No match edge passes the BA covisibility threshold")
    rs = rs[edge_mask]
    ts = ts[edge_mask]
    match_indices = match_indices[edge_mask]
    edge_covisibility = edge_covisibility[edge_mask]
    W_r2t = W_r2t[edge_mask]
    Q_r2t = Q_r2t[edge_mask]
    valid_r2t = valid_r2t[edge_mask]

    # [E, h, w, 2] -> [B, E, h, w, 2]. Coordinate values are continuously
    # scaled into DROID's grid and never rounded.
    target = torch.where(
        valid_r2t.unsqueeze(dim=0).unsqueeze(dim=-1),
        (W_r2t / stride).unsqueeze(dim=0),
        torch.zeros(1, *W_r2t.shape, device=device, dtype=dtype),
    ).contiguous()
    weight = (Q_r2t * valid_r2t.to(dtype)).unsqueeze(dim=0).unsqueeze(dim=-1)
    weight = weight.expand(-1, -1, -1, -1, 2).contiguous()

    intrinsics = torch.stack(
        (
            frames.Ks[:, 0, 0],
            frames.Ks[:, 1, 1],
            frames.Ks[:, 0, 2],
            frames.Ks[:, 1, 2],
        ),
        dim=-1,
    ).unsqueeze(dim=0) / stride
    source_indices = torch.unique(rs)
    damping = torch.full(
        (1, source_indices.numel(), low_height, low_width),
        eta,
        device=device,
        dtype=dtype,
    )
    return DroidFactorGraph(
        rs=rs,
        ts=ts,
        match_indices=match_indices,
        covisibility=edge_covisibility,
        target=target,
        weight=weight,
        disps=disps,
        intrinsics=intrinsics,
        damping=damping,
        stride=stride,
    )


__all__ = [
    "DroidFactorGraph",
    "MIN_EDGE_COVISIBILITY",
    "build_droid_factor_graph",
]
