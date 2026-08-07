"""Glob3R Eq. (2) matching and keyframe-anchored track construction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .frame import Frames


@dataclass(frozen=True)
class Tracks:
    """Sparse multi-view tracks used by Glob3R Eqs. (5) and (6).

    Column ``j`` represents one scene point. ``Xs_Cr[j]`` is the Pi3 point
    sampled in anchor frame ``rs[j]``; ``us[i, j]`` is its 2D observation in
    frame ``i``. All valid observations in a column therefore refer to the
    same optimization variable ``X_j``.
    """

    rs: torch.Tensor       # [P]
    Xs_Cr: torch.Tensor    # [P, 3]
    us: torch.Tensor       # [S, P, 2]
    mask: torch.Tensor     # [S, P]
    ws: torch.Tensor       # [S, P]


def _sample_grid(
    mask: torch.Tensor,
    score: torch.Tensor,
    max_points: int,
) -> torch.Tensor:
    """Select one highest-scoring valid point from each image-grid cell."""

    H, W = mask.shape
    rows = max(min(int(math.sqrt(max_points * H / W)), H), 1)
    columns = max(min(max_points // rows, W), 1)

    ys = torch.arange(H, device=mask.device)
    xs = torch.arange(W, device=mask.device)
    cell_y = torch.div(ys * rows, H, rounding_mode="floor")
    cell_x = torch.div(xs * columns, W, rounding_mode="floor")
    cells = (cell_y[:, None] * columns + cell_x[None]).reshape(-1)

    mask = mask.reshape(-1)
    score = score.reshape(-1)
    cell_count = rows * columns
    best_score = score.new_full((cell_count,), -torch.inf)
    best_score.scatter_reduce_(0, cells[mask], score[mask], reduce="amax")

    indices = torch.arange(H * W, device=mask.device)
    is_best = mask & (score == best_score[cells])
    selected = indices.new_full((cell_count,), H * W)
    selected.scatter_reduce_(0, cells[is_best], indices[is_best], reduce="amin")
    return selected[selected < H * W]


@torch.no_grad()
def match_tracks(
    output,
    frames: Frames,
    reference_index: int,
    points_per_keyframe: int = 512,
    depth_confidence_threshold: float = 0.1,
    warp_confidence_threshold: float = 0.6,
) -> Tracks:
    """Convert one reference's dense Eq. (2) output into uniform sparse tracks."""

    S, _, H, W = frames.Is.shape
    device = frames.Is.device
    dtype = frames.Xs_C.dtype
    r = reference_index

    Xs_C = frames.Xs_C[r].reshape(H * W, 3)
    Cs = frames.Cs[r].reshape(H * W)
    anchor_valid = (
        (Cs > depth_confidence_threshold)
        & torch.isfinite(Xs_C).all(dim=-1)
        & (Xs_C[:, 2] > 0)
    ).reshape(H, W)

    Ws = output.warp_stages[-1] if output.warp_stages else output.coarse_warp
    Qs = (
        output.confidence_stages[-1]
        if output.confidence_stages
        else output.coarse_confidence
    )
    if Ws.shape[-2:] != (H, W):
        Ws = F.interpolate(
            Ws.squeeze(dim=0),
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        ).unsqueeze(dim=0)
        Qs = F.interpolate(
            Qs.squeeze(dim=0),
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        ).unsqueeze(dim=0)

    # [B, T, 2, H, W] -> [T, H, W, 2]
    Ws = Ws.squeeze(dim=0).permute(0, 2, 3, 1)
    # [B, T, 1, H, W] -> [T, H, W]
    Qs = Qs.squeeze(dim=0).squeeze(dim=1)
    ts = torch.as_tensor(output.target_indices, device=device, dtype=torch.long)

    valid = (
        torch.isfinite(Ws).all(dim=-1)
        & (Ws[..., 0] >= 0)
        & (Ws[..., 0] <= W - 1)
        & (Ws[..., 1] >= 0)
        & (Ws[..., 1] <= H - 1)
        & (Qs >= warp_confidence_threshold)
    )
    eligible = anchor_valid & valid.any(dim=0)
    score = frames.Cs[r] * torch.where(valid, Qs, 0).amax(dim=0)
    sampled = _sample_grid(eligible, score, points_per_keyframe)
    P_r = sampled.numel()

    xs = sampled.remainder(W)
    ys = torch.div(sampled, W, rounding_mode="floor")
    us_r = torch.stack((xs, ys), dim=-1).to(dtype)
    Xs_Cr = Xs_C[sampled]
    Cs_r = Cs[sampled]
    Ws = Ws.reshape(-1, H * W, 2)[:, sampled]
    Qs = Qs.reshape(-1, H * W)[:, sampled]
    valid = valid.reshape(-1, H * W)[:, sampled]

    us = torch.zeros(S, P_r, 2, device=device, dtype=dtype)
    mask = torch.zeros(S, P_r, device=device, dtype=torch.bool)
    ws = torch.zeros(S, P_r, device=device, dtype=dtype)
    us[r] = us_r
    mask[r] = True
    ws[r] = Cs_r
    us[ts] = Ws
    mask[ts] = valid
    # Appendix C.1 combines anchor point confidence with warp confidence.
    ws[ts] = Cs_r.unsqueeze(dim=0) * Qs

    # Keep genuinely multi-view tracks: one reference and at least two targets.
    keep = mask.sum(dim=0) >= 3
    return Tracks(
        rs=torch.full((int(keep.sum()),), r, device=device, dtype=torch.long),
        Xs_Cr=Xs_Cr[keep],
        us=us[:, keep],
        mask=mask[:, keep],
        ws=ws[:, keep],
    )


__all__ = ["Tracks", "match_tracks"]
