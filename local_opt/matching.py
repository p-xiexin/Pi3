"""Glob3R Eq. (2) matching and keyframe-anchored track construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .frame import Frames


@dataclass(frozen=True)
class Tracks:
    """Sparse multi-view tracks used by Glob3R Eqs. (5) and (6).

    Column ``j`` represents point ``P_k`` from reference ``r``, where
    ``r = rs[j]`` and ``k = ks[j]``. ``Xs_Cr[j]`` is its Pi3 anchor point and
    ``us[i, j]`` is its 2D observation in frame ``i``. All valid observations
    in a column therefore refer to the same optimization variable ``X_j``.
    """

    rs: torch.Tensor       # [P]
    ks: torch.Tensor       # [P], local P_k index within reference r
    Xs_Cr: torch.Tensor    # [P, 3]
    us: torch.Tensor       # [S, P, 2]
    mask: torch.Tensor     # [S, P]
    ws: torch.Tensor       # [S, P]


def _sample_reliable(
    mask: torch.Tensor,
    max_points: int,
) -> torch.Tensor:
    """Randomly sample pixels from the depth-confidence-filtered region."""

    valid_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
    count = min(max(max_points, 0), valid_indices.numel())
    if count == 0:
        return valid_indices[:0]

    order = torch.randperm(valid_indices.numel(), device=mask.device)[:count]
    return valid_indices[order]


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
    # Appendix C.1: sample from depth-reliable keyframe regions first, then
    # propagate through dense warps and discard low-confidence observations.
    sampled = _sample_reliable(anchor_valid, points_per_keyframe)
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
    P = int(keep.sum())
    return Tracks(
        rs=torch.full((P,), r, device=device, dtype=torch.long),
        ks=torch.arange(P, device=device),
        Xs_Cr=Xs_Cr[keep],
        us=us[:, keep],
        mask=mask[:, keep],
        ws=ws[:, keep],
    )


@torch.no_grad()
def filter_track_frames(
    tracks: Tracks,
    frames: Frames,
    reference_index: int,
    min_triangulation_angle_deg: float = 1.5,
    min_overlap_ratio: float = 0.25,
) -> Tracks:
    """Skip weak reference-frame relations without deleting track columns."""

    r = reference_index
    mask = tracks.mask.clone()

    # GLOMAP-style view-graph filtering adapted to dense Glob3R tracks:
    # o_rt = |O_r intersect O_t| / |O_r|.
    overlap = (
        (mask & mask[r].unsqueeze(dim=0)).sum(dim=1)
        / mask[r].sum().clamp_min(1)
    )
    overlap_keep = overlap >= min_overlap_ratio
    overlap_keep[r] = True
    mask &= overlap_keep.unsqueeze(dim=1)

    # COLMAP triangulation-angle filtering adapted to a reference-frame edge.
    # The median angle between (P_k - c_r) and (P_k - c_t) represents the
    # parallax of frame t; only that frame's observations are skipped.
    # T_WCrs = frames.T_WCs[tracks.rs]
    # Xs_W = torch.einsum(
    #     "pij,pj->pi", T_WCrs[:, :3, :3], tracks.Xs_Cr
    # ) + T_WCrs[:, :3, 3]
    # cs_W = frames.T_WCs[:, :3, 3]
    # rays = Xs_W.unsqueeze(dim=0) - cs_W.unsqueeze(dim=1)
    # anchor_rays = Xs_W - cs_W[r]
    # cosine = torch.einsum("spc,pc->sp", rays, anchor_rays) / (
    #     rays.norm(dim=-1) * anchor_rays.norm(dim=-1).unsqueeze(dim=0)
    # ).clamp_min(1.0e-8)
    # angles = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))
    # angles = angles.masked_fill(~mask, torch.nan)
    # median_angle = torch.nanmedian(angles, dim=1).values
    # parallax_keep = median_angle >= min_triangulation_angle_deg
    # parallax_keep[r] = True
    # mask &= parallax_keep.unsqueeze(dim=1)

    frame_ids = torch.arange(mask.shape[0], device=mask.device)
    low_overlap = frame_ids[(frame_ids != r) & ~overlap_keep]
    low_parallax = frame_ids[
        # (frame_ids != r) & overlap_keep & ~parallax_keep
        (frame_ids != r) & overlap_keep
    ]
    print(
        f"Track frame filtering for reference {r}: "
        f"low-overlap={low_overlap.tolist()}, "
        f"low-parallax={low_parallax.tolist()}"
    )
    return Tracks(
        rs=tracks.rs,
        ks=tracks.ks,
        Xs_Cr=tracks.Xs_Cr,
        us=tracks.us,
        mask=mask,
        ws=tracks.ws * mask,
    )


__all__ = ["Tracks", "filter_track_frames", "match_tracks"]
