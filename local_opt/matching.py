"""Glob3R dense matching for directed reference-to-target pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .frame import Frames


@dataclass
class PairMatch:
    """Dense correspondence carried by one directed edge ``r -> t``."""

    r: int
    t: int
    W_r2t: torch.Tensor       # [H, W, 2]
    valid_r2t: torch.Tensor   # [H, W]
    Q_r2t: torch.Tensor       # [H, W]


@torch.no_grad()
def match_batch(
    model,
    patch_tokens: torch.Tensor,
    encoder_features: Sequence[torch.Tensor],
    frames: Frames,
    keyframes: torch.Tensor,
) -> list[PairMatch]:
    """Match every reference keyframe to every other target frame."""

    height, width = frames.Is.shape[-2:]
    # [N, 3, H, W] -> [B, N, 3, H, W]
    Is = frames.Is.unsqueeze(dim=0)
    matches: list[PairMatch] = []

    for reference_index in keyframes:
        r = int(reference_index)
        output = model.match_pair(
            patch_tokens,
            encoder_features,
            Is,
            reference_index=r,
        )
        W = output.warp_stages[-1] if output.warp_stages else output.coarse_warp
        Q = (
            output.confidence_stages[-1]
            if output.confidence_stages
            else output.coarse_confidence
        )
        # [B, T, 2, H, W] -> [T, 2, H, W]
        W = W.squeeze(dim=0)
        # [B, T, 1, H, W] -> [T, H, W]
        Q = Q.squeeze(dim=0).squeeze(dim=1)

        for target_offset, target_index in enumerate(output.target_indices):
            t = int(target_index)
            # [2, H, W] -> [H, W, 2]. Eq. (2) warp coordinates remain
            # continuous; discretization belongs to the downstream consumer.
            W_r2t = W[target_offset].permute(1, 2, 0)
            Q_r2t = Q[target_offset]
            valid_r2t = (
                torch.isfinite(W_r2t).all(dim=-1)
                & (W_r2t[..., 0] >= 5)
                & (W_r2t[..., 0] <= width - 5)
                & (W_r2t[..., 1] >= 5)
                & (W_r2t[..., 1] <= height - 5)
            )
            match = PairMatch(
                r=r,
                t=t,
                W_r2t=W_r2t,
                valid_r2t=valid_r2t,
                Q_r2t=Q_r2t,
            )
            matches.append(match)

    return matches


__all__ = ["PairMatch", "match_batch"]
