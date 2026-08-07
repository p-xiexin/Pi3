"""Block Cholesky and Schur solvers adapted from DROID-SLAM."""

from __future__ import annotations

import torch


def block_solve(
    H: torch.Tensor,
    b: torch.Tensor,
    ep: float = 1.0e-4,
    lm: float = 1.0e-3,
) -> torch.Tensor:
    """Solve a dense block normal equation ``H x = b``."""

    N, _, D, _ = H.shape
    H = H.permute(0, 2, 1, 3).reshape(N * D, N * D)
    I = torch.eye(N * D, device=H.device, dtype=H.dtype)
    H = H + (ep + lm * H) * I

    L = torch.linalg.cholesky(H)
    x = torch.cholesky_solve(b.reshape(N * D, 1), L)
    return x.reshape(N, D)


def schur_solve(
    H: torch.Tensor,
    E: torch.Tensor,
    C: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    ep: float = 1.0e-4,
    lm: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eliminate point blocks, solve poses, then back-substitute points."""

    M, point_dim, _ = C.shape
    I = torch.eye(point_dim, device=C.device, dtype=C.dtype)
    C = C + (ep + lm * C) * I
    Q = torch.cholesky_solve(
        I.expand(M, point_dim, point_dim),
        torch.linalg.cholesky(C),
    )

    # Schur complement: S = H - E Q E^T, y = v - E Q w.
    EQ = torch.einsum("pmdq,mqk->pmdk", E, Q)
    S = H - torch.einsum("pmdk,rmek->prde", EQ, E)
    Qw = torch.einsum("mqk,mk->mq", Q, w)
    y = v - torch.einsum("pmdq,mq->pd", E, Qw)

    # Solve camera increments and back-substitute shared-point increments.
    dx = block_solve(S, y, ep=ep, lm=lm)
    Et_dx = torch.einsum("pmdq,pd->mq", E, dx)
    dX = torch.einsum("mqk,mk->mq", Q, w - Et_dx)
    return dx, dX


__all__ = ["block_solve", "schur_solve"]
