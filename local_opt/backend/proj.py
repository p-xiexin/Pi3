"""Projective geometry adapted from DROID-SLAM for shared Glob3R points."""

from __future__ import annotations

import torch


MIN_DEPTH = 0.2


def ray_transform(
    cs: torch.Tensor,
    Xs_W: torch.Tensor,
    rays_W: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    jacobian: bool = False,
):
    """Eq. (5) point-to-ray residual after eliminating ray depth."""

    ns = rays_W / rays_W.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    I = torch.eye(3, device=cs.device, dtype=cs.dtype)
    Ps = I - ns.unsqueeze(dim=-1) * ns.unsqueeze(dim=-2)
    residual = torch.einsum("oij,oj->oi", Ps, Xs_W[jj] - cs[ii])

    if jacobian:
        Jc = -Ps
        Jx = Ps
        return residual, (Jc, Jx)

    return residual


def actp(T_CWs, Xs_W: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor, jacobian=False):
    """Apply world-to-camera poses to shared world points."""

    Xs_C = T_CWs[ii] * Xs_W[jj]

    if jacobian:
        X, Y, Z = Xs_C.unbind(dim=-1)
        o = torch.zeros_like(X)
        i = torch.ones_like(X)

        # LieTorch tangent order follows DROID: [translation, rotation].
        J_T = torch.stack(
            (
                i, o, o, o, Z, -Y,
                o, i, o, -Z, o, X,
                o, o, i, Y, -X, o,
            ),
            dim=-1,
        ).reshape(-1, 3, 6)
        J_X = T_CWs.matrix()[ii, :3, :3]
        return Xs_C, (J_T, J_X)

    return Xs_C


def proj(
    Xs_C: torch.Tensor,
    Ks: torch.Tensor,
    deltas: torch.Tensor,
    jacobian: bool = False,
):
    """Project camera-frame points with Brown-Conrady distortion."""

    X, Y, Z = Xs_C.unbind(dim=-1)
    Z_safe = Z.clamp_min(1.0e-8)
    x = X / Z_safe
    y = Y / Z_safe

    k1, k2, p1, p2 = deltas.unbind(dim=-1)
    r2 = x.square() + y.square()
    radial = 1 + k1 * r2 + k2 * r2.square()
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x.square())
    yd = y * radial + p1 * (r2 + 2 * y.square()) + 2 * p2 * x * y

    xy1 = torch.stack((xd, yd, torch.ones_like(xd)), dim=-1)
    ps = torch.einsum("oij,oj->oi", Ks, xy1)
    us = ps[:, :2] / ps[:, 2:3]

    if jacobian:
        dr_dx = 2 * k1 * x + 4 * k2 * r2 * x
        dr_dy = 2 * k1 * y + 4 * k2 * r2 * y
        J_delta = torch.stack(
            (
                radial + x * dr_dx + 2 * p1 * y + 6 * p2 * x,
                x * dr_dy + 2 * p1 * x + 2 * p2 * y,
                y * dr_dx + 2 * p1 * x + 2 * p2 * y,
                radial + y * dr_dy + 6 * p1 * y + 2 * p2 * x,
            ),
            dim=-1,
        ).reshape(-1, 2, 2)
        o = torch.zeros_like(Z_safe)
        J_normalize = torch.stack(
            (
                1 / Z_safe, o, -X / Z_safe.square(),
                o, 1 / Z_safe, -Y / Z_safe.square(),
            ),
            dim=-1,
        ).reshape(-1, 2, 3)
        Jp = Ks[:, :2, :2] @ J_delta @ J_normalize
        return us, Jp

    return us


def projective_transform(
    T_CWs,
    Xs_W: torch.Tensor,
    Ks: torch.Tensor,
    deltas: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    jacobian: bool = False,
):
    """Project shared points into their observing cameras."""

    if jacobian:
        Xs_C, (J_T, J_X) = actp(T_CWs, Xs_W, ii, jj, jacobian=True)
        us, Jp = proj(Xs_C, Ks[ii], deltas[ii], jacobian=True)
        valid = (Xs_C[:, 2] > MIN_DEPTH) & torch.isfinite(us).all(dim=-1)
        return us, valid, (Jp @ J_T, Jp @ J_X)

    Xs_C = actp(T_CWs, Xs_W, ii, jj)
    us = proj(Xs_C, Ks[ii], deltas[ii])
    valid = (Xs_C[:, 2] > MIN_DEPTH) & torch.isfinite(us).all(dim=-1)
    return us, valid


def robust_weights(
    residual: torch.Tensor,
    confidence: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Huber IRLS weights."""

    norm = residual.square().sum(dim=-1).sqrt().clamp_min(1.0e-8)
    return confidence * torch.where(norm <= delta, 1.0, delta / norm)


def robust_objective(
    residual: torch.Tensor,
    confidence: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Confidence-weighted Huber objective."""

    norm = residual.square().sum(dim=-1).sqrt()
    rho = torch.where(
        norm <= delta,
        0.5 * norm.square(),
        delta * (norm - 0.5 * delta),
    )
    return (confidence * rho).sum()


__all__ = [
    "actp",
    "proj",
    "projective_transform",
    "ray_transform",
    "robust_objective",
    "robust_weights",
]
