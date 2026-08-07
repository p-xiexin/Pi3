"""DROID-style Gauss-Newton optimization for Glob3R Eqs. (5) and (6)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from local_opt.matching import Tracks

from . import proj
from .chol import schur_solve
from .gauge import (
    camera_centers,
    center_scale_residual_jacobian,
    pose_scale_residual_jacobian,
    select_baseline_scale_gauge,
)


@dataclass(frozen=True)
class Eq5Result:
    cs: torch.Tensor
    Xs_W: torch.Tensor
    ds: torch.Tensor
    loss: torch.Tensor


@dataclass(frozen=True)
class Eq6Result:
    T_CWs: torch.Tensor
    Xs_W: torch.Tensor
    loss: torch.Tensor


def safe_scatter_add_mat(A, ii, jj, n, m):
    valid = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    output = torch.zeros(
        n * m,
        *A.shape[1:],
        device=A.device,
        dtype=A.dtype,
    )
    output.index_add_(0, ii[valid] * m + jj[valid], A[valid])
    return output.reshape(n, m, *A.shape[1:])


def safe_scatter_add_vec(b, ii, n):
    valid = (ii >= 0) & (ii < n)
    output = torch.zeros(
        n,
        *b.shape[1:],
        device=b.device,
        dtype=b.dtype,
    )
    output.index_add_(0, ii[valid], b[valid])
    return output


def pose_retr(T_CWs, dx, fixedp=1):
    increments = torch.cat((torch.zeros_like(dx[:fixedp]), dx), dim=0)
    return T_CWs.retr(increments)


@torch.no_grad()
def opt_pose_ray(
    Rs: torch.Tensor,
    cs0: torch.Tensor,
    Xs_W0: torch.Tensor,
    vs: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    confidence: torch.Tensor,
    iterations: int = 15,
    scale_prior_weight: float = 1.0e3,
) -> Eq5Result:
    """Glob3R Eq. (5) with DROID-style block Gauss-Newton."""

    if scale_prior_weight <= 0:
        raise ValueError("scale_prior_weight must be positive")
    cs = cs0.clone()
    Xs_W = Xs_W0.clone()
    rays_W = torch.einsum("oij,oj->oi", Rs[ii].transpose(-1, -2), vs)

    fixedp = 1
    fixedx = 0
    P = cs.shape[0] - fixedp
    M = Xs_W.shape[0] - fixedx
    ci = ii - fixedp
    xj = jj - fixedx
    scale_gauge = select_baseline_scale_gauge(cs0)
    scale_ci = scale_gauge.camera_index - fixedp

    for _ in range(iterations):
        # 1. Compute Eq. (5) residuals, Jacobians, and robust weights.
        error, (Jc, Jx) = proj.ray_transform(
            cs, Xs_W, rays_W, ii, jj, jacobian=True
        )
        r = -error.unsqueeze(dim=-1)
        weight = proj.robust_weights(error, confidence, delta=1.0)

        # 2. Construct the camera-point block normal equations.
        wJcT = (weight[:, None, None] * Jc).transpose(1, 2)
        wJxT = (weight[:, None, None] * Jx).transpose(1, 2)

        Hcc = wJcT @ Jc
        Hxx = wJxT @ Jx
        Ecx = wJcT @ Jx
        vc = (wJcT @ r).squeeze(dim=-1)
        vx = (wJxT @ r).squeeze(dim=-1)

        H = safe_scatter_add_mat(Hcc, ci, ci, P, P)
        E = safe_scatter_add_mat(Ecx, ci, xj, P, M)
        C = safe_scatter_add_vec(Hxx, xj, M)
        v = safe_scatter_add_vec(vc, ci, P)
        w = safe_scatter_add_vec(vx, xj, M)

        # Fix only one scalar scale degree of freedom. All coordinates of all
        # 3D points remain active optimization variables.
        scale_error, scale_J = center_scale_residual_jacobian(cs, scale_gauge)
        H[scale_ci, scale_ci] += scale_prior_weight * torch.outer(
            scale_J, scale_J
        )
        v[scale_ci] += scale_prior_weight * scale_J * (-scale_error)

        # 3. Eliminate shared points with the Schur complement.
        dc, dX = schur_solve(H, E, C, v, w)

        # 4. Apply the additive retraction for centers and shared points.
        cs[fixedp:] += dc
        Xs_W += dX

    offsets = Xs_W[jj] - cs[ii]
    ds = (offsets * rays_W).sum(dim=-1) / rays_W.square().sum(dim=-1)
    error = proj.ray_transform(cs, Xs_W, rays_W, ii, jj)
    loss = proj.robust_objective(error, confidence, delta=1.0)
    return Eq5Result(cs=cs, Xs_W=Xs_W, ds=ds, loss=loss)


@torch.no_grad()
def bundle_adjust(
    T_CWs0: torch.Tensor,
    Xs_W0: torch.Tensor,
    Ks: torch.Tensor,
    deltas: torch.Tensor,
    tracks: Tracks,
    iterations: int = 20,
    scale_prior_weight: float = 1.0e3,
) -> Eq6Result:
    """Glob3R Eq. (6) with LieTorch retraction and Schur BA."""

    if scale_prior_weight <= 0:
        raise ValueError("scale_prior_weight must be positive")
    ii, jj = torch.nonzero(tracks.mask, as_tuple=True)
    target = tracks.us[ii, jj]
    confidence = tracks.ws[ii, jj]
    T_CWs = _make_se3(T_CWs0)
    Xs_W = Xs_W0.clone()

    fixedp = 1
    fixedx = 0
    P = T_CWs0.shape[0] - fixedp
    M = Xs_W.shape[0] - fixedx
    ci = ii - fixedp
    xj = jj - fixedx
    scale_gauge = select_baseline_scale_gauge(camera_centers(T_CWs0))
    scale_ci = scale_gauge.camera_index - fixedp

    for _ in range(iterations):
        # 1. Compute Eq. (6) projections, Jacobians, and residuals.
        coords, valid, (Jc, Jx) = proj.projective_transform(
            T_CWs, Xs_W, Ks, deltas, ii, jj, jacobian=True
        )
        r = (target - coords).unsqueeze(dim=-1)
        weight = proj.robust_weights(
            target - coords, confidence, delta=2.0
        ) * valid

        # 2. Construct the pose-point block normal equations.
        wJcT = (weight[:, None, None] * Jc).transpose(1, 2)
        wJxT = (weight[:, None, None] * Jx).transpose(1, 2)

        Hcc = wJcT @ Jc
        Hxx = wJxT @ Jx
        Ecx = wJcT @ Jx
        vc = (wJcT @ r).squeeze(dim=-1)
        vx = (wJxT @ r).squeeze(dim=-1)

        H = safe_scatter_add_mat(Hcc, ci, ci, P, P)
        E = safe_scatter_add_mat(Ecx, ci, xj, P, M)
        C = safe_scatter_add_vec(Hxx, xj, M)
        v = safe_scatter_add_vec(vc, ci, P)
        w = safe_scatter_add_vec(vx, xj, M)

        # The first pose fixes the SE(3) gauge. This one scalar baseline factor
        # fixes only scale; no complete 3D point is frozen.
        scale_error, scale_J = pose_scale_residual_jacobian(
            T_CWs.matrix(), scale_gauge
        )
        H[scale_ci, scale_ci] += scale_prior_weight * torch.outer(
            scale_J, scale_J
        )
        v[scale_ci] += scale_prior_weight * scale_J * (-scale_error)

        # 3. Eliminate shared points with the Schur complement.
        dT, dX = schur_solve(H, E, C, v, w)

        # 4. Retract SE(3) poses and update shared world points.
        T_CWs = pose_retr(T_CWs, dT, fixedp=fixedp)
        Xs_W += dX

    coords, valid = proj.projective_transform(
        T_CWs, Xs_W, Ks, deltas, ii, jj
    )
    residual = target - coords
    loss = proj.robust_objective(
        residual[valid], confidence[valid], delta=2.0
    )
    return Eq6Result(T_CWs=T_CWs.matrix(), Xs_W=Xs_W, loss=loss)


def _matrix_to_xyzw(Rs: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to LieTorch's xyzw quaternion convention."""

    m00, m01, m02 = Rs[..., 0, 0], Rs[..., 0, 1], Rs[..., 0, 2]
    m10, m11, m12 = Rs[..., 1, 0], Rs[..., 1, 1], Rs[..., 1, 2]
    m20, m21, m22 = Rs[..., 2, 0], Rs[..., 2, 1], Rs[..., 2, 2]
    qs = torch.stack(
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
            torch.stack(
                (qs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01),
                dim=-1,
            ),
            torch.stack(
                (m21 - m12, qs[..., 1].square(), m10 + m01, m02 + m20),
                dim=-1,
            ),
            torch.stack(
                (m02 - m20, m10 + m01, qs[..., 2].square(), m12 + m21),
                dim=-1,
            ),
            torch.stack(
                (m10 - m01, m02 + m20, m12 + m21, qs[..., 3].square()),
                dim=-1,
            ),
        ),
        dim=-2,
    )
    candidates = candidates / (2 * qs.clamp_min(1.0e-8)[..., None])
    selector = F.one_hot(qs.argmax(dim=-1), num_classes=4).to(Rs.dtype)
    q_wxyz = (candidates * selector[..., None]).sum(dim=-2)
    return F.normalize(q_wxyz[..., (1, 2, 3, 0)], dim=-1)


def _make_se3(T_CWs: torch.Tensor):
    from lietorch import SE3

    vectors = torch.cat(
        (T_CWs[..., :3, 3], _matrix_to_xyzw(T_CWs[..., :3, :3])),
        dim=-1,
    )
    return SE3.InitFromVec(vectors)


__all__ = ["Eq5Result", "Eq6Result", "bundle_adjust", "opt_pose_ray"]
