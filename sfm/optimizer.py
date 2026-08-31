"""Shared matrix-free Schur optimizer for local and full graph views."""

import torch

from .geometry import camera_centers, se3_exp


MAX_REPROJECTION_ERROR_PX = 4.0
MIN_TRACK_OBSERVATIONS = 3


def _robust_weight(residual, confidence, delta):
    """Return confidence-weighted Huber influence values per observation."""
    norm = residual.norm(dim=-1).clamp_min(1.0e-8)
    return confidence * torch.where(norm <= delta, 1.0, delta / norm)


def _robust_loss(residual, confidence, delta):
    """Evaluate the confidence-weighted Huber objective."""
    norm = residual.norm(dim=-1)
    rho = torch.where(norm <= delta, 0.5 * norm.square(), delta * (norm - 0.5 * delta))
    return (confidence * rho).sum()


def _project(poses, points, K, distortion, ii, jj, jacobian=False):
    """Project indexed landmarks and optionally return camera and point Jacobians."""
    R, t = poses[ii, :3, :3], poses[ii, :3, 3]
    Xc = torch.einsum("oij,oj->oi", R, points[jj]) + t
    X, Y, Z = Xc.unbind(-1)
    Zs = Z.clamp_min(1.0e-8)
    x, y = X / Zs, Y / Zs
    k1, k2, p1, p2 = distortion[ii].unbind(-1)
    r2 = x.square() + y.square()
    radial = 1 + k1 * r2 + k2 * r2.square()
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x.square())
    yd = y * radial + p1 * (r2 + 2 * y.square()) + 2 * p2 * x * y
    xy1 = torch.stack((xd, yd, torch.ones_like(xd)), -1)
    projected = torch.einsum("oij,oj->oi", K[ii], xy1)
    uv = projected[:, :2] / projected[:, 2:3]
    valid = (Z > 0.2) & torch.isfinite(uv).all(-1)
    if not jacobian:
        return uv, valid
    dr_dx = 2 * k1 * x + 4 * k2 * r2 * x
    dr_dy = 2 * k1 * y + 4 * k2 * r2 * y
    Jd = torch.stack(
        (radial + x * dr_dx + 2 * p1 * y + 6 * p2 * x,
         x * dr_dy + 2 * p1 * x + 2 * p2 * y,
         y * dr_dx + 2 * p1 * x + 2 * p2 * y,
         radial + y * dr_dy + 6 * p1 * y + 2 * p2 * x), -1
    ).reshape(-1, 2, 2)
    zero = torch.zeros_like(Zs)
    Jn = torch.stack(
        (1 / Zs, zero, -X / Zs.square(), zero, 1 / Zs, -Y / Zs.square()),
        -1,
    ).reshape(-1, 2, 3)
    Jp = K[ii, :2, :2] @ Jd @ Jn
    one = torch.ones_like(X)
    Jpose = torch.stack(
        (one, zero, zero, zero, Z, -Y,
         zero, one, zero, -Z, zero, X,
         zero, zero, one, Y, -X, zero), -1
    ).reshape(-1, 3, 6)
    return uv, valid, (Jp @ Jpose, Jp @ R)


def _pcg(operator, rhs, preconditioner, iterations=100, tolerance=1.0e-6):
    """Solve a positive-definite linear system with preconditioned CG."""
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    z = preconditioner(residual)
    direction = z.clone()
    rz = torch.dot(residual.reshape(-1), z.reshape(-1))
    initial = residual.norm()
    if initial <= torch.finfo(rhs.dtype).eps:
        return x
    for _ in range(iterations):
        product = operator(direction)
        denominator = torch.dot(direction.reshape(-1), product.reshape(-1))
        if denominator <= torch.finfo(rhs.dtype).eps:
            break
        alpha = rz / denominator
        x += alpha * direction
        residual -= alpha * product
        if residual.norm() <= tolerance * initial:
            break
        z = preconditioner(residual)
        next_rz = torch.dot(residual.reshape(-1), z.reshape(-1))
        direction = z + next_rz / rz.clamp_min(1.0e-20) * direction
        rz = next_rz
    return x


def _schur_step(Jc, Jx, rhs, weight, ii, jj, camera_count, point_count, fixed, gauge=None):
    """Solve one normal-equation step without materializing the reduced camera matrix."""
    camera_dim, point_dim = Jc.shape[-1], Jx.shape[-1]
    active = torch.nonzero(~fixed, as_tuple=False).squeeze(-1)
    active_map = torch.full((camera_count,), -1, device=ii.device, dtype=torch.long)
    active_map[active] = torch.arange(active.numel(), device=ii.device)
    ai = active_map[ii]
    weighted_c = (weight[:, None, None] * Jc).transpose(1, 2)
    weighted_x = (weight[:, None, None] * Jx).transpose(1, 2)
    H = torch.zeros(active.numel(), camera_dim, camera_dim, device=Jc.device, dtype=Jc.dtype)
    C = torch.zeros(point_count, point_dim, point_dim, device=Jc.device, dtype=Jc.dtype)
    bc = torch.zeros(active.numel(), camera_dim, device=Jc.device, dtype=Jc.dtype)
    bp = torch.zeros(point_count, point_dim, device=Jc.device, dtype=Jc.dtype)
    active_observation = ai >= 0
    if active_observation.any():
        H.index_add_(0, ai[active_observation], (weighted_c @ Jc)[active_observation])
        camera_rhs = (weighted_c @ rhs[..., None]).squeeze(-1)
        bc.index_add_(0, ai[active_observation], camera_rhs[active_observation])
    C.index_add_(0, jj, weighted_x @ Jx)
    bp.index_add_(0, jj, (weighted_x @ rhs[..., None]).squeeze(-1))
    if gauge is not None:
        camera, jacobian, residual, strength = gauge
        index = int(active_map[camera])
        if index >= 0:
            H[index] += strength * torch.outer(jacobian, jacobian)
            bc[index] += strength * jacobian * (-residual)
    eye_c = torch.eye(camera_dim, device=Jc.device, dtype=Jc.dtype)
    eye_x = torch.eye(point_dim, device=Jc.device, dtype=Jc.dtype)
    H += torch.diag_embed(1.0e-4 + 1.0e-3 * H.diagonal(dim1=-2, dim2=-1).abs())
    C += torch.diag_embed(1.0e-4 + 1.0e-3 * C.diagonal(dim1=-2, dim2=-1).abs())
    Cinv = torch.linalg.solve(C, eye_x.expand(point_count, -1, -1))
    if not active.numel():
        dc = torch.zeros(
            camera_count, camera_dim, device=Jc.device, dtype=Jc.dtype
        )
        return dc, torch.einsum("mij,mj->mi", Cinv, bp)
    obs = torch.nonzero(active_observation, as_tuple=False).squeeze(-1)
    E = (weighted_c @ Jx)[obs]
    ea, ej = ai[obs], jj[obs]
    Cbp = torch.einsum("mij,mj->mi", Cinv, bp)
    reduced_rhs = bc.clone()
    reduced_rhs.index_add_(0, ea, -torch.einsum("odq,oq->od", E, Cbp[ej]))

    # Apply Hcc - Hcp Hpp^-1 Hpc on demand. Memory therefore scales
    # with observations and block diagonals instead of camera_count squared.
    def operator(x):
        out = torch.einsum("aij,aj->ai", H, x)
        point_sum = torch.zeros(point_count, point_dim, device=x.device, dtype=x.dtype)
        point_sum.index_add_(0, ej, torch.einsum("odq,od->oq", E, x[ea]))
        eliminated = torch.einsum("mij,mj->mi", Cinv, point_sum)
        out.index_add_(0, ea, -torch.einsum("odq,oq->od", E, eliminated[ej]))
        return out

    diagonal = H.clone()
    diagonal.index_add_(0, ea, -E @ Cinv[ej] @ E.transpose(1, 2))
    diagonal += 1.0e-6 * eye_c
    diagonal_inv = torch.linalg.pinv(diagonal)
    dc_active = _pcg(operator, reduced_rhs, lambda x: torch.einsum("aij,aj->ai", diagonal_inv, x))
    point_sum = torch.zeros(point_count, point_dim, device=Jc.device, dtype=Jc.dtype)
    point_sum.index_add_(0, ej, torch.einsum("odq,od->oq", E, dc_active[ea]))
    dpoints = torch.einsum("mij,mj->mi", Cinv, bp - point_sum)
    dc = torch.zeros(camera_count, camera_dim, device=Jc.device, dtype=Jc.dtype)
    dc[active] = dc_active
    return dc, dpoints


def _select_scale_gauge(centers, fixed, active=None, camera=None):
    """Freeze one initial baseline as the monocular scale gauge."""
    if active is None:
        active = torch.ones_like(fixed)
    explicit_camera = camera is not None
    fixed_ids = torch.nonzero(fixed & active, as_tuple=False).squeeze(-1)
    if not fixed_ids.numel():
        if explicit_camera:
            raise RuntimeError("scale gauge root has no active observations")
        return None
    root = int(fixed_ids[0])
    offsets = centers - centers[root]
    if camera is None:
        candidates = torch.nonzero((~fixed) & active, as_tuple=False).squeeze(-1)
        if not candidates.numel():
            return None
        lengths = offsets[candidates].norm(dim=-1)
        camera = int(candidates[int(lengths.argmax())])
    else:
        camera = int(camera)
        if camera < 0 or camera >= centers.shape[0]:
            raise IndexError("scale gauge camera lies outside the optimizer view")
        if not bool(active[camera]):
            raise RuntimeError("scale gauge camera has no active observations")
        if bool(fixed[camera]):
            raise ValueError("scale gauge camera must differ from the fixed root")
    target = offsets[camera].norm()
    if target <= 1.0e-6:
        if explicit_camera:
            raise RuntimeError("scale gauge baseline is degenerate")
        return None
    target = target.detach().clone()
    direction = (offsets[camera] / target).detach().clone()
    return root, camera, direction, target


def _scale_factor(gauge, centers, poses, pose):
    """Linearize the scale gauge for center-only or full-pose updates."""
    if gauge is None:
        return None
    root, camera, direction, target = gauge
    residual = torch.dot(direction, centers[camera] - centers[root]) - target
    if pose:
        zeros = torch.zeros(3, device=poses.device, dtype=poses.dtype)
        jacobian = torch.cat((-(poses[camera, :3, :3] @ direction), zeros))
    else:
        jacobian = direction
    return camera, jacobian, residual, 1.0e3


def _fixed_camera_mask(poses, fixed_ids):
    """Build and validate the camera gauge mask shared by both BA entrypoints."""
    fixed = torch.zeros(poses.shape[0], device=poses.device, dtype=torch.bool)
    fixed_ids = torch.as_tensor(fixed_ids, device=poses.device, dtype=torch.long)
    if fixed_ids.numel() == 0:
        raise ValueError("optimization requires at least one fixed camera")
    if bool(((fixed_ids < 0) | (fixed_ids >= poses.shape[0])).any()):
        raise IndexError("fixed camera index lies outside the optimizer view")
    fixed[fixed_ids] = True
    return fixed


def _stage_fixed_camera_mask(fixed, observation_cameras, camera_count):
    """Freeze cameras without measurements and keep one measured gauge camera."""
    active = torch.bincount(
        observation_cameras, minlength=camera_count
    ) > 0
    if not bool(active.any()):
        raise RuntimeError("optimization stage has no camera observations")
    stage_fixed = fixed | ~active
    if not bool((fixed & active).any()):
        stage_fixed[torch.nonzero(active, as_tuple=False)[0, 0]] = True
    return stage_fixed, active


def _validate_scale_gauge(scale_gauge, fixed, active):
    """Require the persistent gauge baseline to remain in the active problem."""
    if scale_gauge is None:
        return
    root, camera, _, _ = scale_gauge
    if not bool(active[root]) or not bool(active[camera]):
        raise RuntimeError("persistent scale gauge camera has no active observations")
    if not bool(fixed[root]) or bool(fixed[camera]):
        raise RuntimeError("persistent scale gauge endpoints have invalid fixed state")


def _bearing_adjust(poses, points, view, fixed, iterations, scale_gauge=None):
    """Run the Eq. (5)-style center and landmark refinement."""
    ii, jj = view["ii"], view["jj"]
    fixed, active = _stage_fixed_camera_mask(fixed, ii, poses.shape[0])
    centers = camera_centers(poses)
    if scale_gauge is None:
        scale_gauge = _select_scale_gauge(centers, fixed, active)
    _validate_scale_gauge(scale_gauge, fixed, active)
    rotations = poses[:, :3, :3].clone()
    uv1 = torch.cat((view["uv"], torch.ones_like(view["uv"][:, :1])), -1)
    rays_camera = torch.einsum("oij,oj->oi", torch.linalg.inv(view["K"])[ii], uv1)
    rays_world = torch.einsum(
        "oij,oj->oi", rotations[ii].transpose(-1, -2), rays_camera
    )
    for _ in range(int(iterations)):
        rays = rays_world / rays_world.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        eye = torch.eye(3, device=poses.device, dtype=poses.dtype)
        projector = eye - rays[..., None] * rays[..., None, :]
        error = torch.einsum("oij,oj->oi", projector, points[jj] - centers[ii])
        weight = _robust_weight(error, view["weight"], 1.0)
        dc, dpoints = _schur_step(
            -projector,
            projector,
            -error,
            weight,
            ii,
            jj,
            poses.shape[0],
            points.shape[0],
            fixed,
            _scale_factor(scale_gauge, centers, poses, False),
        )
        centers += dc
        points += dpoints
    poses[:, :3, 3] = -torch.einsum("sij,sj->si", rotations, centers)
    return poses, points


def _bundle_adjust(
    poses,
    points,
    view,
    fixed,
    iterations,
    observation_ids=None,
    scale_gauge=None,
):
    """Run matrix-free reprojection BA over all or a selected set of observations."""
    if observation_ids is None:
        observation_ids = torch.arange(
            view["ii"].numel(), device=view["ii"].device, dtype=torch.long
        )
    else:
        observation_ids = torch.as_tensor(
            observation_ids, device=view["ii"].device, dtype=torch.long
        )
    if observation_ids.numel() == 0:
        raise RuntimeError("reprojection BA received no observations")
    ii = view["ii"][observation_ids]
    jj = view["jj"][observation_ids]
    uv = view["uv"][observation_ids]
    confidence = view["weight"][observation_ids]
    fixed, active = _stage_fixed_camera_mask(fixed, ii, poses.shape[0])
    if scale_gauge is None:
        scale_gauge = _select_scale_gauge(camera_centers(poses), fixed, active)
    _validate_scale_gauge(scale_gauge, fixed, active)
    for _ in range(int(iterations)):
        projection, valid, (Jc, Jx) = _project(
            poses, points, view["K"], view["distortion"], ii, jj, True
        )
        residual = torch.where(valid[:, None], uv - projection, 0)
        weight = _robust_weight(residual, confidence, 2.0) * valid
        dpose, dpoints = _schur_step(
            Jc,
            Jx,
            residual,
            weight,
            ii,
            jj,
            poses.shape[0],
            points.shape[0],
            fixed,
            _scale_factor(
                scale_gauge, camera_centers(poses), poses, True
            ),
        )
        poses = se3_exp(dpose) @ poses
        points += dpoints
    return poses, points


@torch.no_grad()
def filter_reprojection_observations(
    view,
    poses,
    points,
    max_error_px=MAX_REPROJECTION_ERROR_PX,
    min_observations=MIN_TRACK_OBSERVATIONS,
    observation_ids=None,
):
    """Apply the local_opt positive-depth, finite, 4 px and track-support gates.

    The returned masks retain the indexing of the input view.  Passing
    ``observation_ids`` restricts the candidate set while preserving that full
    indexing, which lets the second BA round refine only first-round inliers.
    """
    if max_error_px < 0:
        raise ValueError("max_error_px must be non-negative")
    if int(min_observations) < 1:
        raise ValueError("min_observations must be positive")
    observation_count = int(view["ii"].numel())
    if observation_ids is None:
        observation_ids = torch.arange(
            observation_count, device=view["ii"].device, dtype=torch.long
        )
    else:
        observation_ids = torch.as_tensor(
            observation_ids, device=view["ii"].device, dtype=torch.long
        )
    if observation_ids.numel() == 0:
        raise RuntimeError("reprojection filtering received no observations")
    if bool(((observation_ids < 0) | (observation_ids >= observation_count)).any()):
        raise IndexError("observation index lies outside the optimizer view")

    ii = view["ii"][observation_ids]
    jj = view["jj"][observation_ids]
    projection, _ = _project(
        poses, points, view["K"], view["distortion"], ii, jj
    )
    rotations, translations = poses[ii, :3, :3], poses[ii, :3, 3]
    camera_points = torch.einsum("oij,oj->oi", rotations, points[jj]) + translations
    error = torch.linalg.vector_norm(
        projection - view["uv"][observation_ids], dim=-1
    )
    local_inliers = (
        torch.isfinite(projection).all(-1)
        & torch.isfinite(error)
        & torch.isfinite(camera_points[:, 2])
        & (camera_points[:, 2] > 0)
        & (error <= float(max_error_px))
    )
    support = torch.bincount(
        jj[local_inliers], minlength=points.shape[0]
    )
    point_inliers = support >= int(min_observations)
    references = view.get("references")
    if references is not None:
        references = torch.as_tensor(
            references, device=ii.device, dtype=torch.long
        )
        if references.shape != (points.shape[0],):
            raise ValueError("optimizer references must have shape [points]")
        owner_in_view = references >= 0
        if bool((references[owner_in_view] >= view["poses"].shape[0]).any()):
            raise IndexError("optimizer reference camera lies outside the view")
        owner_supported = torch.ones_like(point_inliers)
        owner_observations = local_inliers & (ii == references[jj])
        owner_supported[owner_in_view] = (
            torch.bincount(
                jj[owner_observations], minlength=points.shape[0]
            )[owner_in_view]
            > 0
        )
        point_inliers &= owner_supported
    local_inliers &= point_inliers[jj]
    if not bool(point_inliers.any()):
        raise RuntimeError("post-BA filtering rejected every sparse point")

    observation_inliers = torch.zeros(
        observation_count, device=ii.device, dtype=torch.bool
    )
    observation_inliers[observation_ids] = local_inliers
    reprojection_error = torch.full(
        (observation_count,),
        float("inf"),
        device=error.device,
        dtype=error.dtype,
    )
    reprojection_error[observation_ids] = error
    return {
        "observation_inliers": observation_inliers,
        "point_inliers": point_inliers,
        "reprojection_error": reprojection_error,
        "support": support,
    }


@torch.no_grad()
def optimize_view(view, fixed_ids, iterations):
    """Optimize translation and points first, then run full reprojection BA."""
    poses = view["poses"].clone()
    points = view["points"].clone()
    ii, jj = view["ii"], view["jj"]
    fixed = _fixed_camera_mask(poses, fixed_ids)
    # Bearing-space initialization adjusts centers and landmarks while keeping
    # network rotations fixed. It provides a stable starting point for pixel BA.
    poses, points = _bearing_adjust(poses, points, view, fixed, iterations)
    # The second stage jointly refines SE(3) poses and 3D landmarks against the
    # same observations used by local and global graph views.
    poses, points = _bundle_adjust(poses, points, view, fixed, iterations)
    projection, valid = _project(poses, points, view["K"], view["distortion"], ii, jj)
    valid_observations = valid.sum()
    if not bool(valid_observations):
        raise RuntimeError("BA produced no valid projected observations")
    loss = _robust_loss((view["uv"] - projection)[valid], view["weight"][valid], 2.0)
    # One graph observation is one measured 2D image location. Normalizing the
    # robust objective by this count makes losses comparable across graph sizes.
    loss_per_pixel = loss / valid_observations.to(loss.dtype)
    return {
        "poses": poses,
        "points": points,
        "loss": loss,
        "loss_per_pixel": loss_per_pixel,
        "valid_observations": valid_observations,
    }


@torch.no_grad()
def optimize_view_two_rounds(
    view,
    fixed_ids,
    bearing_iterations=15,
    first_iterations=20,
    second_iterations=10,
    max_error_px=MAX_REPROJECTION_ERROR_PX,
    min_observations=MIN_TRACK_OBSERVATIONS,
    scale_gauge_camera=None,
):
    """Run Eq. (5), BA, hard filtering, a second BA, and final filtering.

    Poses and points retain the compact view indexing.  Observation and point
    masks retain their original lengths so callers can map them back to the
    persistent graph without rebuilding IDs.
    """
    poses = view["poses"].clone()
    points = view["points"].clone()
    fixed = _fixed_camera_mask(poses, fixed_ids)
    stage_fixed, active = _stage_fixed_camera_mask(
        fixed, view["ii"], poses.shape[0]
    )
    scale_gauge = _select_scale_gauge(
        camera_centers(poses),
        fixed if scale_gauge_camera is not None else stage_fixed,
        active,
        camera=scale_gauge_camera,
    )
    if scale_gauge_camera is not None:
        root, camera, _, target = scale_gauge
        frame_ids = view.get("frame_ids")
        root_id = root if frame_ids is None else int(frame_ids[root])
        camera_id = camera if frame_ids is None else int(frame_ids[camera])
        print(
            f"global scale gauge root={root_id} camera={camera_id} "
            f"target={float(target):.6g}"
        )
    poses, points = _bearing_adjust(
        poses,
        points,
        view,
        fixed,
        bearing_iterations,
        scale_gauge=scale_gauge,
    )
    poses, points = _bundle_adjust(
        poses,
        points,
        view,
        fixed,
        first_iterations,
        scale_gauge=scale_gauge,
    )
    first = filter_reprojection_observations(
        view,
        poses,
        points,
        max_error_px=max_error_px,
        min_observations=min_observations,
    )
    first_ids = torch.nonzero(
        first["observation_inliers"], as_tuple=False
    ).squeeze(-1)
    poses, points = _bundle_adjust(
        poses,
        points,
        view,
        fixed,
        second_iterations,
        observation_ids=first_ids,
        scale_gauge=scale_gauge,
    )
    final = filter_reprojection_observations(
        view,
        poses,
        points,
        max_error_px=max_error_px,
        min_observations=min_observations,
        observation_ids=first_ids,
    )
    final_ids = torch.nonzero(
        final["observation_inliers"], as_tuple=False
    ).squeeze(-1)
    projection, _ = _project(
        poses,
        points,
        view["K"],
        view["distortion"],
        view["ii"][final_ids],
        view["jj"][final_ids],
    )
    loss = _robust_loss(
        view["uv"][final_ids] - projection,
        view["weight"][final_ids],
        2.0,
    )
    valid_observations = final["observation_inliers"].sum()
    return {
        "poses": poses,
        "points": points,
        "loss": loss,
        "loss_per_pixel": loss / valid_observations.to(loss.dtype),
        "valid_observations": valid_observations,
        "input_observations": torch.as_tensor(
            view["ii"].numel(), device=poses.device, dtype=torch.long
        ),
        "first_inlier_observations": first["observation_inliers"].sum(),
        "observation_inliers": final["observation_inliers"],
        "first_point_inliers": first["point_inliers"],
        "point_inliers": final["point_inliers"],
        "reprojection_error": final["reprojection_error"],
    }


__all__ = [
    "MAX_REPROJECTION_ERROR_PX",
    "MIN_TRACK_OBSERVATIONS",
    "filter_reprojection_observations",
    "optimize_view",
    "optimize_view_two_rounds",
]
