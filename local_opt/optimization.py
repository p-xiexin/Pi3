"""GPU-compatible motion averaging and bundle adjustment for Glob3R.

The objectives follow Sec. 3.3 exactly.  The paper does not publish its solver
source, but Appendix C.1 specifies GPU residual/Jacobian construction and a
sparse normal-equation solve.  This module therefore uses sparse COO Jacobians
and matrix-free conjugate gradients rather than replacing Eqs. (5)-(6) with a
generic first-order optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from pi3.models.glob3r.geometry import project_with_distortion, robust_penalty


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def so3_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Exponential map used for rotation averaging and BA pose increments."""

    theta2 = rotation_vector.square().sum(dim=-1, keepdim=True)
    theta = theta2.clamp_min(1.0e-16).sqrt()
    small = theta2 < 1.0e-8
    safe_theta = theta.clamp_min(1.0e-8)
    safe_theta2 = theta2.clamp_min(1.0e-8)
    a = torch.where(
        small, 1 - theta2 / 6 + theta2.square() / 120, torch.sin(theta) / safe_theta
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24 + theta2.square() / 720,
        (1 - torch.cos(theta)) / safe_theta2,
    )
    skew = _skew(rotation_vector)
    eye = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype)
    return eye + a[..., None] * skew + b[..., None] * (skew @ skew)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """Stable logarithm map from SO(3) to the axis-angle tangent space."""

    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) * 0.5).clamp(-1, 1)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * vee.square().sum(dim=-1).clamp_min(1.0e-16).sqrt()
    theta = torch.atan2(sine, cosine)
    safe_sine = sine.clamp_min(1.0e-8)
    scale = torch.where(
        theta.abs() < 1.0e-5, 0.5 + theta.square() / 12, theta / (2 * safe_sine)
    )
    return scale[..., None] * vee


def se3_exp(tangent: torch.Tensor) -> torch.Tensor:
    """SE(3) exponential for left-multiplicative world-to-camera updates."""

    rotation_vector, translation_vector = tangent[..., :3], tangent[..., 3:]
    theta2 = rotation_vector.square().sum(dim=-1, keepdim=True)
    theta = theta2.clamp_min(1.0e-16).sqrt()
    small = theta2 < 1.0e-8
    safe_theta = theta.clamp_min(1.0e-8)
    safe_theta2 = theta2.clamp_min(1.0e-8)
    b = torch.where(
        small,
        0.5 - theta2 / 24 + theta2.square() / 720,
        (1 - torch.cos(theta)) / safe_theta2,
    )
    c = torch.where(
        small,
        1 / 6 - theta2 / 120 + theta2.square() / 5040,
        (theta - torch.sin(theta)) / (safe_theta2 * safe_theta),
    )
    skew = _skew(rotation_vector)
    eye3 = torch.eye(3, device=tangent.device, dtype=tangent.dtype)
    left_jacobian = eye3 + b[..., None] * skew + c[..., None] * (skew @ skew)
    rotation = so3_exp(rotation_vector)
    translation = left_jacobian @ translation_vector[..., None]
    top = torch.cat((rotation, translation), dim=-1)
    bottom = tangent.new_tensor((0.0, 0.0, 0.0, 1.0)).expand(*tangent.shape[:-1], 1, 4)
    return torch.cat((top, bottom), dim=-2)


def _huber_irls_weight(residual: torch.Tensor, delta: float) -> torch.Tensor:
    norm = residual.square().sum(dim=-1).clamp_min(1.0e-12).sqrt()
    return torch.where(norm <= delta, torch.ones_like(norm), delta / norm)


def _conjugate_gradient(operator, rhs, max_iterations: int, tolerance: float) -> torch.Tensor:
    solution = torch.zeros_like(rhs)
    residual = rhs - operator(solution)
    direction = residual.clone()
    squared_norm = torch.dot(residual, residual)
    initial_norm = squared_norm.sqrt().clamp_min(torch.finfo(rhs.dtype).eps)
    for _ in range(max_iterations):
        product = operator(direction)
        alpha = squared_norm / torch.dot(direction, product).clamp_min(torch.finfo(rhs.dtype).eps)
        solution = solution + alpha * direction
        residual = residual - alpha * product
        next_squared_norm = torch.dot(residual, residual)
        if next_squared_norm.sqrt() <= tolerance * initial_norm:
            break
        direction = residual + (next_squared_norm / squared_norm.clamp_min(1.0e-20)) * direction
        squared_norm = next_squared_norm
    return solution


def _normal_equation_step(
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    damping: float,
    cg_iterations: int,
    cg_tolerance: float,
) -> torch.Tensor:
    jacobian = jacobian.coalesce()
    transpose = jacobian.transpose(0, 1)
    gradient = torch.sparse.mm(transpose, residual[:, None]).squeeze(1)

    def normal_operator(vector):
        projected = torch.sparse.mm(jacobian, vector[:, None])
        return torch.sparse.mm(transpose, projected).squeeze(1) + damping * vector

    return _conjugate_gradient(normal_operator, -gradient, cg_iterations, cg_tolerance)


def _make_sparse_jacobian(row_parts, column_parts, value_parts, rows: int, columns: int):
    row = torch.cat(row_parts)
    column = torch.cat(column_parts)
    value = torch.cat(value_parts)
    return torch.sparse_coo_tensor(
        torch.stack((row, column)),
        value,
        (rows, columns),
        device=value.device,
        dtype=value.dtype,
        check_invariants=True,
    ).coalesce()


def maximum_spanning_tree_initialization(
    camera_count: int,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    target_from_source: torch.Tensor,
    edge_weight: torch.Tensor,
    root: int = 0,
) -> torch.Tensor:
    """Sec. 3.2 pose initialization from relative poses on a maximum spanning tree."""

    if camera_count < 1:
        raise ValueError("camera_count must be positive")
    device, dtype = target_from_source.device, target_from_source.dtype
    world_to_camera = torch.eye(4, device=device, dtype=dtype).repeat(camera_count, 1, 1)
    visited = {int(root)}
    while len(visited) < camera_count:
        best = None
        for index in range(edge_source.numel()):
            source, target = int(edge_source[index]), int(edge_target[index])
            crosses = (source in visited) ^ (target in visited)
            if crosses and (best is None or float(edge_weight[index]) > best[0]):
                best = (float(edge_weight[index]), index, source, target)
        if best is None:
            missing = sorted(set(range(camera_count)).difference(visited))
            raise RuntimeError(f"pose graph is disconnected; unreachable cameras: {missing}")
        _, index, source, target = best
        relative = target_from_source[index]
        if source in visited:
            world_to_camera[target] = relative @ world_to_camera[source]
            visited.add(target)
        else:
            world_to_camera[source] = torch.linalg.inv(relative) @ world_to_camera[target]
            visited.add(source)
    return world_to_camera


def robust_rotation_averaging(
    initial_world_to_camera: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    target_from_source: torch.Tensor,
    edge_weight: torch.Tensor,
    iterations: int = 15,
    robust_delta: float = 0.1,
    damping: float = 1.0e-5,
    cg_iterations: int = 100,
) -> torch.Tensor:
    """Robust rotation averaging preceding Glob3R Eq. (5)."""

    rotations = initial_world_to_camera[:, :3, :3].clone()
    camera_count = rotations.shape[0]
    if camera_count == 1 or edge_source.numel() == 0:
        return rotations
    edge_rotations = target_from_source[:, :3, :3]
    edge_count = edge_source.numel()
    device, dtype = rotations.device, rotations.dtype
    active_source = edge_source - 1
    active_target = edge_target - 1

    for _ in range(iterations):
        predicted = rotations[edge_target] @ rotations[edge_source].transpose(-1, -2)
        residual = so3_log(edge_rotations.transpose(-1, -2) @ predicted)
        robust = _huber_irls_weight(residual, robust_delta)
        scale = (edge_weight.clamp_min(0) * robust).sqrt()

        def local_residual(source_delta, target_delta, source_rotation, target_rotation, measured):
            estimate = (
                so3_exp(target_delta) @ target_rotation
                @ (so3_exp(source_delta) @ source_rotation).transpose(-1, -2)
            )
            return so3_log(measured.transpose(-1, -2) @ estimate)

        zeros = torch.zeros(edge_count, 3, device=device, dtype=dtype)
        jacobian_fn = torch.func.vmap(torch.func.jacrev(local_residual, argnums=(0, 1)))
        source_jacobian, target_jacobian = jacobian_fn(
            zeros, zeros, rotations[edge_source], rotations[edge_target], edge_rotations
        )
        weighted_residual = (scale[:, None] * residual).reshape(-1)
        rows_base = torch.arange(edge_count, device=device)[:, None, None] * 3
        rows = (rows_base + torch.arange(3, device=device)[None, :, None]).expand(-1, -1, 3)
        local_columns = torch.arange(3, device=device)[None, None, :]
        row_parts, column_parts, value_parts = [], [], []
        for indices, block in ((active_source, source_jacobian), (active_target, target_jacobian)):
            valid = indices >= 0
            if valid.any():
                columns = indices[valid, None, None] * 3 + local_columns
                row_parts.append(rows[valid].reshape(-1))
                column_parts.append(columns.expand(-1, 3, -1).reshape(-1))
                value_parts.append((scale[valid, None, None] * block[valid]).reshape(-1))
        jacobian = _make_sparse_jacobian(
            row_parts, column_parts, value_parts, edge_count * 3, (camera_count - 1) * 3
        )
        step = _normal_equation_step(jacobian, weighted_residual, damping, cg_iterations, 1.0e-6)
        if step.norm() < 1.0e-7:
            break
        rotations[1:] = so3_exp(step.reshape(camera_count - 1, 3)) @ rotations[1:]
    return rotations


@dataclass
class MotionAveragingResult:
    camera_centers: torch.Tensor
    points_3d: torch.Tensor
    ray_depths: torch.Tensor
    objective: torch.Tensor


def opt_pose_ray(
    rotations: torch.Tensor,
    normalized_rays: torch.Tensor,
    observation_camera: torch.Tensor,
    observation_point: torch.Tensor,
    tracking_confidence: torch.Tensor,
    initial_camera_centers: torch.Tensor,
    initial_points_3d: torch.Tensor,
    initial_ray_depths: torch.Tensor,
    iterations: int = 15,
    robust_delta: float = 1.0,
    damping: float = 1.0e-5,
    cg_iterations: int = 200,
) -> MotionAveragingResult:
    """Solve Glob3R Eq. (5) for centers, points, and per-observation depths."""

    camera_count = rotations.shape[0]
    point_count = initial_points_3d.shape[0]
    observation_count = observation_camera.numel()
    if observation_count == 0:
        raise ValueError("translation averaging requires track observations")
    centers = initial_camera_centers.clone()
    depths = initial_ray_depths.clamp_min(1.0e-4).clone()
    world_rays = torch.einsum(
        "oij,oj->oi", rotations[observation_camera].transpose(-1, -2), normalized_rays
    )
    points = initial_points_3d.clone()

    # Similarity gauge: c_0 is fixed and the first observation depth fixes scale.
    center_columns = (camera_count - 1) * 3
    point_columns = point_count * 3
    depth_columns = observation_count - 1
    total_columns = center_columns + point_columns + depth_columns
    device = rotations.device
    rows_base = torch.arange(observation_count, device=device)[:, None] * 3
    component = torch.arange(3, device=device)

    for _ in range(iterations):
        predicted = centers[observation_camera] + depths[:, None] * world_rays
        residual = points[observation_point] - predicted
        robust = _huber_irls_weight(residual, robust_delta)
        scale = (tracking_confidence.clamp_min(0) * robust).sqrt()
        weighted_residual = (scale[:, None] * residual).reshape(-1)
        row_parts, column_parts, value_parts = [], [], []

        movable_camera = observation_camera > 0
        if movable_camera.any():
            rows = (rows_base[movable_camera] + component[None]).reshape(-1)
            columns = ((observation_camera[movable_camera] - 1)[:, None] * 3 + component[None]).reshape(-1)
            row_parts.append(rows)
            column_parts.append(columns)
            value_parts.append((-scale[movable_camera, None]).expand(-1, 3).reshape(-1))

        rows = (rows_base + component[None]).reshape(-1)
        columns = (center_columns + observation_point[:, None] * 3 + component[None]).reshape(-1)
        row_parts.append(rows)
        column_parts.append(columns)
        value_parts.append(scale[:, None].expand(-1, 3).reshape(-1))

        if observation_count > 1:
            rows = (rows_base[1:] + component[None]).reshape(-1)
            columns = (
                center_columns + point_columns + torch.arange(observation_count - 1, device=device)
            )[:, None].expand(-1, 3).reshape(-1)
            row_parts.append(rows)
            column_parts.append(columns)
            value_parts.append((-scale[1:, None] * depths[1:, None] * world_rays[1:]).reshape(-1))

        jacobian = _make_sparse_jacobian(
            row_parts, column_parts, value_parts, observation_count * 3, total_columns
        )
        step = _normal_equation_step(jacobian, weighted_residual, damping, cg_iterations, 1.0e-6)
        if step.norm() < 1.0e-7:
            break
        if camera_count > 1:
            centers[1:] += step[:center_columns].reshape(camera_count - 1, 3)
        points += step[center_columns:center_columns + point_columns].reshape(point_count, 3)
        if observation_count > 1:
            depths[1:] *= torch.exp(step[center_columns + point_columns:].clamp(-1, 1))

    final_residual = points[observation_point] - (
        centers[observation_camera] + depths[:, None] * world_rays
    )
    objective = (
        tracking_confidence
        * robust_penalty(final_residual.square().sum(dim=-1), robust_delta)
    ).sum()
    return MotionAveragingResult(
        camera_centers=centers,
        points_3d=points,
        ray_depths=depths,
        objective=objective,
    )


def intrinsics_to_parameters(intrinsics: torch.Tensor) -> torch.Tensor:
    """Represent focal lengths in log-space and principal point directly."""

    return torch.stack(
        (intrinsics[..., 0, 0].log(), intrinsics[..., 1, 1].log(), intrinsics[..., 0, 2], intrinsics[..., 1, 2]),
        dim=-1,
    )


def parameters_to_intrinsics(parameters: torch.Tensor) -> torch.Tensor:
    log_fx, log_fy, cx, cy = parameters.unbind(dim=-1)
    zero, one = torch.zeros_like(log_fx), torch.ones_like(log_fx)
    return torch.stack(
        (log_fx.exp(), zero, cx, zero, log_fy.exp(), cy, zero, zero, one), dim=-1
    ).reshape(*parameters.shape[:-1], 3, 3)


@dataclass
class BundleAdjustmentResult:
    world_to_camera: torch.Tensor
    points_3d: torch.Tensor
    intrinsics: torch.Tensor
    distortion: torch.Tensor
    objective: torch.Tensor


def bundle_adjust(
    initial_world_to_camera: torch.Tensor,
    initial_points: torch.Tensor,
    initial_intrinsics: torch.Tensor,
    observations: torch.Tensor,
    observation_camera: torch.Tensor,
    observation_point: torch.Tensor,
    tracking_confidence: torch.Tensor,
    initial_distortion: Optional[torch.Tensor] = None,
    optimize_intrinsics: bool = False,
    optimize_distortion: bool = False,
    shared_intrinsics: bool = True,
    iterations: int = 20,
    robust_delta: float = 2.0,
    damping: float = 1.0e-4,
    cg_iterations: int = 250,
) -> BundleAdjustmentResult:
    """Minimize Glob3R Eq. (6) with sparse GPU normal equations."""

    poses = initial_world_to_camera.clone()
    points = initial_points.clone()
    camera_count, point_count = poses.shape[0], points.shape[0]
    observation_count = observation_camera.numel()
    if observation_count == 0:
        raise ValueError("bundle adjustment requires track observations")

    if initial_intrinsics.ndim == 2:
        # [3, 3] -> [1, 3, 3]
        initial_intrinsics = initial_intrinsics.unsqueeze(dim=0)
        # [1, 3, 3] -> [N, 3, 3]
        initial_intrinsics = initial_intrinsics.expand(
            camera_count, -1, -1
        ).clone()
    if initial_intrinsics.shape[0] != camera_count:
        raise ValueError("initial_intrinsics must contain one matrix per camera")
    if shared_intrinsics:
        intrinsic_parameters = intrinsics_to_parameters(initial_intrinsics.mean(dim=0, keepdim=True))
        intrinsic_group = torch.zeros(camera_count, device=poses.device, dtype=torch.long)
    else:
        intrinsic_parameters = intrinsics_to_parameters(initial_intrinsics)
        intrinsic_group = torch.arange(camera_count, device=poses.device)
    group_count = intrinsic_parameters.shape[0]
    if initial_distortion is None:
        distortion = torch.zeros(group_count, 4, device=poses.device, dtype=poses.dtype)
    else:
        initial_distortion = initial_distortion.to(device=poses.device, dtype=poses.dtype).reshape(-1, 4)
        if shared_intrinsics:
            distortion = initial_distortion.mean(dim=0, keepdim=True)
        elif initial_distortion.shape[0] == 1:
            distortion = initial_distortion.expand(camera_count, -1).clone()
        elif initial_distortion.shape[0] == camera_count:
            distortion = initial_distortion.clone()
        else:
            raise ValueError("distortion must contain one vector or one vector per camera")

    # Fix camera 0 and point 0 to remove the BA similarity gauge.
    pose_columns = max(camera_count - 1, 0) * 6
    point_columns = max(point_count - 1, 0) * 3
    intrinsic_columns = group_count * 4 if optimize_intrinsics else 0
    distortion_columns = group_count * 4 if optimize_distortion else 0
    total_columns = pose_columns + point_columns + intrinsic_columns + distortion_columns
    if total_columns == 0:
        raise ValueError("bundle adjustment has no free variables")

    for _ in range(iterations):
        observation_intrinsic_group = intrinsic_group[observation_camera]
        current_intrinsic_parameters = intrinsic_parameters[observation_intrinsic_group]
        current_distortion = distortion[observation_intrinsic_group]

        def local_projection(pose_delta, point_delta, intrinsic_delta, distortion_delta,
                             pose, point, intrinsic_parameter, distortion_parameter):
            updated_pose = se3_exp(pose_delta) @ pose
            updated_point = point + point_delta
            updated_intrinsic = parameters_to_intrinsics(intrinsic_parameter + intrinsic_delta)
            updated_distortion = distortion_parameter + distortion_delta
            camera_point = updated_pose[:3, :3] @ updated_point + updated_pose[:3, 3]
            return project_with_distortion(camera_point, updated_intrinsic, updated_distortion)

        zeros_pose = torch.zeros(observation_count, 6, device=poses.device, dtype=poses.dtype)
        zeros_point = torch.zeros(observation_count, 3, device=poses.device, dtype=poses.dtype)
        zeros_intrinsic = torch.zeros(observation_count, 4, device=poses.device, dtype=poses.dtype)
        zeros_distortion = torch.zeros_like(zeros_intrinsic)
        projection_and_jacobian = torch.func.vmap(
            torch.func.jacrev(local_projection, argnums=(0, 1, 2, 3)),
            in_dims=(0, 0, 0, 0, 0, 0, 0, 0),
        )
        jac_pose, jac_point, jac_intrinsic, jac_distortion = projection_and_jacobian(
            zeros_pose,
            zeros_point,
            zeros_intrinsic,
            zeros_distortion,
            poses[observation_camera],
            points[observation_point],
            current_intrinsic_parameters,
            current_distortion,
        )
        camera_points = (
            torch.einsum("oij,oj->oi", poses[observation_camera, :3, :3], points[observation_point])
            + poses[observation_camera, :3, 3]
        )
        projected = project_with_distortion(
            camera_points,
            parameters_to_intrinsics(current_intrinsic_parameters),
            current_distortion,
        )
        residual = projected - observations
        robust = _huber_irls_weight(residual, robust_delta)
        scale = (tracking_confidence.clamp_min(0) * robust).sqrt()
        weighted_residual = (scale[:, None] * residual).reshape(-1)

        rows_base = torch.arange(observation_count, device=poses.device)[:, None, None] * 2
        rows = (rows_base + torch.arange(2, device=poses.device)[None, :, None])
        row_parts, column_parts, value_parts = [], [], []

        movable_camera = observation_camera > 0
        if movable_camera.any():
            columns = (observation_camera[movable_camera] - 1)[:, None, None] * 6 + torch.arange(
                6, device=poses.device
            )[None, None]
            row_parts.append(rows[movable_camera].expand(-1, -1, 6).reshape(-1))
            column_parts.append(columns.expand(-1, 2, -1).reshape(-1))
            value_parts.append((scale[movable_camera, None, None] * jac_pose[movable_camera]).reshape(-1))

        movable_point = observation_point > 0
        if movable_point.any():
            columns = pose_columns + (observation_point[movable_point] - 1)[:, None, None] * 3 + torch.arange(
                3, device=poses.device
            )[None, None]
            row_parts.append(rows[movable_point].expand(-1, -1, 3).reshape(-1))
            column_parts.append(columns.expand(-1, 2, -1).reshape(-1))
            value_parts.append((scale[movable_point, None, None] * jac_point[movable_point]).reshape(-1))

        if optimize_intrinsics:
            columns = pose_columns + point_columns + observation_intrinsic_group[:, None, None] * 4 + torch.arange(
                4, device=poses.device
            )[None, None]
            row_parts.append(rows.expand(-1, -1, 4).reshape(-1))
            column_parts.append(columns.expand(-1, 2, -1).reshape(-1))
            value_parts.append((scale[:, None, None] * jac_intrinsic).reshape(-1))

        if optimize_distortion:
            columns = (
                pose_columns + point_columns + intrinsic_columns
                + observation_intrinsic_group[:, None, None] * 4
                + torch.arange(4, device=poses.device)[None, None]
            )
            row_parts.append(rows.expand(-1, -1, 4).reshape(-1))
            column_parts.append(columns.expand(-1, 2, -1).reshape(-1))
            value_parts.append((scale[:, None, None] * jac_distortion).reshape(-1))

        jacobian = _make_sparse_jacobian(
            row_parts, column_parts, value_parts, observation_count * 2, total_columns
        )
        step = _normal_equation_step(jacobian, weighted_residual, damping, cg_iterations, 1.0e-6)
        if step.norm() < 1.0e-7:
            break
        offset = 0
        if camera_count > 1:
            pose_step = step[offset:offset + pose_columns].reshape(camera_count - 1, 6)
            poses[1:] = se3_exp(pose_step) @ poses[1:]
            offset += pose_columns
        if point_count > 1:
            points[1:] += step[offset:offset + point_columns].reshape(point_count - 1, 3)
            offset += point_columns
        if optimize_intrinsics:
            intrinsic_parameters += step[offset:offset + intrinsic_columns].reshape(group_count, 4)
            offset += intrinsic_columns
        if optimize_distortion:
            distortion += step[offset:offset + distortion_columns].reshape(group_count, 4)

    expanded_intrinsics = parameters_to_intrinsics(intrinsic_parameters)[intrinsic_group]
    expanded_distortion = distortion[intrinsic_group]
    camera_points = (
        torch.einsum("oij,oj->oi", poses[observation_camera, :3, :3], points[observation_point])
        + poses[observation_camera, :3, 3]
    )
    final_projection = project_with_distortion(
        camera_points,
        expanded_intrinsics[observation_camera],
        expanded_distortion[observation_camera],
    )
    objective = (
        tracking_confidence
        * robust_penalty((final_projection - observations).square().sum(dim=-1), robust_delta)
    ).sum()
    return BundleAdjustmentResult(poses, points, expanded_intrinsics, expanded_distortion, objective)


__all__ = [
    "BundleAdjustmentResult",
    "MotionAveragingResult",
    "bundle_adjust",
    "maximum_spanning_tree_initialization",
    "robust_rotation_averaging",
    "se3_exp",
    "so3_exp",
    "so3_log",
    "opt_pose_ray",
]
