"""Pose-graph initialization for Glob3R Secs. 3.2 and 3.3."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .matching import Tracks


@dataclass(frozen=True)
class PoseGraph:
    """Relative-pose edges weighted by their shared valid track count."""

    source: torch.Tensor
    target: torch.Tensor
    target_from_source: torch.Tensor
    weight: torch.Tensor


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def so3_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Stable exponential map from axis-angle vectors to SO(3)."""

    theta2 = rotation_vector.square().sum(dim=-1, keepdim=True)
    theta = theta2.clamp_min(1.0e-16).sqrt()
    small = theta2 < 1.0e-8
    safe_theta = theta.clamp_min(1.0e-8)
    safe_theta2 = theta2.clamp_min(1.0e-8)
    a = torch.where(
        small,
        1 - theta2 / 6 + theta2.square() / 120,
        torch.sin(theta) / safe_theta,
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
    """Stable logarithm map from SO(3) to axis-angle vectors."""

    cosine = (
        (rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) * 0.5
    ).clamp(-1, 1)
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
    scale = torch.where(
        theta.abs() < 1.0e-5,
        0.5 + theta.square() / 12,
        theta / (2 * sine.clamp_min(1.0e-8)),
    )
    return scale[..., None] * vee


def build_pose_graph(tracks: Tracks, T_CWs: torch.Tensor) -> PoseGraph:
    """Build paper Sec. 3.2 edges from shared valid track observations."""

    camera_count = T_CWs.shape[0]
    if tracks.mask.shape[0] != camera_count:
        raise ValueError("track masks and camera poses have different frame counts")

    observations = tracks.mask.to(dtype=T_CWs.dtype)
    shared = observations @ observations.transpose(0, 1)
    source, target = torch.nonzero(
        torch.triu(shared > 0, diagonal=1), as_tuple=True
    )
    if source.numel() == 0 and camera_count > 1:
        raise RuntimeError("tracks do not induce any pose-graph edges")

    target_from_source = (
        T_CWs[target] @ torch.linalg.inv(T_CWs[source])
    )
    weight = shared[source, target]
    return PoseGraph(source, target, target_from_source, weight)


def maximum_spanning_tree_initialization(
    camera_count: int,
    graph: PoseGraph,
    root: int = 0,
) -> torch.Tensor:
    """Initialize global world-to-camera poses from the maximum spanning tree."""

    if camera_count < 1:
        raise ValueError("camera_count must be positive")
    if not 0 <= root < camera_count:
        raise ValueError(f"root {root} is outside [0, {camera_count})")

    relative = graph.target_from_source
    T_CWs = torch.eye(4, device=relative.device, dtype=relative.dtype).repeat(
        camera_count, 1, 1
    )
    visited = {root}
    while len(visited) < camera_count:
        best = None
        for edge in range(graph.source.numel()):
            source = int(graph.source[edge])
            target = int(graph.target[edge])
            if (source in visited) == (target in visited):
                continue
            candidate = (float(graph.weight[edge]), edge, source, target)
            if best is None or candidate[0] > best[0]:
                best = candidate

        if best is None:
            missing = sorted(set(range(camera_count)).difference(visited))
            raise RuntimeError(
                f"pose graph is disconnected; unreachable cameras: {missing}"
            )

        _, edge, source, target = best
        if source in visited:
            T_CWs[target] = relative[edge] @ T_CWs[source]
            visited.add(target)
        else:
            T_CWs[source] = torch.linalg.inv(relative[edge]) @ T_CWs[target]
            visited.add(source)
    return T_CWs


def _huber_weight(residual: torch.Tensor, delta: float) -> torch.Tensor:
    norm = residual.square().sum(dim=-1).clamp_min(1.0e-16).sqrt()
    return torch.where(norm <= delta, torch.ones_like(norm), delta / norm)


def _conjugate_gradient(
    operator,
    rhs: torch.Tensor,
    max_iterations: int,
    tolerance: float,
) -> torch.Tensor:
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    squared_norm = torch.dot(residual, residual)
    if squared_norm.sqrt() <= torch.finfo(rhs.dtype).eps:
        return solution
    initial_norm = squared_norm.sqrt()
    for _ in range(max_iterations):
        product = operator(direction)
        denominator = torch.dot(direction, product).clamp_min(
            torch.finfo(rhs.dtype).eps
        )
        alpha = squared_norm / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * product
        next_squared_norm = torch.dot(residual, residual)
        if next_squared_norm.sqrt() <= tolerance * initial_norm:
            break
        direction = residual + (
            next_squared_norm / squared_norm.clamp_min(1.0e-20)
        ) * direction
        squared_norm = next_squared_norm
    return solution


def _sparse_normal_step(
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    damping: float,
    cg_iterations: int,
) -> torch.Tensor:
    jacobian = jacobian.coalesce()
    transpose = jacobian.transpose(0, 1)
    gradient = torch.sparse.mm(transpose, residual[:, None]).squeeze(1)

    def operator(vector):
        projected = torch.sparse.mm(jacobian, vector[:, None])
        return (
            torch.sparse.mm(transpose, projected).squeeze(1)
            + damping * vector
        )

    return _conjugate_gradient(operator, -gradient, cg_iterations, 1.0e-6)


def robust_rotation_averaging(
    initial_T_CWs: torch.Tensor,
    graph: PoseGraph,
    iterations: int = 15,
    robust_delta: float = 0.1,
    damping: float = 1.0e-5,
    cg_iterations: int = 100,
) -> torch.Tensor:
    """Robustly average graph rotations before Glob3R Eq. (5)."""

    rotations = initial_T_CWs[:, :3, :3].clone()
    camera_count = rotations.shape[0]
    edge_count = graph.source.numel()
    if camera_count == 1 or edge_count == 0:
        return rotations

    edge_rotations = graph.target_from_source[:, :3, :3]
    active_source = graph.source - 1
    active_target = graph.target - 1
    device, dtype = rotations.device, rotations.dtype

    for _ in range(iterations):
        predicted = (
            rotations[graph.target]
            @ rotations[graph.source].transpose(-1, -2)
        )
        residual = so3_log(edge_rotations.transpose(-1, -2) @ predicted)
        robust = _huber_weight(residual, robust_delta)
        scale = (graph.weight.clamp_min(0) * robust).sqrt()

        def local_residual(ds, dt, Rs, Rt, measured):
            estimate = (
                so3_exp(dt) @ Rt @ (so3_exp(ds) @ Rs).transpose(-1, -2)
            )
            return so3_log(measured.transpose(-1, -2) @ estimate)

        zeros = torch.zeros(edge_count, 3, device=device, dtype=dtype)
        with torch.enable_grad():
            jacobian_fn = torch.func.vmap(
                torch.func.jacrev(local_residual, argnums=(0, 1))
            )
            source_J, target_J = jacobian_fn(
                zeros,
                zeros,
                rotations[graph.source],
                rotations[graph.target],
                edge_rotations,
            )

        rows = (
            torch.arange(edge_count, device=device)[:, None, None] * 3
            + torch.arange(3, device=device)[None, :, None]
        ).expand(-1, -1, 3)
        local_columns = torch.arange(3, device=device)[None, None, :]
        row_parts, column_parts, value_parts = [], [], []
        for indices, block in (
            (active_source, source_J),
            (active_target, target_J),
        ):
            valid = indices >= 0
            if valid.any():
                columns = indices[valid, None, None] * 3 + local_columns
                row_parts.append(rows[valid].reshape(-1))
                column_parts.append(columns.expand(-1, 3, -1).reshape(-1))
                value_parts.append(
                    (scale[valid, None, None] * block[valid]).reshape(-1)
                )

        values = torch.cat(value_parts)
        jacobian = torch.sparse_coo_tensor(
            torch.stack((torch.cat(row_parts), torch.cat(column_parts))),
            values,
        (edge_count * 3, (camera_count - 1) * 3),
        device=device,
        dtype=dtype,
        check_invariants=True,
    ).coalesce()
        step = _sparse_normal_step(
            jacobian,
            (scale[:, None] * residual).reshape(-1),
            damping,
            cg_iterations,
        )
        if step.norm() < 1.0e-7:
            break
        rotations[1:] = (
            so3_exp(step.reshape(camera_count - 1, 3)) @ rotations[1:]
        )
    return rotations


__all__ = [
    "PoseGraph",
    "build_pose_graph",
    "maximum_spanning_tree_initialization",
    "robust_rotation_averaging",
    "so3_exp",
    "so3_log",
]
