"""SO3 helpers and sparse pose-graph initialization."""

import torch


def skew(vector):
    """Convert trailing 3-vectors into skew-symmetric matrices."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        *vector.shape[:-1], 3, 3
    )


def so3_exp(vector):
    """Map axis-angle vectors to rotation matrices with a stable small-angle series."""
    theta2 = vector.square().sum(-1, keepdim=True)
    theta = theta2.clamp_min(1.0e-16).sqrt()
    small = theta2 < 1.0e-8
    a = torch.where(
        small,
        1 - theta2 / 6 + theta2.square() / 120,
        torch.sin(theta) / theta.clamp_min(1.0e-8),
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24 + theta2.square() / 720,
        (1 - torch.cos(theta)) / theta2.clamp_min(1.0e-8),
    )
    K = skew(vector)
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype)
    return eye + a[..., None] * K + b[..., None] * (K @ K)


def so3_log(rotation):
    """Map rotation matrices to axis-angle vectors."""
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        -1,
    )
    sine = 0.5 * vee.square().sum(-1).clamp_min(1.0e-16).sqrt()
    theta = torch.atan2(sine, cosine)
    scale = torch.where(
        theta.abs() < 1.0e-5,
        0.5 + theta.square() / 12,
        theta / (2 * sine.clamp_min(1.0e-8)),
    )
    return scale[..., None] * vee


def se3_exp(vector):
    """Map translation-first twists to homogeneous SE(3) transforms."""
    translation, rotation = vector[..., :3], vector[..., 3:]
    theta2 = rotation.square().sum(-1, keepdim=True)
    theta = theta2.clamp_min(1.0e-16).sqrt()
    small = theta2 < 1.0e-8
    b = torch.where(
        small,
        0.5 - theta2 / 24 + theta2.square() / 720,
        (1 - torch.cos(theta)) / theta2.clamp_min(1.0e-8),
    )
    c = torch.where(
        small,
        1 / 6 - theta2 / 120 + theta2.square() / 5040,
        (theta - torch.sin(theta)) / (theta2 * theta).clamp_min(1.0e-8),
    )
    K = skew(rotation)
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype)
    R = so3_exp(rotation)
    V = eye + b[..., None] * K + c[..., None] * (K @ K)
    T = torch.eye(4, device=vector.device, dtype=vector.dtype)
    T = T.expand(*vector.shape[:-1], 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = torch.einsum("...ij,...j->...i", V, translation)
    return T


def camera_centers(poses):
    """Return world-space centers from world-to-camera transforms."""
    return -torch.einsum("...ji,...j->...i", poses[..., :3, :3], poses[..., :3, 3])


def maximum_spanning_tree(camera_count, source, target, relative, weight, root=0):
    """Propagate absolute poses over the highest-confidence connected tree."""
    poses = torch.eye(4, device=relative.device, dtype=relative.dtype).repeat(camera_count, 1, 1)
    parent = list(range(camera_count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    tree = []
    for edge in torch.argsort(weight, descending=True).tolist():
        i, j = int(source[edge]), int(target[edge])
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        parent[ri] = rj
        tree.append((i, j, edge))
        if len(tree) == camera_count - 1:
            break
    if len(tree) != camera_count - 1:
        raise RuntimeError("pose graph is disconnected")
    adjacency = [[] for _ in range(camera_count)]
    for i, j, edge in tree:
        adjacency[i].append((j, edge, False))
        adjacency[j].append((i, edge, True))
    visited, queue = {int(root)}, [int(root)]
    for i in queue:
        for j, edge, inverse in adjacency[i]:
            if j in visited:
                continue
            transform = torch.linalg.inv(relative[edge]) if inverse else relative[edge]
            poses[j] = transform @ poses[i]
            visited.add(j)
            queue.append(j)
    return poses


def _cg(operator, rhs, iterations=100, tolerance=1.0e-6):
    """Solve the sparse rotation normal equations with conjugate gradients."""
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    norm2 = torch.dot(residual, residual)
    initial = norm2.sqrt()
    if initial <= torch.finfo(rhs.dtype).eps:
        return x
    for _ in range(iterations):
        product = operator(direction)
        denominator = torch.dot(direction, product)
        if denominator <= torch.finfo(rhs.dtype).eps:
            break
        alpha = norm2 / denominator
        x += alpha * direction
        residual -= alpha * product
        next_norm2 = torch.dot(residual, residual)
        if next_norm2.sqrt() <= tolerance * initial:
            break
        direction = residual + next_norm2 / norm2.clamp_min(1.0e-20) * direction
        norm2 = next_norm2
    return x


def average_rotations(initial, source, target, relative, edge_weight, iterations=15):
    """Robustly average relative rotations while fixing camera zero as the gauge."""
    rotations = initial[:, :3, :3].clone()
    if rotations.shape[0] == 1 or source.numel() == 0:
        return rotations
    edge_rotations = relative[:, :3, :3]
    active_source, active_target = source - 1, target - 1
    edge_count = source.numel()
    for _ in range(iterations):
        predicted = rotations[target] @ rotations[source].transpose(-1, -2)
        residual = so3_log(edge_rotations.transpose(-1, -2) @ predicted)
        norm = residual.norm(dim=-1).clamp_min(1.0e-8)
        scale = (edge_weight * torch.where(norm <= 0.1, 1.0, 0.1 / norm)).sqrt()

        def edge_error(ds, dt, Rs, Rt, measured):
            estimate = so3_exp(dt) @ Rt @ (so3_exp(ds) @ Rs).transpose(-1, -2)
            return so3_log(measured.transpose(-1, -2) @ estimate)

        zeros = torch.zeros(edge_count, 3, device=rotations.device, dtype=rotations.dtype)
        with torch.enable_grad():
            Js, Jt = torch.func.vmap(torch.func.jacrev(edge_error, argnums=(0, 1)))(
                zeros, zeros, rotations[source], rotations[target], edge_rotations
            )
        edge_rows = torch.arange(edge_count, device=rotations.device)[:, None, None]
        block_rows = torch.arange(3, device=rotations.device)[None, :, None]
        rows = (edge_rows * 3 + block_rows).expand(-1, -1, 3)
        local_columns = torch.arange(3, device=rotations.device)[None, None]
        row_parts, column_parts, value_parts = [], [], []
        for indices, blocks in ((active_source, Js), (active_target, Jt)):
            valid = indices >= 0
            if valid.any():
                columns = indices[valid, None, None] * 3 + local_columns
                row_parts.append(rows[valid].reshape(-1))
                column_parts.append(columns.expand(-1, 3, -1).reshape(-1))
                value_parts.append((scale[valid, None, None] * blocks[valid]).reshape(-1))
        J = torch.sparse_coo_tensor(
            torch.stack((torch.cat(row_parts), torch.cat(column_parts))), torch.cat(value_parts),
            (edge_count * 3, (rotations.shape[0] - 1) * 3),
            device=rotations.device,
            dtype=rotations.dtype,
            check_invariants=True,
        ).coalesce()
        JT = J.transpose(0, 1)
        weighted_residual = (scale[:, None] * residual).reshape(-1)
        gradient = torch.sparse.mm(JT, weighted_residual[:, None]).squeeze(1)

        def operator(x):
            return torch.sparse.mm(JT, torch.sparse.mm(J, x[:, None])).squeeze(1) + 1.0e-5 * x

        step = _cg(operator, -gradient)
        if step.norm() < 1.0e-7:
            break
        rotations[1:] = so3_exp(step.reshape(-1, 3)) @ rotations[1:]
    return rotations


__all__ = [
    "average_rotations",
    "camera_centers",
    "maximum_spanning_tree",
    "se3_exp",
    "so3_exp",
    "so3_log",
]
