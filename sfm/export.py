"""Point-cloud and camera-wireframe exports for full-sequence SfM results."""

import math
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
import torch


def save_ply(path, points, colors=None, scalar_fields=None):
    """Write a binary PLY with optional RGB colors and scalar metadata."""
    points = points.detach().cpu().numpy().astype(np.float32)
    properties = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None:
        colors = (colors.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        properties += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    arrays = {}
    for name, values in (scalar_fields or {}).items():
        values = values.detach().cpu().numpy()
        if values.ndim != 1 or values.shape[0] != points.shape[0]:
            raise ValueError(f"PLY field {name} must have shape [P]")
        if np.issubdtype(values.dtype, np.integer):
            arrays[name] = values.astype(np.int32)
            properties.append((name, "i4"))
        elif np.issubdtype(values.dtype, np.floating):
            arrays[name] = values.astype(np.float32)
            properties.append((name, "f4"))
        else:
            raise TypeError(f"PLY field {name} has unsupported dtype {values.dtype}")

    vertices = np.empty(points.shape[0], dtype=properties)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    if colors is not None:
        vertices["red"], vertices["green"], vertices["blue"] = colors.T
    for name, values in arrays.items():
        vertices[name] = values
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def save_camera_wireframes(
    path, world_to_camera, scene_points, frame_ids=None
):
    """Write five-vertex camera frustums as OBJ line primitives."""
    if world_to_camera.ndim != 3 or tuple(world_to_camera.shape[-2:]) != (4, 4):
        raise ValueError("camera poses must have shape [C,4,4]")
    camera_count = world_to_camera.shape[0]
    if camera_count == 0:
        raise ValueError("camera wireframe export requires at least one pose")
    if scene_points.ndim != 2 or scene_points.shape[-1] != 3:
        raise ValueError("camera wireframe scene points must have shape [P,3]")

    poses = world_to_camera.detach()
    scene_points = scene_points.detach().to(poses)
    finite_scene = scene_points[torch.isfinite(scene_points).all(-1)]
    size = 0.1
    if finite_scene.numel():
        lower = torch.quantile(finite_scene, 0.05, dim=0)
        upper = torch.quantile(finite_scene, 0.95, dim=0)
        candidate = float(torch.linalg.vector_norm(upper - lower)) * 0.03
        if math.isfinite(candidate) and candidate > 0:
            size = candidate

    camera_vertices = poses.new_tensor(
        (
            (0.0, 0.0, 0.0),
            (-0.6, -0.4, 1.0),
            (0.6, -0.4, 1.0),
            (0.6, 0.4, 1.0),
            (-0.6, 0.4, 1.0),
        )
    ) * size
    camera_edges = torch.tensor(
        ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)),
        device=poses.device,
        dtype=torch.long,
    )
    world_from_camera = torch.linalg.inv(poses)
    vertices = (
        torch.einsum(
            "cij,vj->cvi", world_from_camera[:, :3, :3], camera_vertices
        )
        + world_from_camera[:, None, :3, 3]
    )
    offsets = 5 * torch.arange(camera_count, device=poses.device)[:, None, None]
    edges = (camera_edges[None] + offsets).reshape(-1, 2)

    if frame_ids is None:
        frame_ids = torch.arange(camera_count, device=poses.device)
    else:
        frame_ids = torch.as_tensor(
            frame_ids, device=poses.device, dtype=torch.long
        )
    if tuple(frame_ids.shape) != (camera_count,):
        raise ValueError("camera frame IDs must have shape [C]")

    obj_lines = ["# Camera wireframes"]
    vertices = vertices.cpu()
    edges = edges.cpu()
    for camera_index, frame_id in enumerate(frame_ids.cpu().tolist()):
        obj_lines.append(f"g camera_{frame_id}")
        for x, y, z in vertices[camera_index].tolist():
            obj_lines.append(f"v {x:.9g} {y:.9g} {z:.9g}")
        camera_edges = edges[camera_index * 8:(camera_index + 1) * 8]
        for vertex1, vertex2 in camera_edges.tolist():
            obj_lines.append(f"l {vertex1 + 1} {vertex2 + 1}")
    Path(path).write_text("\n".join(obj_lines) + "\n", encoding="ascii")


__all__ = ["save_camera_wireframes", "save_ply"]
