"""Point-cloud exports for full-sequence SfM results."""

import numpy as np
from plyfile import PlyData, PlyElement


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


__all__ = ["save_ply"]
