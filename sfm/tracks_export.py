"""Compatibility export for the external bundle-adjustment tracks format."""

from pathlib import Path

import numpy as np
import torch


_POINT_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("id", "<u4")
])
_COLOR_DTYPE = np.dtype([("red", "u1"), ("green", "u1"), ("blue", "u1")])
_PROJECTION_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("size", "<f4"), ("id", "<u4"),
    ("z", "<f4"), ("keypoint", "<u8"),
])


def _write_binary_ply(path, properties, records):
    """Write the exact binary-little-endian vertex layout used by image_matcher."""
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {records.shape[0]}",
        *[f"property {kind} {name}" for kind, name in properties],
        "end_header\n",
    ]
    with Path(path).open("wb") as stream:
        stream.write("\n".join(header).encode("utf-8"))
        stream.write(records.tobytes(order="C"))


def _reference_colors(view, frames):
    """Sample one RGB value at each landmark's reference-frame observation."""
    points = view["points"]
    references = view["references"].to(device=points.device, dtype=torch.long)
    observation_points = view["jj"].to(device=points.device, dtype=torch.long)
    observation_frames = view["ii"].to(device=points.device, dtype=torch.long)
    reference_observations = observation_frames == references[observation_points]

    reference_uv = torch.empty(
        points.shape[0], 2, device=points.device, dtype=view["uv"].dtype
    )
    assigned = torch.zeros(points.shape[0], device=points.device, dtype=torch.bool)
    reference_point_ids = observation_points[reference_observations]
    reference_uv[reference_point_ids] = view["uv"][reference_observations]
    assigned[reference_point_ids] = True
    if not bool(assigned.all()):
        missing = torch.nonzero(~assigned, as_tuple=False).squeeze(-1).tolist()
        raise ValueError(f"tracks export points have no reference observation: {missing[:8]}")

    colors = torch.empty(points.shape[0], 3, device=points.device, dtype=torch.float32)
    for reference in references.unique(sorted=True).tolist():
        frame_id = int(view["frame_ids"][reference])
        if frame_id not in frames.dense:
            raise ValueError(f"tracks export reference frame {frame_id} has no stored image")
        image = frames.dense[frame_id][0].to(device=points.device, dtype=torch.float32)
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"reference frame {frame_id} image must have shape [3,H,W]")
        point_ids = torch.nonzero(references == reference, as_tuple=False).squeeze(-1)
        pixels = reference_uv[point_ids].round().long()
        pixels[:, 0].clamp_(0, image.shape[-1] - 1)
        pixels[:, 1].clamp_(0, image.shape[-2] - 1)
        colors[point_ids] = image[:, pixels[:, 1], pixels[:, 0]].transpose(0, 1)
    return colors


def _projection_coordinates(view, pixel_transforms):
    """Map optimizer pixels into the image coordinates consumed by external BA."""
    uv = view["uv"]
    if pixel_transforms is None:
        return uv
    transforms = torch.as_tensor(pixel_transforms, device=uv.device, dtype=uv.dtype)
    if transforms.ndim == 2:
        transforms = transforms.unsqueeze(0).expand(view["frame_ids"].numel(), -1, -1)
    expected = (view["frame_ids"].numel(), 3, 3)
    if tuple(transforms.shape) != expected:
        raise ValueError(f"pixel_transforms must have shape {expected}")
    homogeneous = torch.cat((uv, torch.ones_like(uv[:, :1])), dim=-1)
    mapped = torch.einsum("oij,oj->oi", transforms[view["ii"]], homogeneous)
    if not bool(torch.isfinite(mapped).all()) or bool((mapped[:, 2].abs() < 1.0e-8).any()):
        raise ValueError("pixel transform produced invalid homogeneous coordinates")
    return mapped[:, :2] / mapped[:, 2:3]


def _write_doc_xml(path, projection_counts, point_count):
    """Write the point-cloud manifest while leaving nonessential meta information empty."""
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<point_cloud version="1.7.0">',
        "\t<params>",
        "\t\t<dataType>uint8</dataType>",
        "\t\t<bands>",
        '\t\t\t<band label="Red" />',
        '\t\t\t<band label="Green" />',
        '\t\t\t<band label="Blue" />',
        "\t\t</bands>",
        "\t</params>",
        f'\t<tracks path="tracks.ply" count="{point_count}" />',
        f'\t<points component_id="0" path="points0.ply" count="{point_count}" />',
    ]
    lines.extend(
        f'\t<projections camera_id="{frame_id}" path="p{frame_id}.ply" count="{count}" />'
        for frame_id, count in projection_counts
    )
    lines.extend(("\t<meta />", "</point_cloud>", ""))
    Path(path).write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def save_ba_tracks(output_dir, view, frames, pixel_transforms=None):
    """Export the global BA input as doc.xml, points0.ply, tracks.ply, and pN.ply."""
    required = {"frame_ids", "points", "references", "ii", "jj", "uv"}
    missing = sorted(required.difference(view))
    if missing:
        raise KeyError(f"tracks export view is missing fields: {missing}")
    point_count = int(view["points"].shape[0])
    observation_count = int(view["uv"].shape[0])
    if point_count < 1:
        raise ValueError("tracks export requires at least one point")
    if view["points"].shape != (point_count, 3):
        raise ValueError("tracks export points must have shape [P,3]")
    if view["uv"].shape != (observation_count, 2):
        raise ValueError("tracks export observations must have shape [O,2]")
    if view["ii"].shape != (observation_count,) or view["jj"].shape != (observation_count,):
        raise ValueError("tracks export observation indices must have shape [O]")
    if not bool(torch.isfinite(view["points"]).all()):
        raise ValueError("tracks export points contain non-finite values")
    if bool((view["jj"] < 0).any()) or bool((view["jj"] >= point_count).any()):
        raise ValueError("tracks export contains an invalid point index")
    camera_count = int(view["frame_ids"].numel())
    if bool((view["ii"] < 0).any()) or bool((view["ii"] >= camera_count).any()):
        raise ValueError("tracks export contains an invalid camera index")
    if view["frame_ids"].unique().numel() != camera_count:
        raise ValueError("tracks export frame IDs must be unique")

    colors = _reference_colors(view, frames)
    if not bool(torch.isfinite(colors).all()):
        raise ValueError("tracks export colors contain non-finite values")
    uv = _projection_coordinates(view, pixel_transforms)
    if not bool(torch.isfinite(uv).all()):
        raise ValueError("tracks export observations contain non-finite values")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("p*.ply"):
        if stale.stem[1:].isdigit():
            stale.unlink()
    export_ids = np.arange(1, point_count + 1, dtype=np.uint32)

    point_records = np.empty(point_count, dtype=_POINT_DTYPE)
    xyz = view["points"].detach().cpu().numpy().astype(np.float32)
    point_records["x"], point_records["y"], point_records["z"] = xyz.T
    point_records["id"] = export_ids
    _write_binary_ply(
        output_dir / "points0.ply",
        (("float", "x"), ("float", "y"), ("float", "z"), ("int", "id")),
        point_records,
    )

    color_records = np.empty(point_count, dtype=_COLOR_DTYPE)
    colors = (colors.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    color_records["red"], color_records["green"], color_records["blue"] = colors.T
    _write_binary_ply(
        output_dir / "tracks.ply",
        (("uchar", "red"), ("uchar", "green"), ("uchar", "blue")),
        color_records,
    )

    uv = uv.detach().cpu().numpy().astype(np.float32)
    frame_ids = view["frame_ids"].detach().cpu().numpy().astype(np.int64)
    observation_frames = frame_ids[view["ii"].detach().cpu().numpy()]
    observation_points = view["jj"].detach().cpu().numpy()
    projection_counts = []
    for frame_id in sorted(map(int, frame_ids.tolist())):
        selected = observation_frames == frame_id
        records = np.empty(int(selected.sum()), dtype=_PROJECTION_DTYPE)
        records["x"], records["y"] = uv[selected].T
        records["size"] = 1.0
        records["id"] = export_ids[observation_points[selected]]
        records["z"] = -1.0
        records["keypoint"] = 0
        _write_binary_ply(
            output_dir / f"p{frame_id}.ply",
            (
                ("float", "x"), ("float", "y"), ("float", "size"), ("int", "id"),
                ("float", "z"), ("uint64", "keypoint"),
            ),
            records,
        )
        projection_counts.append((frame_id, records.shape[0]))
    _write_doc_xml(output_dir / "doc.xml", projection_counts, point_count)
    return {
        "path": output_dir,
        "camera_count": len(projection_counts),
        "point_count": point_count,
        "observation_count": observation_count,
    }


__all__ = ["save_ba_tracks"]
