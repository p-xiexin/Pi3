"""Run Glob3R matching and BA on one local image window."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from local_opt.inference import (
    load_calibration,
    load_glob3r_for_sfm,
    load_image_sequence,
)
from local_opt.sfm import Glob3RSfMConfig, Glob3RSfMPipeline
from local_opt.visualization import save_keyframe_matching_overviews


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="directory containing one ordered local window")
    parser.add_argument("--backbone-checkpoint", required=True, help="official Pi3 model.safetensors")
    parser.add_argument("--matching-checkpoint", required=True, help="trained Glob3R refinement checkpoint")
    parser.add_argument("--output", default="outputs/glob3r_sfm/result.pt")
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--calibration",
        required=True,
        help="fixed camera calibration YAML",
    )
    parser.add_argument("--tracking-points", type=int, default=512)
    parser.add_argument(
        "--optimize-intrinsics",
        action="store_true",
        help="optimize one shared fx, fy, cx, cy in bundle adjustment",
    )
    parser.add_argument(
        "--optimize-distortion",
        action="store_true",
        help="optimize one shared k1, k2, p1, p2 in bundle adjustment",
    )
    parser.add_argument(
        "--keyframe-threshold",
        type=float,
        default=0.5,
        help="Eq. (4) valid-projection threshold as a fraction of image pixels",
    )
    return parser.parse_args()


def save_ply(
    path: Path,
    points: torch.Tensor,
    colors: torch.Tensor | None = None,
    scalar_fields: dict[str, torch.Tensor] | None = None,
) -> None:
    points = points.detach().cpu().numpy().astype(np.float32)
    properties = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None:
        colors = (colors.detach().cpu().numpy().clip(0, 1) * 255).astype(
            np.uint8
        )
        properties.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    scalar_fields = scalar_fields or {}
    properties.extend((name, "f4") for name in scalar_fields)
    vertices = np.empty(points.shape[0], dtype=properties)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    if colors is not None:
        vertices["red"], vertices["green"], vertices["blue"] = colors.T
    for name, values in scalar_fields.items():
        vertices[name] = values.detach().cpu().numpy().astype(np.float32)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.height % 14 or args.width % 14:
        raise ValueError("height and width must be divisible by Pi3 patch size 14")
    images, paths = load_image_sequence(args.images, (args.height, args.width))
    print("Input frame order:")
    for index, path in enumerate(paths):
        print(f"  [{index:04d}] {path.name}")

    calibration_path = Path(args.calibration)
    intrinsics, calibration_width, calibration_height = load_calibration(
        calibration_path
    )
    intrinsics[0] *= args.width / calibration_width
    intrinsics[1] *= args.height / calibration_height
    calibration_role = "initial" if args.optimize_intrinsics else "fixed"
    print(f"Using {calibration_role} calibration from {calibration_path}")
    print(f"{calibration_role.capitalize()} resized intrinsics:\n{intrinsics}")
    if args.optimize_distortion:
        print("Lens distortion starts at zero and is optimized as one shared model.")
    else:
        print("Lens distortion is disabled (all coefficients fixed to zero).")
    model = load_glob3r_for_sfm(
        args.backbone_checkpoint, args.matching_checkpoint, device=args.device
    )
    config = Glob3RSfMConfig(
        tracking_points_per_keyframe=args.tracking_points,
        keyframe_projection_threshold=args.keyframe_threshold,
        optimize_intrinsics=args.optimize_intrinsics,
        optimize_distortion=args.optimize_distortion,
    )
    pipeline = Glob3RSfMPipeline(model, config)

    def save_matching(images, matches):
        paths = save_keyframe_matching_overviews(
            output.parent / "matching", images, matches
        )
        print(f"Saved {len(paths)} matching overview(s) to {output.parent / 'matching'}")

    result = pipeline.run(
        images.to(args.device),
        intrinsics.to(args.device),
        matching_callback=save_matching,
    )
    print(f"Selected keyframes: {result.keyframes}")
    if (
        result.raw_points.shape[0] == 0
        or result.raw_colors.shape[0] == 0
        or result.dense_points is None
        or result.dense_colors is None
        or result.dense_points.shape[0] == 0
    ):
        raise RuntimeError(
            "Pi3 or SfM reconstruction produced no point cloud; both PLY results are required."
        )

    torch.save(
        {
            "image_paths": [str(path) for path in paths],
            "world_to_camera": result.world_to_camera.cpu(),
            "camera_to_world": result.camera_to_world.cpu(),
            "points_3d_before_ba": result.points_3d_before_ba.cpu(),
            "points_3d": result.points_3d.cpu(),
            "intrinsics": result.intrinsics.cpu(),
            "distortion": result.distortion.cpu(),
            "keyframes": result.keyframes,
            "observations": result.tracks.observations.cpu(),
            "observation_camera": result.tracks.camera_indices.cpu(),
            "observation_point": result.tracks.point_indices.cpu(),
            "tracking_confidence": result.tracks.confidence.cpu(),
            "predicted_depth": result.tracks.predicted_depth.cpu(),
            "pi3_raw_points": result.raw_points.cpu(),
            "pi3_raw_colors": result.raw_colors.cpu(),
            "pi3_raw_frame_ids": result.raw_frame_ids.cpu(),
            "pi3_sfm_points": result.dense_points.cpu(),
            "pi3_sfm_colors": result.dense_colors.cpu(),
            "pi3_sfm_frame_ids": result.dense_frame_ids.cpu(),
        },
        output,
    )
    raw_point_cloud_output = output.parent / "pi3_raw.ply"
    sfm_point_cloud_output = output.parent / "pi3_sfm.ply"
    sparse_tracks_before_ba_output = output.parent / "sparse_tracks_before_ba.ply"
    sparse_tracks_after_ba_output = output.parent / "sparse_tracks_after_ba.ply"
    save_ply(
        raw_point_cloud_output,
        result.raw_points,
        result.raw_colors,
        {"frame_id": result.raw_frame_ids},
    )
    save_ply(
        sfm_point_cloud_output,
        result.dense_points,
        result.dense_colors,
        {"frame_id": result.dense_frame_ids},
    )
    observation_count = torch.bincount(
        result.tracks.point_indices,
        minlength=result.points_3d.shape[0],
    )
    # Observations are inserted with their reference/keyframe entry first.
    # Preserve that anchor frame per optimized BA point so the sparse and dense
    # clouds can be colored with the same ``frame_id`` field.
    observation_order = torch.arange(
        result.tracks.point_indices.numel(),
        device=result.tracks.point_indices.device,
    )
    first_observation = torch.full(
        (result.points_3d.shape[0],),
        observation_order.numel(),
        device=result.tracks.point_indices.device,
        dtype=torch.long,
    )
    first_observation.scatter_reduce_(
        0,
        result.tracks.point_indices,
        observation_order,
        reduce="amin",
        include_self=True,
    )
    if (first_observation == observation_order.numel()).any():
        raise RuntimeError("a BA point has no track observation")
    sparse_frame_ids = result.tracks.camera_indices[first_observation]
    sparse_scalar_fields = {
        "frame_id": sparse_frame_ids,
        "observation_count": observation_count,
    }
    save_ply(
        sparse_tracks_before_ba_output,
        result.points_3d_before_ba,
        scalar_fields=sparse_scalar_fields,
    )
    save_ply(
        sparse_tracks_after_ba_output,
        result.points_3d,
        scalar_fields=sparse_scalar_fields,
    )
    print(f"Saved SfM result to {output}")
    print(f"Saved raw Pi3 point cloud to {raw_point_cloud_output}")
    print(f"Saved SfM-optimized point cloud to {sfm_point_cloud_output}")
    print(f"Saved sparse tracks before BA to {sparse_tracks_before_ba_output}")
    print(f"Saved sparse tracks after BA to {sparse_tracks_after_ba_output}")


if __name__ == "__main__":
    main()
