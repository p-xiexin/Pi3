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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", required=True, help="directory containing one ordered local window"
    )
    parser.add_argument(
        "--backbone-checkpoint", required=True, help="official Pi3 model.safetensors"
    )
    parser.add_argument(
        "--matching-checkpoint", required=True, help="trained Glob3R refinement checkpoint"
    )
    parser.add_argument("--output", default="outputs/glob3r_sfm/result.pt")
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--calibration",
        required=True,
        help="fixed camera calibration YAML",
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
    Ps_W: torch.Tensor,
    RGBs: torch.Tensor | None = None,
    scalar_fields: dict[str, torch.Tensor] | None = None,
) -> None:
    Ps_W = Ps_W.detach().cpu().numpy().astype(np.float32)
    properties = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if RGBs is not None:
        RGBs = (RGBs.detach().cpu().numpy().clip(0, 1) * 255).astype(
            np.uint8
        )
        properties.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    scalar_fields = scalar_fields or {}
    properties.extend((name, "f4") for name in scalar_fields)
    vertices = np.empty(Ps_W.shape[0], dtype=properties)
    vertices["x"], vertices["y"], vertices["z"] = Ps_W.T
    if RGBs is not None:
        vertices["red"], vertices["green"], vertices["blue"] = RGBs.T
    for name, values in scalar_fields.items():
        vertices[name] = values.detach().cpu().numpy().astype(np.float32)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def main():
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.height % 14 or args.width % 14:
        raise ValueError("height and width must be divisible by Pi3 patch size 14")
    Is, paths = load_image_sequence(args.images, (args.height, args.width))
    print("Input frame order:")
    for index, path in enumerate(paths):
        print(f"  [{index:04d}] {path.name}")

    calibration_path = Path(args.calibration)
    K, calibration_width, calibration_height = load_calibration(
        calibration_path
    )
    K[0] *= args.width / calibration_width
    K[1] *= args.height / calibration_height
    print(f"Using fixed calibration from {calibration_path}")
    print(f"Fixed resized intrinsics:\n{K}")
    print("Lens distortion is disabled for local optimization.")
    model = load_glob3r_for_sfm(
        args.backbone_checkpoint, args.matching_checkpoint, device=args.device
    )
    config = Glob3RSfMConfig(
        keyframe_projection_threshold=args.keyframe_threshold,
    )
    pipeline = Glob3RSfMPipeline(model, config)
    matching_output = output.parent / "matching"

    result = pipeline.run(
        Is.to(args.device),
        K.to(args.device),
        visualization_dir=matching_output,
    )
    print(f"Selected keyframes: {result.keyframes.tolist()}")
    if (
        result.raw_Ps_W.shape[0] == 0
        or result.raw_RGBs.shape[0] == 0
        or result.dense_Ps_W is None
        or result.dense_RGBs is None
        or result.dense_Ps_W.shape[0] == 0
    ):
        raise RuntimeError(
            "Pi3 or SfM reconstruction produced no point cloud; both PLY results are required."
        )

    torch.save(
        {
            "image_paths": [str(path) for path in paths],
            "world_to_camera": result.T_CWs.cpu(),
            "camera_to_world": result.T_WCs.cpu(),
            "intrinsics": result.Ks.cpu(),
            "distortion": result.deltas.cpu(),
            "keyframes": result.keyframes,
            "track_references": result.tracks.rs.cpu(),
            "track_anchor_points": result.tracks.Xs_Cr.cpu(),
            "track_observations": result.tracks.us.cpu(),
            "track_mask": result.tracks.mask.cpu(),
            "track_weights": result.tracks.ws.cpu(),
            "track_inliers": result.track_inliers.cpu(),
            "sparse_points_before_optimization": result.Xs_W0.cpu(),
            "sparse_points": result.Xs_W.cpu(),
            "pi3_raw_points": result.raw_Ps_W.cpu(),
            "pi3_raw_colors": result.raw_RGBs.cpu(),
            "pi3_raw_frame_ids": result.raw_frame_ids.cpu(),
            "pi3_sfm_points": result.dense_Ps_W.cpu(),
            "pi3_sfm_colors": result.dense_RGBs.cpu(),
            "pi3_sfm_frame_ids": result.dense_frame_ids.cpu(),
            "eq5_loss": result.eq5_loss.cpu(),
            "eq6_loss": result.eq6_loss.cpu(),
        },
        output,
    )
    raw_point_cloud_output = output.parent / "pi3_raw.ply"
    sfm_point_cloud_output = output.parent / "pi3_sfm.ply"
    sparse_point_cloud_output = output.parent / "sparse_tracks.ply"
    save_ply(
        raw_point_cloud_output,
        result.raw_Ps_W,
        result.raw_RGBs,
        {"frame_id": result.raw_frame_ids},
    )
    save_ply(
        sfm_point_cloud_output,
        result.dense_Ps_W,
        result.dense_RGBs,
        {"frame_id": result.dense_frame_ids},
    )
    observation_count = result.tracks.mask.sum(dim=0)
    observation_count[~result.track_inliers] = -1
    save_ply(
        sparse_point_cloud_output,
        result.Xs_W,
        scalar_fields={
            "track_id": torch.arange(
                result.Xs_W.shape[0], device=result.Xs_W.device
            ),
            "observation_count": observation_count,
        },
    )
    print(f"Saved SfM result to {output}")
    print(f"Saved raw Pi3 point cloud to {raw_point_cloud_output}")
    print(f"Saved SfM-optimized point cloud to {sfm_point_cloud_output}")
    print(f"Saved optimized sparse tracks to {sparse_point_cloud_output}")


if __name__ == "__main__":
    main()
