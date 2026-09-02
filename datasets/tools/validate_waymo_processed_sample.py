"""Validate processed Waymo geometry and render direct and accumulated overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.waymo_processed_dataset import (
    WaymoPi3XDataset,
    accumulate_depth,
    depth_to_points,
    generate_waymo_processed_index,
    load_frame,
    project_depth,
)


def make_overlay(image_rgb, depth, output_path: Path):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    scaled = (255 * (1 - np.clip(depth, 1, 80) / 80)).astype(np.uint8)
    color = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    kernel = np.ones((3, 3), dtype=np.uint8)
    color = cv2.dilate(color, kernel)
    mask = cv2.dilate((depth > 0).astype(np.uint8), kernel).astype(bool)
    overlay = image_bgr.copy()
    overlay[mask] = (
        0.35 * image_bgr[mask] + 0.65 * color[mask]
    ).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Failed to write {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate WaymoPi3XDataset with processed real data"
    )
    parser.add_argument("data_root", type=Path)
    parser.add_argument("index_file", type=Path)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--frame-num", type=int, default=8)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--depth-accumulate", type=int, default=2)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/waymo_validation"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    index = generate_waymo_processed_index(
        args.data_root,
        args.index_file,
        camera_id=args.camera_id,
    )
    if not index["sequences"]:
        raise RuntimeError(f"No processed sequences found under {args.data_root}")

    record = index["sequences"][0]
    segment = Path(record["segment_dir"])
    frame_ids = [int(value) for value in record["frame_ids"]]
    roundtrip_errors = []
    rotation_determinants = []
    rotation_orthogonality = []
    raw_valid_counts = []

    for frame_id in frame_ids:
        frame = load_frame(segment, frame_id, args.camera_id, image=True)
        depth = np.asarray(frame["depth"], dtype=np.float32)
        reprojection = project_depth(
            depth_to_points(depth, frame["K"]),
            frame["K"],
            *depth.shape,
        )
        mask = depth > 0
        roundtrip_errors.append(
            float(np.max(np.abs(reprojection[mask] - depth[mask])))
        )
        rotation = frame["pose"][:3, :3]
        rotation_determinants.append(float(np.linalg.det(rotation)))
        rotation_orthogonality.append(
            float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
        )
        raw_valid_counts.append(int(mask.sum()))

    center_id = frame_ids[len(frame_ids) // 2]
    center = load_frame(segment, center_id, args.camera_id, image=True)
    _, self_depth, _, _ = accumulate_depth(
        segment,
        center_id,
        [center_id],
        args.camera_id,
        0,
    )
    self_accumulation_error = float(
        np.max(np.abs(self_depth - np.asarray(center["depth"])))
    )

    common = dict(
        data_root=args.data_root,
        index_file=args.index_file,
        camera_id=args.camera_id,
        frame_step=args.frame_step,
        resolution=[[args.width, args.height]],
        frame_num=args.frame_num,
        z_far=80,
        aug_crop=1,
        aug_focal=1.0,
        mode="test",
        shuffle=False,
    )
    direct_views = WaymoPi3XDataset(depth_accumulate=0, **common)[0]
    accumulated_views = WaymoPi3XDataset(
        depth_accumulate=args.depth_accumulate,
        **common,
    )[0]

    direct_counts = [int(view["valid_mask"].sum()) for view in direct_views]
    accumulated_counts = [
        int(view["valid_mask"].sum()) for view in accumulated_views
    ]
    first_pose_identity_error = float(
        np.max(np.abs(direct_views[0]["camera_pose"] - np.eye(4)))
    )

    lo = max(0, len(frame_ids) // 2 - args.depth_accumulate)
    hi = min(len(frame_ids), len(frame_ids) // 2 + args.depth_accumulate + 1)
    image, accumulated_depth, _, _ = accumulate_depth(
        segment,
        center_id,
        frame_ids[lo:hi],
        args.camera_id,
        0,
    )
    direct_overlay = args.output_dir / f"frame_{center_id:03d}_depth_overlay.png"
    accumulated_overlay = (
        args.output_dir / f"frame_{center_id:03d}_accum_overlay.png"
    )
    make_overlay(center["image"], center["depth"], direct_overlay)
    make_overlay(image, accumulated_depth, accumulated_overlay)

    summary = {
        "sequence_id": record["sequence_id"],
        "frames": len(frame_ids),
        "raw_valid_depth_min": min(raw_valid_counts),
        "raw_valid_depth_max": max(raw_valid_counts),
        "depth_roundtrip_max_abs_m": max(roundtrip_errors),
        "self_accumulation_max_abs_m": self_accumulation_error,
        "pose_rotation_det_min": min(rotation_determinants),
        "pose_rotation_det_max": max(rotation_determinants),
        "pose_rotation_orthogonality_max": max(rotation_orthogonality),
        "first_rebased_pose_identity_max_abs": first_pose_identity_error,
        "runtime_direct_valid_depth": direct_counts,
        "runtime_accumulated_valid_depth": accumulated_counts,
        "accumulation_mean_growth": float(
            np.mean(accumulated_counts) / np.mean(direct_counts)
        ),
        "direct_overlay": str(direct_overlay.resolve()),
        "accumulated_overlay": str(accumulated_overlay.resolve()),
    }

    if summary["depth_roundtrip_max_abs_m"] != 0:
        raise AssertionError("Depth backprojection and reprojection changed values")
    if summary["self_accumulation_max_abs_m"] != 0:
        raise AssertionError("Single-frame accumulation changed depth values")
    if abs(summary["pose_rotation_det_min"] - 1.0) > 1e-5:
        raise AssertionError("Camera pose rotation determinant is invalid")
    if summary["pose_rotation_orthogonality_max"] > 1e-5:
        raise AssertionError("Camera pose rotation is not orthogonal")
    if summary["first_rebased_pose_identity_max_abs"] > 1e-5:
        raise AssertionError("First returned camera pose was not rebased to identity")
    if any(value <= 0 for value in direct_counts + accumulated_counts):
        raise AssertionError("A runtime view has no valid depth")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
