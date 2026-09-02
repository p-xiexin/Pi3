"""Build relocatable sampling indexes for Pi3 sequence datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np


INDEX_VERSION = 1


_HYPERSIM_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def _quaternion_rotation(values: list[str] | np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(values, dtype=np.float64)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Zero-length quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _pose(values: list[str]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = _quaternion_rotation(values[3:])
    pose[:3, 3] = np.asarray(values[:3], dtype=np.float32)
    return pose


def _tartanair_pose(values: np.ndarray) -> np.ndarray:
    z, x, y = np.asarray(values[:3], dtype=np.float32)
    qz, qx, qy, qw = np.asarray(values[3:], dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array(
        [
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
        ],
        dtype=np.float32,
    )
    pose[:3, 3] = np.array([x, y, z], dtype=np.float32)
    return pose


def _read_records(path: Path, fields: int) -> list[tuple[float, list[str]]]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != fields + 1:
            raise ValueError(f"{path}:{line_no}: expected {fields + 1} fields")
        records.append((float(parts[0]), parts[1:]))
    return records


def _nearest(
    timestamp: float,
    records: list[tuple[float, list[str]]],
    times: np.ndarray,
    tolerance: float,
) -> tuple[float, list[str]] | None:
    position = int(np.searchsorted(times, timestamp))
    candidates = [i for i in (position - 1, position) if 0 <= i < len(records)]
    if not candidates:
        return None
    index = min(candidates, key=lambda i: abs(records[i][0] - timestamp))
    return records[index] if abs(records[index][0] - timestamp) <= tolerance else None


def _nearest_pose(
    timestamp: float,
    poses: list[tuple[float, list[str]]],
    pose_times: np.ndarray,
    tolerance: float,
) -> np.ndarray | None:
    match = _nearest(timestamp, poses, pose_times, tolerance)
    return None if match is None else _pose(match[1])


def _read_intrinsics(path: Path) -> np.ndarray:
    record = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(record["intrinsic_matrix"], dtype=np.float32).reshape(
        3, 3, order="F"
    )


def _read_trajectory(path: Path) -> list[np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) % 5:
        raise ValueError(f"Invalid Open3D trajectory layout in {path}")
    poses = []
    for start in range(0, len(lines), 5):
        pose = np.asarray(
            [[float(value) for value in lines[start + row].split()] for row in range(1, 5)],
            dtype=np.float32,
        )
        if pose.shape != (4, 4):
            raise ValueError(f"Invalid pose shape {pose.shape} in {path}")
        poses.append(pose)
    return poses


def _load_camera_parameters(path: Path) -> dict[str, dict]:
    parameters = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            parameters[row["scene_name"]] = {
                "width": int(float(row["settings_output_img_width"])),
                "height": int(float(row["settings_output_img_height"])),
                "meters_per_asset_unit": float(row["settings_units_info_meters_scale"]),
                "M_cam_from_uv": np.array(
                    [
                        [float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)]
                        for i in range(3)
                    ],
                    dtype=np.float64,
                ),
            }
    return parameters


def _ray_matrix_and_intrinsics(
    matrix: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    pixel_to_uv = np.array(
        [
            [2.0 / width, 0.0, 1.0 / width - 1.0],
            [0.0, -2.0 / height, 1.0 - 1.0 / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ray_matrix = _HYPERSIM_TO_OPENCV @ matrix @ pixel_to_uv
    intrinsics = np.linalg.inv(ray_matrix)
    intrinsics /= intrinsics[2, 2]
    return ray_matrix.astype(np.float32), intrinsics.astype(np.float32)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Indexed path {path} is outside data root {root}") from exc


def _sequence_id(path: Path, root: Path) -> str:
    relative = _relative(path, root)
    return relative if relative != "." else path.name


def build_tartanair(root: Path, args: argparse.Namespace) -> list[dict]:
    intrinsics = np.array(
        [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    sequences = []
    for pose_path in sorted(root.rglob("pose_left.txt")):
        sequence_dir = pose_path.parent
        image_dir = sequence_dir / "image_left"
        depth_dir = sequence_dir / "depth_left"
        if not image_dir.is_dir() or not depth_dir.is_dir():
            continue
        poses = np.atleast_2d(np.loadtxt(pose_path, dtype=np.float32))
        frames = []
        for image in sorted(image_dir.glob("*_left.png")):
            try:
                frame_id = int(image.name.split("_", 1)[0])
            except ValueError:
                continue
            depth = depth_dir / f"{frame_id:06d}_left_depth.npy"
            if depth.is_file() and frame_id < len(poses):
                frames.append(
                    {
                        "frame_id": frame_id,
                        "image": _relative(image, root),
                        "depth": _relative(depth, root),
                        "camera_pose": _tartanair_pose(poses[frame_id]),
                    }
                )
        if frames:
            sequences.append(
                {
                    "sequence_id": _sequence_id(sequence_dir, root),
                    "intrinsics": intrinsics.copy(),
                    "frames": frames,
                }
            )
    return sequences


def build_tum_rgbd(root: Path, args: argparse.Namespace) -> list[dict]:
    required = ("rgb.txt", "depth.txt", "groundtruth.txt")
    sequence_dirs = [root] if all((root / name).is_file() for name in required) else [
        path.parent
        for path in root.rglob("rgb.txt")
        if all((path.parent / name).is_file() for name in required)
    ]
    sequences = []
    for sequence_dir in sorted(set(sequence_dirs)):
        rgb = _read_records(sequence_dir / "rgb.txt", 1)
        depth = _read_records(sequence_dir / "depth.txt", 1)
        groundtruth = _read_records(sequence_dir / "groundtruth.txt", 7)
        depth_times = np.asarray([record[0] for record in depth])
        pose_times = np.asarray([record[0] for record in groundtruth])
        frames = []
        for timestamp, rgb_values in rgb:
            depth_match = _nearest(
                timestamp, depth, depth_times, args.rgb_depth_tolerance
            )
            pose_match = _nearest(timestamp, groundtruth, pose_times, args.pose_tolerance)
            if depth_match is None or pose_match is None:
                continue
            frames.append(
                {
                    "timestamp": timestamp,
                    "image": _relative(sequence_dir / rgb_values[0], root),
                    "depth": _relative(sequence_dir / depth_match[1][0], root),
                    "camera_pose": _pose(pose_match[1]),
                }
            )
        if frames:
            sequences.append(
                {"sequence_id": _sequence_id(sequence_dir, root), "frames": frames}
            )
    return sequences


def _redwood_roots(root: Path) -> list[Path]:
    candidates = {root, root / "sample"}
    candidates.update(path.parent for path in root.rglob("camera_primesense.json"))
    return sorted(
        path
        for path in candidates
        if (path / "color").is_dir()
        and (path / "depth").is_dir()
        and (path / "camera_primesense.json").is_file()
        and (path / "trajectory.log").is_file()
    )


def build_redwood(root: Path, args: argparse.Namespace) -> list[dict]:
    sequences = []
    for sequence_dir in _redwood_roots(root):
        images = sorted((sequence_dir / "color").glob("*.jpg"))
        depths = sorted((sequence_dir / "depth").glob("*.png"))
        poses = _read_trajectory(sequence_dir / "trajectory.log")
        if not images or len(images) != len(depths) or len(images) != len(poses):
            raise ValueError(
                f"Redwood RGB, depth and pose counts differ in {sequence_dir}: "
                f"{len(images)}, {len(depths)}, {len(poses)}"
            )
        frames = [
            {
                "image": _relative(image, root),
                "depth": _relative(depth, root),
                "camera_pose": pose,
            }
            for image, depth, pose in zip(images, depths, poses)
        ]
        sequences.append(
            {
                "sequence_id": _sequence_id(sequence_dir, root),
                "intrinsics": _read_intrinsics(
                    sequence_dir / "camera_primesense.json"
                ),
                "frames": frames,
            }
        )
    return sequences


def build_eth3d_slam(root: Path, args: argparse.Namespace) -> list[dict]:
    required = ("associated.txt", "calibration.txt", "groundtruth.txt")
    sequence_dirs = [root] if all((root / name).is_file() for name in required) else [
        path.parent
        for path in root.rglob("associated.txt")
        if all((path.parent / name).is_file() for name in required)
    ]
    sequences = []
    for sequence_dir in sorted(set(sequence_dirs)):
        calibration = np.fromstring(
            (sequence_dir / "calibration.txt").read_text(encoding="utf-8"),
            sep=" ",
            dtype=np.float32,
        )
        if calibration.size != 4:
            raise ValueError(f"Expected fx fy cx cy in {sequence_dir / 'calibration.txt'}")
        fx, fy, cx, cy = calibration
        intrinsics = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        poses = _read_records(sequence_dir / "groundtruth.txt", 7)
        pose_times = np.asarray([pose[0] for pose in poses], dtype=np.float64)
        frames = []
        associated_path = sequence_dir / "associated.txt"
        for line_no, line in enumerate(
            associated_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{associated_path}:{line_no}: expected 4 fields")
            timestamp = float(parts[0])
            pose = _nearest_pose(timestamp, poses, pose_times, args.pose_tolerance)
            if pose is None:
                continue
            frames.append(
                {
                    "timestamp": timestamp,
                    "image": _relative(sequence_dir / parts[1], root),
                    "depth": _relative(sequence_dir / parts[3], root),
                    "camera_pose": pose,
                }
            )
        if frames:
            sequences.append(
                {
                    "sequence_id": _sequence_id(sequence_dir, root),
                    "intrinsics": intrinsics,
                    "frames": frames,
                }
            )
    return sequences


def build_hypersim(root: Path, args: argparse.Namespace) -> list[dict]:
    parameter_path = root / "metadata_camera_parameters.csv"
    if not parameter_path.is_file():
        raise FileNotFoundError(f"Hypersim camera parameters not found: {parameter_path}")
    parameters = _load_camera_parameters(parameter_path)
    selected_cameras = set(args.camera or [])
    sequences = []
    for scene_dir in sorted(root.glob("ai_*_*")):
        if not scene_dir.is_dir() or scene_dir.name not in parameters:
            continue
        image_root = scene_dir / "images"
        for preview_dir in sorted(image_root.glob("scene_cam_*_final_preview")):
            camera = preview_dir.name.removeprefix("scene_").removesuffix(
                "_final_preview"
            )
            if selected_cameras and camera not in selected_cameras:
                continue
            detail_dir = scene_dir / "_detail" / camera
            positions = detail_dir / "camera_keyframe_positions.hdf5"
            orientations = detail_dir / "camera_keyframe_orientations.hdf5"
            frame_indices = detail_dir / "camera_keyframe_frame_indices.hdf5"
            if not all(path.is_file() for path in (positions, orientations, frame_indices)):
                continue
            geometry_dir = image_root / f"scene_{camera}_geometry_hdf5"
            frames = []
            for image in sorted(preview_dir.glob("frame.*.tonemap.jpg")):
                frame_id = int(image.name.split(".")[1])
                depth = geometry_dir / f"frame.{frame_id:04d}.depth_meters.hdf5"
                if depth.is_file():
                    frames.append(
                        {
                            "frame_id": frame_id,
                            "image": _relative(image, root),
                            "depth": _relative(depth, root),
                        }
                    )
            if not frames:
                continue
            scene_parameters = parameters[scene_dir.name]
            ray_matrix, intrinsics = _ray_matrix_and_intrinsics(
                scene_parameters["M_cam_from_uv"],
                int(scene_parameters["width"]),
                int(scene_parameters["height"]),
            )
            sequences.append(
                {
                    "sequence_id": f"{scene_dir.name}/{camera}",
                    "scene": scene_dir.name,
                    "camera": camera,
                    "frames": frames,
                    "positions": _relative(positions, root),
                    "orientations": _relative(orientations, root),
                    "frame_indices": _relative(frame_indices, root),
                    "meters_per_asset_unit": float(
                        scene_parameters["meters_per_asset_unit"]
                    ),
                    "ray_matrix": ray_matrix,
                    "intrinsics": intrinsics,
                }
            )
    return sequences


def build_sintel(root: Path, args: argparse.Namespace) -> list[dict]:
    training_root = root / "training"
    if not training_root.is_dir() and (root / "sample" / "training").is_dir():
        training_root = root / "sample" / "training"
    image_root = training_root / args.render_pass
    if not image_root.is_dir():
        raise FileNotFoundError(f"Sintel render pass not found: {image_root}")
    sequences = []
    for sequence_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        frames = []
        for image in sorted(sequence_dir.glob("*.png")):
            depth = training_root / "depth" / sequence_dir.name / f"{image.stem}.dpt"
            camera = (
                training_root / "camdata_left" / sequence_dir.name / f"{image.stem}.cam"
            )
            if depth.is_file() and camera.is_file():
                frames.append(
                    {
                        "image": _relative(image, root),
                        "depth": _relative(depth, root),
                        "camera": _relative(camera, root),
                    }
                )
        if frames:
            sequences.append({"sequence_id": sequence_dir.name, "frames": frames})
    return sequences
BUILDERS: dict[str, Callable[[Path, argparse.Namespace], list[dict]]] = {
    "tartanair": build_tartanair,
    "tum_rgbd": build_tum_rgbd,
    "redwood": build_redwood,
    "eth3d_slam": build_eth3d_slam,
    "hypersim": build_hypersim,
    "sintel": build_sintel,
}


def build_index(dataset: str, root: Path, output: Path, args: argparse.Namespace) -> dict:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")
    sequences = BUILDERS[dataset](root, args)
    if not sequences:
        raise ValueError(f"No valid {dataset} sequences found under {root}")
    payload = {
        "version": INDEX_VERSION,
        "dataset": dataset,
        "sequences": sequences,
    }
    if dataset == "sintel":
        payload["render_pass"] = args.render_pass
    output = output.expanduser()
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, payload, allow_pickle=True)
    image_count = sum(len(sequence["frames"]) for sequence in sequences)
    print(f"index          {output.resolve()}")
    print(f"sequences      {len(sequences)}")
    print(f"images         {image_count}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a relocatable Pi3 training-sampling index"
    )
    parser.add_argument("dataset", choices=sorted(BUILDERS))
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("pi3_index.npy"))
    parser.add_argument("--rgb-depth-tolerance", type=float, default=0.02)
    parser.add_argument("--pose-tolerance", type=float, default=0.02)
    parser.add_argument("--render-pass", default="final")
    parser.add_argument("--camera", action="append")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_index(args.dataset, args.data_root, args.output, args)


if __name__ == "__main__":
    main()
