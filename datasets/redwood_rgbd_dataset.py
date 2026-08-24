from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset


def _read_intrinsics(path: Path) -> np.ndarray:
    record = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(record["intrinsic_matrix"], dtype=np.float32)
    return matrix.reshape(3, 3, order="F")


def _read_trajectory(path: Path) -> list[np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) % 5:
        raise ValueError(f"Invalid Open3D trajectory layout in {path}")

    poses = []
    for start in range(0, len(lines), 5):
        header = lines[start].split()
        if len(header) != 3:
            raise ValueError(f"Invalid Open3D trajectory header {lines[start]}")
        pose = np.asarray(
            [[float(value) for value in lines[start + row].split()] for row in range(1, 5)],
            dtype=np.float32,
        )
        if pose.shape != (4, 4):
            raise ValueError(f"Invalid pose shape {pose.shape} in {path}")
        poses.append(pose)
    return poses


def _find_sequence_root(data_root: Path) -> Path:
    candidates = [data_root, data_root / "sample"]
    for candidate in candidates:
        if (
            (candidate / "color").is_dir()
            and (candidate / "depth").is_dir()
            and (candidate / "camera_primesense.json").is_file()
            and (candidate / "trajectory.log").is_file()
        ):
            return candidate
    raise FileNotFoundError(f"Redwood sample sequence not found under {data_root}")


class RedwoodRGBDDataset(BaseDataset):
    """Open3D's five-frame Redwood RGB-D sample with registered depth and poses."""

    def __init__(
        self,
        data_root: str | Path,
        frame_step: int = 1,
        depth_scale: float = 1000.0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_label = "RedwoodRGBD"
        self.data_root = Path(data_root)
        self.frame_step = int(frame_step)
        if self.frame_step < 1:
            raise ValueError("frame_step must be positive")
        self.depth_scale = float(depth_scale)
        self.verbose = bool(verbose)

        sequence_root = _find_sequence_root(self.data_root)
        images = sorted((sequence_root / "color").glob("*.jpg"))
        depths = sorted((sequence_root / "depth").glob("*.png"))
        poses = _read_trajectory(sequence_root / "trajectory.log")
        if not images or len(images) != len(depths) or len(images) != len(poses):
            raise ValueError(
                f"Redwood RGB, depth and pose counts differ: "
                f"{len(images)}, {len(depths)}, {len(poses)}"
            )
        self.intrinsics = _read_intrinsics(sequence_root / "camera_primesense.json")
        self.records = [
            {
                "image": image,
                "depth": depth,
                "camera_pose": pose,
            }
            for image, depth, pose in zip(images, depths, poses)
        ]
        self.sequences = [sequence_root.name]
        self.num_imgs = {sequence_root.name: len(self.records)}
        print(
            f"[{self.dataset_label}] Found {len(self.records)} frames, "
            f"frame_step={self.frame_step}",
            flush=True,
        )

    def __len__(self) -> int:
        return 1

    def _sample_positions(self, count: int, rng) -> list[int] | None:
        span = (self.frame_num - 1) * self.frame_step + 1
        if count < span:
            return None
        maximum = count - span
        start = maximum // 2 if self.mode == "test" else int(rng.integers(maximum + 1))
        return [start + index * self.frame_step for index in range(self.frame_num)]

    def _get_views(self, index, resolution, rng):
        positions = self._sample_positions(len(self.records), rng)
        self.this_views_info = {"scene": self.sequences[0], "idxs": positions or []}
        if positions is None:
            return []

        views = []
        for position in positions:
            frame = self.records[position]
            image = np.asarray(Image.open(frame["image"]).convert("RGB"))
            depth = np.asarray(Image.open(frame["depth"]), dtype=np.float32) / self.depth_scale
            image, depth, intrinsics = self._crop_resize_if_necessary(
                image,
                depth,
                self.intrinsics.copy(),
                resolution,
                rng=rng,
                info=str(frame["image"]),
            )[:3]
            views.append(
                {
                    "img": image,
                    "depthmap": np.asarray(depth, dtype=np.float32),
                    "camera_pose": frame["camera_pose"].copy(),
                    "camera_intrinsics": np.asarray(intrinsics, dtype=np.float32),
                    "dataset": self.dataset_label,
                    "label": self.sequences[0],
                    "instance": frame["image"].stem,
                    "image_path": str(frame["image"]),
                    "depth_path": str(frame["depth"]),
                    "depth_source": "redwood_registered_depth",
                    "pose_source": "redwood_trajectory_log",
                }
            )
        return views


__all__ = ["RedwoodRGBDDataset"]
