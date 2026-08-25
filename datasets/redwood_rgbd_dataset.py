from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset
from datasets.sample_utils.index_utils import data_path, load_dataset_index


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
        index_file: str | Path = "pi3_index.npy",
        frame_step: int = 1,
        depth_scale: float = 1000.0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_label = "RedwoodRGBD"
        self.data_root, payload = load_dataset_index(
            data_root, index_file, "redwood"
        )
        self.frame_step = int(frame_step)
        if self.frame_step < 1:
            raise ValueError("frame_step must be positive")
        self.depth_scale = float(depth_scale)
        self.verbose = bool(verbose)

        self.records = payload["sequences"]
        self.sequences = [record["sequence_id"] for record in self.records]
        self.num_imgs = {
            record["sequence_id"]: len(record["frames"]) for record in self.records
        }
        print(
            f"[{self.dataset_label}] Found {len(self.records)} sequences, "
            f"frame_step={self.frame_step}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _sample_positions(self, count: int, rng) -> list[int] | None:
        span = (self.frame_num - 1) * self.frame_step + 1
        if count < span:
            return None
        maximum = count - span
        start = maximum // 2 if self.mode == "test" else int(rng.integers(maximum + 1))
        return [start + index * self.frame_step for index in range(self.frame_num)]

    def _get_views(self, index, resolution, rng):
        record = self.records[int(index)]
        positions = self._sample_positions(len(record["frames"]), rng)
        self.this_views_info = {
            "scene": record["sequence_id"],
            "idxs": positions or [],
        }
        if positions is None:
            required = (self.frame_num - 1) * self.frame_step + 1
            raise ValueError(
                f"Redwood sequence {record['sequence_id']} has "
                f"{len(record['frames'])} frames, but requires {required}"
            )

        views = []
        for position in positions:
            frame = record["frames"][position]
            image_path = data_path(self.data_root, frame["image"])
            depth_path = data_path(self.data_root, frame["depth"])
            image = np.asarray(Image.open(image_path).convert("RGB"))
            depth = np.asarray(Image.open(depth_path), dtype=np.float32) / self.depth_scale
            image, depth, intrinsics = self._crop_resize_if_necessary(
                image,
                depth,
                np.asarray(record["intrinsics"], dtype=np.float32).copy(),
                resolution,
                rng=rng,
                info=str(image_path),
            )[:3]
            views.append(
                {
                    "img": image,
                    "depthmap": np.asarray(depth, dtype=np.float32),
                    "camera_pose": frame["camera_pose"].copy(),
                    "camera_intrinsics": np.asarray(intrinsics, dtype=np.float32),
                    "dataset": self.dataset_label,
                    "label": record["sequence_id"],
                    "instance": image_path.stem,
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "depth_source": "redwood_registered_depth",
                    "pose_source": "redwood_trajectory_log",
                }
            )
        return views


__all__ = ["RedwoodRGBDDataset"]
