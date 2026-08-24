from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from datasets.base.base_dataset import BaseDataset
from datasets.sample_utils.index_utils import data_path, load_dataset_index


def _read_rows(path: Path, value_count: int) -> list[tuple[float, list[str]]]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != value_count + 1:
            raise ValueError(f"{path}:{line_no}: expected {value_count + 1} fields")
        records.append((float(parts[0]), parts[1:]))
    return records


def _nearest_pose(
    timestamp: float,
    poses: list[tuple[float, list[str]]],
    pose_times: np.ndarray,
    tolerance: float,
) -> np.ndarray | None:
    position = int(np.searchsorted(pose_times, timestamp))
    candidates = [i for i in (position - 1, position) if 0 <= i < len(poses)]
    if not candidates:
        return None
    index = min(candidates, key=lambda i: abs(poses[i][0] - timestamp))
    if abs(poses[index][0] - timestamp) > tolerance:
        return None
    values = poses[index][1]
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rotation.from_quat(
        np.asarray(values[3:7], dtype=np.float64)
    ).as_matrix().astype(np.float32)
    pose[:3, 3] = np.asarray(values[:3], dtype=np.float32)
    return pose


def _sequence_dirs(data_root: Path) -> list[Path]:
    required = ("associated.txt", "calibration.txt", "groundtruth.txt")
    if all((data_root / name).is_file() for name in required):
        return [data_root]
    return sorted(
        path.parent
        for path in data_root.rglob("associated.txt")
        if all((path.parent / name).is_file() for name in required)
    )


class ETH3DSLAMDataset(BaseDataset):
    """ETH3D SLAM RGB-D frames with calibrated intrinsics and mocap poses."""

    def __init__(
        self,
        data_root: str | Path,
        index_file: str | Path = "pi3_index.npy",
        frame_step: int = 1,
        pose_tolerance: float = 0.02,
        depth_scale: float = 5000.0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_label = "ETH3DSLAM"
        self.data_root, payload = load_dataset_index(
            data_root, index_file, "eth3d_slam"
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
            "frame_step": self.frame_step,
            "idxs": positions or [],
        }
        if positions is None:
            return []

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
                record["intrinsics"].copy(),
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
                    "depth_source": "eth3d_registered_active_depth",
                    "pose_source": "eth3d_motion_capture",
                }
            )
        return views


__all__ = ["ETH3DSLAMDataset"]
