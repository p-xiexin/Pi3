from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from datasets.base.base_dataset import BaseDataset
from datasets.sample_utils.index_utils import data_path, load_dataset_index


TUM_REGISTERED_INTRINSICS = np.array(
    [[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


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


def _pose(values: list[str]) -> np.ndarray:
    translation = np.asarray(values[:3], dtype=np.float32)
    quaternion = np.asarray(values[3:], dtype=np.float64)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rotation.from_quat(quaternion).as_matrix().astype(np.float32)
    pose[:3, 3] = translation
    return pose


def _sequence_dirs(data_root: Path) -> list[Path]:
    if (data_root / "rgb.txt").is_file():
        return [data_root]
    return sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir()
        and (path / "rgb.txt").is_file()
        and (path / "depth.txt").is_file()
        and (path / "groundtruth.txt").is_file()
    )


class TUMRGBDPi3XDataset(BaseDataset):
    """TUM RGB-D RGB, registered depth, calibration and mocap poses."""

    def __init__(
        self,
        data_root: str | Path,
        index_file: str | Path = "pi3_index.npy",
        frame_step: int = 1,
        rgb_depth_tolerance: float = 0.02,
        pose_tolerance: float = 0.02,
        depth_scale: float = 5000.0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_label = "TUMRGBD"
        self.data_root, payload = load_dataset_index(
            data_root, index_file, "tum_rgbd"
        )
        self.frame_step = int(frame_step)
        if self.frame_step < 1:
            raise ValueError("frame_step must be positive")
        self.depth_scale = float(depth_scale)
        self.verbose = bool(verbose)
        self.records: list[dict] = payload["sequences"]

        self.sequences = [record["sequence_id"] for record in self.records]
        self.num_imgs = {
            record["sequence_id"]: len(record["frames"])
            for record in self.records
        }
        print(
            f"[{self.dataset_label}] Found {len(self.records)} sequences, "
            f"frame_step={self.frame_step}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _sample_positions(self, count: int, rng, is_test: bool) -> list[int] | None:
        span = (self.frame_num - 1) * self.frame_step + 1
        if count < span:
            return None
        maximum = count - span
        start = maximum // 2 if is_test else int(rng.integers(0, maximum + 1))
        return [start + i * self.frame_step for i in range(self.frame_num)]

    def _get_views(self, index, resolution, rng, is_test=False):
        record = self.records[int(index)]
        is_test = bool(is_test or self.mode == "test")
        positions = self._sample_positions(len(record["frames"]), rng, is_test)
        if positions is None:
            self.this_views_info = {"scene": record["sequence_id"], "idxs": []}
            return []
        self.this_views_info = {
            "scene": record["sequence_id"],
            "frame_step": self.frame_step,
            "idxs": positions,
        }

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
                TUM_REGISTERED_INTRINSICS.copy(),
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
                    "instance": f"{frame['timestamp']:.6f}",
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "depth_source": "tum_rgbd_registered_depth",
                    "pose_source": "tum_rgbd_motion_capture",
                }
            )
        return views


__all__ = ["TUMRGBDPi3XDataset"]
