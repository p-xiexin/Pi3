from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset
from datasets.sample_utils.index_utils import data_path, load_dataset_index


_HYPERSIM_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def _read_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["dataset"])


def _load_camera_parameters(path: Path) -> dict[str, dict[str, np.ndarray | float | int]]:
    parameters = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            scene = row["scene_name"]
            matrix = np.array(
                [
                    [float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)]
                    for i in range(3)
                ],
                dtype=np.float64,
            )
            parameters[scene] = {
                "width": int(float(row["settings_output_img_width"])),
                "height": int(float(row["settings_output_img_height"])),
                "meters_per_asset_unit": float(
                    row["settings_units_info_meters_scale"]
                ),
                "M_cam_from_uv": matrix,
            }
    return parameters


def _ray_matrix_and_intrinsics(
    M_cam_from_uv: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    pixel_to_uv = np.array(
        [
            [2.0 / width, 0.0, 1.0 / width - 1.0],
            [0.0, -2.0 / height, 1.0 - 1.0 / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ray_matrix = _HYPERSIM_TO_OPENCV @ M_cam_from_uv @ pixel_to_uv
    intrinsics = np.linalg.inv(ray_matrix)
    intrinsics /= intrinsics[2, 2]
    return ray_matrix.astype(np.float32), intrinsics.astype(np.float32)


def _distance_to_z_depth(distance: np.ndarray, ray_matrix: np.ndarray) -> np.ndarray:
    height, width = distance.shape
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    pixels = np.stack((x, y, np.ones_like(x)), axis=0).reshape(3, -1)
    rays = ray_matrix @ pixels
    ray_norm = np.linalg.norm(rays, axis=0)
    z_depth = distance.reshape(-1) * rays[2] / ray_norm
    z_depth = z_depth.reshape(height, width).astype(np.float32)
    valid = np.isfinite(z_depth) & np.isfinite(distance) & (z_depth > 0)
    return np.where(valid, z_depth, 0.0).astype(np.float32)


class HypersimDataset(BaseDataset):
    """Hypersim tonemapped RGB, metric depth and ground-truth camera poses."""

    def __init__(
        self,
        data_root: str | Path,
        index_file: str | Path = "pi3_index.npy",
        camera_parameters_file: str | Path | None = None,
        frame_step: int = 1,
        cameras: list[str] | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_label = "Hypersim"
        self.data_root, payload = load_dataset_index(
            data_root, index_file, "hypersim"
        )
        self.frame_step = int(frame_step)
        if self.frame_step < 1:
            raise ValueError("frame_step must be positive")
        self.verbose = bool(verbose)
        selected_cameras = set(cameras or [])

        self.records = [
            record
            for record in payload["sequences"]
            if not selected_cameras or record["camera"] in selected_cameras
        ]
        if not self.records:
            raise ValueError("Hypersim index has no sequences for the selected cameras")

        self.sequences = [record["sequence_id"] for record in self.records]
        self.num_imgs = {
            record["sequence_id"]: len(record["frames"]) for record in self.records
        }
        self.trajectory_cache = {}
        print(
            f"[{self.dataset_label}] Found {len(self.records)} camera sequences, "
            f"frame_step={self.frame_step}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _trajectory(self, record: dict) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
        key = record["sequence_id"]
        if key not in self.trajectory_cache:
            frame_indices = _read_hdf5(
                data_path(self.data_root, record["frame_indices"])
            ).reshape(-1).astype(int)
            positions = _read_hdf5(
                data_path(self.data_root, record["positions"])
            ).astype(np.float64)
            orientations = _read_hdf5(
                data_path(self.data_root, record["orientations"])
            ).astype(np.float64)
            lookup = {int(frame_id): index for index, frame_id in enumerate(frame_indices)}
            self.trajectory_cache[key] = (lookup, positions, orientations)
        return self.trajectory_cache[key]

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
                f"Hypersim sequence {record['sequence_id']} has "
                f"{len(record['frames'])} frames, but requires {required}"
            )

        frame_lookup, camera_positions, camera_orientations = self._trajectory(record)
        views = []
        for position in positions:
            frame = record["frames"][position]
            frame_id = frame["frame_id"]
            trajectory_index = frame_lookup[frame_id]
            image_path = data_path(self.data_root, frame["image"])
            depth_path = data_path(self.data_root, frame["depth"])
            image = np.asarray(Image.open(image_path).convert("RGB"))
            distance = _read_hdf5(depth_path).astype(np.float32)
            if distance.shape != image.shape[:2]:
                raise ValueError(
                    f"{image_path}: image/depth mismatch "
                    f"{image.shape[:2]} vs {distance.shape}"
                )
            depth = _distance_to_z_depth(distance, record["ray_matrix"])

            camera_pose = np.eye(4, dtype=np.float64)
            camera_pose[:3, :3] = (
                camera_orientations[trajectory_index] @ _HYPERSIM_TO_OPENCV
            )
            camera_pose[:3, 3] = (
                camera_positions[trajectory_index]
                * record["meters_per_asset_unit"]
            )
            intrinsics = record["intrinsics"].copy()
            image, depth, intrinsics = self._crop_resize_if_necessary(
                image,
                depth,
                intrinsics,
                resolution,
                rng=rng,
                info=str(image_path),
            )[:3]
            views.append(
                {
                    "img": image,
                    "depthmap": np.asarray(depth, dtype=np.float32),
                    "camera_pose": camera_pose.astype(np.float32),
                    "camera_intrinsics": np.asarray(intrinsics, dtype=np.float32),
                    "dataset": self.dataset_label,
                    "label": record["sequence_id"],
                    "instance": f"frame.{frame_id:04d}",
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "depth_source": "hypersim_distance_converted_to_z_depth",
                    "pose_source": "hypersim_ground_truth_camera_trajectory",
                }
            )
        return views


__all__ = ["HypersimDataset"]
