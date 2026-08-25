from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset
from datasets.sample_utils.index_utils import data_path, load_dataset_index


SINTEL_TAG = 202021.25


def _read_depth(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        tag = struct.unpack("<f", stream.read(4))[0]
        if tag != SINTEL_TAG:
            raise ValueError(f"Invalid Sintel depth tag {tag} in {path}")
        width, height = struct.unpack("<ii", stream.read(8))
        depth = np.fromfile(stream, dtype="<f4", count=width * height)
    if depth.size != width * height:
        raise ValueError(f"Truncated Sintel depth file {path}")
    return depth.reshape(height, width).astype(np.float32, copy=False)


def _read_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        tag = struct.unpack("<f", stream.read(4))[0]
        if tag != SINTEL_TAG:
            raise ValueError(f"Invalid Sintel camera tag {tag} in {path}")
        intrinsics = np.fromfile(stream, dtype="<f8", count=9).reshape(3, 3)
        world_to_camera = np.fromfile(stream, dtype="<f8", count=12).reshape(3, 4)
    homogeneous = np.eye(4, dtype=np.float64)
    homogeneous[:3] = world_to_camera
    camera_pose = np.linalg.inv(homogeneous)
    return intrinsics.astype(np.float32), camera_pose.astype(np.float32)


class SintelDepthDataset(BaseDataset):
    """MPI Sintel final-pass RGB, metric depth and per-frame cameras."""

    def __init__(
        self,
        data_root: str | Path,
        index_file: str | Path = "pi3_index.npy",
        frame_step: int = 1,
        render_pass: str = "final",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_label = "SintelDepth"
        self.data_root, payload = load_dataset_index(
            data_root, index_file, "sintel"
        )
        self.frame_step = int(frame_step)
        if self.frame_step < 1:
            raise ValueError("frame_step must be positive")
        self.render_pass = render_pass
        self.verbose = bool(verbose)
        indexed_render_pass = payload.get("render_pass", "final")
        if render_pass != indexed_render_pass:
            raise ValueError(
                f"Sintel index uses render pass {indexed_render_pass!r}, "
                f"but loader requested {render_pass!r}"
            )
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
            required = (self.frame_num - 1) * self.frame_step + 1
            raise ValueError(
                f"Sintel sequence {record['sequence_id']} has "
                f"{len(record['frames'])} frames, but requires {required}"
            )

        views = []
        for position in positions:
            frame = record["frames"][position]
            image_path = data_path(self.data_root, frame["image"])
            depth_path = data_path(self.data_root, frame["depth"])
            camera_path = data_path(self.data_root, frame["camera"])
            image = np.asarray(Image.open(image_path).convert("RGB"))
            depth = _read_depth(depth_path)
            intrinsics, camera_pose = _read_camera(camera_path)
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
                    "camera_pose": camera_pose,
                    "camera_intrinsics": np.asarray(intrinsics, dtype=np.float32),
                    "dataset": self.dataset_label,
                    "label": record["sequence_id"],
                    "instance": image_path.stem,
                    "image_path": str(image_path),
                    "depth_path": str(depth_path),
                    "camera_path": str(camera_path),
                    "depth_source": "sintel_metric_depth",
                    "pose_source": "sintel_camera_extrinsics",
                }
            )
        return views


__all__ = ["SintelDepthDataset"]
