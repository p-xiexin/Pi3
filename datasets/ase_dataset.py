from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from datasets.base.base_dataset import BaseDataset

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


# =============================================================================
# Basic IO
# =============================================================================

def read_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)

    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read RGB image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return np.asarray(Image.open(path).convert("RGB"))


def read_ase_distance(path: str | Path) -> np.ndarray:
    """
    ASE depth PNG stores distance along the camera ray in millimeters.
    uint16 max is invalid.
    """
    raw = np.asarray(Image.open(path))

    if raw.ndim == 3:
        raw = raw[..., 0]

    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"Unexpected ASE depth dtype {raw.dtype}: {path}")

    invalid = raw == np.iinfo(raw.dtype).max

    distance = raw.astype(np.float32) / 1000.0
    distance[invalid] = 0.0
    distance[~np.isfinite(distance)] = 0.0

    return distance


def frame_number(path: str | Path) -> int:
    stem = Path(path).stem
    digits = "".join(c for c in stem if c.isdigit())

    if not digits:
        raise ValueError(f"Cannot extract frame number from {path}")

    return int(digits)


# =============================================================================
# ASE trajectory
# =============================================================================

def read_ase_trajectory(path: str | Path) -> dict[str, np.ndarray]:
    """
    Read the native ASE trajectory.csv used by this dataset.

    Expected columns:
        tracking_timestamp_us
        tx_world_device, ty_world_device, tz_world_device
        qx_world_device, qy_world_device, qz_world_device, qw_world_device

    Returns:
        Ts_world_from_device: [N, 4, 4]
        timestamps:           [N]
    """
    df = pd.read_csv(path)

    t_cols = [
        "tx_world_device",
        "ty_world_device",
        "tz_world_device",
    ]
    q_cols = [
        "qx_world_device",
        "qy_world_device",
        "qz_world_device",
        "qw_world_device",
    ]

    required = ["tracking_timestamp_us", *t_cols, *q_cols]
    missing = [name for name in required if name not in df.columns]

    if missing:
        raise ValueError(f"{path}: missing ASE trajectory columns {missing}")

    t = df[t_cols].to_numpy(dtype=np.float32)
    q = df[q_cols].to_numpy(dtype=np.float64)

    R = Rotation.from_quat(q).as_matrix().astype(np.float32)

    T = np.repeat(
        np.eye(4, dtype=np.float32)[None],
        len(df),
        axis=0,
    )
    T[:, :3, :3] = R
    T[:, :3, 3] = t

    return {
        "Ts_world_from_device": T,
        "timestamps": df["tracking_timestamp_us"].to_numpy(dtype=np.int64),
    }


# =============================================================================
# Online fisheye -> pinhole transform
# =============================================================================

def camera_matrix(calib: Any) -> np.ndarray:
    f = np.asarray(calib.get_focal_lengths(), dtype=np.float64).reshape(-1)
    c = np.asarray(calib.get_principal_point(), dtype=np.float64).reshape(-1)

    fx = float(f[0])
    fy = float(f[0] if len(f) == 1 else f[1])
    cx, cy = float(c[0]), float(c[1])

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def transform_matrix(T: Any) -> np.ndarray:
    return np.asarray(T.to_matrix(), dtype=np.float32)


def distance_to_z_depth(
    distance: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Convert Euclidean distance along a pinhole ray to camera-Z depth.
    """
    h, w = distance.shape

    u, v = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]

    depth = distance / np.sqrt(x * x + y * y + 1.0)

    valid = np.isfinite(distance) & (distance > 0.0)
    depth[~valid] = 0.0

    return depth.astype(np.float32)


class ASEOnlineTransform:
    """
    Raw ASE fisheye RGB + ray distance
        -> pinhole RGB + camera-Z depth + pinhole K.

    No converted frame is stored on disk.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        focal: float = 150.0,
        rotate_cw90: bool = True,
    ):
        from projectaria_tools.projects import ase
        from projectaria_tools.core import calibration
        from projectaria_tools.core.image import InterpolationMethod

        self.calibration = calibration
        self.InterpolationMethod = InterpolationMethod
        self.rotate_cw90 = rotate_cw90

        self.raw_calib = ase.get_ase_rgb_calibration()

        self.linear_calib = calibration.get_linear_camera_calibration(
            width,
            height,
            focal,
            "camera-rgb",
            self.raw_calib.get_transform_device_camera(),
        )

        self.output_calib = (
            calibration.rotate_camera_calib_cw90deg(self.linear_calib)
            if rotate_cw90
            else self.linear_calib
        )

        self.K = camera_matrix(self.output_calib)
        self.T_device_camera = transform_matrix(
            self.output_calib.get_transform_device_camera()
        )

    def __call__(
        self,
        rgb: np.ndarray,
        distance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if rgb.shape[:2] != distance.shape:
            raise ValueError(
                f"RGB/depth shape mismatch: {rgb.shape[:2]} vs {distance.shape}"
            )

        rgb = self.calibration.distort_by_calibration(
            rgb,
            self.linear_calib,
            self.raw_calib,
            self.InterpolationMethod.BILINEAR,
        )

        distance = self.calibration.distort_by_calibration(
            distance.astype(np.float32),
            self.linear_calib,
            self.raw_calib,
        )

        rgb = np.asarray(rgb)
        distance = np.asarray(distance, dtype=np.float32)

        if self.rotate_cw90:
            rgb = np.rot90(rgb, k=3).copy()
            distance = np.rot90(distance, k=3).copy()

        distance[~np.isfinite(distance)] = 0.0
        distance[distance < 0.0] = 0.0

        depth = distance_to_z_depth(distance, self.K)

        return (
            rgb,
            depth,
            self.K.copy(),
            self.T_device_camera.copy(),
        )


# =============================================================================
# Lightweight index
# =============================================================================

def is_scene_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "rgb").is_dir()
        and (path / "depth").is_dir()
        and _trajectory_path(path) is not None
    )


def _trajectory_path(scene_dir: Path) -> Path | None:
    for filename in ("trajectory.csv", "trajectory.txt"):
        path = scene_dir / filename
        if path.is_file():
            return path
    return None


def discover_scenes(
    data_root: Path,
    chunks: list[str] | None = None,
) -> list[Path]:
    roots = (
        [data_root / chunk for chunk in chunks]
        if chunks
        else sorted(p for p in data_root.iterdir() if p.is_dir())
    )

    scenes = []

    for root in roots:
        if is_scene_dir(root):
            scenes.append(root)
        elif root.is_dir():
            scenes.extend(
                sorted(p for p in root.iterdir() if is_scene_dir(p))
            )

    return scenes


def generate_ase_index(
    data_root: str | Path,
    output_path: str | Path | None = None,
    chunks: list[str] | None = None,
) -> dict[str, Any]:
    """
    Save only raw paths + frame numbers.
    """
    data_root = Path(data_root)
    records = []

    for scene_dir in tqdm(
        discover_scenes(data_root, chunks),
        desc="[ASE] building index",
        unit="scene",
    ):
        scene = scene_dir.relative_to(data_root).as_posix()
        trajectory_path = _trajectory_path(scene_dir)
        if trajectory_path is None:
            continue

        rgb_dir = scene_dir / "rgb"
        depth_dir = scene_dir / "depth"

        depth_map = {
            frame_number(path): path
            for path in depth_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        }

        frames = []

        for image_path in sorted(rgb_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            frame_no = frame_number(image_path)
            depth_path = depth_map.get(frame_no)

            if depth_path is None:
                continue

            frames.append(
                {
                    "frame_no": frame_no,
                    "frame_id": f"{frame_no:07d}",
                    "image": str(image_path.resolve()),
                    "depth": str(depth_path.resolve()),
                }
            )

        frames.sort(key=lambda x: x["frame_no"])

        if not frames:
            continue

        records.append(
            {
                "sequence_id": scene,
                "scene_dir": str(scene_dir.resolve()),
                "trajectory": str(trajectory_path.resolve()),
                "frames": frames,
            }
        )

    index = {
        "version": 1,
        "sequences": records,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as f:
            np.save(f, index, allow_pickle=True)

    return index


# =============================================================================
# Dataset
# =============================================================================

class AriaSyntheticEnvironmentsPi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root: str | Path,
        index_file: str | Path,
        frame_step: int = 1,
        rectify_width: int = 512,
        rectify_height: int = 512,
        rectify_focal: float = 150.0,
        rotate_cw90: bool = True,
        chunks: list[str] | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataset_label = "AriaSyntheticEnvironmentsPi3X"
        self.data_root = Path(data_root)
        self.frame_step = int(frame_step)
        self.verbose = verbose

        self.rectifier = ASEOnlineTransform(
            width=rectify_width,
            height=rectify_height,
            focal=rectify_focal,
            rotate_cw90=rotate_cw90,
        )

        index_path = Path(index_file)
        if not index_path.is_absolute() and not index_path.exists():
            index_path = self.data_root / index_path

        index = np.load(index_path, allow_pickle=True).item()

        selected_chunks = set(chunks or [])

        self.records = []

        for record in index["sequences"]:
            scene = record["sequence_id"]
            chunk = Path(scene).parts[0]

            if selected_chunks and chunk not in selected_chunks:
                continue

            self.records.append(record)

        self.sequences = [r["sequence_id"] for r in self.records]
        self.trajectory_cache = {}

        print(
            f"[{self.dataset_label}] Found {len(self.records)} scenes",
            flush=True,
        )

    def __len__(self):
        return len(self.records)

    def _trajectory(self, record):
        scene = record["sequence_id"]

        if scene not in self.trajectory_cache:
            self.trajectory_cache[scene] = read_ase_trajectory(
                record["trajectory"]
            )

        return self.trajectory_cache[scene]

    def _sample_indices(self, n, rng, is_test):
        span = (self.frame_num - 1) * self.frame_step + 1

        if n < span:
            return None

        max_start = n - span
        start = (
            max_start // 2
            if is_test
            else int(rng.integers(0, max_start + 1))
        )

        return [
            start + i * self.frame_step
            for i in range(self.frame_num)
        ]

    def _get_views(
        self,
        index,
        resolution,
        rng,
        is_test=False,
    ):
        is_test = is_test or self.mode == "test"

        record = self.records[index]
        frames = record["frames"]

        idxs = self._sample_indices(
            len(frames),
            rng,
            is_test,
        )

        if idxs is None:
            self.this_views_info = {
                "scene": record["sequence_id"],
                "idxs": [],
            }
            required = (self.frame_num - 1) * self.frame_step + 1
            raise ValueError(
                f"ASE sequence {record['sequence_id']} has {len(frames)} frames, "
                f"but frame_num={self.frame_num} and frame_step={self.frame_step} "
                f"require {required} frames"
            )

        self.this_views_info = {
            "scene": record["sequence_id"],
            "idxs": idxs,
        }

        trajectory = self._trajectory(record)
        poses = trajectory["Ts_world_from_device"]

        views = []

        for idx in idxs:
            frame = frames[idx]
            frame_no = int(frame["frame_no"])

            if frame_no >= len(poses):
                raise IndexError(
                    f"frame_no={frame_no}, trajectory_len={len(poses)}"
                )

            rgb = read_rgb(frame["image"])
            distance = read_ase_distance(frame["depth"])

            rgb, depth, K, T_device_camera = self.rectifier(
                rgb,
                distance,
            )

            T_world_camera = (
                poses[frame_no] @ T_device_camera
            ).astype(np.float32)

            rgb, depth, K = self._crop_resize_if_necessary(
                rgb,
                depth,
                K,
                resolution,
                rng=rng,
                info=frame["image"],
            )[:3]

            views.append(
                {
                    "img": rgb,
                    "depthmap": np.asarray(depth, dtype=np.float32),
                    "camera_pose": T_world_camera,
                    "camera_intrinsics": np.asarray(K, dtype=np.float32),
                    "dataset": self.dataset_label,
                    "label": record["sequence_id"],
                    "instance": frame["frame_id"],
                    "prefix": (
                        f"{record['sequence_id']}_{frame['frame_id']}"
                    ),
                    "image_path": frame["image"],
                    "depth_path": frame["depth"],
                    "frame_no": frame_no,
                }
            )

        return views


# =============================================================================
# Build index
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build a lightweight ASE index")
    parser.add_argument("--data-root", type=Path, default=Path("data/ASE"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dataset_cache/ase_minimal.npy"),
    )
    parser.add_argument("--chunks", nargs="*")
    args = parser.parse_args()

    index = generate_ase_index(
        args.data_root,
        args.output,
        chunks=args.chunks,
    )

    print(
        f"Saved {len(index['sequences'])} scenes to {args.output}"
    )


if __name__ == "__main__":
    main()
