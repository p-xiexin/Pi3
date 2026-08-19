"""Fast indexed ScanNet loader for Glob3R."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import lanczos


def scene_number(scene: str) -> int | None:
    if not scene.startswith("scene") or "_" not in scene:
        return None
    try:
        return int(scene.split("_", 1)[0][5:])
    except ValueError:
        return None


def numeric_stems(directory: Path, suffix: str) -> set[int]:
    """Faster than Path.glob for large NAS directories."""
    if not directory.is_dir():
        return set()

    out = set()
    with os.scandir(directory) as it:
        for entry in it:
            if not entry.is_file() or not entry.name.endswith(suffix):
                continue
            try:
                out.add(int(entry.name[:-len(suffix)]))
            except ValueError:
                pass
    return out


def read_matrix_txt(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Much faster than np.loadtxt for tiny fixed-size matrices."""
    values = np.fromstring(Path(path).read_text(), sep=" ", dtype=np.float32)
    if values.size != shape[0] * shape[1]:
        raise ValueError(f"{path}: expected {shape}, got {values.size} values")
    return values.reshape(shape)


def load_invalid_list(path: str | Path | None) -> dict[str, set[int]]:
    if path is None or not Path(path).is_file():
        return {}
    raw = json.loads(Path(path).read_text())
    return {str(scene): {int(x) for x in (frames or [])} for scene, frames in raw.items()}


def _scan_scene(base: Path, invalid_frames: set[int]):
    color_ids = numeric_stems(base / "color", ".jpg")
    depth_ids = numeric_stems(base / "depth", ".png")
    pose_ids = numeric_stems(base / "pose", ".txt")
    frame_ids = sorted((color_ids & depth_ids & pose_ids) - invalid_frames)

    intrinsic_path = base / "intrinsic" / "intrinsic_depth.txt"
    if not frame_ids or not intrinsic_path.is_file():
        return None

    intrinsic = read_matrix_txt(intrinsic_path, (4, 4))[:3, :3]
    return {
        "scene": base.name,
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "intrinsic": intrinsic.astype(np.float32),
    }


def generate_scannet_index(
    data_root: str | Path,
    output_path: str | Path,
    invalid_list_path: str | Path | None = None,
    workers: int = 16,
) -> dict:
    """
    Fast index builder.

    Only caches:
      - valid frame ids
      - intrinsic

    Poses are NOT parsed here. They are read lazily only for sampled frames.
    """
    data_root = Path(data_root)
    invalid = load_invalid_list(invalid_list_path)

    scene_dirs = sorted(
        p for p in data_root.iterdir()
        if p.is_dir() and scene_number(p.name) is not None
    )

    def task(base):
        return _scan_scene(base, invalid.get(base.name, set()))

    scenes = {}
    workers = max(1, int(workers))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(task, scene_dirs), 1):
            if result is not None:
                scenes[result["scene"]] = {
                    "frame_ids": result["frame_ids"],
                    "intrinsic": result["intrinsic"],
                }

            if i % 100 == 0 or i == len(scene_dirs):
                print(f"[ScanNet index] {i}/{len(scene_dirs)} scenes", flush=True)

    index = {"version": 3, "scenes": scenes}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, index, allow_pickle=True)

    num_frames = sum(len(x["frame_ids"]) for x in scenes.values())
    print(
        f"[ScanNet index] saved {len(scenes)} scenes / {num_frames} frames "
        f"-> {output_path}",
        flush=True,
    )
    return index


class PoseCache:
    def __init__(self, capacity=4096):
        self.capacity = int(capacity)
        self.data = OrderedDict()

    def get(self, path: Path):
        key = str(path)
        pose = self.data.get(key)
        if pose is not None:
            self.data.move_to_end(key)
            return pose

        pose = read_matrix_txt(path, (4, 4))
        if not np.isfinite(pose).all():
            raise ValueError(f"Non-finite camera pose: {path}")

        if self.capacity > 0:
            self.data[key] = pose
            self.data.move_to_end(key)
            while len(self.data) > self.capacity:
                self.data.popitem(last=False)

        return pose


class Glob3RScannetValidationDataset(BaseDataset):
    def __init__(
        self,
        data_root: str,
        index_file: str,
        frame_step: int = 1,
        sample_seed: int = 0,
        pose_cache_size: int = 4096,
        verbose: bool = False,
        **kwargs,
    ):
        if frame_step < 1:
            raise ValueError("frame_step must be positive")

        super().__init__(shuffle=False, random_sample_thres=0.0, **kwargs)

        self.dataset_label = "ScanNet"
        self.data_root = Path(data_root).expanduser().resolve()
        self.frame_step = int(frame_step)
        self.sample_seed = int(sample_seed)
        self.verbose = bool(verbose)
        self.pose_cache = PoseCache(pose_cache_size)

        index = np.load(index_file, allow_pickle=True).item()
        if index.get("version") != 3:
            raise ValueError(
                f"ScanNet index version={index.get('version')}; regenerate version 3"
            )

        required_span = (self.frame_num - 1) * self.frame_step + 1
        self.scene_data = {}

        for scene, data in index["scenes"].items():
            number = scene_number(scene)
            in_split = number <= 660 if self.mode == "train" else number > 660
            if in_split and len(data["frame_ids"]) >= required_span:
                self.scene_data[scene] = data

        self.sequences = sorted(self.scene_data)
        self.num_imgs = {
            scene: len(self.scene_data[scene]["frame_ids"])
            for scene in self.sequences
        }

        print(
            f"[{self.dataset_label}] Loaded index: {len(self.sequences)} usable videos "
            f"for mode={self.mode}",
            flush=True,
        )

    def __len__(self):
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        scene = self.sequences[int(index)]
        data = self.scene_data[scene]
        frame_ids = data["frame_ids"]
        intrinsic = data["intrinsic"]

        required_span = (self.frame_num - 1) * self.frame_step + 1
        scene_rng = np.random.default_rng(self.sample_seed + int(index))
        start = int(scene_rng.integers(0, len(frame_ids) - required_span + 1))

        positions = start + np.arange(self.frame_num) * self.frame_step
        window_ids = [int(frame_ids[p]) for p in positions]

        shuffled_positions = positions.copy()
        rng.shuffle(shuffled_positions)
        selected_ids = [int(frame_ids[p]) for p in shuffled_positions]

        self.this_views_info = {
            "scene": scene,
            "sampling": "local_window_random_order",
            "frame_step": self.frame_step,
            "window_idxs": window_ids,
            "idxs": selected_ids,
            "reference": selected_ids[0],
        }

        base = self.data_root / scene
        views = []

        for pos in shuffled_positions:
            frame_id = int(frame_ids[int(pos)])
            image_path = base / "color" / f"{frame_id}.jpg"
            depth_path = base / "depth" / f"{frame_id}.png"
            pose_path = base / "pose" / f"{frame_id}.txt"

            camera_pose = self.pose_cache.get(pose_path)

            with Image.open(image_path) as f:
                image = np.asarray(
                    f.convert("RGB").resize((640, 480), resample=lanczos)
                )

            with Image.open(depth_path) as f:
                depth = np.asarray(f, dtype=np.float32) / 1000.0

            image, depth, K = self._crop_resize_if_necessary(
                image,
                depth,
                intrinsic.copy(),
                resolution,
                rng=scene_rng,
                info=str(image_path),
            )

            views.append({
                "img": image,
                "depthmap": depth,
                "camera_pose": camera_pose.copy(),
                "camera_intrinsics": K.astype(np.float32),
                "dataset": self.dataset_label,
                "label": scene,
                "instance": str(frame_id),
            })

        return views


def main():
    data_root = Path("/starmap/nas/workspace/pxx/data/scannet")
    output_path = Path(
        "/starmap/nas/workspace/pxx/Pi3/data/dataset_cache/scannet_index.npy"
    )
    invalid_list = Path(
        "/starmap/nas/workspace/pxx/Pi3/data/scannet_invalid_list.json"
    )

    generate_scannet_index(
        data_root,
        output_path,
        invalid_list_path=invalid_list,
        workers=16,
    )


if __name__ == "__main__":
    main()


__all__ = ["Glob3RScannetValidationDataset", "generate_scannet_index"]
