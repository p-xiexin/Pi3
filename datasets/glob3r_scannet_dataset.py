"""High-overlap ScanNet windows with randomized in-window view order."""

from __future__ import annotations

import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import lanczos


class Glob3RScannetValidationDataset(BaseDataset):
    """Return one high-overlap frame window per scene with randomized view order.

    Expected directory layout::

        data_root/
          scene0000_00/
            color/0.jpg
            depth/0.png
            pose/0.txt
            intrinsic/intrinsic_depth.txt

    Train/test splitting follows the original Pi3 ScanNet loader:
    scene id <= 660 is train and scene id > 660 is test.
    """

    def __init__(
        self,
        data_root: str,
        frame_step: int = 1,
        sample_seed: int = 0,
        invalid_list_path: str | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        if not data_root:
            raise ValueError("data_root must be provided")
        if frame_step < 1:
            raise ValueError("frame_step must be positive")

        # BaseDataset owns mode, frame_num, transform and resolution handling.
        super().__init__(shuffle=False, random_sample_thres=0.0, **kwargs)

        self.dataset_label = "ScanNet"
        self.data_root = osp.abspath(osp.expanduser(data_root))
        self.frame_step = int(frame_step)
        self.sample_seed = int(sample_seed)
        self.verbose = bool(verbose)

        if not osp.isdir(self.data_root):
            raise FileNotFoundError(f"ScanNet data_root does not exist: {self.data_root}")

        all_sequences = sorted(
            entry.name
            for entry in Path(self.data_root).iterdir()
            if entry.is_dir() and self._scene_number(entry.name) is not None
        )

        if self.mode == "train":
            self.sequences = [
                scene for scene in all_sequences
                if self._scene_number(scene) <= 660
            ]
        else:
            self.sequences = [
                scene for scene in all_sequences
                if self._scene_number(scene) > 660
            ]

        if not self.sequences:
            raise RuntimeError(
                f"No ScanNet scenes found for mode={self.mode!r} in {self.data_root}"
            )

        self.invalid_list = self._load_invalid_list(invalid_list_path)

        # Build frame ids from the actual file intersection. This avoids stale
        # num_imgs caches and does not assume that frame ids are contiguous.
        self.scene_frames: dict[str, list[int]] = {}
        skipped_scenes: list[str] = []
        for scene in self.sequences:
            frames = self._find_complete_frames(scene)
            invalid_frames = self.invalid_list.get(scene, set())
            frames = [frame for frame in frames if frame not in invalid_frames]

            required_span = (self.frame_num - 1) * self.frame_step + 1
            if len(frames) >= required_span:
                self.scene_frames[scene] = frames
            else:
                skipped_scenes.append(scene)

        self.sequences = [
            scene for scene in self.sequences
            if scene in self.scene_frames
        ]

        if not self.sequences:
            raise RuntimeError(
                "No scene contains enough complete, valid frames for "
                f"frame_num={self.frame_num}, frame_step={self.frame_step}"
            )

        print(
            f"[{self.dataset_label}] Found {len(self.sequences)} usable videos "
            f"in {self.data_root} for mode={self.mode}; "
            f"skipped {len(skipped_scenes)} short/incomplete scenes",
            flush=True,
        )
        if self.verbose and skipped_scenes:
            print(f"[{self.dataset_label}] Skipped scenes: {skipped_scenes}")

    @staticmethod
    def _scene_number(scene: str) -> int | None:
        """Extract XXXX from sceneXXXX_YY, returning None for other names."""
        if not scene.startswith("scene") or "_" not in scene:
            return None
        try:
            return int(scene.split("_", 1)[0][5:])
        except ValueError:
            return None

    def _load_invalid_list(
        self,
        invalid_list_path: str | None,
    ) -> dict[str, set[int]]:
        if invalid_list_path is None:
            project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
            invalid_list_path = osp.join(
                project_root,
                "data",
                "scannet_invalid_list.json",
            )
        else:
            invalid_list_path = osp.abspath(osp.expanduser(invalid_list_path))

        if not osp.isfile(invalid_list_path):
            print(
                f"[{self.dataset_label}] Warning: invalid-list file not found: "
                f"{invalid_list_path}; continuing with an empty invalid list",
                flush=True,
            )
            return {}

        with open(invalid_list_path, "r", encoding="utf-8") as file:
            raw = json.load(file)

        if not isinstance(raw, dict):
            raise TypeError(
                f"{invalid_list_path} must contain "
                '{"sceneXXXX_YY": [frame_id, ...]}, '
                f"but got {type(raw).__name__}"
            )

        result: dict[str, set[int]] = {}
        for scene, frames in raw.items():
            if frames is None:
                frames = []
            if not isinstance(frames, (list, tuple)):
                raise TypeError(
                    f"Invalid frames for {scene!r} must be a list, "
                    f"but got {type(frames).__name__}"
                )
            result[str(scene)] = {int(frame) for frame in frames}
        return result

    def _find_complete_frames(self, scene: str) -> list[int]:
        base = Path(self.data_root) / scene

        color_ids = self._numeric_stems(base / "color", ".jpg")
        depth_ids = self._numeric_stems(base / "depth", ".png")
        pose_ids = self._numeric_stems(base / "pose", ".txt")

        return sorted(color_ids & depth_ids & pose_ids)

    @staticmethod
    def _numeric_stems(directory: Path, suffix: str) -> set[int]:
        if not directory.is_dir():
            return set()

        result: set[int] = set()
        for path in directory.glob(f"*{suffix}"):
            try:
                result.add(int(path.stem))
            except ValueError:
                continue
        return result

    def __len__(self) -> int:
        return len(self.sequences)

    def _get_views(self, index, resolution, rng):
        scene = self.sequences[int(index)]
        valid_frames = self.scene_frames[scene]
        required_span = (self.frame_num - 1) * self.frame_step + 1

        scene_rng = np.random.default_rng(self.sample_seed + int(index))
        start = int(
            scene_rng.integers(
                0,
                len(valid_frames) - required_span + 1,
            )
        )
        positions = start + np.arange(self.frame_num) * self.frame_step
        window_frame_indices = [
            valid_frames[int(position)]
            for position in positions
        ]
        # Select a temporally local window first, then randomize only its view
        # order. The matching model still uses batch position 0 as reference,
        # but that reference is no longer forced to be the earliest frame.
        frame_indices = window_frame_indices.copy()
        rng.shuffle(frame_indices)

        self.this_views_info = {
            "scene": scene,
            "sampling": "local_window_random_order",
            "frame_step": self.frame_step,
            "window_idxs": window_frame_indices,
            "idxs": frame_indices,
            "reference": frame_indices[0],
        }

        base_path = osp.join(self.data_root, scene)
        intrinsic_path = osp.join(
            base_path,
            "intrinsic",
            "intrinsic_depth.txt",
        )
        intrinsic = np.loadtxt(
            intrinsic_path,
            dtype=np.float32,
        ).reshape(4, 4)[:3, :3]

        views = []
        for frame in frame_indices:
            image_path = osp.join(base_path, "color", f"{frame}.jpg")
            depth_path = osp.join(base_path, "depth", f"{frame}.png")
            pose_path = osp.join(base_path, "pose", f"{frame}.txt")

            camera_pose = np.loadtxt(
                pose_path,
                dtype=np.float32,
            ).reshape(4, 4)
            if not np.isfinite(camera_pose).all():
                raise ValueError(f"Non-finite camera pose: {pose_path}")

            with Image.open(image_path) as image_file:
                image = np.asarray(
                    image_file.convert("RGB").resize(
                        (640, 480),
                        resample=lanczos,
                    )
                )

            with Image.open(depth_path) as depth_file:
                depth = (
                    np.asarray(depth_file, dtype=np.float32)
                    / 1000.0
                )

            image, depth, resized_intrinsic = (
                self._crop_resize_if_necessary(
                    image,
                    depth,
                    intrinsic.copy(),
                    resolution,
                    rng=scene_rng,
                    info=image_path,
                )
            )

            views.append(
                {
                    "img": image,
                    "depthmap": depth,
                    "camera_pose": camera_pose.astype(np.float32),
                    "camera_intrinsics": resized_intrinsic.astype(np.float32),
                    "dataset": self.dataset_label,
                    "label": scene,
                    "instance": str(frame),
                }
            )

        return views


__all__ = ["Glob3RScannetValidationDataset"]
