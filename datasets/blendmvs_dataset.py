from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from datasets.base.base_dataset import BaseDataset


def read_pfm(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as f:
        header = f.readline().decode("ascii").strip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"{path}: invalid PFM header {header!r}")
        color = header == "PF"

        line = f.readline().decode("ascii").strip()
        while line.startswith("#"):
            line = f.readline().decode("ascii").strip()

        width, height = map(int, line.split())
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 3 if color else 1
        data = np.fromfile(f, dtype=endian + "f4", count=width * height * channels)

    shape = (height, width, 3) if color else (height, width)
    return np.flipud(data.reshape(shape)).astype(np.float32)


def read_cam(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """MVSNet cam.txt: extrinsic is world->camera; BaseDataset needs camera->world."""
    lines = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if len(lines) < 9 or lines[0].lower() != "extrinsic" or lines[5].lower() != "intrinsic":
        raise ValueError(f"{path}: malformed MVSNet camera file")

    E = np.array([[float(x) for x in lines[i].split()] for i in range(1, 5)], dtype=np.float64)
    K = np.array([[float(x) for x in lines[i].split()] for i in range(6, 9)], dtype=np.float32)
    return K, np.linalg.inv(E).astype(np.float32)


def read_pair_file(path: str | Path) -> dict[int, list[int]]:
    lines = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if not lines:
        return {}

    pairs, cursor = {}, 1
    for _ in range(int(lines[0])):
        if cursor + 1 >= len(lines):
            break

        ref_id = int(lines[cursor])
        tokens = lines[cursor + 1].split()
        cursor += 2

        num_src = min(int(tokens[0]), (len(tokens) - 1) // 2)
        pairs[ref_id] = [int(tokens[1 + 2 * i]) for i in range(num_src)]

    return pairs


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def generate_blendedmvs_index(
    data_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    data_root = Path(data_root)
    records = []

    for scene_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        image_dir = scene_dir / "blended_images"
        cam_dir = scene_dir / "cams"
        depth_dir = scene_dir / "rendered_depth_maps"
        pair_path = cam_dir / "pair.txt"

        if not (image_dir.is_dir() and cam_dir.is_dir() and depth_dir.is_dir() and pair_path.is_file()):
            continue

        image_ids = {int(p.stem) for p in image_dir.glob("*.jpg") if not p.stem.endswith("_masked")}
        depth_ids = {int(p.stem) for p in depth_dir.glob("*.pfm")}
        cam_ids = {int(p.stem.replace("_cam", "")) for p in cam_dir.glob("*_cam.txt")}
        valid_ids = image_ids & depth_ids & cam_ids
        if not valid_ids:
            continue
        

        raw_pairs = read_pair_file(pair_path)
        pairs = {
            ref: [src for src in srcs if src in valid_ids and src != ref]
            for ref, srcs in raw_pairs.items()
            if ref in valid_ids
        }
        pairs = {ref: srcs for ref, srcs in pairs.items() if srcs}
        if not pairs:
            continue

        records.append({
            "sequence_id": scene_dir.name,
            "scene_dir": str(scene_dir.resolve()),
            "view_ids": np.asarray(sorted(valid_ids), dtype=np.int32),
            "pairs": pairs,
        })

    index = {"version": 1, "format": "blendedmvs_mvsnet", "sequences": records}

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, index, allow_pickle=True)

    return index


class BlendedMVSPi3XDataset(BaseDataset):
    def __init__(
        self,
        data_root,
        index_file,
        pair_topk=10,
        max_pair_hops=2,
        min_depth_valid_ratio=0.01,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataset_label = "BlendedMVS"
        self.data_root = Path(data_root)
        self.pair_topk = int(pair_topk)
        self.max_pair_hops = int(max_pair_hops)
        self.min_depth_valid_ratio = float(min_depth_valid_ratio)
        self.verbose = bool(verbose)

        index_path = Path(index_file)
        if not index_path.is_absolute() and not index_path.exists():
            index_path = self.data_root / index_path

        index = np.load(index_path, allow_pickle=True).item()
        self.records = list(index["sequences"])
        self.sequences = [r["sequence_id"] for r in self.records]
        self.num_imgs = {r["sequence_id"]: len(r["view_ids"]) for r in self.records}

        print(f"[{self.dataset_label}] Found {len(self.records)} scenes", flush=True)

    def __len__(self):
        return len(self.records)

    def _candidate_sources(self, ref_id: int, pairs: dict[int, list[int]]) -> list[int]:
        selected, visited, frontier = [], {ref_id}, [ref_id]

        for _ in range(max(1, self.max_pair_hops)):
            next_frontier = []
            for node in frontier:
                for src_id in pairs.get(node, [])[: self.pair_topk]:
                    if src_id in visited:
                        continue
                    visited.add(src_id)
                    selected.append(src_id)
                    next_frontier.append(src_id)

            if not next_frontier:
                break
            frontier = next_frontier

        return selected

    def _sample_view_ids(self, record, rng, is_test):
        pairs = record["pairs"]
        ref_ids = list(pairs.keys())
        if not ref_ids:
            return None

        ref_id = int(ref_ids[0] if is_test else rng.choice(ref_ids))
        pair_ids = self._candidate_sources(ref_id, pairs)
        if not is_test and pair_ids:
            pair_ids = list(rng.permutation(pair_ids))

        used = {ref_id, *map(int, pair_ids)}
        fallback = [int(x) for x in record["view_ids"] if int(x) not in used]
        if not is_test and fallback:
            fallback = list(rng.permutation(fallback))

        return [ref_id] + [int(x) for x in pair_ids] + fallback

    def _load_view(self, scene_dir: Path, view_id: int):
        name = f"{view_id:08d}"
        image = read_rgb(scene_dir / "blended_images" / f"{name}.jpg")
        depth = read_pfm(scene_dir / "rendered_depth_maps" / f"{name}.pfm")
        K, camera_pose = read_cam(scene_dir / "cams" / f"{name}_cam.txt")

        if depth.ndim == 3:
            depth = depth[..., 0]

        depth = np.asarray(depth, dtype=np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0.0] = 0.0

        if image.shape[:2] != depth.shape:
            raise ValueError(
                f"{scene_dir.name}/{name}: image/depth mismatch {image.shape[:2]} vs {depth.shape}"
            )

        return image, depth, K, camera_pose

    def _get_views(self, index, resolution, rng, is_test=False):
        is_test = bool(is_test or self.mode == "test")
        record = self.records[index]
        view_ids = self._sample_view_ids(record, rng, is_test)

        if view_ids is None:
            self.this_views_info = {
                "scene": record["sequence_id"],
                "view_ids": [],
            }
            raise RuntimeError(
                f"{record['sequence_id']}: cannot sample {self.frame_num} views"
            )

        scene_dir = Path(record["scene_dir"])
        views, selected_ids = [], []

        for view_id in view_ids:
            image, depth, K, camera_pose = self._load_view(scene_dir, view_id)

            valid_ratio = np.count_nonzero(depth > 0) / depth.size
            if valid_ratio < self.min_depth_valid_ratio:
                continue

            image, depth, K = self._crop_resize_if_necessary(
                image, depth, K, resolution, rng=rng,
                info=f"{record['sequence_id']}/{view_id:08d}",
            )[:3]

            views.append({
                "img": image,
                "depthmap": np.asarray(depth, dtype=np.float32),
                "camera_pose": np.asarray(camera_pose, dtype=np.float32),
                "camera_intrinsics": np.asarray(K, dtype=np.float32),
                "dataset": self.dataset_label,
                "sequence": record["sequence_id"],
                "label": f"{view_id:08d}",
                "instance": str(view_id),
                "prefix": f"{record['sequence_id']}_{view_id:08d}",
                "view_id": view_id,
            })
            selected_ids.append(view_id)

            if len(views) == self.frame_num:
                break

        self.this_views_info = {"scene": record["sequence_id"], "view_ids": selected_ids}
        if len(views) != self.frame_num:
            raise RuntimeError(
                f"{record['sequence_id']}: only {len(views)}/{self.frame_num} valid views "
                f"after checking {len(view_ids)} candidates"
            )

        T_ref_world = np.linalg.inv(views[0]["camera_pose"].astype(np.float64))
        for view in views:
            view["camera_pose"] = (
                T_ref_world @ view["camera_pose"].astype(np.float64)
            ).astype(np.float32)

        if self.verbose:
            ratios = [
                np.count_nonzero(view["depthmap"] > 0) / view["depthmap"].size
                for view in views
            ]
            print(
                f"[BlendedMVS] {record['sequence_id']} views={view_ids} "
                f"depth_valid={[round(x * 100, 1) for x in ratios]}%",
                flush=True,
            )

        return views


def main():
    data_root = Path("/starmap/nas/workspace/pxx/data/blendmvs")
    output_path = Path(
        "/starmap/nas/workspace/pxx/Pi3/data/dataset_cache/blendedmvs.npy"
    )

    index = generate_blendedmvs_index(data_root, output_path)
    num_views = sum(len(r["view_ids"]) for r in index["sequences"])

    print(
        f"Saved {len(index['sequences'])} scenes / {num_views} views to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()