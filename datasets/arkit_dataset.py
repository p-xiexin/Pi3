from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets.base.base_dataset import BaseDataset

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]


def _path_roots(roots: dict[str, str | Path] | None) -> dict[str, Path]:
    return {key: Path(value) for key, value in (roots or {}).items() if value is not None}


def _optional_path_roots(roots: dict[str, str | Path | None] | None) -> dict[str, Path | None]:
    return {key: None if value is None else Path(value) for key, value in (roots or {}).items()}


def _require_dir(path: Path, name: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{name} directory not found: {path}")
    return path


def _resolve_existing_path(data_root: Path, value: str | Path, name: str) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [path, data_root / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found: {candidates[-1]}")


def _absolute(path: Path) -> str:
    return str(path.resolve())


def _read_rgb_image(path: Path) -> np.ndarray | None:
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _read_depth_png_meters(path: Path) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    return depth / 1000.0


def _timestamp_from_stem(path: Path) -> float:
    stem = path.stem
    if "_" in stem:
        stem = stem.rsplit("_", 1)[-1]
    return float(stem)


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    if cv2 is not None:
        matrix, _ = cv2.Rodrigues(axis_angle)
        return matrix.astype(np.float64)
    axis = axis_angle / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _read_pincam(path: Path) -> dict[str, Any]:
    """ARKitScenes .pincam: width height fx fy cx cy."""
    values = [float(item) for item in path.read_text(encoding="utf-8").split()]
    if len(values) != 6:
        raise ValueError(f"{path}: expected width height fx fy cx cy")
    width, height, fx, fy, cx, cy = values
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return {"width": int(round(width)), "height": int(round(height)), "K": K}


def _read_trajectory(path: Path) -> dict[float, np.ndarray]:
    """
    Parse lowres_wide.traj and return camera->world poses.

    The matrix formed directly from axis-angle + translation is world->camera
    in the official ARKitScenes loader, so it must be inverted before being
    stored as BaseDataset camera_pose.
    """
    poses: dict[float, np.ndarray] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 7:
            raise ValueError(f"{path}:{line_no}: expected timestamp axis-angle xyz translation")

        # Keep timestamps out of float32; millisecond matching matters here.
        timestamp = float(parts[0])
        axis_angle = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
        translation = np.array([float(parts[4]), float(parts[5]), float(parts[6])], dtype=np.float64)

        T_w2c = np.eye(4, dtype=np.float64)
        T_w2c[:3, :3] = _axis_angle_to_matrix(axis_angle)
        T_w2c[:3, 3] = translation
        poses[timestamp] = np.linalg.inv(T_w2c).astype(np.float32)
    return poses


def _nearest_timestamp(timestamp: float, items: dict[float, Any], tolerance: float) -> tuple[float, Any] | None:
    if not items:
        return None
    key = min(items.keys(), key=lambda t: abs(t - timestamp))
    if abs(key - timestamp) > tolerance:
        return None
    return key, items[key]


def _align_lowres_rgb_to_depth_canvas(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    Put ARKitScenes lowres_wide RGB into the lowres_depth pixel canvas.

    Official ARKitScenes handling:
      RGB   = 256x192
      depth = 384x288
      RGB is copied unchanged into canvas[y=48:240, x=64:320].
    """
    dh, dw = depth.shape
    ih, iw = rgb.shape[:2]

    if (ih, iw) == (dh, dw):
        return rgb

    if (ih, iw) == (192, 256) and (dh, dw) == (288, 384):
        canvas = np.zeros((288, 384, 3), dtype=rgb.dtype)
        canvas[48:48 + 192, 64:64 + 256] = rgb
        return canvas

    raise ValueError(
        "Unsupported ARKitScenes RGB/depth geometry: "
        f"rgb={iw}x{ih}, depth={dw}x{dh}. "
        "Do not independently resize RGB; add an explicit verified alignment rule."
    )


def _validate_geometry(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, pose_c2w: np.ndarray, info: str) -> None:
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"{info}: RGB/depth mismatch after alignment: {rgb.shape[:2]} vs {depth.shape}")
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError(f"{info}: invalid intrinsics")
    if pose_c2w.shape != (4, 4) or not np.isfinite(pose_c2w).all():
        raise ValueError(f"{info}: invalid camera pose")

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    h, w = depth.shape
    if fx <= 0 or fy <= 0:
        raise ValueError(f"{info}: non-positive focal length fx={fx}, fy={fy}")
    if not (0.0 <= cx < w and 0.0 <= cy < h):
        raise ValueError(f"{info}: principal point ({cx:.3f},{cy:.3f}) outside {w}x{h}")

    det_R = float(np.linalg.det(pose_c2w[:3, :3].astype(np.float64)))
    if not np.isfinite(det_R) or abs(det_R - 1.0) > 1e-2:
        raise ValueError(f"{info}: det(R)={det_R}, expected ~1")


def _default_scans_root(data_root: Path) -> Path:
    if (data_root / "raw").is_dir():
        return data_root / "raw"
    return data_root


def _is_raw_scan_dir(scan_dir: Path) -> bool:
    return (
        (scan_dir / "lowres_wide").is_dir()
        and (scan_dir / "lowres_depth").is_dir()
        and (scan_dir / "lowres_wide_intrinsics").is_dir()
        and (scan_dir / "lowres_wide.traj").is_file()
    )


def _find_scan_dir(scans_root: Path, scan_id: str, splits: tuple[str, ...]) -> Path:
    candidates = [scans_root / split / scan_id for split in splits] + [scans_root / scan_id]
    for candidate in candidates:
        if candidate.is_dir() and _is_raw_scan_dir(candidate):
            return candidate
    raise FileNotFoundError(f"ARKitScenes scan not found under {scans_root}: {scan_id}")


def _discover_scan_dirs(scans_root: Path, splits: tuple[str, ...], scan_ids: list[str] | None) -> dict[str, Path]:
    if scan_ids:
        return {scan_id: _find_scan_dir(scans_root, scan_id, splits) for scan_id in scan_ids}

    scan_dirs: dict[str, Path] = {}
    roots = [scans_root / split for split in splits if (scans_root / split).is_dir()]
    if not roots:
        roots = [scans_root]
    for root in roots:
        for path in sorted(root.iterdir()):
            if path.is_dir() and _is_raw_scan_dir(path):
                scan_dirs[path.name] = path
    return scan_dirs


def generate_arkit_scenes_index(
    data_root: str | Path,
    output_path: str | Path | None = None,
    scan_ids: list[str] | None = None,
    splits: tuple[str, ...] = ("Training", "Validation"),
    roots: dict[str, str | Path] | None = None,
    rgb_tolerance_s: float = 0.005,
    intrinsics_tolerance_s: float = 0.002,
    pose_tolerance_s: float = 0.005,
) -> dict[str, Any]:
    """Build a strict index anchored on lowres_depth timestamps."""
    data_root = Path(data_root)
    scans_root = _require_dir(_path_roots(roots).get("scans", _default_scans_root(data_root)), "roots.scans")
    scan_dirs = _discover_scan_dirs(scans_root, splits, scan_ids)

    records: list[dict[str, Any]] = []
    for scan_id in tqdm(sorted(scan_dirs), desc="[ARKitScenes] building index", unit="scan"):
        scan_dir = scan_dirs[scan_id]
        image_dir = scan_dir / "lowres_wide"
        depth_dir = scan_dir / "lowres_depth"
        confidence_dir = scan_dir / "confidence"
        intrinsics_dir = scan_dir / "lowres_wide_intrinsics"
        pose_path = scan_dir / "lowres_wide.traj"

        for name, path in (
            ("lowres_wide", image_dir),
            ("lowres_depth", depth_dir),
            ("lowres_wide_intrinsics", intrinsics_dir),
        ):
            _require_dir(path, f"{scan_id}.{name}")
        if not pose_path.is_file():
            raise FileNotFoundError(f"ARKitScenes trajectory not found: {pose_path}")

        poses = _read_trajectory(pose_path)
        rgb_by_time = {_timestamp_from_stem(path): path for path in image_dir.glob("*.png")}
        intrinsics_by_time = {_timestamp_from_stem(path): _read_pincam(path) for path in intrinsics_dir.glob("*.pincam")}
        confidence_by_time = (
            {_timestamp_from_stem(path): path for path in confidence_dir.glob("*.png")}
            if confidence_dir.is_dir() else {}
        )
        depth_paths = sorted(depth_dir.glob("*.png"), key=_timestamp_from_stem)

        frames: list[dict[str, Any]] = []
        rejected = 0
        for depth_path in depth_paths:
            timestamp = _timestamp_from_stem(depth_path)
            rgb_match = _nearest_timestamp(timestamp, rgb_by_time, rgb_tolerance_s)
            intrinsic_match = _nearest_timestamp(timestamp, intrinsics_by_time, intrinsics_tolerance_s)
            pose_match = _nearest_timestamp(timestamp, poses, pose_tolerance_s)
            if rgb_match is None or intrinsic_match is None or pose_match is None:
                rejected += 1
                continue

            rgb_timestamp, image_path = rgb_match
            intrinsics_timestamp, intrinsic_record = intrinsic_match
            pose_timestamp, pose_c2w = pose_match

            confidence_path = None
            if confidence_by_time:
                conf_match = _nearest_timestamp(timestamp, confidence_by_time, rgb_tolerance_s)
                if conf_match is not None:
                    confidence_path = conf_match[1]

            frames.append({
                "frame_id": depth_path.stem,
                "timestamp": timestamp,
                "rgb_timestamp": rgb_timestamp,
                "intrinsics_timestamp": intrinsics_timestamp,
                "pose_timestamp": pose_timestamp,
                "image": _absolute(image_path),
                "depth": _absolute(depth_path),
                "confidence": None if confidence_path is None else _absolute(confidence_path),
                "camera_intrinsics": intrinsic_record["K"].astype(np.float32).tolist(),
                "pincam_size": [int(intrinsic_record["width"]), int(intrinsic_record["height"])],
                "camera_pose": pose_c2w.astype(np.float32).tolist(),
            })

        if frames:
            records.append({"sequence_id": scan_id, "scan_dir": _absolute(scan_dir), "frames": frames})
        if rejected:
            print(
                f"[ARKitScenes] {scan_id}: kept={len(frames)}, "
                f"rejected_timestamp_mismatch={rejected}",
                file=sys.stderr,
                flush=True,
            )

    index = {
        "version": 2,
        "pose_convention": "camera_to_world",
        "frame_anchor": "lowres_depth_timestamp",
        "sequences": records,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            np.save(handle, index, allow_pickle=True)
    return index


class ARKitScenesPi3XDataset(BaseDataset):
    """
    ARKitScenes loader for Pi3 / Glob3R.

    Guarantees:
      - camera_pose is camera->world;
      - RGB/depth/K share one pixel canvas before BaseDataset crop/resize;
      - timestamp association is bounded by explicit tolerances;
      - views are sampled from a local temporal window.
    """

    def __init__(
        self,
        data_root: str | Path,
        verbose: bool = False,
        index_file: str | Path | None = None,
        scan_ids: list[str] | None = None,
        splits: tuple[str, ...] = ("Training", "Validation"),
        roots: dict[str, str | Path] | None = None,
        optional_roots: dict[str, str | Path | None] | None = None,
        frame_step: int = 1,
        rgb_tolerance_s: float = 0.005,
        intrinsics_tolerance_s: float = 0.002,
        pose_tolerance_s: float = 0.005,
        **kwargs: Any,
    ) -> None:
        self.verbose = verbose
        self.frame_step = int(frame_step)
        if self.frame_step <= 0:
            raise ValueError(f"frame_step must be positive, got {frame_step}")

        super().__init__(**kwargs)
        self.dataset_label = "ARKitScenesPi3X"
        self.data_root = Path(data_root)
        component_roots = _path_roots(roots)
        self.optional_roots = _optional_path_roots(optional_roots)
        self.scans_root = _require_dir(
            component_roots.get("scans", _default_scans_root(self.data_root)),
            "roots.scans",
        )
        self.splits = splits

        if index_file is None:
            index = generate_arkit_scenes_index(
                self.data_root,
                scan_ids=scan_ids,
                splits=splits,
                roots=roots,
                rgb_tolerance_s=rgb_tolerance_s,
                intrinsics_tolerance_s=intrinsics_tolerance_s,
                pose_tolerance_s=pose_tolerance_s,
            )
        else:
            index_file_path = _resolve_existing_path(self.data_root, index_file, "index_file")
            index = np.load(index_file_path, allow_pickle=True).item()
            if int(index.get("version", 1)) < 2:
                raise ValueError(
                    f"Old ARKitScenes index detected: {index_file_path}. "
                    "Regenerate it: version-1 indexes may contain the old pose/timestamp associations."
                )

        selected = set(scan_ids or [])
        self.sequences: list[str] = []
        self.frames: dict[str, list[dict[str, Any]]] = {}
        self.scan_dirs: dict[str, str] = {}

        for record in index.get("sequences", []):
            scan_id = record["sequence_id"]
            if selected and scan_id not in selected:
                continue
            frames = sorted(record.get("frames", []), key=lambda x: float(x.get("timestamp", 0.0)))
            if not frames:
                continue
            self.sequences.append(scan_id)
            self.frames[scan_id] = frames
            self.scan_dirs[scan_id] = record.get("scan_dir", "")

        self.num_imgs = {scan_id: len(frames) for scan_id, frames in self.frames.items()}
        if self.verbose:
            print(f"[{self.dataset_label}] sequences={self.sequences}", flush=True)
        print(
            f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.scans_root}",
            file=sys.stderr,
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def _sample_local_indices(
        self,
        num_frames: int,
        rng: np.random.Generator,
        is_test: bool,
    ) -> list[int] | None:
        span = (self.frame_num - 1) * self.frame_step + 1
        if num_frames < span:
            return None
        max_start = num_frames - span
        start = max_start // 2 if is_test else int(rng.integers(0, max_start + 1))
        return [start + i * self.frame_step for i in range(self.frame_num)]

    def _get_views(
        self,
        index: int,
        resolution: list[int],
        rng: np.random.Generator,
        is_test: bool = False,
    ) -> list[dict[str, Any]]:
        scene = self.sequences[index]
        frames = self.frames.get(scene, [])
        idxs = self._sample_local_indices(len(frames), rng, is_test)

        if idxs is None:
            self.this_views_info = {
                "scene": scene,
                "frame_num": self.frame_num,
                "frame_step": self.frame_step,
                "available": len(frames),
                "idxs": [],
            }
            required = (self.frame_num - 1) * self.frame_step + 1
            raise ValueError(
                f"ARKitScenes sequence {scene} has {len(frames)} frames, "
                f"but requires {required}"
            )

        self.this_views_info = {
            "scene": scene,
            "frame_num": self.frame_num,
            "frame_step": self.frame_step,
            "idxs": idxs,
        }

        views: list[dict[str, Any]] = []
        for idx in idxs:
            frame = frames[idx]
            image_path = Path(frame["image"])
            depth_path = Path(frame["depth"])

            rgb = _read_rgb_image(image_path)
            if rgb is None:
                raise RuntimeError(f"Failed to read RGB image: {image_path}")
            depthmap = _read_depth_png_meters(depth_path)

            # Critical ordering: align RGB to depth/K canvas BEFORE Pi3 crop/resize.
            rgb = _align_lowres_rgb_to_depth_canvas(rgb, depthmap)
            intrinsics = np.asarray(frame["camera_intrinsics"], dtype=np.float32).copy()
            camera_pose = np.asarray(frame["camera_pose"], dtype=np.float32).copy()

            _validate_geometry(
                rgb,
                depthmap,
                intrinsics,
                camera_pose,
                info=f"{scene}/{frame['frame_id']}",
            )

            img, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=str(image_path),
            )[:3]

            views.append({
                "img": img,
                "depthmap": np.asarray(depthmap, dtype=np.float32),
                "camera_pose": camera_pose,
                "camera_intrinsics": np.asarray(intrinsics, dtype=np.float32),
                "dataset": self.dataset_label,
                "sequence": scene,
                "path": self.scan_dirs.get(scene, ""),
                "label": scene,
                "instance": frame["frame_id"],
                "prefix": f"{scene}_{frame['frame_id']}",
                "image_path": str(image_path),
                "depth_path": str(depth_path),
                "timestamp": float(frame["timestamp"]),
                "rgb_timestamp": float(frame["rgb_timestamp"]),
                "pose_timestamp": float(frame["pose_timestamp"]),
                "intrinsics_timestamp": float(frame["intrinsics_timestamp"]),
                "depth_source": "arkitscenes_lowres_depth",
                "pose_source": "arkitscenes_traj_c2w",
                "intrinsics_source": "arkitscenes_pincam",
                "pseudo_label": False,
                "valid_mask_required": True,
            })

        if len(views) != self.frame_num:
            raise RuntimeError(f"{scene}: expected {self.frame_num} views, got {len(views)}")
        return views

def main():
    data_root = Path("/starmap/nas/workspace/pxx/data/arkitscenes")
    output_path = Path(
        "/starmap/nas/workspace/pxx/Pi3/data/dataset_cache/arkitscenes.npy"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    index = generate_arkit_scenes_index(
        data_root=data_root,
        output_path=output_path,
        scan_ids=None,  # None = 全部场景
        splits=("Training", "Validation"),
        roots=None,
        rgb_tolerance_s=0.005,
        intrinsics_tolerance_s=0.002,
        pose_tolerance_s=0.005,
    )

    num_sequences = len(index["sequences"])
    num_frames = sum(
        len(sequence["frames"])
        for sequence in index["sequences"]
    )

    print("\nARKitScenes index generated successfully.")
    print(f"Sequences : {num_sequences}")
    print(f"Frames    : {num_frames}")
    print(f"Saved to  : {output_path}")


if __name__ == "__main__":
    main()
