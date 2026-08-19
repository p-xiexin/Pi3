from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from datasets.base.base_dataset import BaseDataset


EARTH_RADIUS_METERS = 6378137.0
DEFAULT_CAMERAS = ("image_02",)


def _read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_depth(path):
    """KITTI Depth Completion groundtruth PNG: uint16 depth / 256 = meters."""
    depth = np.asarray(Image.open(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / 256.0


def _camera_name(camera):
    if isinstance(camera, int):
        return f"image_{camera:02d}"
    camera = str(camera)
    if camera.startswith("image_"):
        return f"image_{int(camera.split('_')[-1]):02d}"
    return f"image_{int(camera):02d}"


def _parse_calib(path):
    records = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)

        try:
            values = [float(x) for x in value.split()]
        except ValueError:
            continue

        if len(values) == 12:
            records[key] = np.asarray(values, dtype=np.float64).reshape(3, 4)
        elif len(values) == 9:
            records[key] = np.asarray(values, dtype=np.float64).reshape(3, 3)
        elif len(values) == 3:
            records[key] = np.asarray(values, dtype=np.float64)

    return records


def _transform(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def _rotation_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotation_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _oxts_pose(packet, scale):
    lat, lon, alt, roll, pitch, yaw = packet[:6]
    tx = scale * lon * math.pi * EARTH_RADIUS_METERS / 180.0
    ty = scale * EARTH_RADIUS_METERS * math.log(math.tan((90.0 + lat) * math.pi / 360.0))
    R = _rotation_z(yaw) @ _rotation_y(pitch) @ _rotation_x(roll)
    return _transform(R, [tx, ty, alt])


def _load_oxts_poses(oxts_dir):
    files = sorted(oxts_dir.glob("*.txt"))
    if not files:
        return []

    packets = []
    for path in files:
        values = np.asarray([float(x) for x in path.read_text().split()], dtype=np.float64)
        if len(values) < 6:
            raise ValueError(f"Invalid OXTS packet: {path}")
        packets.append(values)

    scale = math.cos(float(packets[0][0]) * math.pi / 180.0)
    poses = [_oxts_pose(packet, scale) for packet in packets]
    T0_inv = np.linalg.inv(poses[0])
    return [(T0_inv @ pose).astype(np.float32) for pose in poses]


def _camera_geometry(date_root, cameras):
    """
    Return:
      T_camera_imu[camera] : IMU -> rectified camera
      K[camera]            : rectified camera intrinsics
    """
    cam = _parse_calib(date_root / "calib_cam_to_cam.txt")
    velo = _parse_calib(date_root / "calib_velo_to_cam.txt")
    imu = _parse_calib(date_root / "calib_imu_to_velo.txt")

    R_rect = np.eye(4, dtype=np.float64)
    R_rect[:3, :3] = cam["R_rect_00"]
    T_cam0_velo = _transform(velo["R"], velo["T"])
    T_velo_imu = _transform(imu["R"], imu["T"])
    T_rect_cam0_imu = R_rect @ T_cam0_velo @ T_velo_imu

    T_camera_imu, intrinsics = {}, {}
    for camera in cameras:
        suffix = int(camera.split("_")[-1])
        P = cam[f"P_rect_{suffix:02d}"]
        K = P[:3, :3].copy()

        # P_rect_i = K_i [I | t_i] expressed from rectified cam0 coordinates.
        T_camera_cam0 = np.eye(4, dtype=np.float64)
        T_camera_cam0[0, 3] = P[0, 3] / P[0, 0]
        T_camera_cam0[1, 3] = P[1, 3] / P[1, 1]

        T_camera_imu[camera] = T_camera_cam0 @ T_rect_cam0_imu
        intrinsics[camera] = K.astype(np.float32)

    return T_camera_imu, intrinsics


def _splat_depth(depth, radius):
    """Optional Waymo-style local support expansion; nearest depth wins."""
    radius = int(radius)
    if radius <= 0:
        return np.asarray(depth, dtype=np.float32)

    depth = np.asarray(depth, dtype=np.float32)
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    if len(x) == 0:
        return np.zeros_like(depth, dtype=np.float32)

    z = depth[y, x]
    H, W = depth.shape
    xs = np.concatenate([x + dx for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)])
    ys = np.concatenate([y + dy for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)])
    zs = np.tile(z, (2 * radius + 1) ** 2)

    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, zs = xs[valid], ys[valid], zs[valid]

    order = np.argsort(zs)
    pixel = ys[order] * W + xs[order]
    _, first = np.unique(pixel, return_index=True)
    keep = order[first]

    out = np.zeros_like(depth, dtype=np.float32)
    out[ys[keep], xs[keep]] = zs[keep]
    return out


def generate_kitti_index(
    raw_root,
    depth_root,
    output_path=None,
    cameras=DEFAULT_CAMERAS,
    splits=("train", "val"),
):
    """
    Build the index from official Depth Completion groundtruth.

    Expected layout:
      raw_root/
        2011_09_26/
          calib_*.txt
          2011_09_26_drive_xxxx_sync/
            image_02/data/*.png
            oxts/data/*.txt

      depth_root/
        train|val/
          2011_09_26_drive_xxxx_sync/
            proj_depth/groundtruth/image_02/*.png
    """
    raw_root, depth_root = Path(raw_root), Path(depth_root)
    cameras = tuple(_camera_name(camera) for camera in cameras)
    records, geometry_cache = [], {}

    for split in splits:
        split_root = depth_root / split
        if not split_root.is_dir():
            continue

        for drive_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            sequence, date = drive_dir.name, drive_dir.name[:10]
            raw_sequence = raw_root / date / sequence
            date_root = raw_root / date
            oxts_dir = raw_sequence / "oxts" / "data"
            if not raw_sequence.is_dir() or not oxts_dir.is_dir():
                continue

            if date not in geometry_cache:
                geometry_cache[date] = _camera_geometry(date_root, cameras)
            T_camera_imu, intrinsics = geometry_cache[date]
            poses_world_imu = _load_oxts_poses(oxts_dir)

            for camera in cameras:
                depth_dir = drive_dir / "proj_depth" / "groundtruth" / camera
                image_dir = raw_sequence / camera / "data"
                if not depth_dir.is_dir() or not image_dir.is_dir() or camera not in T_camera_imu:
                    continue

                T_imu_camera = np.linalg.inv(T_camera_imu[camera])
                frames = []

                for depth_path in sorted(depth_dir.glob("*.png")):
                    frame_id = depth_path.stem
                    frame_no = int(frame_id)
                    image_path = image_dir / f"{frame_id}.png"
                    if not image_path.is_file() or frame_no >= len(poses_world_imu):
                        continue

                    T_world_camera = poses_world_imu[frame_no].astype(np.float64) @ T_imu_camera
                    frames.append({
                        "frame_id": frame_id,
                        "frame_no": frame_no,
                        "image": str(image_path.resolve()),
                        "depth": str(depth_path.resolve()),
                        "camera_pose": T_world_camera.astype(np.float32).tolist(),
                        "camera_intrinsics": intrinsics[camera].tolist(),
                    })

                if frames:
                    records.append({
                        "sequence_id": f"{split}/{sequence}/{camera}",
                        "split": split,
                        "drive": sequence,
                        "date": date,
                        "camera": camera,
                        "frames": frames,
                    })

    index = {"version": 2, "sequences": records}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, index, allow_pickle=True)
    return index


class KITTIPi3XDataset(BaseDataset):
    """
    KITTI Raw RGB/pose + official Depth Completion groundtruth.

    Depth is never reconstructed from Velodyne here; the official aligned
    semi-dense groundtruth PNG is used directly.
    """

    def __init__(
        self,
        raw_root,
        depth_root,
        index_file=None,
        cameras=DEFAULT_CAMERAS,
        splits=("train",),
        frame_step=1,
        splat_radius=0,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataset_label = "KITTIPi3X"
        self.raw_root, self.depth_root = Path(raw_root), Path(depth_root)
        self.cameras = tuple(_camera_name(camera) for camera in cameras)
        self.splits = set(splits)
        self.frame_step = int(frame_step)
        self.splat_radius = int(splat_radius)
        self.verbose = bool(verbose)

        if index_file is None:
            index = generate_kitti_index(self.raw_root, self.depth_root, cameras=self.cameras, splits=tuple(self.splits))
        else:
            index = np.load(index_file, allow_pickle=True).item()

        self.records = [
            record for record in index["sequences"]
            if record["camera"] in self.cameras and record["split"] in self.splits
        ]
        self.sequences = [record["sequence_id"] for record in self.records]
        self.num_imgs = {record["sequence_id"]: len(record["frames"]) for record in self.records}

        print(
            f"[{self.dataset_label}] Found {len(self.records)} sequences, "
            f"cameras={self.cameras}, splat_radius={self.splat_radius}",
            flush=True,
        )

    def __len__(self):
        return len(self.records)

    def _sample_positions(self, n, rng, is_test):
        span = (self.frame_num - 1) * self.frame_step + 1
        if n < span:
            return None
        max_start = n - span
        start = max_start // 2 if is_test else int(rng.integers(0, max_start + 1))
        return [start + i * self.frame_step for i in range(self.frame_num)]

    def _get_views(self, index, resolution, rng, is_test=False):
        is_test = bool(is_test or self.mode == "test")
        record = self.records[index]
        frames = record["frames"]
        positions = self._sample_positions(len(frames), rng, is_test)

        if positions is None:
            self.this_views_info = {"scene": record["sequence_id"], "idxs": []}
            return []

        self.this_views_info = {
            "scene": record["sequence_id"],
            "drive": record["drive"],
            "camera": record["camera"],
            "idxs": positions,
            "frame_step": self.frame_step,
        }

        views = []
        for pos in positions:
            frame = frames[pos]
            image = _read_rgb(frame["image"])
            depth = _read_depth(frame["depth"])
            if self.splat_radius > 0:
                depth = _splat_depth(depth, self.splat_radius)

            K = np.asarray(frame["camera_intrinsics"], dtype=np.float32)
            pose = np.asarray(frame["camera_pose"], dtype=np.float32)

            if self.verbose:
                valid = np.isfinite(depth) & (depth > 0)
                print(
                    f"[KITTI] {record['drive']} {record['camera']} frame={frame['frame_id']} "
                    f"depth_valid={valid.mean():.2%}",
                    flush=True,
                )

            image, depth, K = self._crop_resize_if_necessary(
                image, depth, K, resolution, rng=rng, info=frame["image"]
            )[:3]

            views.append({
                "img": image,
                "depthmap": np.asarray(depth, dtype=np.float32),
                "camera_pose": pose,
                "camera_intrinsics": np.asarray(K, dtype=np.float32),
                "dataset": self.dataset_label,
                "sequence": record["sequence_id"],
                "label": record["drive"],
                "instance": frame["frame_id"],
                "prefix": f"{record['drive']}_{record['camera']}_{frame['frame_id']}",
                "image_path": frame["image"],
                "depth_path": frame["depth"],
                "camera_name": record["camera"],
                "depth_source": "kitti_official_depth_completion_groundtruth",
                "depth_definition": "camera_z_m",
                "pose_source": "kitti_raw_oxts",
                "pseudo_label": False,
                "valid_mask_required": True,
            })

        # Local sample frame, same convention used by Waymo/nuScenes datasets.
        T0_inv = np.linalg.inv(views[0]["camera_pose"].astype(np.float64))
        for view in views:
            view["camera_pose"] = (
                T0_inv @ view["camera_pose"].astype(np.float64)
            ).astype(np.float32)

        return views


def main():
    raw_root = Path("/starmap/nas/workspace/pxx/data/kitti_raw")
    depth_root = Path("/starmap/nas/workspace/pxx/data/kitti_depth")
    output_path = Path("/starmap/nas/workspace/pxx/Pi3/data/dataset_cache/kitti.npy")

    index = generate_kitti_index(
        raw_root,
        depth_root,
        output_path,
        cameras=("image_02",),
        splits=("train", "val"),
    )
    print(f"Saved {len(index['sequences'])} sequences to {output_path}", flush=True)


if __name__ == "__main__":
    main()