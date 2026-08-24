from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from datasets.base.base_dataset import BaseDataset


def read_matrix(path, shape):
    x = np.loadtxt(path, dtype=np.float64)
    if x.shape != shape:
        raise ValueError(f"{path}: expected {shape}, got {x.shape}")
    return x


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def frame_key(frame_id, camera_id=0):
    return f"{frame_id:06d}_{camera_id}"


def load_pose(segment, frame_id, camera_id=0):
    """
    Processed Waymo convention:

      ego_pose/{frame}_{camera}.txt   = T_world_vehicle
      extrinsics/{frame}_{camera}.txt = T_vehicle_camera

    The processed camera frame is already standard pinhole/OpenCV:
      +x right, +y down, +z forward.
    """
    key = frame_key(frame_id, camera_id)

    T_world_vehicle = read_matrix(
        segment / "ego_pose" / f"{key}.txt",
        (4, 4),
    )
    T_vehicle_camera = read_matrix(
        segment / "extrinsics" / f"{key}.txt",
        (4, 4),
    )

    return T_world_vehicle @ T_vehicle_camera


def load_frame(segment, frame_id, camera_id=0, image=True):
    key = frame_key(frame_id, camera_id)

    K = read_matrix(
        segment / "intrinsics" / f"{key}.txt",
        (3, 3),
    ).astype(np.float32)

    depth = np.load(
        segment / "lidars" / f"{key}.npy",
        mmap_mode="r",
    )

    if depth.ndim != 2:
        raise ValueError(f"{key}: depth must be HxW, got {depth.shape}")

    out = {
        "K": K,
        "depth": depth,
        "pose": load_pose(segment, frame_id, camera_id),
    }

    if image:
        img = read_rgb(segment / "images" / f"{key}.png")

        if img.shape[:2] != depth.shape:
            raise ValueError(
                f"{key}: image/depth mismatch "
                f"{img.shape[:2]} vs {depth.shape}"
            )

        out["image"] = img

    return out


def depth_to_points(depth, K):
    """Sparse camera-Z depth -> camera-frame XYZ."""
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0))

    if len(u) == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = np.asarray(depth[v, u], dtype=np.float32)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z

    return np.stack([x, y, z], axis=1)


def transform_points(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def project_depth(points, K, height, width, splat_radius=0):
    """Camera-frame XYZ -> Z-depth, nearest point wins."""
    depth = np.zeros((height, width), dtype=np.float32)

    if len(points) == 0:
        return depth

    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-4)

    xyz = points[valid]
    z = xyz[:, 2]

    if len(xyz) == 0:
        return depth

    u = np.rint(K[0, 0] * xyz[:, 0] / z + K[0, 2]).astype(np.int32)
    v = np.rint(K[1, 1] * xyz[:, 1] / z + K[1, 2]).astype(np.int32)

    r = int(splat_radius)

    if r > 0:
        u0, v0, z0 = u, v, z
        u = np.concatenate(
            [u0 + du for dv in range(-r, r + 1) for du in range(-r, r + 1)]
        )
        v = np.concatenate(
            [v0 + dv for dv in range(-r, r + 1) for du in range(-r, r + 1)]
        )
        z = np.tile(z0, (2 * r + 1) ** 2)

    valid = (
        (u >= 0) & (u < width)
        & (v >= 0) & (v < height)
        & np.isfinite(z) & (z > 0)
    )

    u, v, z = u[valid], v[valid], z[valid].astype(np.float32)

    if len(z) == 0:
        return depth

    order = np.argsort(z)
    pixel = v[order] * width + u[order]
    _, first = np.unique(pixel, return_index=True)
    keep = order[first]

    depth[v[keep], u[keep]] = z[keep]
    return depth


def accumulate_depth(
    segment,
    target_id,
    source_ids,
    camera_id=0,
    splat_radius=0,
):
    """
    Fuse existing processed Z-depth maps into target camera.

    This is much cheaper than going back to Waymo TFRecord/LiDAR.
    """
    target = load_frame(
        segment,
        target_id,
        camera_id,
        image=True,
    )

    T_target_world = np.linalg.inv(target["pose"])
    points_target = []

    for source_id in source_ids:
        src = load_frame(
            segment,
            source_id,
            camera_id,
            image=False,
        )

        pts = depth_to_points(src["depth"], src["K"])
        if len(pts) == 0:
            continue

        pts_world = transform_points(
            src["pose"],
            pts.astype(np.float64),
        )

        points_target.append(
            transform_points(T_target_world, pts_world)
        )

    points_target = (
        np.concatenate(points_target, axis=0)
        if points_target
        else np.empty((0, 3), dtype=np.float64)
    )

    depth = project_depth(
        points_target,
        target["K"],
        target["image"].shape[0],
        target["image"].shape[1],
        splat_radius,
    )

    return target["image"], depth, target["K"], target["pose"]


def generate_waymo_processed_index(
    data_root,
    output_path=None,
    camera_id=0,
):
    """
    Scan processed segment directories.

    Required per frame:
      images/{frame}_{camera}.png
      lidars/{frame}_{camera}.npy
      intrinsics/{frame}_{camera}.txt
      extrinsics/{frame}_{camera}.txt
      ego_pose/{frame}_{camera}.txt
    """
    data_root = Path(data_root)
    records = []

    for segment in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if not all(
            (segment / d).is_dir()
            for d in ("images", "lidars", "intrinsics", "extrinsics", "ego_pose")
        ):
            continue

        frame_ids = []

        for depth_path in sorted(
            (segment / "lidars").glob(f"*_{camera_id}.npy")
        ):
            frame_id = int(depth_path.stem.split("_")[0])
            key = frame_key(frame_id, camera_id)

            required = (
                segment / "images" / f"{key}.png",
                segment / "intrinsics" / f"{key}.txt",
                segment / "extrinsics" / f"{key}.txt",
                segment / "ego_pose" / f"{key}.txt",
            )

            if all(p.is_file() for p in required):
                frame_ids.append(frame_id)

        if frame_ids:
            records.append(
                {
                    "sequence_id": segment.name,
                    "segment_dir": str(segment.resolve()),
                    "frame_ids": np.asarray(frame_ids, dtype=np.int32),
                    "camera_id": int(camera_id),
                }
            )

    index = {
        "version": 1,
        "format": "processed_waymo_z_depth",
        "sequences": records,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, index, allow_pickle=True)

    return index


class WaymoPi3XDataset(BaseDataset):
    """
    Fast Dataset for already-processed Waymo data.

    depth_accumulate = 0:
        directly use existing aligned sparse depth.

    depth_accumulate = 2:
        fuse t-2 ... t+2 using existing Z-depth + poses.

    No TensorFlow, TFRecord, Waymo API, or raw LiDAR conversion is used.
    """

    def __init__(
        self,
        data_root,
        index_file,
        camera_id=0,
        frame_step=1,
        depth_accumulate=0,
        splat_radius=0,
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataset_label = "WaymoPi3X"
        self.data_root = Path(data_root)
        self.camera_id = int(camera_id)
        self.frame_step = int(frame_step)
        self.depth_accumulate = int(depth_accumulate)
        self.splat_radius = int(splat_radius)
        self.verbose = bool(verbose)

        index_path = Path(index_file)
        if not index_path.is_absolute() and not index_path.exists():
            index_path = self.data_root / index_path

        index = np.load(index_path, allow_pickle=True).item()

        self.records = list(index["sequences"])
        self.sequences = [r["sequence_id"] for r in self.records]
        self.num_imgs = {
            r["sequence_id"]: len(r["frame_ids"])
            for r in self.records
        }

        print(
            f"[{self.dataset_label}] Found {len(self.records)} sequences, "
            f"camera={self.camera_id}, "
            f"depth_accumulate_radius={self.depth_accumulate}",
            flush=True,
        )

    def __len__(self):
        return len(self.records)

    def _sample_positions(self, n, rng, is_test):
        span = (self.frame_num - 1) * self.frame_step + 1

        if n < span:
            return None

        max_start = n - span
        start = max_start // 2 if is_test else int(
            rng.integers(0, max_start + 1)
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
        is_test = bool(is_test or self.mode == "test")

        record = self.records[index]
        segment = Path(record["segment_dir"])
        frame_ids = np.asarray(record["frame_ids"], dtype=np.int32)

        positions = self._sample_positions(
            len(frame_ids),
            rng,
            is_test,
        )

        if positions is None:
            self.this_views_info = {
                "scene": record["sequence_id"],
                "idxs": [],
            }
            required = (self.frame_num - 1) * self.frame_step + 1
            raise ValueError(
                f"{record['sequence_id']}: {len(frame_ids)} frames cannot "
                f"provide frame_num={self.frame_num} with "
                f"frame_step={self.frame_step}; required span={required}"
            )

        target_ids = [int(frame_ids[p]) for p in positions]

        self.this_views_info = {
            "scene": record["sequence_id"],
            "idxs": target_ids,
            "depth_accumulate": self.depth_accumulate,
        }

        views = []

        for pos, target_id in zip(positions, target_ids):
            if self.depth_accumulate == 0:
                frame = load_frame(
                    segment,
                    target_id,
                    self.camera_id,
                    image=True,
                )

                image = frame["image"]
                depth = np.asarray(frame["depth"], dtype=np.float32)
                K = frame["K"]
                pose = frame["pose"]

            else:
                lo = max(0, pos - self.depth_accumulate)
                hi = min(len(frame_ids), pos + self.depth_accumulate + 1)
                source_ids = [int(x) for x in frame_ids[lo:hi]]

                image, depth, K, pose = accumulate_depth(
                    segment,
                    target_id,
                    source_ids,
                    self.camera_id,
                    self.splat_radius,
                )

            if self.verbose:
                print(
                    f"[Waymo] frame={target_id} "
                    f"valid_depth={np.count_nonzero(depth > 0)}",
                    flush=True,
                )

            image, depth, K = self._crop_resize_if_necessary(
                image,
                depth,
                K,
                resolution,
                rng=rng,
                info=f"{record['sequence_id']}/{target_id}",
            )[:3]

            views.append(
                {
                    "img": image,
                    "depthmap": np.asarray(depth, dtype=np.float32),
                    "camera_pose": np.asarray(pose, dtype=np.float32),
                    "camera_intrinsics": np.asarray(K, dtype=np.float32),
                    "dataset": self.dataset_label,
                    "sequence": record["sequence_id"],
                    "label": record["sequence_id"],
                    "instance": str(target_id),
                    "prefix": (
                        f"{record['sequence_id']}_{target_id:06d}_{self.camera_id}"
                    ),
                    "frame_id": target_id,
                    "camera_id": self.camera_id,
                }
            )

        # Remove the large Waymo global translation.
        T_ref_world = np.linalg.inv(
            views[0]["camera_pose"].astype(np.float64)
        )

        for view in views:
            view["camera_pose"] = (
                T_ref_world
                @ view["camera_pose"].astype(np.float64)
            ).astype(np.float32)

        return views


def main():
    data_root = Path(
        "/starmap/nas/workspace/pxx/data/waymo_processed"
    )

    output_path = Path(
        "/starmap/nas/workspace/pxx/Pi3/"
        "data/dataset_cache/waymo_processed.npy"
    )

    index = generate_waymo_processed_index(
        data_root,
        output_path,
        camera_id=0,
    )

    num_frames = sum(
        len(record["frame_ids"])
        for record in index["sequences"]
    )

    print(
        f"Saved {len(index['sequences'])} sequences / "
        f"{num_frames} frames to {output_path}"
    )


if __name__ == "__main__":
    main()
