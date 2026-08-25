from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from datasets.base.base_dataset import BaseDataset


CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
)


def _api():
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import points_in_box, transform_matrix, view_points
        from pyquaternion import Quaternion
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "NuScenesPi3XDataset requires nuscenes-devkit. "
            "Install with: python -m pip install nuscenes-devkit"
        ) from exc
    return NuScenes, LidarPointCloud, points_in_box, transform_matrix, view_points, Quaternion


def _read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _sensor_pose_global(nusc, sample_data_token):
    """Return T_global_sensor."""
    _, _, _, transform_matrix, _, Quaternion = _api()
    sd = nusc.get("sample_data", sample_data_token)
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    T_ego_sensor = transform_matrix(calib["translation"], Quaternion(calib["rotation"]), inverse=False)
    T_global_ego = transform_matrix(ego["translation"], Quaternion(ego["rotation"]), inverse=False)
    return (T_global_ego @ T_ego_sensor).astype(np.float64)


def _transform_points(T, points):
    return points @ T[:3, :3].T + T[:3, 3]


def _load_static_lidar_global(nusc, sample_token, min_lidar_distance=1.0, remove_annotated_objects=True, box_scale=1.05):
    """Load one keyframe LIDAR_TOP, remove annotated objects, transform to global."""
    _, LidarPointCloud, points_in_box, _, _, _ = _api()
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]

    pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))
    xyz = pc.points[:3].astype(np.float64)
    valid = np.isfinite(xyz).all(axis=0) & (np.linalg.norm(xyz[:2], axis=0) > float(min_lidar_distance))
    xyz = xyz[:, valid]

    if remove_annotated_objects and xyz.shape[1] > 0:
        ann_tokens = [
            token for token in sample["anns"]
            if nusc.get("sample_annotation", token).get("num_lidar_pts", 0) > 0
        ]
        _, boxes, _ = nusc.get_sample_data(lidar_token, selected_anntokens=ann_tokens)
        keep = np.ones(xyz.shape[1], dtype=bool)
        for box in boxes:
            keep &= ~points_in_box(box, xyz, wlh_factor=float(box_scale))
        xyz = xyz[:, keep]

    return _transform_points(_sensor_pose_global(nusc, lidar_token), xyz.T).astype(np.float32)


def _voxel_downsample(points, voxel_size):
    voxel_size = float(voxel_size)
    if voxel_size <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def _build_shared_static_map(nusc, scene_tokens, min_lidar_distance=1.0, remove_annotated_objects=True, box_scale=1.05, voxel_size=0.0):
    """
    Build one static map from ALL keyframe LiDAR scans in the scene.
    Supervision cameras may use only a small sampled window, but every view
    is rendered from this same scene-level point set.
    """
    point_sets = []
    for token in scene_tokens:
        points = _load_static_lidar_global(
            nusc, token, min_lidar_distance=min_lidar_distance,
            remove_annotated_objects=remove_annotated_objects, box_scale=box_scale,
        )
        if len(points):
            point_sets.append(points)

    if not point_sets:
        return np.empty((0, 3), dtype=np.float32)
    return _voxel_downsample(np.concatenate(point_sets, axis=0), voxel_size).astype(np.float32)


def _splat_depth(uv, z, height, width, radius=1):
    """Projected points -> camera-Z depth; nearest depth wins."""
    depth = np.zeros((height, width), dtype=np.float32)
    if len(z) == 0:
        return depth

    valid = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
    uv, z = uv[valid], z[valid].astype(np.float32)
    if len(z) == 0:
        return depth

    u, v = np.rint(uv[:, 0]).astype(np.int32), np.rint(uv[:, 1]).astype(np.int32)
    r = int(radius)
    if r > 0:
        u0, v0, z0 = u, v, z
        u = np.concatenate([u0 + du for dv in range(-r, r + 1) for du in range(-r, r + 1)])
        v = np.concatenate([v0 + dv for dv in range(-r, r + 1) for du in range(-r, r + 1)])
        z = np.tile(z0, (2 * r + 1) ** 2)

    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height) & np.isfinite(z) & (z > 0)
    u, v, z = u[inside], v[inside], z[inside]
    if len(z) == 0:
        return depth

    order = np.argsort(z)
    pixel = v[order] * width + u[order]
    _, first = np.unique(pixel, return_index=True)
    keep = order[first]
    depth[v[keep], u[keep]] = z[keep]
    return depth


def _render_shared_map_to_camera(nusc, sample_token, camera, points_global, splat_radius=1, min_camera_depth=0.5):
    """Render the same scene-level global point map into one target camera."""
    _, _, _, _, view_points, _ = _api()
    sample = nusc.get("sample", sample_token)
    camera_token = sample["data"][camera]
    camera_sd = nusc.get("sample_data", camera_token)
    calib = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])

    image_path = nusc.get_sample_data_path(camera_token)
    image = _read_rgb(image_path)
    K = np.asarray(calib["camera_intrinsic"], dtype=np.float32)

    T_global_camera = _sensor_pose_global(nusc, camera_token)
    points_camera = _transform_points(np.linalg.inv(T_global_camera), points_global.astype(np.float64))
    valid = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > float(min_camera_depth))
    points_camera = points_camera[valid]

    if len(points_camera):
        z = points_camera[:, 2]
        projected = view_points(points_camera.T, K, normalize=True)
        u, v = projected[0], projected[1]
        fov = (
            np.isfinite(u) & np.isfinite(v)
            & (u >= 0) & (u < image.shape[1])
            & (v >= 0) & (v < image.shape[0])
        )
        uv, z = np.stack([u[fov], v[fov]], axis=1), z[fov]
    else:
        uv, z = np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    depth = _splat_depth(uv, z, image.shape[0], image.shape[1], radius=splat_radius)
    return image, depth, K, T_global_camera.astype(np.float32), str(image_path)


def generate_nuscenes_index(data_root, output_path=None, version="v1.0-trainval", cameras=CAMERAS):
    """One sequence = one scene + one fixed camera."""
    NuScenes, _, _, _, _, _ = _api()
    nusc = NuScenes(version=version, dataroot=str(data_root), verbose=False)
    records = []

    for scene in nusc.scene:
        tokens, token = [], scene["first_sample_token"]
        while token:
            sample = nusc.get("sample", token)
            tokens.append(token)
            token = sample["next"]

        for camera in cameras:
            valid_tokens = [
                token for token in tokens
                if camera in nusc.get("sample", token)["data"] and "LIDAR_TOP" in nusc.get("sample", token)["data"]
            ]
            if valid_tokens:
                records.append({
                    "sequence_id": f"{scene['name']}/{camera}",
                    "scene_name": scene["name"],
                    "scene_token": scene["token"],
                    "camera": camera,
                    "sample_tokens": valid_tokens,
                })

    index = {
        "version": 3,
        "dataset_version": version,
        "geometry": "scene_shared_static_keyframe_lidar_map",
        "sequences": records,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, index, allow_pickle=True)
    return index


class NuScenesPi3XDataset(BaseDataset):
    """
    Local-shared-static-map nuScenes Dataset.

    Camera supervision samples `frame_num` views. The shared static map uses
    the continuous keyframe interval spanning those views plus `map_context`
    keyframes on both sides. Thus frames skipped by `frame_step` still
    contribute LiDAR points to the shared map.
    """

    def __init__(
        self, data_root, index_file=None, version="v1.0-mini", cameras=None,
        frame_step=1, lidar_sweeps=None, min_lidar_distance=1.0, min_camera_depth=0.5,
        splat_radius=1, remove_annotated_objects=True, box_scale=1.05,
        voxel_size=0.0, map_context=4, scan_cache_size=64,
        verbose=False, **kwargs,
    ):
        super().__init__(**kwargs)
        NuScenes, _, _, _, _, _ = _api()

        self.dataset_label = "NuScenesPi3X"
        self.data_root, self.version = Path(data_root), str(version)
        self.frame_step, self.splat_radius = int(frame_step), int(splat_radius)
        self.min_lidar_distance, self.min_camera_depth = float(min_lidar_distance), float(min_camera_depth)
        self.remove_annotated_objects, self.box_scale = bool(remove_annotated_objects), float(box_scale)
        self.voxel_size, self.verbose = float(voxel_size), bool(verbose)
        self.map_context = int(map_context)
        self.scan_cache_size = int(scan_cache_size)
        self.lidar_sweeps = lidar_sweeps  # old config compatibility only
        self.nusc = NuScenes(version=self.version, dataroot=str(self.data_root), verbose=False)

        selected_cameras = set(cameras or CAMERAS)
        if index_file is None:
            index = generate_nuscenes_index(self.data_root, version=self.version, cameras=tuple(selected_cameras))
        else:
            index_path = Path(index_file)
            if not index_path.is_absolute() and not index_path.exists():
                index_path = self.data_root / index_path
            index = np.load(index_path, allow_pickle=True).item()

        self.records = [r for r in index["sequences"] if r["camera"] in selected_cameras]
        self.sequences = [r["sequence_id"] for r in self.records]
        self.num_imgs = {r["sequence_id"]: len(r["sample_tokens"]) for r in self.records}

        # Per-worker LRU of already filtered keyframe LiDAR in global coordinates.
        # Overlapping training windows can reuse most scans without rebuilding them.
        self._scan_cache = OrderedDict()

        print(
            f"[{self.dataset_label}] Found {len(self.records)} sequences, "
            f"geometry=local_shared_static_map, context=±{self.map_context}, "
            f"splat_radius={self.splat_radius}, remove_boxes={self.remove_annotated_objects}",
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

    def _static_scan(self, token):
        if token in self._scan_cache:
            points = self._scan_cache.pop(token)
            self._scan_cache[token] = points
            return points

        points = _load_static_lidar_global(
            self.nusc, token, min_lidar_distance=self.min_lidar_distance,
            remove_annotated_objects=self.remove_annotated_objects, box_scale=self.box_scale,
        )
        self._scan_cache[token] = points
        while len(self._scan_cache) > self.scan_cache_size:
            self._scan_cache.popitem(last=False)
        return points

    def _local_map(self, record, positions):
        tokens = record["sample_tokens"]
        lo = max(0, min(positions) - self.map_context)
        hi = min(len(tokens), max(positions) + self.map_context + 1)

        # Always use every keyframe in the covered interval. `frame_step` only
        # controls supervision-view sampling; skipped frames such as 11/13/15
        # are still valuable LiDAR observations for the shared map.
        map_tokens = tokens[lo:hi]

        point_sets = [self._static_scan(token) for token in map_tokens]
        point_sets = [points for points in point_sets if len(points)]
        if not point_sets:
            return np.empty((0, 3), dtype=np.float32), (lo, hi, len(map_tokens))

        points = _voxel_downsample(np.concatenate(point_sets, axis=0), self.voxel_size).astype(np.float32)
        return points, (lo, hi, len(map_tokens))

    def _get_views(self, index, resolution, rng, is_test=False):
        is_test = bool(is_test or self.mode == "test")
        record = self.records[index]
        positions = self._sample_positions(len(record["sample_tokens"]), rng, is_test)
        if positions is None:
            self.this_views_info = {"scene": record["sequence_id"], "idxs": []}
            required = (self.frame_num - 1) * self.frame_step + 1
            raise ValueError(
                f"NuScenes sequence {record['sequence_id']} has "
                f"{len(record['sample_tokens'])} frames, but requires {required}"
            )

        selected_tokens = [record["sample_tokens"][pos] for pos in positions]
        self.this_views_info = {
            "scene": record["sequence_id"], "camera": record["camera"],
            "idxs": positions, "geometry": "local_shared_static_map",
        }

        shared_map, map_info = self._local_map(record, positions)
        if self.verbose:
            lo, hi, n_scans = map_info
            print(
                f"[nuScenes] {record['scene_name']} map_frames=[{lo},{hi}) "
                f"scans={n_scans} shared_map_points={len(shared_map)}",
                flush=True,
            )

        views = []
        for pos, token in zip(positions, selected_tokens):
            image, depth, K, pose, image_path = _render_shared_map_to_camera(
                self.nusc, token, record["camera"], shared_map,
                splat_radius=self.splat_radius, min_camera_depth=self.min_camera_depth,
            )
            if self.verbose:
                print(f"[nuScenes] frame={pos} depth_valid={np.count_nonzero(depth > 0) / depth.size:.2%}", flush=True)

            image, depth, K = self._crop_resize_if_necessary(
                image, depth, K, resolution, rng=rng, info=image_path
            )[:3]

            views.append({
                "img": image, "depthmap": np.asarray(depth, dtype=np.float32),
                "camera_pose": np.asarray(pose, dtype=np.float32),
                "camera_intrinsics": np.asarray(K, dtype=np.float32),
                "dataset": self.dataset_label, "sequence": record["sequence_id"],
                "label": record["sequence_id"], "instance": token,
                "prefix": f"{record['sequence_id']}_{pos:03d}",
                "image_path": image_path, "sample_token": token,
                "camera_name": record["camera"],
                "depth_source": "nuscenes_local_shared_static_keyframe_lidar",
                "depth_definition": "camera_z_m", "pseudo_label": False,
                "valid_mask_required": True,
            })

        T0_inv = np.linalg.inv(views[0]["camera_pose"].astype(np.float64))
        for view in views:
            view["camera_pose"] = (T0_inv @ view["camera_pose"].astype(np.float64)).astype(np.float32)
        return views


def main():
    data_root = Path("/starmap/nas184/open_source/nuScenes/v1.0-mini")
    output_path = Path("/starmap/nas/workspace/pxx/Pi3/data/dataset_cache/nuscenes.npy")
    index = generate_nuscenes_index(data_root, output_path, version="v1.0-mini", cameras=CAMERAS)
    print(f"Saved {len(index['sequences'])} scene-camera sequences to {output_path}", flush=True)


if __name__ == "__main__":
    main()
