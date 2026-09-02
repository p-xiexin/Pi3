"""Convert a small official Waymo TFRecord to the processed dataset layout."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
from pathlib import Path
import struct
import sys
import zlib

import cv2
import google_crc32c
import numpy as np


_MASK_DELTA = 0xA282EAD8
_RAW_FROM_OPENCV = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _masked_crc32c(data: bytes) -> int:
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + _MASK_DELTA) & 0xFFFFFFFF


def iter_tfrecord(path: Path):
    """Read complete records and tolerate a deliberately truncated final record."""
    with path.open("rb") as stream:
        record_index = 0
        while True:
            header = stream.read(12)
            if not header:
                return
            if len(header) != 12:
                print(f"Ignoring truncated TFRecord header at record {record_index}")
                return

            length_bytes = header[:8]
            length = struct.unpack("<Q", length_bytes)[0]
            stored_length_crc = struct.unpack("<I", header[8:])[0]
            if stored_length_crc != _masked_crc32c(length_bytes):
                raise ValueError(f"Bad TFRecord length CRC at record {record_index}")

            payload = stream.read(length)
            footer = stream.read(4)
            if len(payload) != length or len(footer) != 4:
                print(f"Ignoring truncated TFRecord payload at record {record_index}")
                return

            stored_payload_crc = struct.unpack("<I", footer)[0]
            if stored_payload_crc != _masked_crc32c(payload):
                raise ValueError(f"Bad TFRecord payload CRC at record {record_index}")

            yield payload
            record_index += 1


def compile_minimal_proto(work_dir: Path):
    proto_root = Path(__file__).resolve().parent / "waymo_proto"
    proto_path = proto_root / "waymo_open_dataset" / "dataset_minimal.proto"
    generated_root = work_dir / "generated_proto"
    generated_path = (
        generated_root / "waymo_open_dataset" / "dataset_minimal_pb2.py"
    )

    if not generated_path.is_file() or generated_path.stat().st_mtime < proto_path.stat().st_mtime:
        from grpc_tools import protoc

        generated_root.mkdir(parents=True, exist_ok=True)
        result = protoc.main(
            [
                "grpc_tools.protoc",
                f"-I{proto_root}",
                f"--python_out={generated_root}",
                str(proto_path),
            ]
        )
        if result != 0:
            raise RuntimeError(f"protoc failed with exit code {result}")

    spec = importlib.util.spec_from_file_location(
        "waymo_dataset_minimal_pb2",
        generated_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def matrix(transform) -> np.ndarray:
    values = np.asarray(transform.transform, dtype=np.float64)
    if values.size != 16:
        raise ValueError(f"Expected a 4x4 transform, got {values.size} values")
    return values.reshape(4, 4)


def parse_matrix_float(blob: bytes, proto_module) -> np.ndarray:
    message = proto_module.MatrixFloat()
    message.ParseFromString(zlib.decompress(blob))
    shape = tuple(message.shape.dims)
    values = np.asarray(message.data, dtype=np.float32)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"MatrixFloat shape {shape} has {values.size} values")
    return values.reshape(shape)


def rotation_matrix(roll, pitch, yaw):
    """Waymo 3-2-1 Euler rotation, vectorized over leading dimensions."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    out = np.empty(np.shape(roll) + (3, 3), dtype=np.float64)
    out[..., 0, 0] = cy * cp
    out[..., 0, 1] = cy * sp * sr - sy * cr
    out[..., 0, 2] = cy * sp * cr + sy * sr
    out[..., 1, 0] = sy * cp
    out[..., 1, 1] = sy * sp * sr + cy * cr
    out[..., 1, 2] = sy * sp * cr - cy * sr
    out[..., 2, 0] = -sp
    out[..., 2, 1] = cp * sr
    out[..., 2, 2] = cp * cr
    return out


def inclinations(calibration, height: int) -> np.ndarray:
    if calibration.beam_inclinations:
        values = np.asarray(calibration.beam_inclinations, dtype=np.float64)
    else:
        ratio = (np.arange(height, dtype=np.float64) + 0.5) / height
        values = (
            calibration.beam_inclination_min
            + ratio
            * (
                calibration.beam_inclination_max
                - calibration.beam_inclination_min
            )
        )
    if values.size != height:
        raise ValueError(f"Expected {height} beam inclinations, got {values.size}")
    return values[::-1]


def range_image_to_vehicle(
    range_image: np.ndarray,
    calibration,
    pixel_pose: np.ndarray | None,
    frame_pose: np.ndarray,
) -> np.ndarray:
    height, width = range_image.shape[:2]
    ranges = range_image[..., 0].astype(np.float64)
    extrinsic = matrix(calibration.extrinsic)

    ratios = (np.arange(width, 0, -1, dtype=np.float64) - 0.5) / width
    azimuth = (
        (ratios * 2.0 - 1.0) * np.pi
        - np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    )
    inclination = inclinations(calibration, height)

    cos_inclination = np.cos(inclination)[:, None]
    points_lidar = np.stack(
        [
            np.cos(azimuth)[None, :] * cos_inclination * ranges,
            np.sin(azimuth)[None, :] * cos_inclination * ranges,
            np.sin(inclination)[:, None] * np.ones((1, width)) * ranges,
        ],
        axis=-1,
    )
    points_vehicle = (
        points_lidar @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    )

    if pixel_pose is not None:
        rotations = rotation_matrix(
            pixel_pose[..., 0],
            pixel_pose[..., 1],
            pixel_pose[..., 2],
        )
        points_world = (
            np.einsum("...ij,...j->...i", rotations, points_vehicle)
            + pixel_pose[..., 3:6]
        )
        world_to_vehicle = np.linalg.inv(frame_pose)
        points_vehicle = (
            points_world @ world_to_vehicle[:3, :3].T
            + world_to_vehicle[:3, 3]
        )

    return points_vehicle


def frame_points_vehicle(frame, proto_module) -> np.ndarray:
    calibrations = {item.name: item for item in frame.context.laser_calibrations}
    frame_pose = matrix(frame.pose)
    top_pose = None
    points = []

    for laser in frame.lasers:
        calibration = calibrations[laser.name]
        for return_index, range_return in enumerate(
            (laser.ri_return1, laser.ri_return2)
        ):
            if not range_return.range_image_compressed:
                continue
            range_image = parse_matrix_float(
                range_return.range_image_compressed,
                proto_module,
            )
            if laser.name == proto_module.LaserName.TOP:
                if return_index == 0 and range_return.range_image_pose_compressed:
                    top_pose = parse_matrix_float(
                        range_return.range_image_pose_compressed,
                        proto_module,
                    )
                pixel_pose = top_pose
            else:
                pixel_pose = None

            points_vehicle = range_image_to_vehicle(
                range_image,
                calibration,
                pixel_pose,
                frame_pose,
            )
            mask = range_image[..., 0] > 0
            points.append(points_vehicle[mask])

    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(points, axis=0)


def camera_matrices(calibration):
    intrinsic = np.asarray(calibration.intrinsic, dtype=np.float64)
    if intrinsic.size != 9:
        raise ValueError(f"Expected 9 camera intrinsic values, got {intrinsic.size}")
    fu, fv, cu, cv, k1, k2, p1, p2, k3 = intrinsic
    K = np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]])
    distortion = np.array([k1, k2, p1, p2, k3])
    vehicle_from_camera = matrix(calibration.extrinsic) @ _RAW_FROM_OPENCV
    return K, distortion, vehicle_from_camera


def z_buffer_depth(points_camera, K, height, width):
    z = points_camera[:, 2]
    valid = np.isfinite(points_camera).all(axis=1) & (z > 1e-4)
    points_camera = points_camera[valid]
    z = points_camera[:, 2]

    u = np.rint(K[0, 0] * points_camera[:, 0] / z + K[0, 2]).astype(np.int32)
    v = np.rint(K[1, 1] * points_camera[:, 1] / z + K[1, 2]).astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[valid], v[valid], z[valid].astype(np.float32)

    depth = np.zeros((height, width), dtype=np.float32)
    order = np.argsort(z)
    flat = v[order] * width + u[order]
    _, first = np.unique(flat, return_index=True)
    keep = order[first]
    depth[v[keep], u[keep]] = z[keep]
    return depth


def convert_frame(frame, proto_module, official_camera_id: int):
    images = {image.name: image for image in frame.images}
    calibrations = {
        calibration.name: calibration
        for calibration in frame.context.camera_calibrations
    }
    image_proto = images[official_camera_id]
    calibration = calibrations[official_camera_id]

    encoded = np.frombuffer(image_proto.image, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("Failed to decode Waymo camera JPEG")

    K, distortion, vehicle_from_camera = camera_matrices(calibration)
    image_bgr = cv2.undistort(image_bgr, K, distortion, None, K)

    points_vehicle = frame_points_vehicle(frame, proto_module)
    world_from_frame_vehicle = matrix(frame.pose)
    world_from_image_vehicle = matrix(image_proto.pose)
    world_from_camera = world_from_image_vehicle @ vehicle_from_camera

    points_world = (
        points_vehicle @ world_from_frame_vehicle[:3, :3].T
        + world_from_frame_vehicle[:3, 3]
    )
    camera_from_world = np.linalg.inv(world_from_camera)
    points_camera = (
        points_world @ camera_from_world[:3, :3].T
        + camera_from_world[:3, 3]
    )
    depth = z_buffer_depth(
        points_camera,
        K,
        image_bgr.shape[0],
        image_bgr.shape[1],
    )
    return image_bgr, depth, K, world_from_image_vehicle, vehicle_from_camera


def save_frame(segment_dir: Path, frame_id: int, camera_id: int, values):
    image_bgr, depth, K, world_from_vehicle, vehicle_from_camera = values
    key = f"{frame_id:06d}_{camera_id}"
    if not cv2.imwrite(str(segment_dir / "images" / f"{key}.png"), image_bgr):
        raise RuntimeError(f"Failed to write image {key}")
    np.save(segment_dir / "lidars" / f"{key}.npy", depth)
    np.savetxt(segment_dir / "intrinsics" / f"{key}.txt", K, fmt="%.12g")
    np.savetxt(
        segment_dir / "extrinsics" / f"{key}.txt",
        vehicle_from_camera,
        fmt="%.12g",
    )
    np.savetxt(
        segment_dir / "ego_pose" / f"{key}.txt",
        world_from_vehicle,
        fmt="%.12g",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a small Waymo TFRecord sample for WaymoPi3XDataset"
    )
    parser.add_argument("tfrecord", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--official-camera-id", type=int, default=1)
    parser.add_argument("--output-camera-id", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/waymo_sample"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_frames < 1:
        raise ValueError("--max-frames must be positive")

    proto_module = compile_minimal_proto(args.work_dir)
    frames = iter_tfrecord(args.tfrecord)
    first_payload = next(frames, None)
    if first_payload is None:
        raise RuntimeError("No complete frame found in TFRecord")

    first_frame = proto_module.Frame()
    first_frame.ParseFromString(first_payload)
    sequence_id = first_frame.context.name
    if not sequence_id:
        raise ValueError("Waymo frame has an empty context name")

    segment_dir = args.output_root / sequence_id
    for name in ("images", "lidars", "intrinsics", "extrinsics", "ego_pose"):
        (segment_dir / name).mkdir(parents=True, exist_ok=True)

    count = 0
    for payload in itertools.chain([first_payload], frames):
        if count >= args.max_frames:
            break
        frame = proto_module.Frame()
        frame.ParseFromString(payload)
        if frame.context.name != sequence_id:
            raise ValueError("TFRecord contains multiple context names")
        values = convert_frame(frame, proto_module, args.official_camera_id)
        save_frame(segment_dir, count, args.output_camera_id, values)
        print(
            f"frame={count:03d} timestamp={frame.timestamp_micros} "
            f"valid_depth={np.count_nonzero(values[1])}",
            flush=True,
        )
        count += 1

    if count == 0:
        raise RuntimeError("No frames were converted")
    print(f"Converted {count} frames to {segment_dir.resolve()}")


if __name__ == "__main__":
    main()
