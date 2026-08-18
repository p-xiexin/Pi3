"""Generate a list of ScanNet frames whose camera poses are invalid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def generate_invalid_list(data_root: Path) -> dict[str, list[int]]:
    invalid_list: dict[str, list[int]] = {}

    for scene_path in tqdm(sorted(data_root.iterdir()), desc="Scanning scenes"):
        if not scene_path.is_dir():
            continue

        pose_dir = scene_path / "pose"
        if not pose_dir.is_dir():
            continue

        invalid_frames: list[int] = []
        for pose_file in sorted(pose_dir.glob("*.txt")):
            try:
                frame_id = int(pose_file.stem)
            except ValueError:
                continue

            try:
                camera_pose = np.loadtxt(pose_file, dtype=np.float32).reshape(4, 4)
            except (OSError, ValueError, IndexError):
                invalid_frames.append(frame_id)
                continue

            if not np.isfinite(camera_pose).all():
                invalid_frames.append(frame_id)

        if invalid_frames:
            invalid_list[scene_path.name] = invalid_frames

    return invalid_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_root",
        type=Path,
        help="ScanNet root containing sceneXXXX_YY/pose directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/scannet_invalid_list.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"ScanNet data root does not exist: {data_root}")

    invalid_list = generate_invalid_list(data_root)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(invalid_list, file, indent=4, sort_keys=True)
        file.write("\n")

    invalid_frame_count = sum(len(frames) for frames in invalid_list.values())
    print(f"Generated invalid list at {output_path}")
    print(
        f"Found {invalid_frame_count} invalid poses "
        f"across {len(invalid_list)} scenes."
    )


if __name__ == "__main__":
    main()
