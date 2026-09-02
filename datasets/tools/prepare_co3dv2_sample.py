"""Convert an official CO3Dv2 category subset to this repository's legacy index."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def positive_depth(path: Path) -> bool:
    encoded = np.asarray(Image.open(path), dtype=np.uint16)
    depth = encoded.view(np.float16)
    return bool(np.any(np.isfinite(depth) & (depth > 0)))


def flatten(annotation: dict) -> dict:
    viewpoint = annotation["viewpoint"]
    return {
        "filepath": annotation["image"]["path"],
        "R": viewpoint["R"],
        "T": viewpoint["T"],
        "focal_length": viewpoint["focal_length"],
        "principal_point": viewpoint["principal_point"],
        "depth_scale_adjustment": annotation["depth"].get("scale_adjustment", 1.0),
        "mask_path": annotation["mask"]["path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("category")
    parser.add_argument("--sequence")
    parser.add_argument("--frames-per-split", type=int, default=16)
    args = parser.parse_args()

    category_root = args.data_root / args.category
    with gzip.open(category_root / "frame_annotations.jgz", "rt", encoding="utf-8") as handle:
        annotations = json.load(handle)

    sequences: dict[str, list[dict]] = defaultdict(list)
    for annotation in annotations:
        sequences[annotation["sequence_name"]].append(annotation)
    sequence = args.sequence or sorted(sequences)[0]
    if sequence not in sequences:
        raise KeyError(f"Sequence {sequence} not found in {args.category}")

    splits = {"train": [], "test": []}
    for annotation in sorted(sequences[sequence], key=lambda item: item["frame_number"]):
        frame_type = annotation.get("meta", {}).get("frame_type", "")
        split = "train" if frame_type.endswith("known") else "test"
        if len(splits[split]) >= args.frames_per_split:
            continue
        depth_path = args.data_root / annotation["depth"]["path"]
        if positive_depth(depth_path):
            splits[split].append(flatten(annotation))

    for split, frames in splits.items():
        if len(frames) < args.frames_per_split:
            raise RuntimeError(
                f"Only {len(frames)} valid {split} frames in {args.category}/{sequence}"
            )
        output = args.data_root / f"{args.category}_{split}.jgz"
        with gzip.open(output, "wt", encoding="utf-8") as handle:
            json.dump({sequence: frames}, handle)
        print(f"Saved {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
