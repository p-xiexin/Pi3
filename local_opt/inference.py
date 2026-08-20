"""Checkpoint loading and image I/O for the standalone Glob3R SfM pipeline."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from pi3.models.pi3 import Pi3

from .glob3r_sfm import Glob3RSfM
from .image_utils import crop_resize


def _natural_sort_key(path: Path):
    return [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
    ]


def _checkpoint_file(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = [
            path / "model.safetensors",
            path / "pytorch_model.bin",
            path / "pytorch_model_1.bin",
        ]
        existing = [candidate for candidate in candidates if candidate.is_file()]
        if len(existing) == 1:
            return existing[0]
        if not existing:
            raise FileNotFoundError(f"no model checkpoint found in {path}")
        raise RuntimeError(f"ambiguous checkpoint directory {path}: {existing}")
    raise FileNotFoundError(path)


def _read_state_dict(path: str | Path) -> Mapping[str, torch.Tensor]:
    path = _checkpoint_file(path)
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, Mapping) and isinstance(state.get(key), Mapping):
            state = state[key]
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint {path} does not contain a state dict")
    return state


def _strip_prefixes(state: Mapping[str, torch.Tensor], prefixes) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        normalized[key] = value
    return normalized


def load_glob3r_for_sfm(
    backbone_checkpoint: str | Path,
    matching_checkpoint: str | Path,
    device: str | torch.device = "cuda",
) -> Glob3RSfM:
    """Load Eq. (1) Pi3 geometry and the trained Eq. (2) matching/refinement head."""

    backbone = Pi3(pos_type="rope100", decoder_size="large")
    backbone_state = _strip_prefixes(
        _read_state_dict(backbone_checkpoint), ("module.", "model.", "backbone.")
    )
    result = backbone.load_state_dict(backbone_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"incomplete Pi3 inference checkpoint: {result}")

    model = Glob3RSfM(
        backbone,
        encoder_layers=(5, 11, 17, 23),
        enable_refinement=True,
        matching_checkpoint=None,
    )
    matching_state = _read_state_dict(matching_checkpoint)
    prefix = "glob3r_matching_head."
    matching_state = {
        key.split(prefix, 1)[1]: value
        for key, value in matching_state.items()
        if prefix in key
    } or dict(matching_state)
    model.glob3r_matching_head.load_state_dict(matching_state, strict=True)
    return model.to(device).eval()


def load_image_sequence(
    directory: str | Path,
    size: tuple[int, int],
) -> tuple[torch.Tensor, list[Path]]:
    """Load an ordered image directory as ``[N,3,H,W]`` in the [0,1] range."""

    directory = Path(directory)
    paths = sorted(
        (
            path
            for path in directory.iterdir()
            if path.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ),
        key=_natural_sort_key,
    )
    if len(paths) < 2:
        raise RuntimeError(f"{directory} must contain at least two images")
    height, width = size
    Is = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image, _, _ = crop_resize(image, np.eye(3), size)
            array = np.asarray(image, dtype=np.float32) / 255.0
        Is.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(Is), paths


def load_calibration(
    path: str | Path,
) -> tuple[torch.Tensor, int, int]:
    """Load camera intrinsics and their native image size from YAML."""

    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    fx, fy, cx, cy = map(float, config["calibration"][:4])
    K = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return K, int(config["width"]), int(config["height"])


__all__ = [
    "Glob3RSfM",
    "load_calibration",
    "load_glob3r_for_sfm",
    "load_image_sequence",
]
