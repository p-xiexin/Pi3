"""Small YAML loader for the full-sequence pipeline."""

from pathlib import Path

import yaml


def load_config(path):
    """Load and validate the complete YAML contract for one SfM run."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("configuration root must be a YAML mapping")
    required = {
        "images", "output", "backbone_checkpoint", "tracks_model", "device",
        "image_size", "window_size", "points_per_keyframe", "local_iterations",
        "global_iterations",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise KeyError(f"missing configuration keys: {missing}")
    height, width = map(int, config["image_size"])
    if height % 14 or width % 14:
        raise ValueError("image_size must be divisible by Pi3 patch size 14")
    if int(config["window_size"]) < 3:
        raise ValueError("window_size must be at least three for three-view tracks")
    if config["tracks_model"] not in {"glob3r", "vgg"}:
        raise ValueError("tracks_model must be glob3r or vgg")
    model_keys = {
        "glob3r": ("matching_checkpoint",),
        "vgg": ("vggsfm_root", "vggsfm_checkpoint"),
    }[config["tracks_model"]]
    missing = [key for key in model_keys if not config.get(key)]
    if missing:
        raise KeyError(f"missing {config['tracks_model']} configuration keys: {missing}")
    for key in ("points_per_keyframe", "local_iterations", "global_iterations"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    return config


__all__ = ["load_config"]
