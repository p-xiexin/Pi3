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
        "images", "data_h5", "backbone_checkpoint", "tracks_model", "device",
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
    config["superpoint_detection_threshold"] = float(
        config.get("superpoint_detection_threshold", 0.005)
    )
    config["vgg_track_visibility_threshold"] = float(
        config.get("vgg_track_visibility_threshold", 0.05)
    )
    config["vgg_track_score_threshold"] = float(
        config.get("vgg_track_score_threshold", 0.5)
    )
    config["pi3_mask_confidence_threshold"] = float(
        config.get("pi3_mask_confidence_threshold", 0.3)
    )
    config["keyframe_projection_threshold"] = float(
        config.get("keyframe_projection_threshold", 0.7)
    )
    config["keyframe_confidence_threshold"] = float(
        config.get("keyframe_confidence_threshold", 0.1)
    )
    config["keyframe_max_interval"] = int(
        config.get("keyframe_max_interval", 5)
    )
    config["random_seed"] = int(config.get("random_seed", 0))
    for key in (
        "vgg_track_visibility_threshold",
        "vgg_track_score_threshold",
        "pi3_mask_confidence_threshold",
        "keyframe_projection_threshold",
        "keyframe_confidence_threshold",
    ):
        if not 0.0 <= config[key] <= 1.0:
            raise ValueError(f"{key} must be within [0, 1]")
    if config["keyframe_max_interval"] < 1:
        raise ValueError("keyframe_max_interval must be positive")
    if config["random_seed"] < 0:
        raise ValueError("random_seed must be nonnegative")
    if not 0.0 <= config["superpoint_detection_threshold"] <= 1.0:
        raise ValueError("superpoint_detection_threshold must be within [0, 1]")
    config["loop"] = bool(config.get("loop", True))
    config["data_loop_h5"] = config.get(
        "data_loop_h5", str(Path(config["data_h5"]).with_name("data_loop.h5"))
    )
    if Path(config["data_h5"]).resolve() == Path(config["data_loop_h5"]).resolve():
        raise ValueError("data_h5 and data_loop_h5 must be different files")
    config["loop_similarity_threshold"] = float(
        config.get("loop_similarity_threshold", 0.5)
    )
    config["loop_batch_size"] = int(config.get("loop_batch_size", 16))
    if not -1.0 <= config["loop_similarity_threshold"] <= 1.0:
        raise ValueError("loop_similarity_threshold must be within [-1, 1]")
    if config["loop_batch_size"] < 1:
        raise ValueError("loop_batch_size must be positive")
    if config["loop"] and not config.get("dino_salad_checkpoint"):
        raise KeyError("dino_salad_checkpoint is required when loop detection is enabled")
    config["global_ba_backend"] = config.get("global_ba_backend", "native")
    if config["global_ba_backend"] not in {"native", "colmap"}:
        raise ValueError("global_ba_backend must be native or colmap")
    model_keys = {
        "glob3r": ("matching_checkpoint",),
        "vgg": ("vggsfm_root", "vggsfm_checkpoint"),
    }[config["tracks_model"]]
    missing = [key for key in model_keys if not config.get(key)]
    if missing:
        raise KeyError(f"missing {config['tracks_model']} configuration keys: {missing}")
    for key in (
        "points_per_keyframe",
        "local_iterations",
        "global_iterations",
    ):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    return config


__all__ = ["load_config"]
