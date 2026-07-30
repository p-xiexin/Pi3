"""Lazy public API for Glob3R single-window association and local optimization."""

from importlib import import_module


_EXPORTS = {
    "Glob3RSfM": (".glob3r_sfm", "Glob3RSfM"),
    "Glob3RSfMConfig": (".sfm", "Glob3RSfMConfig"),
    "Glob3RSfMPipeline": (".sfm", "Glob3RSfMPipeline"),
    "SfMResult": (".sfm", "SfMResult"),
    "bundle_adjust": (".optimization", "bundle_adjust"),
    "load_glob3r_for_sfm": (".inference", "load_glob3r_for_sfm"),
    "maximum_spanning_tree_initialization": (
        ".optimization",
        "maximum_spanning_tree_initialization",
    ),
    "robust_rotation_averaging": (".optimization", "robust_rotation_averaging"),
    "save_keyframe_matching_overviews": (
        ".visualization",
        "save_keyframe_matching_overviews",
    ),
    "translation_averaging": (".optimization", "translation_averaging"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
