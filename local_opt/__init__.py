"""Lazy public API for Glob3R single-window association and local optimization."""

from importlib import import_module


_EXPORTS = {
    "Frame": (".frame", "Frame"),
    "Frames": (".frame", "Frames"),
    "Glob3RSfM": (".glob3r_sfm", "Glob3RSfM"),
    "Glob3RSfMConfig": (".sfm", "Glob3RSfMConfig"),
    "Glob3RSfMPipeline": (".sfm", "Glob3RSfMPipeline"),
    "Tracks": (".matching", "Tracks"),
    "SfMResult": (".sfm", "SfMResult"),
    "Eq5Result": (".backend", "Eq5Result"),
    "Eq6Result": (".backend", "Eq6Result"),
    "bundle_adjust": (".backend", "bundle_adjust"),
    "load_glob3r_for_sfm": (".inference", "load_glob3r_for_sfm"),
    "match_tracks": (".matching", "match_tracks"),
    "opt_pose_ray": (".backend", "opt_pose_ray"),
    "PoseGraph": (".pose_graph", "PoseGraph"),
    "build_pose_graph": (".pose_graph", "build_pose_graph"),
    "maximum_spanning_tree_initialization": (
        ".pose_graph",
        "maximum_spanning_tree_initialization",
    ),
    "robust_rotation_averaging": (
        ".pose_graph",
        "robust_rotation_averaging",
    ),
    "save_matching_matrix": (
        ".visualization",
        "save_matching_matrix",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
