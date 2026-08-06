"""Lazy public API for Glob3R single-window association and local optimization."""

from importlib import import_module


_EXPORTS = {
    "Frame": (".frame", "Frame"),
    "Frames": (".frame", "Frames"),
    "DroidFactorGraph": (".factor_graph", "DroidFactorGraph"),
    "DepthRatioFilterConfig": (".filter", "DepthRatioFilterConfig"),
    "DepthRatioFilterResult": (".filter", "DepthRatioFilterResult"),
    "Glob3RSfM": (".glob3r_sfm", "Glob3RSfM"),
    "Glob3RSfMConfig": (".sfm", "Glob3RSfMConfig"),
    "Glob3RSfMPipeline": (".sfm", "Glob3RSfMPipeline"),
    "PairMatch": (".matching", "PairMatch"),
    "SfMResult": (".sfm", "SfMResult"),
    "DroidBAResult": (".droid_ba.adapter", "DroidBAResult"),
    "optimize_droid_ba": (".droid_ba.adapter", "optimize_droid_ba"),
    "load_glob3r_for_sfm": (".inference", "load_glob3r_for_sfm"),
    "match_batch": (".matching", "match_batch"),
    "build_droid_factor_graph": (
        ".factor_graph",
        "build_droid_factor_graph",
    ),
    "filter_depth_ratios": (".filter", "filter_depth_ratios"),
    "source_support_from_factors": (
        ".filter",
        "source_support_from_factors",
    ),
    "save_keyframe_matching_overviews": (
        ".visualization",
        "save_keyframe_matching_overviews",
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
