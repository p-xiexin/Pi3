"""Lazy public API for Glob3R.

Keeping imports lazy lets the standalone matching-head smoke test import
``glob3r.model`` without loading the complete training model.
"""

from importlib import import_module


_EXPORTS = {
    "GeometryPrediction": (".model", "GeometryPrediction"),
    "Glob3RMatchingHead": (".model", "Glob3RMatchingHead"),
    "MatchingOutput": (".model", "MatchingOutput"),
    "load_romav2_refinement": (".model", "load_romav2_refinement"),
    "Glob3RMatchingLoss": (".loss", "Glob3RMatchingLoss"),
    "PoseRayGeometryLoss": (".loss", "PoseRayGeometryLoss"),
    "Glob3R": (".glob3r_training", "Glob3R"),
    "WarpSupervision": (".geometry", "WarpSupervision"),
    "build_ground_truth_warp": (".geometry", "build_ground_truth_warp"),
    "bundle_adjustment_objective": (".geometry", "bundle_adjustment_objective"),
    "keyframe_projection_count": (".geometry", "keyframe_projection_count"),
    "motion_averaging_objective": (".geometry", "motion_averaging_objective"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
