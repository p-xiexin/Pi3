"""Lazy exports for the local Glob3R nonlinear least-squares backend."""

from importlib import import_module


_EXPORTS = {
    "Eq5Result": (".ba", "Eq5Result"),
    "Eq6Result": (".ba", "Eq6Result"),
    "bundle_adjust": (".ba", "bundle_adjust"),
    "opt_pose_ray": (".ba", "opt_pose_ray"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
