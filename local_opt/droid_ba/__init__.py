"""Lazy public API for the vendored DROID solvers and local adapters."""

from importlib import import_module


_EXPORTS = {
    "DroidBAResult": (".adapter", "DroidBAResult"),
    "optimize_droid_ba": (".adapter", "optimize_droid_ba"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
