"""DROID-style nonlinear least-squares backend for Glob3R."""

from .ba import Eq5Result, Eq6Result, bundle_adjust, opt_pose_ray

__all__ = ["Eq5Result", "Eq6Result", "bundle_adjust", "opt_pose_ray"]
