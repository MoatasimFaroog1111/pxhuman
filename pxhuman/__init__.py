"""
pxhuman -- parametric synthetic human body generator.

Bodies are generated entirely from numeric anthropometric measurements.
Nothing in this package derives shape, texture or identity from a photograph
or from any real individual.
"""
from .anthro import BodyParams
from .sdf import BodyField, build_joints, HIERARCHY
from .mesh import build_mesh, verify, measure_circumference
from .rig import skin_weights, apply_pose, POSE_PRESETS
from . import export

__version__ = "1.0.0"
__all__ = [
    "BodyParams", "BodyField", "build_joints", "HIERARCHY",
    "build_mesh", "verify", "measure_circumference",
    "skin_weights", "apply_pose", "POSE_PRESETS", "export",
]
