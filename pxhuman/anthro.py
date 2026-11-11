"""
Anthropometric parameter model.

Everything the body mesh needs is derived from a small set of real-world
measurements (centimetres). Segment lengths that the user does not supply are
derived from stature using the Drillis & Contini segment-ratio table, which is
the standard public-domain anthropometric reference used in biomechanics.

No part of this module is derived from any photograph or any real individual.
Shape is driven only by numeric measurements.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Drillis & Contini segment ratios, expressed as a fraction of stature H.
# ---------------------------------------------------------------------------
RATIO = {
    "head_height":      0.130,   # crown -> chin
    "acromion_h":       0.818,   # shoulder joint height
    "chest_h":          0.720,   # chest / bust level
    "elbow_h":          0.630,
    "waist_h":          0.620,   # natural waist (narrowest)
    "wrist_h":          0.485,
    "hip_h":            0.530,   # greater trochanter
    "crotch_h":         0.485,
    "knee_h":           0.285,
    "ankle_h":          0.039,
    "biacromial_w":     0.259,   # shoulder breadth
    "bihip_w":          0.191,
    "foot_length":      0.152,
    "foot_breadth":     0.055,
    "hand_length":      0.108,
    "hand_breadth":     0.044,
}

# Two neutral proportion templates. These are statistical body-shape
# templates, not depictions of anyone. `proportion` blends between them.
TEMPLATE_A = {   # broader shoulder girdle, narrower pelvis
    "chest_over_stature":    0.571,
    "waist_over_stature":    0.486,
    "hip_over_stature":      0.560,
    "neck_over_stature":     0.217,
    "shoulder_scale":        1.000,
    "hip_width_scale":       1.000,
    "waist_level":           0.600,
    "chest_depth_ratio":     0.72,   # depth / width at chest
    "waist_depth_ratio":     0.74,
    "hip_depth_ratio":       0.78,
    "limb_girth_scale":      1.000,
    "bust_projection":       0.00,
}
TEMPLATE_B = {   # narrower shoulder girdle, wider pelvis
    "chest_over_stature":    0.558,
    "waist_over_stature":    0.448,
    "hip_over_stature":      0.600,
    "neck_over_stature":     0.200,
    "shoulder_scale":        0.915,
    "hip_width_scale":       1.065,
    "waist_level":           0.625,
    "chest_depth_ratio":     0.78,
    "waist_depth_ratio":     0.76,
    "hip_depth_ratio":       0.74,
    "limb_girth_scale":      0.930,
    "bust_projection":       0.055,
}


def _blend(a: Dict[str, float], b: Dict[str, float], t: float) -> Dict[str, float]:
    return {k: a[k] * (1.0 - t) + b[k] * t for k in a}


@dataclass
class BodyParams:
    """Measurement-driven description of a synthetic body.

    All lengths and circumferences in centimetres. Any circumference left as
    None is derived from `stature` via the blended proportion template.
    """

    stature: float = 172.0          # total standing height
    proportion: float = 0.0         # 0..1 blend between proportion templates
    muscle: float = 0.5             # 0..1 limb and torso girth distribution
    bodyfat: float = 0.5            # 0..1 soft-tissue smoothing / girth bias
    definition: float = 0.0         # 0..1 anatomical surface relief amplitude
    posture: float = 0.0            # -1 .. 1  (lordotic .. kyphotic spine curve)

    chest: float | None = None      # chest / bust circumference
    waist: float | None = None      # natural waist circumference
    hip: float | None = None        # hip circumference at widest
    neck: float | None = None       # neck circumference
    shoulder_width: float | None = None   # biacromial breadth
    inseam: float | None = None     # crotch height
    arm_length: float | None = None # acromion -> wrist
    thigh_girth: float | None = None
    calf_girth: float | None = None
    upperarm_girth: float | None = None
    forearm_girth: float | None = None
    head_height: float | None = None

    # A-pose arm abduction from vertical, in degrees (0 = arms down, 90 = T-pose)
    arm_abduction: float = 45.0
    leg_stance: float = 1.5         # outward foot splay from the hip, degrees

    # resolved values, filled by resolve()
    derived: Dict[str, float] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    def resolve(self) -> "BodyParams":
        H = self.stature
        tpl = _blend(TEMPLATE_A, TEMPLATE_B, _clamp01(self.proportion))

        # girth bias from muscle / bodyfat, centred so 0.5 is neutral
        gb = 1.0 + 0.16 * (self.muscle - 0.5) + 0.22 * (self.bodyfat - 0.5)
        limb_gb = 1.0 + 0.26 * (self.muscle - 0.5) + 0.16 * (self.bodyfat - 0.5)

        d: Dict[str, float] = {}

        # ---- circumferences -------------------------------------------
        d["chest"] = self.chest if self.chest else H * tpl["chest_over_stature"] * gb
        d["waist"] = self.waist if self.waist else H * tpl["waist_over_stature"] * gb
        d["hip"] = self.hip if self.hip else H * tpl["hip_over_stature"] * gb
        d["neck"] = self.neck if self.neck else H * tpl["neck_over_stature"] * (1 + 0.1 * (gb - 1))

        # ---- key heights ----------------------------------------------
        d["head_height"] = self.head_height or H * RATIO["head_height"]
        d["acromion_h"] = H * RATIO["acromion_h"]
        d["chest_h"] = H * RATIO["chest_h"]
        d["waist_h"] = H * tpl["waist_level"] * (RATIO["waist_h"] / 0.620)
        d["hip_h"] = H * RATIO["hip_h"]
        d["crotch_h"] = self.inseam if self.inseam else H * RATIO["crotch_h"]
        d["knee_h"] = H * RATIO["knee_h"]
        d["ankle_h"] = H * RATIO["ankle_h"]
        d["elbow_h"] = H * RATIO["elbow_h"]
        d["wrist_h"] = H * RATIO["wrist_h"]
        d["chin_h"] = H - d["head_height"]
        d["neck_base_h"] = d["acromion_h"] + 0.022 * H

        # ---- breadths --------------------------------------------------
        d["shoulder_width"] = (
            self.shoulder_width
            if self.shoulder_width
            else H * RATIO["biacromial_w"] * tpl["shoulder_scale"]
        )
        d["hip_width"] = H * RATIO["bihip_w"] * tpl["hip_width_scale"] * gb
        d["foot_length"] = H * RATIO["foot_length"]
        d["foot_breadth"] = H * RATIO["foot_breadth"]
        d["hand_length"] = H * RATIO["hand_length"]
        d["hand_breadth"] = H * RATIO["hand_breadth"]

        # ---- segment lengths -------------------------------------------
        if self.arm_length:
            total = self.arm_length
            ua = total * (RATIO["acromion_h"] - RATIO["elbow_h"]) / (
                RATIO["acromion_h"] - RATIO["wrist_h"])
            d["upperarm_len"] = ua
            d["forearm_len"] = total - ua
        else:
            d["upperarm_len"] = H * (RATIO["acromion_h"] - RATIO["elbow_h"])
            d["forearm_len"] = H * (RATIO["elbow_h"] - RATIO["wrist_h"])
        d["arm_length"] = d["upperarm_len"] + d["forearm_len"]
        d["thigh_len"] = d["crotch_h"] - d["knee_h"] + 0.04 * H
        d["shank_len"] = d["knee_h"] - d["ankle_h"]

        # ---- limb girths ------------------------------------------------
        lg = tpl["limb_girth_scale"] * limb_gb
        d["thigh_girth"] = self.thigh_girth or d["hip"] * 0.585 * lg
        d["knee_girth"] = d["thigh_girth"] * 0.68
        d["calf_girth"] = self.calf_girth or d["thigh_girth"] * 0.665 * lg
        d["ankle_girth"] = d["calf_girth"] * 0.615
        d["upperarm_girth"] = self.upperarm_girth or d["chest"] * 0.315 * lg
        d["elbow_girth"] = d["upperarm_girth"] * 0.88
        d["forearm_girth"] = self.forearm_girth or d["upperarm_girth"] * 0.885 * lg
        d["wrist_girth"] = d["forearm_girth"] * 0.615

        # ---- shape qualifiers --------------------------------------------
        d["chest_depth_ratio"] = tpl["chest_depth_ratio"]
        d["waist_depth_ratio"] = tpl["waist_depth_ratio"]
        d["hip_depth_ratio"] = tpl["hip_depth_ratio"]
        d["bust_projection"] = tpl["bust_projection"] * (1.0 + 0.5 * (self.bodyfat - 0.5))
        d["posture"] = self.posture
        d["arm_abduction"] = self.arm_abduction
        d["leg_stance"] = self.leg_stance
        d["definition"] = self.definition

        # head geometry (generic ovoid; no facial features are generated)
        d["head_width"] = d["head_height"] * 0.72
        d["head_depth"] = d["head_height"] * 0.84

        self.derived = d
        return self

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop("derived", None)
        out["derived"] = self.derived
        return out

    def summary(self) -> str:
        d = self.derived
        rows = [
            ("stature", self.stature), ("chest", d["chest"]), ("waist", d["waist"]),
            ("hip", d["hip"]), ("neck", d["neck"]),
            ("shoulder width", d["shoulder_width"]), ("inseam", d["crotch_h"]),
            ("arm length", d["arm_length"]), ("thigh girth", d["thigh_girth"]),
            ("calf girth", d["calf_girth"]), ("upper-arm girth", d["upperarm_girth"]),
        ]
        return "\n".join(f"  {k:<16} {v:7.1f} cm" for k, v in rows)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------------------------------------------------------------------
# Ellipse perimeter helpers (Ramanujan II) - used to turn a requested
# circumference into cross-section semi-axes.
# ---------------------------------------------------------------------------
def ellipse_perimeter(p: float, q: float) -> float:
    h = ((p - q) ** 2) / ((p + q) ** 2) if (p + q) else 0.0
    return math.pi * (p + q) * (1.0 + 3.0 * h / (10.0 + math.sqrt(4.0 - 3.0 * h)))


def semi_axes_for_circumference(circ: float, depth_ratio: float) -> Tuple[float, float]:
    """Return (half_width, half_depth) of an ellipse with the given perimeter."""
    unit = ellipse_perimeter(1.0, depth_ratio)
    s = circ / unit
    return s, s * depth_ratio
