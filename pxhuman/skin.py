"""
Skin layer: ambient occlusion, tone, per-vertex colour and UVs.

The tone model is numeric and generic. Nothing here samples, references or is
derived from any photograph or any real individual; the head carries no
features and none are added.

Ambient occlusion is computed from the distance field rather than by ray
casting. The field is already in hand, so marching a few steps along each
vertex normal and comparing the field value to the distance travelled gives
the standard SDF occlusion estimate at a fraction of the cost: where the
surface is concave the field stays smaller than the step, and the shortfall is
the occlusion. This is what puts real depth into armpits, the groin, the neck,
under the pectorals and between the fingers.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

# Fitzpatrick-style base albedos in linear-ish sRGB. These are tone presets,
# not descriptions of any person.
TONES: Dict[str, Tuple[float, float, float]] = {
    "I":   (0.941, 0.855, 0.812),
    "II":  (0.906, 0.796, 0.741),
    "III": (0.847, 0.714, 0.639),
    "IV":  (0.737, 0.588, 0.494),
    "V":   (0.573, 0.427, 0.345),
    "VI":  (0.400, 0.286, 0.231),
}

# Blood-perfusion tint added where skin is thin or vascular.
PERFUSION = np.array([0.42, 0.10, 0.11])
# Colour the light picks up as it scatters under the surface.
SUBSURFACE = np.array([0.78, 0.34, 0.29])


def ambient_occlusion(field, verts: np.ndarray, normals: np.ndarray,
                      stature: float, steps: int = 6, reach: float = 0.055,
                      chunk: int = 60000) -> np.ndarray:
    """Per-vertex AO in [0,1]; 1 is fully open, 0 fully occluded."""
    span = reach * stature
    occ = np.zeros(len(verts))
    weight_total = 0.0
    for i in range(1, steps + 1):
        t = (i / steps) ** 1.35 * span
        w = 1.0 / (2.0 ** i)
        weight_total += w
        vals = np.empty(len(verts))
        for a in range(0, len(verts), chunk):
            b = min(a + chunk, len(verts))
            vals[a:b] = field(verts[a:b] + normals[a:b] * t)
        # outside the surface the field should read t; any shortfall is
        # nearby geometry closing the gap
        occ += w * np.clip((t - vals) / max(t, 1e-6), 0.0, 1.0)
    ao = 1.0 - np.clip(occ / max(weight_total, 1e-9), 0.0, 1.0)
    return np.clip(ao, 0.0, 1.0) ** 1.25


def smooth_scalar(values: np.ndarray, faces: np.ndarray,
                  iterations: int = 4, strength: float = 0.65) -> np.ndarray:
    """Laplacian smoothing of a per-vertex scalar across the mesh.

    Occlusion sampled straight off the field picks up every crease in it,
    including the C1 seam where a clamped muscle belly meets the surface.
    Geometrically that seam is sub-millimetre, but unsmoothed AO turns it into
    a hard dark line that reads as a scratch rather than as anatomy.
    """
    n = len(values)
    out = np.asarray(values, np.float64).copy()
    a = faces[:, [0, 1, 2]].ravel()
    b = faces[:, [1, 2, 0]].ravel()
    cnt = np.zeros(n)
    np.add.at(cnt, a, 1.0)
    np.add.at(cnt, b, 1.0)
    cnt = np.maximum(cnt, 1.0)
    for _ in range(iterations):
        acc = np.zeros(n)
        np.add.at(acc, a, out[b])
        np.add.at(acc, b, out[a])
        out = out * (1.0 - strength) + (acc / cnt) * strength
    return out


def curvature_proxy(field, verts, normals, stature, delta: float = 0.012):
    """Cheap convexity estimate: compare the field just off the surface with
    the step taken. Convex ridges read positive, creases negative."""
    t = delta * stature
    vals = np.empty(len(verts))
    for a in range(0, len(verts), 60000):
        b = min(a + 60000, len(verts))
        vals[a:b] = field(verts[a:b] + normals[a:b] * t)
    return np.clip((vals - t) / t, -1.0, 1.0)


def skin_colours(verts: np.ndarray, normals: np.ndarray, ao: np.ndarray,
                 curv: np.ndarray, params, derived: Dict[str, float],
                 tone: str = "II", redness: float = 0.5,
                 translucency: float = 0.5) -> np.ndarray:
    """Per-vertex linear RGB.

    Real skin is not one flat value. Tone is modulated by where the vertex
    sits on the body: extremities and joints carry more blood, the torso reads
    slightly cooler and paler, thin regions let subsurface colour through, and
    occlusion both darkens and warms rather than simply darkening.
    """
    d, H = derived, params.stature
    base = np.array(TONES.get(tone, TONES["II"]), float)
    col = np.tile(base, (len(verts), 1))

    y = verts[:, 1]
    r_axis = np.sqrt(verts[:, 0] ** 2 + verts[:, 2] ** 2)

    # --- regional perfusion -------------------------------------------
    def band(centre, width):
        return np.exp(-0.5 * ((y - centre) / max(width, 1e-6)) ** 2)

    perf = np.zeros(len(verts))
    perf += 0.55 * band(d["knee_h"], 0.030 * H)            # knees
    perf += 0.45 * band(d["elbow_h"], 0.028 * H)           # elbows
    perf += 0.30 * band(d["ankle_h"] + 0.02 * H, 0.030 * H)
    perf += 0.40 * np.clip((r_axis - 0.17 * H) / (0.10 * H), 0, 1)   # hands
    perf += 0.35 * band(d["chin_h"] + d["head_height"] * 0.45, 0.055 * H)
    perf += 0.22 * band(d["neck_base_h"], 0.030 * H)
    # cooler, paler across the abdomen and chest
    cool = 0.40 * band((d["waist_h"] + d["chest_h"]) * 0.5, 0.10 * H) \
        * np.clip(verts[:, 2] / (0.10 * H), 0, 1)

    perf = np.clip(perf, 0.0, 1.2) * redness
    col += PERFUSION[None, :] * perf[:, None] * 0.26
    col -= np.array([0.015, 0.004, -0.012])[None, :] * cool[:, None]

    # --- subsurface bleed in thin, convex regions ----------------------
    thin = np.clip(curv, 0.0, 1.0) ** 1.4
    col += SUBSURFACE[None, :] * (thin * translucency * 0.085)[:, None]

    # --- occlusion: darkens and warms ----------------------------------
    shade = 0.32 + 0.68 * ao
    col *= shade[:, None]
    warm = (1.0 - ao)[:, None] * np.array([0.085, 0.022, 0.012])[None, :]
    col += warm * translucency

    # --- fine tonal break-up so large areas are not flat ----------------
    n = (np.sin(verts[:, 0] * 1.7) * np.sin(verts[:, 1] * 1.3)
         * np.sin(verts[:, 2] * 1.9)
         + 0.5 * np.sin(verts[:, 0] * 4.1 + verts[:, 1] * 3.3))
    col *= (1.0 + 0.016 * n)[:, None]

    return np.clip(col, 0.0, 1.0)


def cylindrical_uv(verts: np.ndarray, params) -> np.ndarray:
    """A usable, seamed cylindrical unwrap.

    Not a production unwrap -- limbs held away from the axis distort and there
    is a seam at the back -- but it makes the mesh paintable in a DCC instead
    of leaving it with no UVs at all.
    """
    H = params.stature
    u = (np.arctan2(verts[:, 2], verts[:, 0]) / (2.0 * np.pi)) % 1.0
    v = np.clip(verts[:, 1] / H, 0.0, 1.0)
    return np.stack([u, v], axis=1)


def roughness_for(ao: np.ndarray, curv: np.ndarray) -> np.ndarray:
    """Skin is not uniformly rough: creases hold moisture and read smoother,
    exposed convex areas read drier."""
    r = 0.62 - 0.10 * (1.0 - ao) + 0.06 * np.clip(curv, 0, 1)
    return np.clip(r, 0.35, 0.80)
