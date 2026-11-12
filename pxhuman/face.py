"""
Parametric head and face.

The face is an ORIGINAL synthetic design built from canonical proportions and
numeric parameters. It is not derived from, modelled on, or fitted to any
photograph or any real person, and no likeness of a real individual is
produced or intended.

Features sit at the centimetre scale, far finer than the body's limbs, so the
head is built as its own mesh in its own bounding box. At ~0.10 cm voxels a
head is a few million samples; the same voxel over a whole body would be
billions.

Local frame: origin at the chin point, +y up to the crown, +z forward (the
direction the face looks), +x to the subject's left.
"""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field as dc_field
from typing import Dict

from .sdf import (smin, smax, ssub, sd_sphere, sd_ellipsoid, sd_round_cone,
                  sd_ellipsoid_oriented, frame_from)


def rot_x(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


@dataclass
class FaceParams:
    """All proportions are fractions of head height unless noted."""
    head_height: float = 22.0        # chin to crown, cm
    width_ratio: float = 0.655       # head width / head height
    depth_ratio: float = 0.830       # head depth / head height

    # vertical landmarks, measured from the chin
    y_mouth: float = 0.192
    y_nose_base: float = 0.325
    y_eye: float = 0.500
    y_brow: float = 0.575
    y_hairline: float = 0.730

    # feature scale
    eye_sep: float = 0.232           # pupil offset from centreline, x
    eye_size: float = 0.088          # palpebral fissure half-width
    eyeball_r: float = 0.058
    iris_r: float = 0.026
    nose_len: float = 0.185
    nose_width: float = 0.098
    nose_proj: float = 0.118         # how far the tip stands off the face
    mouth_width: float = 0.150
    lip_upper: float = 0.028
    lip_lower: float = 0.038
    chin_width: float = 0.150
    jaw_width: float = 0.400
    cheek_height: float = 0.440
    cheek_out: float = 0.470
    brow_thick: float = 0.022
    ear_len: float = 0.185

    # character dials
    softness: float = 0.55           # rounder, softer transitions
    cheek_fullness: float = 0.50
    jaw_taper: float = 0.62          # 1 = square jaw, 0 = very pointed
    nose_turn: float = 0.35          # upward tilt of the tip
    eye_tilt: float = 6.0            # outer canthus lift, degrees
    lip_fullness: float = 0.55
    brow_arch: float = 0.55

    hair: str = "bob"                # none | bob | long | bun
    hair_volume: float = 0.055

    def derived(self) -> Dict[str, float]:
        H = self.head_height
        return dict(H=H, W=H * self.width_ratio, D=H * self.depth_ratio)


class FaceField:
    """Signed-distance field for the head."""

    def __init__(self, p: FaceParams):
        self.p = p
        d = p.derived()
        self.H, self.W, self.D = d["H"], d["W"], d["D"]
        self.k = 0.012 * self.H

    # ------------------------------------------------------------------
    def skull(self, P: np.ndarray) -> np.ndarray:
        """Head mass.

        Built from FEW large forms with WIDE blends. Many small ellipsoids
        each protruding a little do not add up to a face -- they add up to a
        bag of lumps, because every one of them carries its own silhouette.
        The cranium and the face-jaw mass carry the shape; everything else
        only modulates it from inside.
        """
        p, H, W, D = self.p, self.H, self.W, self.D
        kb = 0.075 * H                      # broad blend between major masses

        # The masses are pushed BACK so the face plane lands near z = +0.30 H.
        # Centred near the origin they reach further forward than the nose and
        # the lips do, and the features end up buried inside the skull.
        f = sd_ellipsoid(P, (0.0, 0.690 * H, -0.120 * H),
                         (W * 0.500, H * 0.315, D * 0.450))
        jw = W * (0.330 + 0.130 * p.jaw_taper)
        f = smin(f, sd_ellipsoid(P, (0.0, 0.400 * H, -0.060 * H),
                                 (W * 0.455, H * 0.290, D * 0.395)), kb)
        f = smin(f, sd_ellipsoid(P, (0.0, 0.185 * H, -0.020 * H),
                                 (jw, H * 0.195, D * 0.330)), kb * 0.95)
        f = smin(f, sd_ellipsoid(P, (0.0, 0.082 * H, 0.055 * H),
                                 (W * p.chin_width * 1.15, H * 0.090,
                                  D * 0.195)), kb * 0.75)
        # occiput
        f = smin(f, sd_ellipsoid(P, (0.0, 0.600 * H, -0.250 * H),
                                 (W * 0.405, H * 0.230, D * 0.265)), kb)

        # --- modulation: these stay INSIDE the silhouette -----------------
        # brow ridge, shallow
        f = smin(f, sd_ellipsoid(P, (0.0, p.y_brow * H, 0.170 * H),
                                 (W * 0.355, H * 0.036, D * 0.105)), kb * 0.55)
        # cheekbones, pulled in so they shade rather than bulge
        for sx in (1.0, -1.0):
            f = smin(f, sd_ellipsoid(
                P, (sx * W * 0.315, p.cheek_height * H, 0.150 * H),
                (W * 0.145, H * 0.062, D * 0.150)), kb * 0.85)
            f = smin(f, sd_ellipsoid(
                P, (sx * W * 0.255, 0.330 * H, 0.165 * H),
                (W * 0.145, H * 0.080,
                 D * 0.120 * (0.7 + 0.5 * p.cheek_fullness))), kb * 0.95)
        return f

    # ------------------------------------------------------------------
    def eyes(self, P: np.ndarray, f: np.ndarray) -> np.ndarray:
        p, H, W, D = self.p, self.H, self.W, self.D
        k = self.k
        ey = p.y_eye * H
        ez = 0.222 * H
        for sx in (1.0, -1.0):
            ex = sx * W * p.eye_sep
            c = np.array([ex, ey, ez])
            # orbit: a shallow scoop. Deep sockets bury the eye and read as
            # hollows rather than as a seated eyeball.
            f = ssub(f, sd_ellipsoid(P, c + np.array([0, 0, 0.055 * H]),
                                     (W * 0.145, H * 0.070, D * 0.085)),
                     k * 2.2)
            # eyeball
            ball = sd_sphere(P, c, p.eyeball_r * H)
            f = np.minimum(f, ball)
            # lids: a shell around the ball, opened by a lens-shaped cut
            lid = sd_sphere(P, c, p.eyeball_r * H * 1.115)
            lid = smax(lid, -sd_sphere(P, c, p.eyeball_r * H * 0.96), k * 0.5)
            tilt = frame_from((0, 0, 1))
            rot = rot_z(sx * p.eye_tilt)
            opening = sd_ellipsoid_oriented(
                P, c + np.array([0, 0, p.eyeball_r * H * 0.55]),
                (p.eye_size * H, p.eye_size * H * 0.42, p.eyeball_r * H * 1.4),
                rot @ tilt)
            lid = smax(lid, -opening, k * 0.45)
            f = smin(f, lid, k * 0.9)
            # brow
            bx = ex * 0.98
            by = p.y_brow * H + 0.012 * H
            # narrower, and set wide enough apart that the two never bridge
            # into a single bar across the face
            brow = sd_ellipsoid_oriented(
                P, (bx * 1.06, by, ez + 0.040 * H),
                (W * 0.122, H * p.brow_thick, D * 0.045),
                rot_z(sx * (4.0 + 10.0 * p.brow_arch)) @ frame_from((0, 0, 1)))
            f = smin(f, brow, k * 0.9)
        return f

    # ------------------------------------------------------------------
    def nose(self, P: np.ndarray, f: np.ndarray) -> np.ndarray:
        p, H, W, D = self.p, self.H, self.W, self.D
        k = self.k
        y_top = p.y_brow * H - 0.015 * H
        y_base = p.y_nose_base * H
        z_root = 0.215 * H
        z_tip = (0.215 + p.nose_proj) * H

        # bridge from the root down to just above the tip
        f = smin(f, sd_round_cone(
            P, (0.0, y_top, z_root),
            (0.0, y_base + 0.030 * H, z_tip - 0.012 * H),
            W * 0.052, W * 0.070), k * 2.2)
        # tip
        tip = np.array([0.0, y_base + 0.022 * H, z_tip])
        f = smin(f, sd_ellipsoid(P, tip, (W * 0.072, H * 0.040, D * 0.058)),
                 k * 1.6)
        # alae
        for sx in (1.0, -1.0):
            f = smin(f, sd_ellipsoid(
                P, (sx * W * p.nose_width * 0.72, y_base + 0.012 * H,
                    z_tip - 0.028 * H),
                (W * 0.050, H * 0.034, D * 0.050)), k * 1.5)
            # nostril
            f = ssub(f, sd_ellipsoid(
                P, (sx * W * p.nose_width * 0.40, y_base - 0.004 * H,
                    z_tip - 0.020 * H),
                (W * 0.026, H * 0.024, D * 0.034)), k * 0.7)
        # undercut so the base reads separate from the lip
        f = ssub(f, sd_ellipsoid(
            P, (0.0, y_base - 0.030 * H, z_tip - 0.010 * H),
            (W * 0.120, H * 0.030, D * 0.055)), k * 1.0)
        return f

    # ------------------------------------------------------------------
    def mouth(self, P: np.ndarray, f: np.ndarray) -> np.ndarray:
        p, H, W, D = self.p, self.H, self.W, self.D
        k = self.k
        my = p.y_mouth * H
        mz = 0.252 * H
        fu = 0.7 + 0.6 * p.lip_fullness

        # upper lip, with a slight downward turn at the corners
        f = smin(f, sd_ellipsoid(
            P, (0.0, my + 0.020 * H, mz),
            (W * p.mouth_width * 1.05, H * p.lip_upper * fu, D * 0.062 * fu)),
            k * 1.5)
        # lower lip, fuller
        f = smin(f, sd_ellipsoid(
            P, (0.0, my - 0.026 * H, mz - 0.002 * H),
            (W * p.mouth_width * 0.95, H * p.lip_lower * fu, D * 0.070 * fu)),
            k * 1.5)
        # mouth line
        f = ssub(f, sd_ellipsoid(
            P, (0.0, my - 0.002 * H, mz + 0.028 * H),
            (W * p.mouth_width * 0.97, H * 0.0085, D * 0.045)), k * 0.55)
        # philtrum
        f = ssub(f, sd_ellipsoid(
            P, (0.0, my + 0.055 * H, mz + 0.030 * H),
            (W * 0.038, H * 0.034, D * 0.030)), k * 0.9)
        # mentolabial sulcus, the crease under the lower lip
        f = ssub(f, sd_ellipsoid(
            P, (0.0, my - 0.062 * H, mz + 0.010 * H),
            (W * 0.170, H * 0.020, D * 0.038)), k * 1.3)
        return f

    # ------------------------------------------------------------------
    def ears(self, P: np.ndarray, f: np.ndarray) -> np.ndarray:
        p, H, W, D = self.p, self.H, self.W, self.D
        k = self.k
        ey = 0.455 * H
        for sx in (1.0, -1.0):
            ex = sx * W * 0.495
            c = np.array([ex, ey, -0.045 * H])
            fr = rot_y(sx * -16.0) @ frame_from((sx, 0.0, 0.0))
            shell = sd_ellipsoid_oriented(
                P, c, (H * p.ear_len * 0.52, H * p.ear_len * 0.34, W * 0.030), fr)
            f = smin(f, shell, k * 1.6)
            # concha
            f = ssub(f, sd_ellipsoid_oriented(
                P, c + np.array([sx * W * 0.020, -0.004 * H, 0.004 * H]),
                (H * p.ear_len * 0.26, H * p.ear_len * 0.18, W * 0.030), fr),
                k * 0.7)
        return f

    # ------------------------------------------------------------------
    def neck(self, P: np.ndarray, f: np.ndarray) -> np.ndarray:
        H, W, D = self.H, self.W, self.D
        k = self.k
        f = smin(f, sd_round_cone(P, (0.0, 0.130 * H, -0.060 * H),
                                  (0.0, -0.430 * H, -0.075 * H),
                                  W * 0.300, W * 0.360), k * 3.4)
        return f

    # ------------------------------------------------------------------
    def hair_field(self, P: np.ndarray, skull: np.ndarray) -> np.ndarray:
        """Hair as a shaped mass. Strand-level hair is a different pipeline
        (cards or curves); this is a volume that reads correctly in silhouette
        and takes the hair colour."""
        p, H, W, D = self.p, self.H, self.W, self.D
        if p.hair == "none":
            return np.full(P.shape[0], 1e6)
        k = self.k
        t = p.hair_volume * H

        # A close cap, only a little larger than the skull. Oversizing it is
        # what turns hair into a helmet that swallows the head's silhouette.
        cap = sd_ellipsoid(P, (0.0, 0.695 * H, -0.055 * H),
                           (W * 0.500 + t, H * 0.315 + t * 0.8, D * 0.495 + t))
        # open the face: everything forward of the hairline is skin
        cap = smax(cap, -(sd_ellipsoid(
            P, (0.0, p.y_hairline * H - 0.075 * H, 0.520 * H),
            (W * 0.420, H * 0.235, D * 0.430))), k * 1.6)

        if p.hair == "bob":
            mass = sd_ellipsoid(P, (0.0, 0.520 * H, -0.085 * H),
                                (W * 0.520 + t, H * 0.350, D * 0.515 + t))
            # clear the face and jaw
            mass = smax(mass, -sd_ellipsoid(
                P, (0.0, 0.400 * H, 0.430 * H),
                (W * 0.400, H * 0.330, D * 0.430)), k * 2.0)
            # bob stops around the jaw line
            mass = smax(mass, -(0.235 * H - P[:, 1]), k * 1.6)
            cap = smin(cap, mass, k * 2.0)
        elif p.hair == "long":
            mass = sd_round_cone(P, (0.0, 0.620 * H, -0.130 * H),
                                 (0.0, -0.280 * H, -0.175 * H),
                                 W * 0.480 + t, W * 0.430)
            mass = smax(mass, -sd_ellipsoid(
                P, (0.0, 0.330 * H, 0.450 * H),
                (W * 0.430, H * 0.430, D * 0.430)), k * 2.2)
            cap = smin(cap, mass, k * 2.2)
        elif p.hair == "bun":
            cap = smin(cap, sd_ellipsoid(
                P, (0.0, 0.790 * H, -0.400 * H),
                (W * 0.215, H * 0.145, D * 0.190)), k * 2.0)

        # never let hair sit inside the skull
        return smax(cap, -skull, k * 0.8)

    # ------------------------------------------------------------------
    def __call__(self, P: np.ndarray, with_hair: bool = True) -> np.ndarray:
        f = self.skull(P)
        f = self.neck(f=f, P=P)
        f = self.eyes(P, f)
        f = self.nose(P, f)
        f = self.mouth(P, f)
        f = self.ears(P, f)
        if with_hair and self.p.hair != "none":
            f = np.minimum(f, self.hair_field(P, self.skull(P)))
        return f

    # ------------------------------------------------------------------
    def bounds(self, pad: float = 1.2):
        H, W, D = self.H, self.W, self.D
        t = self.p.hair_volume * H + pad
        lo = np.array([-W * 0.62 - t, -0.46 * H, -D * 0.68 - t])
        hi = np.array([W * 0.62 + t, 1.06 * H + t, D * 0.62 + t])
        return lo, hi

    def landmarks(self) -> Dict[str, np.ndarray]:
        p, H, W = self.p, self.H, self.W
        return dict(
            eye_l=np.array([W * p.eye_sep, p.y_eye * H, 0.222 * H]),
            eye_r=np.array([-W * p.eye_sep, p.y_eye * H, 0.222 * H]),
            mouth=np.array([0.0, p.y_mouth * H, 0.252 * H]),
            nose_tip=np.array([0.0, p.y_nose_base * H + 0.022 * H,
                               (0.215 + p.nose_proj) * H]),
            hairline_y=p.y_hairline * H,
            eyeball_r=p.eyeball_r * H,
            iris_r=p.iris_r * H,
        )
