"""
Anatomical muscle layer.

Surface relief is added as a set of muscle bellies anchored to the *generated*
surface, not to assumed coordinates: each belly queries the torso profile or
the limb taper at its anchor, then sits so that it protrudes by a controlled
depth. Relief therefore follows whatever form the measurements produced.

Amplitude is governed by `definition` (0 = smooth mannequin) scaled by
`muscle`. Girth is left to the calibration pass, which re-solves the torso
profile against the muscled surface, so adding relief does not silently
inflate the requested chest, waist or hip.

Angles follow the torso convention: theta = pi/2 is anterior (+z),
3*pi/2 is posterior. A belly at theta mirrors to pi - theta.
"""
from __future__ import annotations

import math
import numpy as np
from typing import List, Tuple

from .sdf import (smin, smax, ssub, frame_from, sd_ellipsoid_oriented,
                  sd_round_cone, TAU)

FRONT = math.pi / 2
BACK = 3 * math.pi / 2


class MuscleLayer:
    def __init__(self, params, derived, profile, joints):
        self.p = params
        self.d = derived
        self.prof = profile
        self.J = joints
        self.H = params.stature
        # 0 when definition is 0; muscle biases how pronounced the relief is.
        # Kept well under 1.0: relief reads as anatomy at a few millimetres and
        # turns into hard-edged lumps long before it reaches a centimetre.
        #
        # Subcutaneous fat masks definition, so relief falls as bodyfat rises.
        # That is how real bodies behave, and it also keeps relief from pushing
        # already-thick limbs into contact on heavy builds.
        fat = float(np.clip(params.bodyfat, 0.0, 1.0))
        self.amp = (float(np.clip(params.definition, 0.0, 1.0))
                    * (0.34 + 0.52 * float(np.clip(params.muscle, 0.0, 1.0)))
                    * (1.0 - 0.62 * fat ** 1.4))

    # How deep the ellipsoid sits relative to its protrusion.
    DEPTH_FACTOR = 2.2
    # How far past the current surface a belly is allowed to reach, as a
    # multiple of relief. Every belly is clamped to this shell, which is what
    # keeps the layer well behaved: an ellipsoid anchored on an analytic
    # surface estimate can otherwise float free where the real blended surface
    # differs, or -- if it is wide enough to wrap around the body -- surface
    # again on the far side as a detached blob.
    SHELL = 1.7

    @property
    def active(self) -> bool:
        return self.amp > 1e-3

    def _clamp(self, belly: np.ndarray, f: np.ndarray, relief: float) -> np.ndarray:
        margin = max(relief * self.SHELL, 1e-3)
        return smax(belly, f - margin, margin * 0.6)

    # ------------------------------------------------------------------
    # anchoring helpers
    # ------------------------------------------------------------------
    def _belly_on_torso(self, P, f, theta, y, around, along, depth,
                        tilt=0.0, gain=1.0):
        """Add one belly sitting on the torso surface at (y, theta)."""
        H = self.H
        y = float(np.clip(y, self.prof.y_bottom + 0.004 * H,
                          self.prof.y_top - 0.004 * H))
        pt, nrm = self.prof.surface_at(y, theta)
        relief = depth * H * self.amp * gain
        c = max(relief * self.DEPTH_FACTOR, 1e-4)
        semi = np.array([around * H, along * H, c])
        fr = frame_from(nrm)
        if tilt:
            ct, st = math.cos(tilt), math.sin(tilt)
            rot = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
            fr = rot @ fr
        centre = pt - nrm * (c - relief)
        belly = sd_ellipsoid_oriented(P, centre, semi, fr)
        belly = self._clamp(belly, f, relief)
        return smin(f, belly, max(relief * 3.0, 0.10))

    def _groove_on_torso(self, P, f, theta, y0, y1, radius, depth, steps=7):
        """Carve a vertical furrow along the torso surface."""
        H = self.H
        r = radius * H
        cut_d = depth * H * self.amp
        if cut_d <= 1e-3:
            return f
        pts = []
        for i in range(steps):
            y = y0 + (y1 - y0) * i / (steps - 1)
            y = float(np.clip(y, self.prof.y_bottom + 0.004 * H,
                              self.prof.y_top - 0.004 * H))
            pt, nrm = self.prof.surface_at(y, theta)
            pts.append(pt + nrm * (r - cut_d))
        for a, b in zip(pts[:-1], pts[1:]):
            f = ssub(f, sd_round_cone(P, a, b, r, r), r * 1.45)
        return f

    def _limb_frame(self, a, b, side_sign):
        axis = np.asarray(b, float) - np.asarray(a, float)
        L = float(np.linalg.norm(axis))
        axis = axis / max(L, 1e-9)
        ant = np.array([0.0, 0.0, 1.0])
        ant = ant - axis * float(axis @ ant)
        n = np.linalg.norm(ant)
        ant = ant / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
        lat = np.cross(axis, ant)
        lat /= max(np.linalg.norm(lat), 1e-9)
        if lat[0] * side_sign < 0:          # keep +phi pointing away from midline
            lat = -lat
        return axis, ant, lat, L

    def _belly_on_limb(self, P, f, a, b, r_a, r_b, t, phi, side_sign,
                       around, along, depth, gain=1.0):
        """Add one belly on a tapered limb segment at parameter t, angle phi."""
        H = self.H
        axis, ant, lat, L = self._limb_frame(a, b, side_sign)
        base = np.asarray(a, float) + axis * (t * L)
        r = r_a + (r_b - r_a) * t
        dirv = ant * math.cos(phi) + lat * math.sin(phi)
        dirv /= max(np.linalg.norm(dirv), 1e-9)

        relief = depth * H * self.amp * gain
        c = max(relief * self.DEPTH_FACTOR, 1e-4)
        semi = np.array([around * H, along * H, c])
        fr = frame_from(dirv, up_hint=axis)
        centre = base + dirv * (r - (c - relief))
        belly = sd_ellipsoid_oriented(P, centre, semi, fr)
        belly = self._clamp(belly, f, relief)
        return smin(f, belly, max(relief * 3.0, 0.10))

    # ------------------------------------------------------------------
    # torso
    # ------------------------------------------------------------------
    def torso(self, P: np.ndarray, f: np.ndarray) -> np.ndarray:
        if not self.active:
            return f
        d, H = self.d, self.H

        for sgn in (1.0, -1.0):
            m = lambda off: FRONT + sgn * off          # anterior, mirrored
            b = lambda off: BACK + sgn * off           # posterior, mirrored

            # Pectoralis major: broad and flat, angled toward the sternum.
            # Equal width and height makes it a dome rather than a sheet.
            f = self._belly_on_torso(P, f, m(0.46), d["chest_h"] + 0.020 * H,
                                     0.078, 0.032, 0.0072, tilt=sgn * 0.34)
            # serratus anterior digitations -- barely there; at any more they
            # read as parallel stripes cut into the ribcage
            for i in range(3):
                f = self._belly_on_torso(
                    P, f, m(0.96 + 0.05 * i), d["chest_h"] - (0.026 + 0.028 * i) * H,
                    0.024, 0.010, 0.0018, tilt=sgn * 0.55, gain=0.55)
            # Rectus abdominis. One continuous sheath carries the column, with
            # shallow bands on top: separate bellies alone read as a stack of
            # sausages rather than as a single muscle divided by tendon.
            f = self._belly_on_torso(P, f, m(0.150),
                                     d["waist_h"] - 0.052 * H + 0.060 * H,
                                     0.032, 0.090, 0.0052, gain=1.0)
            for i in range(4):
                yy = d["waist_h"] - 0.050 * H + i * 0.040 * H
                w = 0.030 - 0.002 * i
                f = self._belly_on_torso(P, f, m(0.150), yy, w, 0.020,
                                         0.0030, gain=0.9)
            # external oblique
            f = self._belly_on_torso(P, f, m(0.80), d["waist_h"] - 0.030 * H,
                                     0.040, 0.044, 0.0068, tilt=sgn * 0.42,
                                     gain=0.85)
            # Trapezius: sits below the acromion. Placed at the very top of the
            # profile it spreads across the shoulder girdle and reads as a
            # shelf around the base of the neck.
            f = self._belly_on_torso(P, f, b(-0.38), d["acromion_h"] - 0.034 * H,
                                     0.058, 0.044, 0.0062, tilt=sgn * 0.34)
            # latissimus dorsi
            f = self._belly_on_torso(P, f, b(-0.74), d["chest_h"] - 0.034 * H,
                                     0.074, 0.066, 0.0062, tilt=sgn * 0.30)
            # erector spinae column
            f = self._belly_on_torso(P, f, b(-0.105),
                                     (d["hip_h"] + d["chest_h"]) * 0.5,
                                     0.017, 0.080, 0.0046)
            # gluteus medius
            f = self._belly_on_torso(P, f, b(-1.12), d["hip_h"] + 0.030 * H,
                                     0.030, 0.030, 0.0058, gain=0.8)

        # Midline furrows: wide and shallow. A narrow deep cut reads as a slot
        # gouged into the body rather than as the seam between two muscle
        # bellies, so the radius stays large and the depth small.
        f = self._groove_on_torso(P, f, FRONT, d["waist_h"] - 0.058 * H,
                                  d["chest_h"] - 0.034 * H, 0.0150, 0.0021)
        f = self._groove_on_torso(P, f, BACK, d["hip_h"] + 0.014 * H,
                                  d["acromion_h"] - 0.030 * H, 0.0180, 0.0026)
        return f

    # ------------------------------------------------------------------
    # limbs
    # ------------------------------------------------------------------
    def limbs(self, P: np.ndarray, f: np.ndarray, side: str) -> np.ndarray:
        if not self.active:
            return f
        d, J, H = self.d, self.J, self.H
        sg = 1.0 if side == "l" else -1.0

        sh, el, wr = J[f"shoulder_{side}"], J[f"elbow_{side}"], J[f"wrist_{side}"]
        r_sh = d["upperarm_girth"] / TAU * 1.12
        r_el = d["elbow_girth"] / TAU
        r_fa = d["forearm_girth"] / TAU * 1.08
        r_wr = d["wrist_girth"] / TAU

        # Deltoid: three heads, staggered along the arm. Placed at the same t
        # they wrap into a continuous ring that reads as a band around the
        # upper arm rather than as a shoulder cap.
        for t, phi, g in ((0.16, math.pi / 2 - 0.80, 0.80),
                          (0.09, math.pi / 2, 1.00),
                          (0.17, math.pi / 2 + 0.85, 0.78)):
            f = self._belly_on_limb(P, f, sh, el, r_sh, r_el, t, phi, sg,
                                    0.023, 0.030, 0.0072, gain=g)
        # biceps brachii / triceps
        f = self._belly_on_limb(P, f, sh, el, r_sh, r_el, 0.46, 0.0, sg,
                                0.024, 0.062, 0.0092)
        f = self._belly_on_limb(P, f, sh, el, r_sh, r_el, 0.42, math.pi, sg,
                                0.030, 0.068, 0.0085)
        # forearm flexor and extensor mass, bulked proximally and offset along
        # the limb so the two do not close into a ring
        f = self._belly_on_limb(P, f, el, wr, r_fa, r_wr, 0.28, 0.60, sg,
                                0.020, 0.050, 0.0048)
        f = self._belly_on_limb(P, f, el, wr, r_fa, r_wr, 0.21, math.pi - 0.62,
                                sg, 0.020, 0.046, 0.0044)

        hp, kn, an = J[f"hip_{side}"], J[f"knee_{side}"], J[f"ankle_{side}"]
        top_y = min(d["crotch_h"] + 0.012 * H, d["hip_h"] - 0.010 * H)
        th_a = np.array([hp[0], top_y, hp[2]])
        r_th = d["thigh_girth"] / TAU * 1.04
        r_kn = d["knee_girth"] / TAU
        r_cf = d["calf_girth"] / TAU
        r_an = d["ankle_girth"] / TAU

        # rectus femoris, vastus lateralis, vastus medialis
        f = self._belly_on_limb(P, f, th_a, kn, r_th, r_kn, 0.50, 0.0, sg,
                                0.034, 0.098, 0.0092)
        f = self._belly_on_limb(P, f, th_a, kn, r_th, r_kn, 0.56, 0.80, sg,
                                0.030, 0.082, 0.0085)
        f = self._belly_on_limb(P, f, th_a, kn, r_th, r_kn, 0.80, -0.70, sg,
                                0.026, 0.044, 0.0085)
        # hamstrings and adductor group
        f = self._belly_on_limb(P, f, th_a, kn, r_th, r_kn, 0.46, math.pi, sg,
                                0.038, 0.100, 0.0088)
        f = self._belly_on_limb(P, f, th_a, kn, r_th, r_kn, 0.38, -1.45, sg,
                                0.028, 0.072, 0.0072)
        # gastrocnemius heads and tibialis anterior
        cf = kn + (an - kn) * 0.30
        for phi, g in ((math.pi - 0.42, 1.0), (math.pi + 0.42, 0.92)):
            f = self._belly_on_limb(P, f, kn, cf, r_kn, r_cf, 0.75, phi, sg,
                                    0.024, 0.050, 0.0092, gain=g)
        f = self._belly_on_limb(P, f, kn, an, r_cf, r_an, 0.30, 0.30, sg,
                                0.020, 0.058, 0.0062)
        return f
