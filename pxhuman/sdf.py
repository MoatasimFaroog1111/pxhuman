"""
Vectorised signed-distance primitives and the body field assembly.

The body is expressed as a smooth union of analytic primitives plus one
lofted generalised cylinder for the torso. The torso profile is renormalised
numerically at every control height so the requested chest / waist / hip
circumferences are reproduced exactly on the generated surface.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Dict, List, Tuple

from .anthro import BodyParams, semi_axes_for_circumference

TAU = 2.0 * math.pi


# ---------------------------------------------------------------------------
# smooth operators
# ---------------------------------------------------------------------------
def smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth minimum (C1 continuous)."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def smax(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth maximum -- the dual of smin."""
    if k <= 0:
        return np.maximum(a, b)
    return -smin(-a, -b, k)


def ssub(f: np.ndarray, cut: np.ndarray, k: float) -> np.ndarray:
    """Smoothly carve `cut` out of `f` -- used for anatomical grooves."""
    return smax(f, -cut, k)


# ---------------------------------------------------------------------------
# primitives -- all take P of shape (N, 3) and return (N,)
# ---------------------------------------------------------------------------
def sd_sphere(P: np.ndarray, c, r: float) -> np.ndarray:
    return np.linalg.norm(P - np.asarray(c, float), axis=1) - r


def sd_ellipsoid(P: np.ndarray, c, radii) -> np.ndarray:
    r = np.asarray(radii, float)
    p = P - np.asarray(c, float)
    k0 = np.linalg.norm(p / r, axis=1)
    k1 = np.linalg.norm(p / (r * r), axis=1)
    out = np.zeros_like(k0)
    nz = k1 > 1e-12
    out[nz] = k0[nz] * (k0[nz] - 1.0) / k1[nz]
    out[~nz] = -r.min()
    return out


def sd_round_cone(P: np.ndarray, a, b, r1: float, r2: float) -> np.ndarray:
    """Capsule with a different radius at each end (tapered limb segment)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ba = b - a
    l2 = float(ba @ ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2

    pa = P - a
    y = pa @ ba
    z = y - l2
    x2 = np.einsum("ij,ij->i", pa * l2 - np.outer(y, ba), pa * l2 - np.outer(y, ba))
    y2 = y * y * l2
    z2 = z * z * l2

    k = np.sign(rr) * rr * rr * x2
    d = np.empty(P.shape[0])

    m1 = (np.sign(z) * a2 * z2) > k
    m2 = (np.sign(y) * a2 * y2) < k
    m3 = ~(m1 | m2)

    d[m1] = np.sqrt(x2[m1] + z2[m1]) * il2 - r2
    d[m2] = np.sqrt(x2[m2] + y2[m2]) * il2 - r1
    d[m3] = (np.sqrt(np.maximum(x2[m3] * a2 * il2, 0.0)) + y[m3] * rr) * il2 - r1
    return d


def frame_from(normal, up_hint=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Orthonormal 3x3 frame whose third row is `normal`.

    Rows are (around, along, normal) so a muscle belly can be sized in
    anatomically meaningful directions.
    """
    n = np.asarray(normal, float)
    n = n / max(np.linalg.norm(n), 1e-9)
    u = np.asarray(up_hint, float)
    if abs(float(u @ n)) > 0.985:                 # degenerate, pick another
        u = np.array([0.0, 0.0, 1.0])
    a = np.cross(u, n)
    a /= max(np.linalg.norm(a), 1e-9)
    al = np.cross(n, a)
    return np.stack([a, al, n])


def sd_ellipsoid_oriented(P: np.ndarray, c, radii, frame: np.ndarray) -> np.ndarray:
    """Ellipsoid with arbitrary orientation; `frame` rows are its local axes."""
    q = (P - np.asarray(c, float)) @ frame.T
    r = np.asarray(radii, float)
    k0 = np.linalg.norm(q / r, axis=1)
    k1 = np.linalg.norm(q / (r * r), axis=1)
    out = np.zeros_like(k0)
    nz = k1 > 1e-12
    out[nz] = k0[nz] * (k0[nz] - 1.0) / k1[nz]
    out[~nz] = -r.min()
    return out


def scaled(fn, P: np.ndarray, scale) -> np.ndarray:
    """Evaluate a primitive in an anisotropically scaled space."""
    s = np.asarray(scale, float)
    return fn(P / s) * float(s.min())


# ---------------------------------------------------------------------------
# lofted torso profile
# ---------------------------------------------------------------------------
class TorsoProfile:
    """Generalised cylinder whose cross-section perimeter is exact at each
    control height. Stored as a radius lookup table over (height, angle)."""

    N_THETA = 192
    N_Y = 320

    def __init__(self, sections: List[dict], y_bottom: float, y_top: float):
        self.y_bottom = y_bottom
        self.y_top = y_top
        self.theta = np.linspace(0.0, TAU, self.N_THETA, endpoint=False)
        self.ys = np.linspace(y_bottom, y_top, self.N_Y)

        ctrl_y = np.array([s["y"] for s in sections], float)
        ctrl_r = np.stack([self._section_radii(s) for s in sections])    # (K, T)
        ctrl_z = np.array([s.get("z_off", 0.0) for s in sections], float)

        order = np.argsort(ctrl_y)
        ctrl_y, ctrl_r, ctrl_z = ctrl_y[order], ctrl_r[order], ctrl_z[order]

        from scipy.interpolate import PchipInterpolator
        self.R = PchipInterpolator(ctrl_y, ctrl_r, axis=0)(self.ys)       # (NY, T)
        self.Z = PchipInterpolator(ctrl_y, ctrl_z)(self.ys)               # (NY,)
        self.R = np.maximum(self.R, 0.4)
        # wrap column for periodic interpolation in theta
        self.Rw = np.concatenate([self.R, self.R[:, :1]], axis=1)

    # -- build one cross-section, perimeter-exact -------------------------
    def _section_radii(self, s: dict) -> np.ndarray:
        th = self.theta
        pw, pd = semi_axes_for_circumference(s["circ"], s["depth_ratio"])
        # base ellipse radius at angle theta (x = width axis, z = depth axis)
        r = (pw * pd) / np.sqrt((pd * np.cos(th)) ** 2 + (pw * np.sin(th)) ** 2)

        # anterior projection (front of the torso), smooth and local
        bp = s.get("front_proj", 0.0)
        if bp:
            lobe = np.exp(-((np.sin(th) - 1.0) ** 2) / 0.28) * (np.sin(th) > -0.1)
            r = r * (1.0 + bp * lobe)
        # posterior flattening (back of the torso is flatter than the front)
        bf = s.get("back_flat", 0.0)
        if bf:
            lobe = np.exp(-((np.sin(th) + 1.0) ** 2) / 0.45) * (np.sin(th) < 0.1)
            r = r * (1.0 - bf * lobe)
        # posterior projection (gluteal mass) -- kept inside the profile so the
        # section perimeter stays exact after renormalisation
        pp = s.get("back_proj", 0.0)
        if pp:
            lobe = np.exp(-((np.sin(th) + 1.0) ** 2) / 0.30) * (np.sin(th) < 0.1)
            r = r * (1.0 + pp * lobe)
        # lateral squaring of the shoulder girdle
        sq = s.get("square", 0.0)
        if sq:
            r = r * (1.0 + sq * np.abs(np.cos(th)) ** 3)

        # renormalise so the closed curve perimeter equals the requested value
        per = self._perimeter(r)
        return r * (s["circ"] / per)

    def _perimeter(self, r: np.ndarray) -> float:
        dth = TAU / self.N_THETA
        dr = (np.roll(r, -1) - np.roll(r, 1)) / (2.0 * dth)
        return float(np.sum(np.sqrt(r ** 2 + dr ** 2)) * dth)

    def section_circumference(self, y: float) -> float:
        iy = int(np.clip(np.searchsorted(self.ys, y), 0, self.N_Y - 1))
        return self._perimeter(self.R[iy])

    def radius_at(self, y: float, theta: float) -> float:
        iy = int(np.clip(np.searchsorted(self.ys, y), 0, self.N_Y - 1))
        ft = (np.mod(theta, TAU) / TAU) * self.N_THETA
        j0 = int(np.floor(ft)) % self.N_THETA
        j1 = (j0 + 1) % self.N_THETA
        w = ft - np.floor(ft)
        return float(self.R[iy, j0] * (1 - w) + self.R[iy, j1] * w)

    def surface_at(self, y: float, theta: float):
        """Point on the torso surface at (height, angle), with its outward
        normal. Muscle bellies are anchored here so relief follows whatever
        shape the measurements produced rather than assumed coordinates."""
        iy = int(np.clip(np.searchsorted(self.ys, y), 0, self.N_Y - 1))
        r = self.radius_at(y, theta)
        zc = float(self.Z[iy])
        ct, st = math.cos(theta), math.sin(theta)
        pt = np.array([r * ct, y, zc + r * st])

        # normal of the radial curve, including the dR/dtheta tilt
        h = TAU / self.N_THETA
        dr = (self.radius_at(y, theta + h) - self.radius_at(y, theta - h)) / (2 * h)
        tang = np.array([dr * ct - r * st, 0.0, dr * st + r * ct])
        nrm = np.array([tang[2], 0.0, -tang[0]])
        n = np.linalg.norm(nrm)
        nrm = nrm / n if n > 1e-9 else np.array([ct, 0.0, st])
        if nrm[0] * ct + nrm[2] * st < 0:
            nrm = -nrm
        return pt, nrm

    # -- field -------------------------------------------------------------
    def __call__(self, P: np.ndarray) -> np.ndarray:
        y = P[:, 1]
        t = np.clip((y - self.y_bottom) / (self.y_top - self.y_bottom), 0.0, 1.0)
        fy = t * (self.N_Y - 1)
        i0 = np.floor(fy).astype(np.int32)
        i1 = np.minimum(i0 + 1, self.N_Y - 1)
        wy = (fy - i0)[:, None]

        zc = self.Z[i0] * (1 - wy[:, 0]) + self.Z[i1] * wy[:, 0]
        dx = P[:, 0]
        dz = P[:, 2] - zc
        rp = np.sqrt(dx * dx + dz * dz)
        th = np.mod(np.arctan2(dz, dx), TAU)

        ft = th / TAU * self.N_THETA
        j0 = np.floor(ft).astype(np.int32)
        j1 = j0 + 1
        wt = ft - j0

        r00 = self.Rw[i0, j0]; r01 = self.Rw[i0, j1]
        r10 = self.Rw[i1, j0]; r11 = self.Rw[i1, j1]
        r0 = r00 * (1 - wt) + r01 * wt
        r1 = r10 * (1 - wt) + r11 * wt
        R = r0 * (1 - wy[:, 0]) + r1 * wy[:, 0]

        d_rad = rp - R
        # cap the generalised cylinder at both ends
        d_cap = np.maximum(y - self.y_top, self.y_bottom - y)
        return np.maximum(d_rad, d_cap)


# ---------------------------------------------------------------------------
# skeleton landmarks
# ---------------------------------------------------------------------------
def build_joints(p: BodyParams) -> Dict[str, np.ndarray]:
    d = p.derived
    H = p.stature
    J: Dict[str, np.ndarray] = {}

    # femoral head, not the greater trochanter: the joint centre sits well
    # medial to the lateral hip breadth
    hip_x = 0.047 * H
    drop = d["hip_h"] - d["ankle_h"]

    # leg_stance splays the feet outward from the hip joint. The floors below
    # guarantee the two legs keep a real gap all the way down. The thighs are
    # allowed to meet at the top -- that is anatomically right -- but if the
    # legs touch AGAIN lower down with a gap between the two contacts, the
    # pair closes a ring and the mesh gains a handle. Foot breadth alone is
    # not enough: on heavy builds the knees and calves are the narrow point.
    gap = 0.018 * H
    knee_r = d["knee_girth"] / (2.0 * math.pi)
    calf_r = d["calf_girth"] / (2.0 * math.pi)
    # knee_x = 0.62*ankle_x + 0.38*hip_x ; calf sits 30% down from the knee
    need_knee = (knee_r + gap - 0.38 * hip_x) / 0.62
    need_calf = (calf_r + gap - 0.266 * hip_x) / 0.734
    foot_floor = d["foot_breadth"] * 0.62 + 0.022 * H
    ankle_x = max(foot_floor, need_knee, need_calf,
                  hip_x + math.tan(math.radians(d["leg_stance"])) * drop)

    J["root"] = np.array([0.0, d["hip_h"], 0.0])
    J["spine_01"] = np.array([0.0, d["hip_h"] + 0.055 * H, 0.0])
    J["spine_02"] = np.array([0.0, d["waist_h"] + 0.045 * H, -0.004 * H])
    J["spine_03"] = np.array([0.0, d["chest_h"] + 0.030 * H, -0.006 * H])
    J["neck"] = np.array([0.0, d["neck_base_h"], -0.004 * H])
    J["head"] = np.array([0.0, d["chin_h"] + 0.012 * H, 0.004 * H])
    J["head_top"] = np.array([0.0, H, 0.0])

    abd = math.radians(d["arm_abduction"])
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        sh = np.array([sgn * d["shoulder_width"] * 0.455, d["acromion_h"], 0.0])
        dirv = np.array([sgn * math.sin(abd), -math.cos(abd), 0.0])
        el = sh + dirv * d["upperarm_len"]
        wr = el + dirv * d["forearm_len"]
        hn = wr + dirv * d["hand_length"]
        J[f"clavicle_{side}"] = np.array([sgn * d["shoulder_width"] * 0.17,
                                          d["acromion_h"] - 0.012 * H, 0.0])
        J[f"shoulder_{side}"] = sh
        J[f"elbow_{side}"] = el
        J[f"wrist_{side}"] = wr
        J[f"hand_{side}"] = hn

        hp = np.array([sgn * hip_x, d["hip_h"], 0.0])
        kn = np.array([sgn * (ankle_x + (hip_x - ankle_x) * 0.38), d["knee_h"], 0.004 * H])
        an = np.array([sgn * ankle_x, d["ankle_h"], 0.0])
        to = np.array([sgn * ankle_x, d["ankle_h"] * 0.62, d["foot_length"] * 0.70])
        J[f"hip_{side}"] = hp
        J[f"knee_{side}"] = kn
        J[f"ankle_{side}"] = an
        J[f"toe_{side}"] = to
    return J


HIERARCHY = {
    "root": None,
    "spine_01": "root", "spine_02": "spine_01", "spine_03": "spine_02",
    "neck": "spine_03", "head": "neck", "head_top": "head",
    "clavicle_l": "spine_03", "shoulder_l": "clavicle_l",
    "elbow_l": "shoulder_l", "wrist_l": "elbow_l", "hand_l": "wrist_l",
    "clavicle_r": "spine_03", "shoulder_r": "clavicle_r",
    "elbow_r": "shoulder_r", "wrist_r": "elbow_r", "hand_r": "wrist_r",
    "hip_l": "root", "knee_l": "hip_l", "ankle_l": "knee_l", "toe_l": "ankle_l",
    "hip_r": "root", "knee_r": "hip_r", "ankle_r": "knee_r", "toe_r": "ankle_r",
}


# ---------------------------------------------------------------------------
# full body field
# ---------------------------------------------------------------------------
class BodyField:
    def __init__(self, p: BodyParams, gains: Dict[str, float] | None = None,
                 hand_detail: str = "mitt"):
        self.p = p
        self.d = p.derived
        self.hand_detail = hand_detail
        # multiplicative corrections solved by the calibration pass so the
        # measured surface, not the input profile, matches the target girths
        self.gains = {"chest": 1.0, "waist": 1.0, "hip": 1.0, **(gains or {})}
        self.J = build_joints(p)
        self.profile = self._build_profile()
        self.k_blend = 0.016 * p.stature          # joint blend radius
        from .muscles import MuscleLayer
        self.muscles = MuscleLayer(p, self.d, self.profile, self.J)

    # -- torso control sections -------------------------------------------
    def _build_profile(self) -> TorsoProfile:
        d = self.d
        H = self.p.stature
        post = d["posture"]
        g = self.gains
        c_chest = d["chest"] * g["chest"]
        c_waist = d["waist"] * g["waist"]
        c_hip = d["hip"] * g["hip"]

        # shoulder girdle circumference implied by biacromial breadth
        c_shoulder = c_chest * 1.045 + d["shoulder_width"] * 0.30
        c_neckbase = d["neck"] * 1.30

        glute = 0.22 + 0.10 * self.p.proportion + 0.08 * (self.p.bodyfat - 0.5)

        S = [
            dict(y=d["crotch_h"] - 0.012 * H, circ=c_hip * 0.880,
                 depth_ratio=d["hip_depth_ratio"] * 1.02, z_off=0.0,
                 back_proj=glute * 0.55),
            dict(y=d["hip_h"], circ=c_hip,
                 depth_ratio=d["hip_depth_ratio"], z_off=-0.004 * H,
                 back_proj=glute),
            dict(y=(d["hip_h"] + d["waist_h"]) * 0.5, circ=(c_hip + c_waist) * 0.505,
                 depth_ratio=(d["hip_depth_ratio"] + d["waist_depth_ratio"]) * 0.5,
                 z_off=-0.006 * H + 0.004 * H * post, back_flat=0.06,
                 back_proj=glute * 0.30),
            dict(y=d["waist_h"], circ=c_waist,
                 depth_ratio=d["waist_depth_ratio"],
                 z_off=-0.008 * H + 0.006 * H * post, back_flat=0.08),
            dict(y=(d["waist_h"] + d["chest_h"]) * 0.5, circ=(c_waist + c_chest) * 0.515,
                 depth_ratio=(d["waist_depth_ratio"] + d["chest_depth_ratio"]) * 0.5,
                 z_off=-0.004 * H + 0.002 * H * post, back_flat=0.10,
                 front_proj=d["bust_projection"] * 0.45),
            dict(y=d["chest_h"], circ=c_chest,
                 depth_ratio=d["chest_depth_ratio"], z_off=0.0, back_flat=0.11,
                 front_proj=d["bust_projection"]),
            dict(y=d["chest_h"] + 0.045 * H, circ=c_chest * 0.985,
                 depth_ratio=d["chest_depth_ratio"] * 0.96,
                 z_off=0.002 * H - 0.004 * H * post, back_flat=0.09,
                 front_proj=d["bust_projection"] * 0.30, square=0.05),
            dict(y=d["acromion_h"], circ=c_shoulder,
                 depth_ratio=0.56, z_off=0.0, back_flat=0.05, square=0.16),
            dict(y=d["neck_base_h"], circ=c_neckbase,
                 depth_ratio=0.80, z_off=-0.004 * H, square=0.02),
        ]
        return TorsoProfile(S, S[0]["y"], S[-1]["y"])

    # -- evaluate -----------------------------------------------------------
    def __call__(self, P: np.ndarray) -> np.ndarray:
        """Field assembly.

        Midline structures (torso, neck, head) are built first. Each side is
        then blended onto its OWN copy of that base and the two sides are
        combined with a hard union. Opposite limbs therefore can never smooth
        -blend into one another -- on heavy builds the inner thighs and calves
        would otherwise web together while the knees stayed apart, ringing a
        hole between the legs.
        """
        base = self._midline(P)
        left = self._side(P, base, "l")
        right = self._side(P, base, "r")
        return np.minimum(left, right)

    def _midline(self, P: np.ndarray) -> np.ndarray:
        d, H = self.d, self.p.stature
        J, k = self.J, self.k_blend

        f = self.profile(P)

        # neck
        nr = d["neck"] / TAU
        f = smin(f, sd_round_cone(P, J["neck"] - np.array([0, 0.02 * H, 0]),
                                  np.array([0.0, d["chin_h"] + 0.012 * H, 0.006 * H]),
                                  nr * 1.14, nr * 0.95), k * 2.0)

        # head -- generic ovoid, no facial features are generated
        hc = np.array([0.0, d["chin_h"] + d["head_height"] * 0.52, 0.004 * H])
        f = smin(f, sd_ellipsoid(P, hc, (d["head_width"] * 0.5,
                                         d["head_height"] * 0.52,
                                         d["head_depth"] * 0.5)), k * 1.2)
        jaw = np.array([0.0, d["chin_h"] + d["head_height"] * 0.20,
                        d["head_depth"] * 0.10])
        f = smin(f, sd_ellipsoid(P, jaw, (d["head_width"] * 0.40,
                                          d["head_height"] * 0.24,
                                          d["head_depth"] * 0.38)), k * 1.8)

        f = self.muscles.torso(P, f)
        return f

    def _side(self, P: np.ndarray, base: np.ndarray, side: str) -> np.ndarray:
        """One side's limbs blended onto its own copy of the midline field."""
        d, H = self.d, self.p.stature
        J, k = self.J, self.k_blend
        f = base

        # --- arm ----------------------------------------------------------
        sh, el, wr = J[f"shoulder_{side}"], J[f"elbow_{side}"], J[f"wrist_{side}"]
        r_sh = d["upperarm_girth"] / TAU * 1.12
        r_el = d["elbow_girth"] / TAU
        r_fa = d["forearm_girth"] / TAU
        r_wr = d["wrist_girth"] / TAU
        # The deltoid sphere alone carries the shoulder junction. The upper arm
        # uses a HARD union: a smooth blend there can bridge the arm to the
        # ribcage a second time lower down, ringing the armpit into a handle.
        f = smin(f, sd_sphere(P, sh, r_sh * 1.06), k * 2.0)              # deltoid
        f = np.minimum(f, sd_round_cone(P, sh, el, r_sh, r_el))
        f = smin(f, sd_round_cone(P, el, wr, r_fa * 1.08, r_wr), k * 0.55)
        f = smin(f, self._hand(P, side), k * 0.45)

        # --- leg ------------------------------------------------------------
        hp, kn, an = J[f"hip_{side}"], J[f"knee_{side}"], J[f"ankle_{side}"]
        r_th = d["thigh_girth"] / TAU
        r_kn = d["knee_girth"] / TAU
        r_cf = d["calf_girth"] / TAU
        r_an = d["ankle_girth"] / TAU
        # the thigh must not add girth above the crotch: the torso profile
        # already owns the hip circumference there
        top_y = min(d["crotch_h"] + 0.012 * H, d["hip_h"] - 0.010 * H)
        hp_up = np.array([hp[0], top_y, hp[2]])
        f = smin(f, sd_round_cone(P, hp_up, kn, r_th * 1.04, r_kn), k * 1.5)
        cf = kn + (an - kn) * 0.30          # calf belly sits high on the shank
        f = smin(f, sd_round_cone(P, kn, cf, r_kn, r_cf), k * 1.6)
        f = smin(f, sd_round_cone(P, cf, an, r_cf, r_an), k * 1.6)
        f = smin(f, self._foot(P, side), k * 1.2)
        f = self.muscles.limbs(P, f, side)
        return f

    # -- extremities ---------------------------------------------------------
    def _hand(self, P: np.ndarray, side: str) -> np.ndarray:
        """Hand. `fingers` mode needs voxel <= 0.35 cm to resolve the gaps
        between digits; below that they fuse and introduce spurious handles,
        so the default is a mitt with a separated thumb."""
        d, J = self.d, self.J
        wr, hn = J[f"wrist_{side}"], J[f"hand_{side}"]
        axis = hn - wr
        L = float(np.linalg.norm(axis))
        axis = axis / L
        up = np.array([0.0, 0.0, 1.0])
        lat = np.cross(axis, up)
        lat /= np.linalg.norm(lat)
        HB = d["hand_breadth"]
        thick = HB * 0.30

        palm_c = wr + axis * (L * 0.34)
        pal = sd_round_cone(P, wr + axis * L * 0.06, wr + axis * L * 0.56,
                            HB * 0.46, HB * 0.52)
        pal = np.maximum(pal, np.abs((P - palm_c) @ up) - thick)
        out = pal

        if self.hand_detail == "fingers":
            n_f = 4
            for i in range(n_f):
                off = (i - (n_f - 1) / 2.0) * HB * 0.280
                base = wr + axis * (L * 0.54) + lat * off
                tip = base + axis * (L * (0.40 - 0.085 * abs(i - 1.1)))
                fr = HB * 0.095
                out = smin(out, sd_round_cone(P, base, tip, fr, fr * 0.80), 0.12)
        else:
            mitt_c = wr + axis * (L * 0.74)
            mitt = sd_round_cone(P, wr + axis * L * 0.50, wr + axis * L * 0.92,
                                 HB * 0.52, HB * 0.34)
            mitt = np.maximum(mitt, np.abs((P - mitt_c) @ up) - thick * 0.80)
            out = smin(out, mitt, 0.6)

        # thumb -- separated in both modes
        tb_base = wr + axis * (L * 0.18) + lat * (HB * 0.44)
        tb_tip = tb_base + (axis * 0.60 + lat * 0.52) * (L * 0.34)
        out = smin(out, sd_round_cone(P, tb_base, tb_tip, HB * 0.155, HB * 0.115), 0.45)
        return out

    def _foot(self, P: np.ndarray, side: str) -> np.ndarray:
        """Foot. Evaluated in a vertically compressed space so the section is
        an ellipse rather than a circle -- a plain round cone would be as tall
        as it is wide and read as a slab."""
        d, J = self.d, self.J
        an = J[f"ankle_{side}"]
        FL, FB = d["foot_length"], d["foot_breadth"]
        sy = 0.52                       # vertical compression of the foot
        y0 = d["ankle_h"] * 0.30        # compression centre, near the sole

        Q = P.copy()
        Q[:, 1] = (P[:, 1] - y0) / sy + y0

        heel = np.array([an[0], y0 + (an[1] - d["ankle_h"] * 0.10 - y0) / sy,
                         an[2] - FL * 0.13])
        ball = np.array([an[0], y0 + (an[1] - d["ankle_h"] * 0.30 - y0) / sy,
                         an[2] + FL * 0.36])
        toe = np.array([an[0], y0 + (an[1] - d["ankle_h"] * 0.36 - y0) / sy,
                        an[2] + FL * 0.54])

        f = sd_round_cone(Q, heel, ball, FB * 0.48, FB * 0.52)
        f = smin(f, sd_round_cone(Q, ball, toe, FB * 0.50, FB * 0.32), 0.9)
        f = f * sy
        # instep ramps up to meet the ankle
        f = smin(f, sd_round_cone(P, an + np.array([0.0, d["ankle_h"] * 0.30, 0.0]),
                                  np.array([an[0], y0 * 0.9, an[2] + FL * 0.10]),
                                  d["ankle_girth"] / TAU * 1.02,
                                  FB * 0.52), 1.2)
        # flat sole on the ground plane
        return np.maximum(f, -P[:, 1])
