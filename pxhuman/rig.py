"""Skeleton, linear-blend skinning weights, and posing."""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from .sdf import HIERARCHY

# bones that should not capture vertices (leaf markers)
NO_CAPTURE = {"head_top"}

# per-bone influence multiplier -- keeps the torso from stealing limb vertices
BONE_GAIN = {
    "root": 1.00, "spine_01": 1.00, "spine_02": 1.00, "spine_03": 0.95,
    "neck": 0.90, "head": 1.10,
    "clavicle_l": 0.55, "clavicle_r": 0.55,
    "shoulder_l": 1.25, "shoulder_r": 1.25,
    "elbow_l": 1.35, "elbow_r": 1.35,
    "wrist_l": 1.45, "wrist_r": 1.45,
    "hip_l": 1.15, "hip_r": 1.15,
    "knee_l": 1.30, "knee_r": 1.30,
    "ankle_l": 1.40, "ankle_r": 1.40,
}


def bone_list() -> List[Tuple[str, str]]:
    """(parent, child) pairs, parent is the joint that drives the bone."""
    return [(par, ch) for ch, par in HIERARCHY.items()
            if par is not None and ch not in NO_CAPTURE]


def _seg_distance(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-9:
        return np.linalg.norm(P - a, axis=1)
    t = np.clip((P - a) @ ab / L2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(P - proj, axis=1)


def skin_weights(verts: np.ndarray, joints: Dict[str, np.ndarray],
                 max_influences: int = 4, power: float = 4.0
                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Inverse-distance linear blend skinning weights against bone segments.

    Returns (indices (N,max_inf) int32, weights (N,max_inf) float32, joint order).
    The driving joint of each bone is the deforming joint.
    """
    bones = bone_list()
    names = list(joints.keys())
    jidx = {n: i for i, n in enumerate(names)}

    D = np.empty((verts.shape[0], len(bones)), dtype=np.float64)
    for bi, (par, ch) in enumerate(bones):
        d = _seg_distance(verts, joints[par], joints[ch])
        D[:, bi] = d / max(BONE_GAIN.get(ch, 1.0), 1e-3)

    W = 1.0 / np.power(D + 0.75, power)
    order = np.argsort(-W, axis=1)[:, :max_influences]
    rows = np.arange(verts.shape[0])[:, None]
    w = W[rows, order]
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)

    # map bone index -> deforming joint index (the child joint of the bone)
    bone_joint = np.array([jidx[ch] for (_, ch) in bones], dtype=np.int32)
    idx = bone_joint[order].astype(np.int32)
    return idx, w.astype(np.float32), names


# ---------------------------------------------------------------------------
# posing
# ---------------------------------------------------------------------------
def _rot(axis_angle_deg) -> np.ndarray:
    v = np.asarray(axis_angle_deg, float)
    ang = np.linalg.norm(v)
    if ang < 1e-9:
        return np.eye(3)
    k = v / ang
    a = np.radians(ang)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def global_transforms(joints: Dict[str, np.ndarray],
                      pose: Dict[str, Tuple[float, float, float]] | None = None
                      ) -> Dict[str, np.ndarray]:
    """4x4 global transforms for a pose given as per-joint axis-angle (degrees)."""
    pose = pose or {}
    out: Dict[str, np.ndarray] = {}

    def solve(name: str) -> np.ndarray:
        if name in out:
            return out[name]
        par = HIERARCHY[name]
        R = _rot(pose.get(name, (0.0, 0.0, 0.0)))
        M = np.eye(4)
        if par is None:
            M[:3, :3] = R
            M[:3, 3] = joints[name]
        else:
            Mp = solve(par)
            local = joints[name] - joints[par]
            M[:3, :3] = Mp[:3, :3] @ R
            M[:3, 3] = Mp[:3, 3] + Mp[:3, :3] @ local
        out[name] = M
        return M

    for n in joints:
        solve(n)
    return out


def apply_pose(verts: np.ndarray, joints: Dict[str, np.ndarray],
               idx: np.ndarray, w: np.ndarray, names: List[str],
               pose: Dict[str, Tuple[float, float, float]]) -> np.ndarray:
    G = global_transforms(joints, pose)
    R = np.stack([G[n][:3, :3] for n in names])
    T = np.stack([G[n][:3, 3] for n in names])
    J = np.stack([joints[n] for n in names])

    out = np.zeros_like(verts)
    for s in range(idx.shape[1]):
        ji = idx[:, s]
        local = verts - J[ji]
        moved = np.einsum("nij,nj->ni", R[ji], local) + T[ji]
        out += w[:, s][:, None] * moved
    return out


POSE_PRESETS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "a_pose": {},
    "t_pose": {"shoulder_l": (0, 0, -45), "shoulder_r": (0, 0, 45)},
    "relaxed": {"shoulder_l": (0, 0, 33), "shoulder_r": (0, 0, -33),
                "elbow_l": (0, 0, -8), "elbow_r": (0, 0, 8)},
    "stride": {"hip_l": (-22, 0, 0), "knee_l": (14, 0, 0),
               "hip_r": (18, 0, 0), "knee_r": (-26, 0, 0),
               "shoulder_l": (16, 0, 30), "shoulder_r": (-16, 0, -30),
               "elbow_l": (-26, 0, 0), "elbow_r": (-20, 0, 0)},
    "sit": {"hip_l": (-88, 0, 0), "hip_r": (-88, 0, 0),
            "knee_l": (86, 0, 0), "knee_r": (86, 0, 0),
            "shoulder_l": (0, 0, 28), "shoulder_r": (0, 0, -28)},
}
