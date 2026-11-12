"""
Node hierarchy of a glTF: rest pose, posed world transforms, skinning.

Retargeting has to reason about where bones point in the world, not about
the numbers stored on each node. Two rigs can share every bone name and
still store completely different local rotations for the same visible pose:
MetaPerson's thigh carries a ~177 degree rest rotation because its bone axis
points down, a rig built with identity rests carries none. Copying one set
of local numbers onto the other is what turned the legs upside down.

Everything here is vectorised over frames: world rotations are (N, K, 3, 3)
for N frames and K nodes.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .gltf_anim import Glb

_CTYPE = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
          5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_accessor(g: Glb, idx: int) -> np.ndarray:
    acc = g.json["accessors"][idx]
    n, nc = acc["count"], _NCOMP[acc["type"]]
    fmt, size = _CTYPE[acc["componentType"]]
    bv = g.json["bufferViews"][acc["bufferView"]]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or nc * size
    dt = np.dtype("<" + fmt)
    if stride == nc * size:
        out = np.frombuffer(bytes(g.bin[base: base + n * stride]), dt).reshape(n, nc)
    else:
        out = np.stack([np.frombuffer(bytes(g.bin[base + i * stride:
                                                  base + i * stride + nc * size]), dt)
                        for i in range(n)])
    out = out.astype(np.float32)
    if acc.get("normalized") and fmt != "f":
        out /= float(np.iinfo(dt).max)
    return out if nc > 1 else out[:, 0]


# ---------------------------------------------------------------------------
# rotations (glTF quaternion order is x, y, z, w)

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrices -> unit quaternions (xyzw), w kept non-negative."""
    m = np.asarray(m, np.float64)
    shp = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    # largest of 4w^2, 4x^2, 4y^2, 4z^2 picks the numerically safe branch
    cand = np.stack([1 + tr,
                     1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
                     1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                     1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], 1)
    k = np.argmax(cand, 1)
    s = np.sqrt(np.maximum(cand[np.arange(len(m)), k], 1e-12)) * 2
    q = np.empty((len(m), 4))
    a = k == 0
    q[a, 3] = s[a] / 4
    q[a, 0] = (m[a, 2, 1] - m[a, 1, 2]) / s[a]
    q[a, 1] = (m[a, 0, 2] - m[a, 2, 0]) / s[a]
    q[a, 2] = (m[a, 1, 0] - m[a, 0, 1]) / s[a]
    a = k == 1
    q[a, 0] = s[a] / 4
    q[a, 3] = (m[a, 2, 1] - m[a, 1, 2]) / s[a]
    q[a, 1] = (m[a, 0, 1] + m[a, 1, 0]) / s[a]
    q[a, 2] = (m[a, 0, 2] + m[a, 2, 0]) / s[a]
    a = k == 2
    q[a, 1] = s[a] / 4
    q[a, 3] = (m[a, 0, 2] - m[a, 2, 0]) / s[a]
    q[a, 0] = (m[a, 0, 1] + m[a, 1, 0]) / s[a]
    q[a, 2] = (m[a, 1, 2] + m[a, 2, 1]) / s[a]
    a = k == 3
    q[a, 2] = s[a] / 4
    q[a, 3] = (m[a, 1, 0] - m[a, 0, 1]) / s[a]
    q[a, 0] = (m[a, 0, 2] + m[a, 2, 0]) / s[a]
    q[a, 1] = (m[a, 1, 2] + m[a, 2, 1]) / s[a]
    q *= np.where(q[:, 3:4] < 0, -1.0, 1.0)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(shp + (4,))


def arc_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest rotation taking direction a onto direction b (3x3)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:                       # opposite: any perpendicular axis
        p = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def angle_deg(R: np.ndarray) -> np.ndarray:
    """Rotation angle of matrices, degrees."""
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


# ---------------------------------------------------------------------------
def _decompose(M: np.ndarray):
    t = M[:3, 3].copy()
    A = M[:3, :3]
    s = np.linalg.norm(A, axis=0)
    R = A / np.maximum(s, 1e-12)
    # re-orthonormalise; exporters write matrices with float noise
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    return t, R, float(np.mean(s))


class Skeleton:
    """Rest pose and forward kinematics for every node of a glTF.

    Scale is treated as uniform per node, which holds for every rig this is
    aimed at (a centimetre armature under a 0.01 root is the common case).

    use_bind: take the rest rotation of skin joints from the inverse bind
    matrices rather than from the node TRS. The bind pose is the pose the
    mesh was built in; a file saved mid-animation can carry posed node TRS,
    and then the node values are not a rest pose at all.
    """

    def __init__(self, g: Glb, use_bind: bool = True):
        self.g = g
        nodes = g.json.get("nodes", [])
        K = len(nodes)
        self.K = K
        self.names = [n.get("name", "") for n in nodes]
        self.parent = np.full(K, -1, int)
        for i, n in enumerate(nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self.t = np.zeros((K, 3))
        self.R = np.tile(np.eye(3), (K, 1, 1))
        self.s = np.ones(K)
        for i, n in enumerate(nodes):
            if "matrix" in n:
                M = np.array(n["matrix"], np.float64).reshape(4, 4).T
                self.t[i], self.R[i], self.s[i] = _decompose(M)
            else:
                self.t[i] = n.get("translation", [0, 0, 0])
                self.R[i] = quat_to_mat(n.get("rotation", [0, 0, 0, 1]))
                sc = n.get("scale", [1, 1, 1])
                self.s[i] = float(np.mean(sc))
        self.order = self._topo()
        self.rebound = 0
        if use_bind and g.json.get("skins"):
            self._rest_from_bind()

    def _topo(self) -> List[int]:
        order, seen = [], set()
        kids: Dict[int, list] = {}
        for i, p in enumerate(self.parent):
            kids.setdefault(int(p), []).append(i)
        stack = list(reversed(kids.get(-1, [])))
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            order.append(i)
            stack.extend(reversed(kids.get(i, [])))
        return order

    def _rest_from_bind(self):
        sk = self.g.json["skins"][0]
        joints = sk["joints"]
        if "inverseBindMatrices" not in sk:
            return
        ibm = read_accessor(self.g, sk["inverseBindMatrices"]).reshape(-1, 4, 4)
        ibm = ibm.transpose(0, 2, 1).astype(np.float64)       # column-major
        bind = {j: np.linalg.inv(ibm[k]) for k, j in enumerate(joints)}
        for j in joints:
            p = int(self.parent[j])
            if p not in bind:
                continue
            L = np.linalg.inv(bind[p]) @ bind[j]
            _, R, _ = _decompose(L)
            if angle_deg(R.T @ self.R[j]) > 0.05:
                self.rebound += 1
            self.R[j] = R

    # ------------------------------------------------------------------
    def index(self, canonical_name: str, canon) -> Optional[int]:
        for i, nm in enumerate(self.names):
            if canon(nm) == canonical_name:
                return i
        return None

    def children(self, i: int) -> List[int]:
        return [int(c) for c in np.where(self.parent == i)[0]]

    def world(self, rot: Optional[Dict[int, np.ndarray]] = None,
              trans: Optional[Dict[int, np.ndarray]] = None, n: int = 1):
        """World rotations (n,K,3,3), positions (n,K,3) and scales (K,).

        rot / trans override a node's local rotation (n,3,3) or
        translation (n,3) per frame; other nodes keep their rest values.
        """
        rot = rot or {}
        trans = trans or {}
        WR = np.empty((n, self.K, 3, 3))
        WP = np.empty((n, self.K, 3))
        WS = np.ones(self.K)
        for i in self.order:
            Rl = rot.get(i)
            if Rl is None:
                Rl = np.broadcast_to(self.R[i], (n, 3, 3))
            tl = trans.get(i)
            if tl is None:
                tl = np.broadcast_to(self.t[i], (n, 3))
            p = self.parent[i]
            if p < 0:
                WR[:, i] = Rl
                WP[:, i] = tl
                WS[i] = self.s[i]
            else:
                WR[:, i] = WR[:, p] @ Rl
                WP[:, i] = WP[:, p] + WS[p] * np.einsum("nij,nj->ni", WR[:, p], tl)
                WS[i] = WS[p] * self.s[i]
        return WR, WP, WS


# ---------------------------------------------------------------------------
def animation_overrides(g: Glb, sk: Skeleton, frames: np.ndarray,
                        anim: int = 0):
    """Local rotation / translation overrides of an animation at given times."""
    a = g.json["animations"][anim]
    rot, trans = {}, {}
    for c in a["channels"]:
        path = c["target"]["path"]
        if path not in ("rotation", "translation"):
            continue
        smp = a["samplers"][c["sampler"]]
        t = read_accessor(g, smp["input"])
        v = read_accessor(g, smp["output"])
        if smp.get("interpolation") == "CUBICSPLINE":
            v = v[1::3]
        idx = np.clip(np.searchsorted(t, frames, side="right") - 1, 0, len(t) - 1)
        idx2 = np.minimum(idx + 1, len(t) - 1)
        span = np.maximum(t[idx2] - t[idx], 1e-8)
        w = np.clip((frames - t[idx]) / span, 0, 1)[:, None]
        if smp.get("interpolation") == "STEP":
            w = np.zeros_like(w)
        a0, a1 = v[idx], v[idx2]
        if path == "rotation":
            a1 = np.where(((a0 * a1).sum(1, keepdims=True) < 0), -a1, a1)
            rot[c["target"]["node"]] = quat_to_mat(a0 * (1 - w) + a1 * w)
        else:
            trans[c["target"]["node"]] = a0 * (1 - w) + a1 * w
    return rot, trans


def skinned_vertices(g: Glb, sk: Skeleton, rot=None, trans=None, n: int = 1,
                     meshes: Optional[List[int]] = None, stride: int = 1):
    """Linear-blend-skinned vertex positions, (n, V, 3), over skinned meshes."""
    WR, WP, WS = sk.world(rot, trans, n)
    skin = g.json["skins"][0]
    joints = skin["joints"]
    ibm = read_accessor(g, skin["inverseBindMatrices"]).reshape(-1, 4, 4)
    ibm = ibm.transpose(0, 2, 1).astype(np.float64)
    J = np.empty((n, len(joints), 4, 4))
    for k, j in enumerate(joints):
        W = np.zeros((n, 4, 4))
        W[:, :3, :3] = WR[:, j] * WS[j]
        W[:, :3, 3] = WP[:, j]
        W[:, 3, 3] = 1
        J[:, k] = W @ ibm[k]
    out = []
    nodes = g.json["nodes"]
    for ni, nd in enumerate(nodes):
        if "mesh" not in nd or "skin" not in nd:
            continue
        if meshes is not None and nd["mesh"] not in meshes:
            continue
        for prim in g.json["meshes"][nd["mesh"]]["primitives"]:
            at = prim["attributes"]
            if "JOINTS_0" not in at:
                continue
            V = read_accessor(g, at["POSITION"])[::stride].astype(np.float64)
            jo = read_accessor(g, at["JOINTS_0"])[::stride].astype(int)
            we = read_accessor(g, at["WEIGHTS_0"])[::stride].astype(np.float64)
            Vh = np.c_[V, np.ones(len(V))]
            acc = np.zeros((n, len(V), 3))
            for k in range(4):
                M = J[:, jo[:, k]]                     # (n, V, 4, 4)
                acc += we[None, :, k:k + 1] * np.einsum("nvij,vj->nvi", M, Vh)[..., :3]
            out.append(acc)
    return np.concatenate(out, axis=1) if out else np.zeros((n, 0, 3))
