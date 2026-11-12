"""
Load a skeletal animation from one glTF and retarget it onto another.

Bones are matched by name. MetaPerson uses Mixamo's names (Hips, Spine,
Spine1, Spine2, Neck, Head, LeftArm, LeftForeArm, ...), and Mixamo's
"mixamorig:" prefix is stripped, so a Mixamo clip matches with no mapping.

Matching names is not the same as matching rotations. A glTF node's
rotation is relative to its parent *and to the bone's own axis convention*,
and that convention differs between rigs: MetaPerson's thigh carries a ~177
degree rest rotation because its bone axis points down the leg, while a rig
built with identity rests carries none. Copying local rotations by name
therefore puts the clip's numbers into the wrong frame -- on MetaPerson it
turned both legs upside down.

So the clip is transferred as motion in world space, not as numbers:

    delta(t)  = Ws(t) * Ws_rest^-1            how the source bone moved
    Wt(t)     = delta(t) * A * Wt_rest        same motion on the target
    local(t)  = Wt_parent(t)^-1 * Wt(t)       back into the target hierarchy

A aligns the two rest *poses*, and only where they are really different
poses. Two rigs never share an anatomy: against Mixamo, MetaPerson's spine
is curved 19 degrees further, its foot bone points 22 degrees lower and its
clavicles 37 degrees higher. Those are shapes, and forcing them to the
source's directions bends the avatar into the source's body -- the feet
alone tipped the toes 4.6 cm through the floor. A pose difference (arms in
an A-pose against a T-pose, legs spread) shows up at the root of a limb, so
alignment is applied there only, at the upper arm and thigh, and only past
ALIGN_MIN_DEG; the rest of the limb inherits it. Mixamo on MetaPerson needs
none; an A-pose clip gets its arms aligned and nothing else. Twist about
the bone is carried by the delta.

Hip translation (--root-motion) is the source hips' displacement from their
own rest position, scaled by the ratio of leg lengths, added to the
target's rest position. Copying the absolute value is what dropped the
avatar 94 cm into the floor: a clip's hips sit wherever its rig was built,
often at the origin, and in its own units.
"""
from __future__ import annotations

import numpy as np
import re
from typing import Dict, Tuple

from .gltf_anim import Glb
from .skeleton import Skeleton, quat_to_mat, mat_to_quat, arc_between, angle_deg

PREFIX = re.compile(r"^(mixamorig\d*:|Armature\|)", re.IGNORECASE)


def canonical(name: str) -> str:
    return PREFIX.sub("", name or "").strip()


def _read(g: Glb, idx: int) -> np.ndarray:
    import struct
    acc = g.json["accessors"][idx]
    ncomp = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc["type"]]
    fmt, size = {5126: ("f", 4), 5123: ("H", 2), 5125: ("I", 4),
                 5121: ("B", 1), 5122: ("h", 2), 5120: ("b", 1)}[acc["componentType"]]
    bv = g.json["bufferViews"][acc["bufferView"]]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or ncomp * size
    out = np.empty((acc["count"], ncomp), np.float32)
    for i in range(acc["count"]):
        out[i] = struct.unpack_from("<" + fmt * ncomp, g.bin, base + i * stride)
    return out if ncomp > 1 else out[:, 0]


class BodyClip:
    """Per-bone rotation (and hip translation) tracks sampled on a shared clock."""

    def __init__(self, path, animation: int | str = 0):
        g = Glb(path)
        self.src = g
        anims = g.json.get("animations", [])
        if not anims:
            raise ValueError(f"{path} contains no animation")
        if isinstance(animation, str):
            idx = next((i for i, a in enumerate(anims)
                        if a.get("name") == animation), 0)
        else:
            idx = min(animation, len(anims) - 1)
        self.anim = anims[idx]
        self.name = self.anim.get("name") or f"clip{idx}"

        self.rot: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.trans: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        nodes = g.json["nodes"]
        for ch in self.anim["channels"]:
            tgt = ch["target"]
            node = nodes[tgt["node"]]
            bone = canonical(node.get("name", ""))
            if not bone:
                continue
            smp = self.anim["samplers"][ch["sampler"]]
            t = _read(g, smp["input"])
            v = _read(g, smp["output"])
            if tgt["path"] == "rotation":
                if v.ndim == 2 and v.shape[0] == len(t) * 3:      # CUBICSPLINE
                    v = v[1::3]
                self.rot[bone] = (t, v)
            elif tgt["path"] == "translation":
                if v.ndim == 2 and v.shape[0] == len(t) * 3:
                    v = v[1::3]
                self.trans[bone] = (t, v)

        allt = [t for t, _ in list(self.rot.values()) + list(self.trans.values())]
        self.duration = float(max((t[-1] for t in allt if len(t)), default=0.0))

    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"<BodyClip '{self.name}' {len(self.rot)} rot bones, "
                f"{self.duration:.2f}s>")

    def sample_rotation(self, bone: str, times: np.ndarray,
                        loop: bool = True) -> np.ndarray | None:
        """Quaternions at `times`, looping the clip if it is shorter."""
        if bone not in self.rot:
            return None
        src_t, src_v = self.rot[bone]
        if len(src_t) == 0:
            return None
        t = np.mod(times, self.duration) if (loop and self.duration > 0) else \
            np.clip(times, src_t[0], src_t[-1])
        idx = np.clip(np.searchsorted(src_t, t) - 1, 0, len(src_t) - 2)
        t0, t1 = src_t[idx], src_t[idx + 1]
        a = np.clip((t - t0) / np.maximum(t1 - t0, 1e-8), 0, 1)[:, None]
        q0, q1 = src_v[idx], src_v[idx + 1]
        # shortest-arc slerp, done as normalised lerp: with dense keys the
        # error is far below what the eye resolves, and it cannot flip
        flip = (q0 * q1).sum(axis=1, keepdims=True) < 0
        q1 = np.where(flip, -q1, q1)
        q = q0 * (1 - a) + q1 * a
        return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)

    def sample_translation(self, bone: str, times: np.ndarray,
                           loop: bool = True) -> np.ndarray | None:
        if bone not in self.trans:
            return None
        src_t, src_v = self.trans[bone]
        t = np.mod(times, self.duration) if (loop and self.duration > 0) else \
            np.clip(times, src_t[0], src_t[-1])
        idx = np.clip(np.searchsorted(src_t, t) - 1, 0, len(src_t) - 2)
        t0, t1 = src_t[idx], src_t[idx + 1]
        a = np.clip((t - t0) / np.maximum(t1 - t0, 1e-8), 0, 1)[:, None]
        return src_v[idx] * (1 - a) + src_v[idx + 1] * a

    def bones(self):
        return sorted(self.rot.keys())


def match_report(clip: BodyClip, target: Glb) -> Tuple[list, list]:
    """Which of the clip's bones exist on the target skeleton."""
    have = {canonical(n.get("name", "")) for n in target.json.get("nodes", [])}
    matched = [b for b in clip.bones() if b in have]
    missing = [b for b in clip.bones() if b not in have]
    return matched, missing


# ---------------------------------------------------------------------------
LEG = ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot"]

# where a rest *pose* difference shows (see module docstring); below this
# angle a difference is anatomy -- bone placement -- and is left alone
LIMB_ROOTS = ("LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg")
ALIGN_MIN_DEG = 10.0


def _continuous(q: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so consecutive keys never take the long way."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


def _chain_length(P: np.ndarray, idx: Dict[str, int]) -> float | None:
    if not all(b in idx for b in LEG):
        return None
    pts = [P[idx[b]] for b in LEG]
    return float(sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(3)))


class Retarget:
    """Transfer a BodyClip onto a target skeleton through world space."""

    def __init__(self, clip: BodyClip, target: Glb, align: bool = True):
        self.clip = clip
        self.src = Skeleton(clip.src)
        self.tgt = Skeleton(target)
        self.s_idx: Dict[str, int] = {}
        for i, nm in enumerate(self.src.names):
            c = canonical(nm)
            if c:
                self.s_idx.setdefault(c, i)
        self.t_idx: Dict[str, int] = {}
        for i, nm in enumerate(self.tgt.names):
            c = canonical(nm)
            if c:
                self.t_idx.setdefault(c, i)
        # only bones that are joints of the target's skin are worth driving;
        # name collisions with mesh nodes ("Head" vs "AvatarHead") stay out
        skin_j = set()
        for sk in target.json.get("skins", []):
            skin_j.update(sk["joints"])
        self.pairs: Dict[int, int] = {
            ti: self.s_idx[n] for n, ti in self.t_idx.items()
            if n in self.s_idx and (not skin_j or ti in skin_j)}

        RS0, PS0, self.SS0 = self.src.world()
        RT0, PT0, self.ST0 = self.tgt.world()
        self.RS0, self.PS0 = RS0[0], PS0[0]
        self.RT0, self.PT0 = RT0[0], PT0[0]

        # rest alignment, per target bone, applied at limb roots only
        self.A: Dict[int, np.ndarray] = {}
        self.mismatch: Dict[str, float] = {}     # aligned: limb root -> deg
        self.shape: Dict[str, float] = {}        # left alone: bone -> deg
        I = np.eye(3)
        for ti in self.tgt.order:
            if ti not in self.pairs:
                continue
            si = self.pairs[ti]
            name = canonical(self.tgt.names[ti])
            p = self.tgt.parent[ti]             # nearest aligned ancestor
            while p >= 0 and p not in self.A:
                p = self.tgt.parent[p]
            Ap = self.A[p] if p >= 0 else I
            A = Ap
            if name in LIMB_ROOTS:
                Rres = self._fit_limb(ti, Ap)
                d = float(angle_deg(Rres)) if Rres is not None else 0.0
                if align and d >= ALIGN_MIN_DEG:
                    A = Rres @ Ap
                    self.mismatch[name] = d
            kids = [c for c in self.tgt.children(ti) if c in self.pairs]
            if len(kids) == 1:
                c = kids[0]
                dt = self.PT0[c] - self.PT0[ti]
                ds = self.PS0[self.pairs[c]] - self.PS0[si]
                if np.linalg.norm(dt) > 1e-6 and np.linalg.norm(ds) > 1e-9:
                    d = float(angle_deg(arc_between(A @ dt, ds)))
                    if d >= 1.0:
                        self.shape[name] = d
            self.A[ti] = A

        ls = _chain_length(self.PS0, self.s_idx)
        lt = _chain_length(self.PT0, self.t_idx)
        if ls and lt and ls > 1e-9:
            self.scale = lt / ls
        else:
            hs = self.PS0[self.s_idx["Hips"], 1] if "Hips" in self.s_idx else 0
            ht = self.PT0[self.t_idx["Hips"], 1] if "Hips" in self.t_idx else 0
            self.scale = ht / hs if abs(hs) > 1e-9 else 1.0

    def _fit_limb(self, root: int, Ap: np.ndarray) -> np.ndarray | None:
        """Best rotation taking the target limb's rest onto the source's.

        Fitted over every matched joint below the limb root (Kabsch, joint
        offsets from the root as unit vectors weighted by length), not just
        the first bone: a single direction leaves the twist about that bone
        free, and the hand and fingers would land rotated around the arm.
        For a limb that differs by a rigid rotation the fit is exact; where
        the two differ in shape it averages the differences out.
        """
        si = self.pairs[root]
        H = np.zeros((3, 3))
        stack = list(self.tgt.children(root))
        n = 0
        while stack:
            j = stack.pop()
            stack.extend(self.tgt.children(j))
            if j not in self.pairs:
                continue
            vt = Ap @ (self.PT0[j] - self.PT0[root])
            vs = self.PS0[self.pairs[j]] - self.PS0[si]
            lt, ls = np.linalg.norm(vt), np.linalg.norm(vs)
            if lt < 1e-6 or ls < 1e-9:
                continue
            H += lt * np.outer(vt / lt, vs / ls)
            n += 1
        if n == 0:
            return None
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
        return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    # ------------------------------------------------------------------
    def solve(self, times: np.ndarray, loop: bool = True,
              root_motion: bool = False, min_deg: float = 0.01):
        """Returns ({bone: quats (n,4)}, {bone: translations (n,3)})."""
        n = len(times)
        src_rot = {}
        for bone in self.clip.rot:
            si = self.s_idx.get(bone)
            q = self.clip.sample_rotation(bone, times, loop=loop)
            if si is not None and q is not None:
                src_rot[si] = quat_to_mat(q)
        src_trans = {}
        hip = "Hips"
        if root_motion and hip in self.clip.trans and hip in self.s_idx:
            src_trans[self.s_idx[hip]] = self.clip.sample_translation(
                hip, times, loop=loop)
        RS, PS, _ = self.src.world(src_rot, src_trans, n)

        G: Dict[int, np.ndarray] = {}
        local: Dict[int, np.ndarray] = {}
        for ti in self.tgt.order:
            p = self.tgt.parent[ti]
            if ti in self.pairs:
                si = self.pairs[ti]
                Gi = RS[:, si] @ (self.RS0[si].T @ self.A[ti] @ self.RT0[ti])
                L = Gi if p < 0 else np.swapaxes(G[p], 1, 2) @ Gi
                local[ti] = L
                G[ti] = Gi
            else:
                G[ti] = (np.broadcast_to(self.tgt.R[ti], (n, 3, 3)) if p < 0
                         else G[p] @ self.tgt.R[ti])

        rot_out: Dict[str, np.ndarray] = {}
        for ti, L in local.items():
            dev = angle_deg(np.swapaxes(L, 1, 2) @ self.tgt.R[ti]).max()
            if dev < min_deg:
                continue
            rot_out[canonical(self.tgt.names[ti])] = _continuous(
                mat_to_quat(L).astype(np.float32))

        trans_out: Dict[str, np.ndarray] = {}
        if src_trans and hip in self.t_idx:
            si, ti = self.s_idx[hip], self.t_idx[hip]
            dp = (PS[:, si] - self.PS0[si]) * self.scale
            want = self.PT0[ti] + dp
            p = self.tgt.parent[ti]
            if p < 0:
                tl = want
            else:
                Rp, Pp = G[p], None
                # parent position: the target's hips parent is not driven
                # by the clip in any rig this targets, so its rest stands
                _, PW, SW = self.tgt.world()
                Pp = PW[0, p]
                tl = np.einsum("nji,nj->ni", Rp, want - Pp) / SW[p]
            trans_out[hip] = tl.astype(np.float32)
        return rot_out, trans_out

    def report(self) -> str:
        """One line on the rest poses: what was aligned, and why not more."""
        if self.mismatch:
            s = "aligned " + ", ".join(
                f"{b} {d:.0f}deg" for b, d in
                sorted(self.mismatch.items(), key=lambda kv: -kv[1]))
        else:
            s = "same pose, nothing to align"
        if self.shape:
            s += (f" ({len(self.shape)} bones differ in shape only, "
                  f"up to {max(self.shape.values()):.0f}deg; kept as the "
                  f"avatar's own)")
        return s
