"""
Load a skeletal animation from one glTF and retarget it onto another.

Written for Mixamo clips, which is nearly a no-op case: MetaPerson's skeleton
uses Mixamo's own bone names (Hips, Spine, Spine1, Spine2, Neck, Head,
LeftArm, LeftForeArm, LeftHandThumb1 ...), so bones match by name and no
retargeting solve is needed. Mixamo's FBX exporter prefixes every bone with
"mixamorig:", which is stripped here.

Rotation is copied; translation is not, except optionally on the hips. Two
rigs that share bone names rarely share limb lengths or units, and copying
translation across them is what tears a character apart. Rotation-only
retargeting is scale-free and is the safe default.
"""
from __future__ import annotations

import numpy as np
import re
from typing import Dict, Tuple

from .gltf_anim import Glb

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
