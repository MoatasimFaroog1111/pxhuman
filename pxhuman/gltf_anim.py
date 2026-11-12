"""
Read a GLB, append an animation, write it back.

Works directly on the glTF JSON and binary chunk rather than going through a
scene library, because morph-target weight animation plus an existing skin is
exactly the combination most importers round-trip badly. Everything already in
the file — skin, joints, materials, textures, morph targets — is left byte for
byte as it was; only new accessors, bufferViews and one animation are added.
"""
from __future__ import annotations

import json
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

GLB_MAGIC = 0x46546C67
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942

COMP_FLOAT = 5126


class Glb:
    def __init__(self, path):
        data = Path(path).read_bytes()
        magic, ver, total = struct.unpack_from("<III", data, 0)
        if magic != GLB_MAGIC:
            raise ValueError("not a GLB")
        off = 12
        self.json = None
        self.bin = bytearray()
        while off < total:
            clen, ctype = struct.unpack_from("<II", data, off)
            chunk = data[off + 8: off + 8 + clen]
            if ctype == JSON_CHUNK:
                self.json = json.loads(chunk.decode("utf-8"))
            elif ctype == BIN_CHUNK:
                self.bin = bytearray(chunk)
            off += 8 + clen + (-clen % 4)
        if self.json is None:
            raise ValueError("GLB has no JSON chunk")

    # ------------------------------------------------------------------
    def mesh_index_by_name(self, name: str) -> int | None:
        """Find a mesh by name.

        Exporters are inconsistent about where the readable name lives: it may
        be on the mesh, or only on the node that references it. MetaPerson puts
        it on the node, so a mesh-only lookup silently finds nothing.
        """
        for i, m in enumerate(self.json.get("meshes", [])):
            if m.get("name") == name:
                return i
        for n in self.json.get("nodes", []):
            if n.get("name") == name and "mesh" in n:
                return n["mesh"]
        return None

    def node_by_name_with_mesh(self, name: str) -> int | None:
        for i, n in enumerate(self.json.get("nodes", [])):
            if n.get("name") == name and "mesh" in n:
                return i
        return None

    def node_for_mesh(self, mesh_index: int) -> int | None:
        for i, n in enumerate(self.json.get("nodes", [])):
            if n.get("mesh") == mesh_index:
                return i
        return None

    def node_index_by_name(self, name: str) -> int | None:
        for i, n in enumerate(self.json.get("nodes", [])):
            if n.get("name") == name:
                return i
        return None

    def target_names(self, mesh_index: int) -> List[str]:
        m = self.json["meshes"][mesh_index]
        ex = m.get("extras") or {}
        names = ex.get("targetNames")
        if names:
            return list(names)
        n = len(m["primitives"][0].get("targets", []))
        return [f"target_{i}" for i in range(n)]

    def num_targets(self, mesh_index: int) -> int:
        return len(self.json["meshes"][mesh_index]["primitives"][0].get("targets", []))

    # ------------------------------------------------------------------
    def _add_buffer_view(self, payload: bytes) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        offset = len(self.bin)
        self.bin.extend(payload)
        while len(self.bin) % 4:
            self.bin.append(0)
        bvs = self.json.setdefault("bufferViews", [])
        bvs.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload)})
        return len(bvs) - 1

    def add_accessor(self, array: np.ndarray, atype: str) -> int:
        a = np.ascontiguousarray(array, dtype=np.float32)
        bv = self._add_buffer_view(a.tobytes())
        ncomp = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[atype]
        count = a.size // ncomp
        flat = a.reshape(count, ncomp)
        acc = {
            "bufferView": bv, "componentType": COMP_FLOAT, "count": int(count),
            "type": atype,
            "min": [float(x) for x in flat.min(axis=0)],
            "max": [float(x) for x in flat.max(axis=0)],
        }
        accs = self.json.setdefault("accessors", [])
        accs.append(acc)
        return len(accs) - 1

    # ------------------------------------------------------------------
    def add_animation(self, name: str, tracks: List[dict]) -> None:
        """tracks: [{node, path, times (N,), values (N,k)}]"""
        samplers, channels = [], []
        for t in tracks:
            atype = {"weights": "SCALAR", "rotation": "VEC4",
                     "translation": "VEC3", "scale": "VEC3"}[t["path"]]
            ti = self.add_accessor(np.asarray(t["times"], np.float32), "SCALAR")
            vi = self.add_accessor(np.asarray(t["values"], np.float32), atype)
            samplers.append({"input": ti, "output": vi, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1,
                             "target": {"node": int(t["node"]), "path": t["path"]}})
        anims = self.json.setdefault("animations", [])
        anims.append({"name": name, "samplers": samplers, "channels": channels})

    # ------------------------------------------------------------------
    def save(self, path) -> Path:
        self.json.setdefault("buffers", [{}])
        self.json["buffers"][0]["byteLength"] = len(self.bin)
        self.json["buffers"][0].pop("uri", None)

        jb = json.dumps(self.json, separators=(",", ":")).encode("utf-8")
        jb += b" " * (-len(jb) % 4)
        bb = bytes(self.bin)
        bb += b"\0" * (-len(bb) % 4)

        total = 12 + 8 + len(jb) + 8 + len(bb)
        out = bytearray()
        out += struct.pack("<III", GLB_MAGIC, 2, total)
        out += struct.pack("<II", len(jb), JSON_CHUNK) + jb
        out += struct.pack("<II", len(bb), BIN_CHUNK) + bb
        p = Path(path)
        p.write_bytes(bytes(out))
        return p


# ---------------------------------------------------------------------------
def quat_from_euler(pitch, yaw, roll) -> np.ndarray:
    """XYZ euler (radians) -> xyzw quaternions, vectorised over frames."""
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    x = sp * cy * cr + cp * sy * sr
    y = cp * sy * cr - sp * cy * sr
    z = cp * cy * sr - sp * sy * cr
    w = cp * cy * cr + sp * sy * sr
    return np.stack([x, y, z, w], axis=1)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, xyzw, broadcasting over frames."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)
