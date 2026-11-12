#!/usr/bin/env python3
"""
Verify a talking animation by evaluating it and rendering the mouth.

Writing a file without an exception is not evidence that the animation does
anything. This reads the morph targets back, applies the animated weights at
several times, and renders the result, so the mouth can be seen moving.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pxhuman.gltf_anim import Glb                      # noqa: E402

CTYPE = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2),
         5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_accessor(g: Glb, idx: int) -> np.ndarray:
    acc = g.json["accessors"][idx]
    n = acc["count"]
    nc = NCOMP[acc["type"]]
    fmt, size = CTYPE[acc["componentType"]]
    bv = g.json["bufferViews"][acc["bufferView"]]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride") or nc * size
    out = np.empty((n, nc), np.float32)
    buf = g.bin
    for i in range(n):
        off = base + i * stride
        out[i] = struct.unpack_from("<" + fmt * nc, buf, off)
    return out if nc > 1 else out[:, 0]


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "out/talking.glb")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "out/talk_check.png")
    g = Glb(src)

    mi = g.mesh_index_by_name("AvatarHead")
    node = g.node_by_name_with_mesh("AvatarHead")
    prim = g.json["meshes"][mi]["primitives"][0]
    base = read_accessor(g, prim["attributes"]["POSITION"])
    tri = read_accessor(g, prim["indices"]).astype(np.int64).reshape(-1, 3)
    names = g.target_names(mi)
    print(f"AvatarHead node {node}: {len(base):,} verts, {len(names)} targets")

    anim = g.json["animations"][0]
    ch = next(c for c in anim["channels"]
              if c["target"]["node"] == node and c["target"]["path"] == "weights")
    smp = anim["samplers"][ch["sampler"]]
    times = read_accessor(g, smp["input"])
    W = read_accessor(g, smp["output"]).reshape(len(times), len(names))
    print(f"animation '{anim['name']}': {len(times)} frames, "
          f"{times[0]:.2f}..{times[-1]:.2f}s")

    peak = W.max(axis=0)
    active = [(names[i], float(peak[i])) for i in np.argsort(-peak)[:10]]
    print("\ntop driven targets (peak weight):")
    for nm, v in active:
        print(f"   {nm:18s} {v:.3f}")
    if peak.max() < 1e-3:
        print("\nFAIL: no target ever leaves zero")
        return 1

    # lazily load only the targets that actually move
    moving = np.where(peak > 0.02)[0]
    deltas = {}
    for i in moving:
        t = prim["targets"][i]
        if "POSITION" in t:
            deltas[i] = read_accessor(g, t["POSITION"])
    print(f"\nloaded {len(deltas)} moving targets")

    # pick frames spread across the clip
    picks = np.linspace(0, len(times) - 1, 6).astype(int)
    scale = 1.0 / (base[:, 1].max() - base[:, 1].min())
    tiles = []
    for fi in picks:
        V = base.copy()
        for i, d in deltas.items():
            w = W[fi, i]
            if w > 1e-3:
                V += d * w
        tiles.append(render_mouth(V, tri, times[fi], W[fi], names))

    H, Wd = tiles[0].shape[:2]
    pad = 6
    sheet = np.full((H + 2 * pad, (Wd + pad) * len(tiles) + pad, 3), 248, np.uint8)
    for i, t in enumerate(tiles):
        sheet[pad:pad + H, pad + i * (Wd + pad): pad + i * (Wd + pad) + Wd] = t
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(dst)
    print(f"wrote {dst}")
    return body_check(g, anim)


HEAD_ONLY = {"Head", "Neck", "Neck1", "Neck2"}


def body_check(g: Glb, anim) -> int:
    """Pose the skeleton from the file and check the body is still a body.

    The mouth check above cannot see the body: a file whose legs point at
    the sky passes it. This poses every joint from the animation, skins the
    meshes, and checks the figure stays upright and on its floor.
    """
    from pxhuman.skeleton import Skeleton, animation_overrides, skinned_vertices
    from pxhuman.bodyanim import canonical

    nodes = g.json["nodes"]
    bones = {canonical(nodes[c["target"]["node"]].get("name", ""))
             for c in anim["channels"] if c["target"]["path"] != "weights"}
    if not (bones - HEAD_ONLY):
        print("\nbody: head and neck only, nothing to check")
        return 0
    sk = Skeleton(g)
    tmax = max(read_accessor(g, anim["samplers"][c["sampler"]]["input"])[-1]
               for c in anim["channels"])
    times = np.linspace(0, tmax, 24).astype(np.float32)
    rot, trans = animation_overrides(g, sk, times)
    _, P, _ = sk.world(rot, trans, len(times))
    V = skinned_vertices(g, sk, rot, trans, len(times), stride=5)
    V0 = skinned_vertices(g, sk, stride=5)[0]

    idx = {canonical(n): i for i, n in enumerate(sk.names)}
    fails = []
    if all(b in idx for b in ("Hips", "LeftFoot", "RightFoot", "Head")):
        hy = P[:, idx["Hips"], 1]
        if not (np.all(P[:, idx["LeftFoot"], 1] < hy) and
                np.all(P[:, idx["RightFoot"], 1] < hy)):
            fails.append("feet above the hips")
        if not np.all(P[:, idx["Head"], 1] > hy):
            fails.append("head below the hips")
    h0, f0 = V0[:, 1].max() - V0[:, 1].min(), V0[:, 1].min()
    floor = V[:, :, 1].min(1)
    height = V[:, :, 1].max(1) - floor
    if floor.min() < f0 - 0.05 * h0:
        fails.append(f"sinks {100*(f0-floor.min()):.0f} cm through the floor")
    if height.min() < 0.5 * h0:
        fails.append(f"collapsed to {height.min():.2f} m")
    if height.max() > 1.25 * h0:
        fails.append("stretched beyond its rest height")

    print(f"\nbody: {len(bones - HEAD_ONLY)} bones animated, "
          f"height {height.min():.2f}-{height.max():.2f} m (rest {h0:.2f}), "
          f"lowest point {floor.min():+.3f} m (rest {f0:+.3f})")
    if fails:
        print("FAIL: " + "; ".join(fails))
        return 1
    print("body OK: upright, on the floor")
    return 0


def render_mouth(V, tri, t, w, names, W_=380, H_=380):
    """Flat-shaded close-up of the lower face."""
    n = np.zeros_like(V)
    a = V[tri[:, 0]]; b = V[tri[:, 1]]; c = V[tri[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, tri[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-9)

    ymin, ymax = V[:, 1].min(), V[:, 1].max()
    hh = ymax - ymin
    cy = ymin + hh * 0.30          # mouth sits low on the head
    half = hh * 0.30
    cx = V[:, 0].mean()

    sc = H_ / (2 * half)
    xs = ((V[:, 0] - cx) * sc + W_ / 2).astype(np.int32)
    ys = (H_ / 2 - (V[:, 1] - cy) * sc).astype(np.int32)
    light = np.array([0.3, 0.45, 0.84]); light /= np.linalg.norm(light)
    sh = 0.25 + 0.75 * np.clip(n @ light, 0, 1)
    col = np.clip(sh[:, None] * np.array([0.86, 0.74, 0.69]) * 255, 0, 255)

    SX, SY, SZ, SC = [], [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            SX.append(xs + dx); SY.append(ys + dy)
            SZ.append(V[:, 2]); SC.append(col)
    X = np.concatenate(SX); Y = np.concatenate(SY)
    Z = np.concatenate(SZ); C = np.concatenate(SC)
    ok = (X >= 0) & (X < W_) & (Y >= 0) & (Y < H_)
    X, Y, Z, C = X[ok], Y[ok], Z[ok], C[ok]
    o = np.argsort(Z)
    img = np.full((H_, W_, 3), 248.0)
    img[Y[o], X[o]] = C[o]
    return img.astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
