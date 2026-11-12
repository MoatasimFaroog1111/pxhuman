#!/usr/bin/env python3
"""
Render a talking animation to an animated GIF.

    python preview_talk.py out/talking.glb out/talking.gif [--head|--full]

Evaluates the morph-target weights straight out of the written file, so what
you see is what the file contains rather than what the generator intended.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pxhuman.gltf_anim import Glb                      # noqa: E402
from verify_talk import read_accessor                  # noqa: E402

LIGHT = np.array([0.30, 0.45, 0.84])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
SKIN = np.array([0.88, 0.76, 0.71])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("out", nargs="?", default="out/talking.gif")
    ap.add_argument("--mesh", default="AvatarHead")
    ap.add_argument("--framing", choices=["head", "mouth"], default="head")
    ap.add_argument("--step", type=int, default=2, help="keep every Nth frame")
    ap.add_argument("--size", type=int, default=420)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    g = Glb(args.glb)
    mi = g.mesh_index_by_name(args.mesh)
    node = g.node_by_name_with_mesh(args.mesh)
    if mi is None or node is None:
        print(f"mesh '{args.mesh}' not found")
        return 1
    prim = g.json["meshes"][mi]["primitives"][0]
    base = read_accessor(g, prim["attributes"]["POSITION"])
    tri = read_accessor(g, prim["indices"]).astype(np.int64).reshape(-1, 3)
    names = g.target_names(mi)

    anim = g.json["animations"][0]
    ch = next((c for c in anim["channels"]
               if c["target"]["node"] == node and c["target"]["path"] == "weights"), None)
    if ch is None:
        print("no weight track for that mesh")
        return 1
    smp = anim["samplers"][ch["sampler"]]
    times = read_accessor(g, smp["input"])
    W = read_accessor(g, smp["output"]).reshape(len(times), len(names))

    peak = W.max(axis=0)
    moving = [i for i in np.where(peak > 0.02)[0] if "POSITION" in prim["targets"][i]]
    deltas = {i: read_accessor(g, prim["targets"][i]["POSITION"]) for i in moving}
    print(f"{args.mesh}: {len(base):,} verts, {len(names)} targets, "
          f"{len(deltas)} moving, {len(times)} frames")

    ymin, ymax = base[:, 1].min(), base[:, 1].max()
    hh = ymax - ymin
    if args.framing == "mouth":
        cy, half = ymin + hh * 0.30, hh * 0.30
    else:
        cy, half = ymin + hh * 0.42, hh * 0.46
    cx = base[:, 0].mean()

    S = args.size
    sc = S / (2 * half)
    frames = []
    for fi in range(0, len(times), max(1, args.step)):
        V = base.copy()
        for i, d in deltas.items():
            w = W[fi, i]
            if w > 1e-3:
                V += d * w
        frames.append(Image.fromarray(shade(V, tri, cx, cy, sc, S)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {out}  ({len(frames)} frames, {out.stat().st_size/1024:.0f} KB)")
    return 0


def shade(V, tri, cx, cy, sc, S):
    n = np.zeros_like(V)
    a, b, c = V[tri[:, 0]], V[tri[:, 1]], V[tri[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, tri[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

    xs = ((V[:, 0] - cx) * sc + S / 2).astype(np.int32)
    ys = (S / 2 - (V[:, 1] - cy) * sc).astype(np.int32)
    sh = 0.25 + 0.75 * np.clip(n @ LIGHT, 0, 1)
    col = np.clip(sh[:, None] * SKIN * 255, 0, 255)

    SX, SY, SZ, SC = [], [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            SX.append(xs + dx); SY.append(ys + dy)
            SZ.append(V[:, 2]); SC.append(col)
    X = np.concatenate(SX); Y = np.concatenate(SY)
    Z = np.concatenate(SZ); C = np.concatenate(SC)
    ok = (X >= 0) & (X < S) & (Y >= 0) & (Y < S)
    X, Y, Z, C = X[ok], Y[ok], Z[ok], C[ok]
    o = np.argsort(Z)
    img = np.full((S, S, 3), 248.0)
    img[Y[o], X[o]] = C[o]
    return img.astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
