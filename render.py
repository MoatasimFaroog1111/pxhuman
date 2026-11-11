#!/usr/bin/env python3
"""Offscreen preview renderer: orthographic z-buffer splatting with
normal-based shading. Fast, dependency-light, good enough to eyeball a build.

    python render.py out/body.obj out/preview.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_obj(path):
    V, N, F = [], [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vn "):
            N.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(V), (np.array(N) if N else None), np.array(F)


def vertex_normals(V, F):
    n = np.zeros_like(V)
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(ln, 1e-9)


def rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def render_view(V, N, yaw, W=420, H=680, splat=2, bg=(246, 246, 248)):
    R = rot_y(yaw)
    P = V @ R.T
    Nr = N @ R.T

    lo, hi = P.min(0), P.max(0)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    span = max(hi[1] - lo[1], (hi[0] - lo[0]) * H / W) * 1.08
    sc = H / span

    xs = ((P[:, 0] - cx) * sc + W / 2).astype(np.int32)
    ys = (H / 2 - (P[:, 1] - cy) * sc).astype(np.int32)
    zs = P[:, 2]

    light = np.array([0.38, 0.52, 0.76])
    light /= np.linalg.norm(light)
    lam = np.clip(Nr @ light, 0, 1)
    rim = np.clip(1.0 - np.abs(Nr[:, 2]), 0, 1) ** 3
    shade = 0.20 + 0.74 * lam ** 0.85 + 0.16 * rim
    base = np.array([0.80, 0.80, 0.83])
    col = np.clip(shade[:, None] * base * 255, 0, 255)

    # Gather every splat sample first, then sort the whole set far-to-near and
    # assign once. Sorting per offset pass and assigning in a loop lets a later
    # pass paint far samples over near ones, which flattens the shading.
    SX, SY, SZ, SC = [], [], [], []
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            if dx * dx + dy * dy > splat * splat + 1:
                continue
            SX.append(xs + dx); SY.append(ys + dy); SZ.append(zs); SC.append(col)
    X = np.concatenate(SX); Y = np.concatenate(SY)
    Z = np.concatenate(SZ); C = np.concatenate(SC)

    ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
    X, Y, Z, C = X[ok], Y[ok], Z[ok], C[ok]
    # fancy-index assignment lets the LAST duplicate win, so sort ascending by
    # depth: the nearest sample (largest z, camera sits at +z) is written last
    order = np.argsort(Z)
    X, Y, C = X[order], Y[order], C[order]

    img = np.full((H, W, 3), bg, dtype=np.float64)
    img[Y, X] = C
    return img.astype(np.uint8)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "out/body.obj")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "out/preview.png")
    views = [0, 35, 90, 180]
    labels = ["front", "3/4", "side", "back"]

    V, _, F = load_obj(src)
    # always recompute from the winding: file normals may be inconsistently
    # oriented after remeshing, which shades the whole body flat and dark
    N = vertex_normals(V, F)
    c = V.mean(0)
    if np.mean(np.einsum("ij,ij->i", N, V - c)) < 0:
        N = -N

    tiles = [render_view(V, N, y) for y in views]
    H, W = tiles[0].shape[:2]
    pad = 10
    sheet = np.full((H + 2 * pad, (W + pad) * len(tiles) + pad, 3), 246, np.uint8)
    for i, t in enumerate(tiles):
        x0 = pad + i * (W + pad)
        sheet[pad:pad + H, x0:x0 + W] = t

    im = Image.fromarray(sheet)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)
    print(f"wrote {dst}  ({im.size[0]}x{im.size[1]})  views: {', '.join(labels)}")


if __name__ == "__main__":
    main()
