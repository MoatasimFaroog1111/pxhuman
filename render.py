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


def render_view(V, N, yaw, W=420, H=680, splat=2, bg=(246, 246, 248),
                albedo=None, rough=None):
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
    ndl = Nr @ light
    rim = np.clip(1.0 - np.abs(Nr[:, 2]), 0, 1) ** 3

    if albedo is None:
        lam = np.clip(ndl, 0, 1)
        shade = 0.20 + 0.74 * lam ** 0.85 + 0.16 * rim
        base = np.array([0.80, 0.80, 0.83])
        col = np.clip(shade[:, None] * base * 255, 0, 255)
    else:
        # Wrap diffuse: light keeps carrying past the terminator because it
        # scatters under the surface instead of stopping at it. A hard
        # Lambert cutoff is what makes skin read as painted plastic.
        wrap = 0.38
        lam = np.clip((ndl + wrap) / (1.0 + wrap), 0.0, 1.0)
        # colour the scattered component: light that travels through tissue
        # comes back red
        bleed = np.clip((wrap - np.abs(ndl)) / wrap, 0.0, 1.0) ** 2
        sss = np.array([0.52, 0.12, 0.08])[None, :] * bleed[:, None] * 0.55

        # Keep the brightest lit skin near 0.80 BEFORE the gamma encode. Let
        # the diffuse term reach 1.0 and the highlights clip to flat white,
        # which erases all the tonal variation and leaves only the darkest
        # creases visible as streaks.
        fill = 0.17 + 0.13 * np.clip(Nr[:, 1] * 0.5 + 0.5, 0, 1)
        diff = (fill + 0.56 * lam ** 1.15)[:, None] * albedo + sss * 0.75

        # broad, weak specular -- skin is rough and never mirror-like
        half = light + np.array([0.0, 0.0, 1.0])
        half /= np.linalg.norm(half)
        r = 0.62 if rough is None else rough
        shin = np.clip(2.0 / np.maximum(np.asarray(r) ** 4, 1e-4), 2.0, 200.0)
        spec = np.clip(Nr @ half, 0, 1) ** shin
        spec = spec * 0.045 * np.clip(ndl * 3.0, 0, 1)

        col = diff + spec[:, None] + (rim * 0.035)[:, None]
        # soft shoulder instead of a hard clip, so the brightest areas roll off
        # rather than flattening into a single white value
        col = col / (1.0 + col * 0.26)
        col = np.clip(col, 0.0, 1.0) ** (1.0 / 2.2)      # to display space
        col = col * 255.0

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

    # pick up a skin sidecar written alongside the mesh, if there is one
    albedo = rough = None
    side = src.with_name(src.stem + "_skin.npz")
    if side.exists():
        z = np.load(side)
        albedo = z["colour_linear"]
        rough = float(np.mean(z["roughness"]))
        print(f"skin: {side.name}  ({len(albedo):,} vertex colours)")

    V, _, F = load_obj(src)
    # always recompute from the winding: file normals may be inconsistently
    # oriented after remeshing, which shades the whole body flat and dark
    N = vertex_normals(V, F)
    c = V.mean(0)
    if np.mean(np.einsum("ij,ij->i", N, V - c)) < 0:
        N = -N

    if albedo is not None and len(albedo) != len(V):
        print(f"  note: sidecar has {len(albedo):,} colours for {len(V):,} "
              f"vertices -- ignoring it")
        albedo = None

    tiles = [render_view(V, N, y, albedo=albedo, rough=rough) for y in views]
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
