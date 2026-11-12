#!/usr/bin/env python3
"""
Generate the parametric head.

    python generate_head.py --height 22 --hair bob --voxel 0.11 --out out/head

The head is built separately from the body because facial features sit at the
centimetre scale: resolving them needs roughly a millimetre voxel, which is
tractable over a head-sized box and not over a whole body.

The face is an ORIGINAL synthetic design driven by canonical proportions and
the numeric parameters below. It is not derived from, modelled on, or fitted
to any photograph or any real person.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pxhuman.face import FaceParams, FaceField          # noqa: E402
from pxhuman.mesh import sample_volume, extract_surface, clean_mesh  # noqa: E402
from pxhuman import skin as skinmod, export             # noqa: E402

HAIR_COLOURS = {
    "black":    (0.055, 0.048, 0.046),
    "darkbrown": (0.105, 0.070, 0.052),
    "brown":    (0.175, 0.112, 0.072),
    "auburn":   (0.230, 0.105, 0.062),
    "blonde":   (0.455, 0.345, 0.195),
}
IRIS_COLOURS = {
    "brown":  (0.160, 0.092, 0.050),
    "hazel":  (0.245, 0.170, 0.072),
    "green":  (0.130, 0.205, 0.130),
    "blue":   (0.135, 0.215, 0.290),
    "grey":   (0.215, 0.230, 0.240),
}


def colourise(verts, normals, ao, curv, ff: FaceField, args):
    """Per-vertex colour: skin tone, then the face regions on top."""
    p = ff.p
    L = ff.landmarks()
    H, W = ff.H, ff.W
    n = len(verts)

    base = np.array(skinmod.TONES.get(args.tone, skinmod.TONES["II"]), float)
    col = np.tile(base, (n, 1))

    y, z = verts[:, 1], verts[:, 2]
    ax = np.abs(verts[:, 0])

    # --- warmth variation across the face ----------------------------
    cheek = np.exp(-0.5 * (((y - p.cheek_height * H) / (0.085 * H)) ** 2)) \
        * np.clip((ax - W * 0.16) / (W * 0.28), 0, 1) * np.clip(z / (0.18 * H), 0, 1)
    col += skinmod.PERFUSION[None, :] * (cheek * args.redness * 0.17)[:, None]
    nosetip = np.exp(-0.5 * ((np.linalg.norm(
        verts - L["nose_tip"][None, :], axis=1) / (0.055 * H)) ** 2))
    col += skinmod.PERFUSION[None, :] * (nosetip * args.redness * 0.10)[:, None]

    # --- hair -----------------------------------------------------------
    # Decide by comparing the two fields, not by distance from the skull:
    # the neck is not part of the skull either, so a distance test paints it
    # as hair.
    is_hair = np.zeros(n, bool)
    if p.hair != "none":
        f_skin = ff(verts, with_hair=False)
        f_hair = ff.hair_field(verts, ff.skull(verts))
        is_hair = f_hair < f_skin - 1e-4
        hc = np.array(HAIR_COLOURS.get(args.hair_colour, HAIR_COLOURS["darkbrown"]))
        col[is_hair] = hc

    # --- lips -------------------------------------------------------------
    # Elliptical falloff. A box test gives a rectangular patch of colour
    # sitting on the face instead of lips.
    m = L["mouth"]
    du = verts[:, 0] / (W * p.mouth_width * 1.05)
    dv = (y - m[1]) / (0.060 * H)
    dw = np.clip((m[2] - 0.090 * H - z) / (0.090 * H), 0, 1)
    rad = np.sqrt(du ** 2 + dv ** 2)
    lip_t = np.clip(1.0 - rad, 0, 1) ** 0.55 * (1.0 - dw)
    lip_t[is_hair] = 0.0
    lipc = np.array([0.455, 0.180, 0.175])
    col = col * (1 - 0.80 * lip_t[:, None]) + lipc * (0.80 * lip_t[:, None])

    # --- eyes ------------------------------------------------------------
    r_ball, r_iris = L["eyeball_r"], L["iris_r"]
    sclera = np.array([0.880, 0.868, 0.860])
    iris = np.array(IRIS_COLOURS.get(args.eye_colour, IRIS_COLOURS["brown"]))
    for key in ("eye_l", "eye_r"):
        c = L[key]
        d = np.linalg.norm(verts - c[None, :], axis=1)
        on_ball = (np.abs(d - r_ball) < 0.016 * H) & (z > c[2] - r_ball * 0.2) \
            & ~is_hair
        col[on_ball] = sclera
        radial = np.linalg.norm(verts[:, :2] - c[:2][None, :], axis=1)
        in_iris = on_ball & (radial < r_iris)
        col[in_iris] = iris
        # limbal ring darkens the iris edge, pupil at the centre
        ring = in_iris & (radial > r_iris * 0.80)
        col[ring] = iris * 0.45
        col[in_iris & (radial < r_iris * 0.42)] = np.array([0.020, 0.018, 0.018])

    # --- brows ------------------------------------------------------------
    by = p.y_brow * H + 0.012 * H
    brow = (np.abs(y - by) < 0.028 * H) & (ax > W * 0.055) & (ax < W * 0.40) \
        & (z > 0.22 * H) & ~is_hair
    bc = np.array(HAIR_COLOURS.get(args.hair_colour, HAIR_COLOURS["darkbrown"]))
    col[brow] = col[brow] * 0.30 + bc * 0.70

    # --- lashes: darken the lid rim --------------------------------------
    for key in ("eye_l", "eye_r"):
        c = L[key]
        d = np.linalg.norm(verts - c[None, :], axis=1)
        rim = (np.abs(d - r_ball * 1.10) < 0.013 * H) & (z > c[2]) \
            & (y > c[1]) & ~is_hair
        col[rim] = col[rim] * 0.35

    # --- occlusion ---------------------------------------------------------
    shade = 0.30 + 0.70 * ao
    col *= shade[:, None]
    warm = (1.0 - ao)[:, None] * np.array([0.075, 0.020, 0.010])[None, :]
    col += warm * args.translucency * (~is_hair)[:, None]

    thin = np.clip(curv, 0.0, 1.0) ** 1.4
    col += skinmod.SUBSURFACE[None, :] * (
        thin * args.translucency * 0.055 * (~is_hair))[:, None]

    return np.clip(col, 0.0, 1.0), is_hair


def main() -> int:
    ap = argparse.ArgumentParser(description="Parametric synthetic head")
    ap.add_argument("--height", type=float, default=22.0, help="chin to crown, cm")
    ap.add_argument("--voxel", type=float, default=0.11)
    ap.add_argument("--out", default="out/head")

    ap.add_argument("--hair", default="bob", choices=["none", "bob", "long", "bun"])
    ap.add_argument("--hair-colour", default="darkbrown", choices=list(HAIR_COLOURS))
    ap.add_argument("--eye-colour", default="brown", choices=list(IRIS_COLOURS))
    ap.add_argument("--tone", default="II", choices=list("I II III IV V VI".split()))
    ap.add_argument("--redness", type=float, default=0.55)
    ap.add_argument("--translucency", type=float, default=0.6)

    ap.add_argument("--softness", type=float, default=0.55)
    ap.add_argument("--cheek-fullness", type=float, default=0.50)
    ap.add_argument("--jaw-taper", type=float, default=0.62)
    ap.add_argument("--lip-fullness", type=float, default=0.55)
    ap.add_argument("--eye-tilt", type=float, default=6.0)
    ap.add_argument("--brow-arch", type=float, default=0.55)
    ap.add_argument("--nose-proj", type=float, default=0.118)
    args = ap.parse_args()

    fp = FaceParams(head_height=args.height, hair=args.hair,
                    softness=args.softness, cheek_fullness=args.cheek_fullness,
                    jaw_taper=args.jaw_taper, lip_fullness=args.lip_fullness,
                    eye_tilt=args.eye_tilt, brow_arch=args.brow_arch,
                    nose_proj=args.nose_proj)
    ff = FaceField(fp)

    lo, hi = ff.bounds()
    print(f"head {args.height} cm   bounds {np.round(lo,1)} -> {np.round(hi,1)}"
          f"   voxel {args.voxel} cm")

    t0 = time.time()
    vol, _ = sample_volume(ff, lo, hi, args.voxel, slab=16)
    verts, faces, normals = extract_surface(vol, lo, args.voxel)
    print(f"  surface {len(verts):,} vertices  {len(faces):,} triangles")

    import trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh, info = clean_mesh(mesh)
    mesh.fix_normals()
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.vertex_normals)

    print("  baking occlusion")
    ao = skinmod.ambient_occlusion(ff, verts, normals, args.height * 3.2,
                                   steps=6, reach=0.040)
    ao = skinmod.smooth_scalar(ao, faces, iterations=4, strength=0.65)
    curv = skinmod.curvature_proxy(ff, verts, normals, args.height * 3.2,
                                   delta=0.010)
    curv = skinmod.smooth_scalar(curv, faces, iterations=3, strength=0.65)

    colours, is_hair = colourise(verts, normals, ao, curv, ff, args)
    rough = np.where(is_hair, 0.42, skinmod.roughness_for(ao, curv))

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    export.write_obj(f"{stem}.obj", verts, faces, normals, name="synthetic_head")
    out, _ = export.write_trimesh_formats(stem, verts, faces, ("glb", "ply"),
                                          colours=colours,
                                          roughness=float(rough.mean()))
    export.write_skin_sidecar(f"{stem}_skin.npz", colours, ao, rough,
                              np.zeros((len(verts), 2)), args.tone)

    report = dict(
        face_parameters={k: getattr(fp, k) for k in
                         ("head_height", "width_ratio", "depth_ratio", "y_eye",
                          "y_nose_base", "y_mouth", "y_brow", "hair")},
        appearance=dict(tone=args.tone, hair=args.hair,
                        hair_colour=args.hair_colour, eye_colour=args.eye_colour),
        mesh=dict(vertices=int(len(verts)), faces=int(len(faces)),
                  voxel_cm=args.voxel, watertight=bool(mesh.is_watertight),
                  genus=int((2 - mesh.euler_number) // 2),
                  components=int(mesh.body_count),
                  fragments_removed=info["removed"]),
        note=("Original synthetic design generated from numeric parameters. "
              "Not derived from any photograph or any real individual."),
    )
    Path(f"{stem}_report.json").write_text(json.dumps(report, indent=2))

    m = report["mesh"]
    print(f"  topology  watertight={m['watertight']} genus={m['genus']} "
          f"components={m['components']}"
          + (f"  (removed {info['removed']} fragment(s))" if info["removed"] else ""))
    print(f"\nWrote in {time.time()-t0:.1f}s -> {stem}.obj/.glb/.ply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
