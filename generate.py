#!/usr/bin/env python3
"""
Generate a parametric synthetic human body mesh from measurements.

    python generate.py --stature 178 --chest 104 --waist 92 --hip 101 \
                       --out out/body --quality high --pose a_pose

Every dimension of the output is determined by the numeric arguments below.
Nothing is derived from a photograph or from any real individual.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pxhuman import BodyParams, build_mesh, export            # noqa: E402
from pxhuman.mesh import calibrate, audit, verify             # noqa: E402
from pxhuman.rig import skin_weights, apply_pose, POSE_PRESETS  # noqa: E402
from pxhuman.sdf import build_joints                           # noqa: E402

QUALITY = {"draft": 1.1, "medium": 0.80, "high": 0.60, "ultra": 0.42}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parametric synthetic human body generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("measurements (centimetres)")
    g.add_argument("--stature", type=float, default=172.0)
    g.add_argument("--chest", type=float)
    g.add_argument("--waist", type=float)
    g.add_argument("--hip", type=float)
    g.add_argument("--neck", type=float)
    g.add_argument("--shoulder-width", type=float)
    g.add_argument("--inseam", type=float)
    g.add_argument("--arm-length", type=float)
    g.add_argument("--thigh-girth", type=float)
    g.add_argument("--calf-girth", type=float)
    g.add_argument("--upperarm-girth", type=float)
    g.add_argument("--forearm-girth", type=float)

    s = ap.add_argument_group("shape")
    s.add_argument("--proportion", type=float, default=0.0,
                   help="0..1 blend between the two proportion templates")
    s.add_argument("--muscle", type=float, default=0.5)
    s.add_argument("--bodyfat", type=float, default=0.5)
    s.add_argument("--definition", type=float, default=0.0,
                   help="0..1 anatomical muscle relief; 0 is a smooth base mesh")
    s.add_argument("--posture", type=float, default=0.0, help="-1..1 spine curve")
    s.add_argument("--arm-abduction", type=float, default=45.0)
    s.add_argument("--leg-stance", type=float, default=1.5)

    o = ap.add_argument_group("output")
    o.add_argument("--out", default="out/body")
    o.add_argument("--quality", choices=list(QUALITY), default="high")
    o.add_argument("--voxel", type=float, help="override voxel size in cm")
    o.add_argument("--hands", choices=["mitt", "fingers"], default="mitt",
                   help="'fingers' needs --quality ultra to resolve the gaps")
    sk = ap.add_argument_group("skin")
    sk.add_argument("--skin", action="store_true",
                    help="bake ambient occlusion, skin tone and per-vertex colour")
    sk.add_argument("--tone", default="II", choices=list("I II III IV V VI".split()),
                    help="Fitzpatrick-style tone preset")
    sk.add_argument("--redness", type=float, default=0.5,
                    help="0..1 blood perfusion at joints and extremities")
    sk.add_argument("--translucency", type=float, default=0.5,
                    help="0..1 subsurface colour bleed")

    o.add_argument("--pose", choices=list(POSE_PRESETS), default="a_pose")
    o.add_argument("--no-calibrate", action="store_true")
    o.add_argument("--formats", default="obj,stl,glb",
                   help="comma separated: obj,stl,glb,ply")
    args = ap.parse_args()

    voxel = args.voxel or QUALITY[args.quality]
    if args.hands == "fingers" and voxel > 0.45:
        print("note: --hands fingers needs voxel <= 0.45 cm; falling back to mitt")
        args.hands = "mitt"

    p = BodyParams(
        stature=args.stature, chest=args.chest, waist=args.waist, hip=args.hip,
        neck=args.neck, shoulder_width=args.shoulder_width, inseam=args.inseam,
        arm_length=args.arm_length, thigh_girth=args.thigh_girth,
        calf_girth=args.calf_girth, upperarm_girth=args.upperarm_girth,
        forearm_girth=args.forearm_girth, proportion=args.proportion,
        muscle=args.muscle, bodyfat=args.bodyfat, posture=args.posture,
        definition=args.definition,
        arm_abduction=args.arm_abduction, leg_stance=args.leg_stance,
    ).resolve()

    print("\nRequested measurements")
    print(p.summary())

    t0 = time.time()
    gains = {"chest": 1.0, "waist": 1.0, "hip": 1.0}
    if not args.no_calibrate:
        print("\nCalibrating profile against measured output")
        gains = calibrate(p, voxel=max(voxel, 0.85), iters=3, hand_detail=args.hands)

    print("\nBuilding surface")
    verts, faces, normals, field = build_mesh(p, voxel=voxel, gains=gains,
                                              hand_detail=args.hands)

    import trimesh
    from pxhuman.mesh import clean_mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh, clean_info = clean_mesh(mesh)
    mesh.fix_normals()
    verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    normals = np.asarray(mesh.vertex_normals)

    print("\nRigging")
    joints = build_joints(p)
    idx, w, names = skin_weights(verts, joints)
    print(f"  {len(names)} joints, {idx.shape[1]} influences per vertex")

    posed = verts
    if args.pose != "a_pose":
        posed = apply_pose(verts, joints, idx, w, names, POSE_PRESETS[args.pose])
        print(f"  applied pose preset '{args.pose}'")

    colours = ao = rough = uv = None
    rough_mean = 0.62
    if args.skin:
        print("\nSkin")
        from pxhuman import skin as skinmod
        ao = skinmod.ambient_occlusion(field, verts, normals, p.stature)
        ao = skinmod.smooth_scalar(ao, faces, iterations=5, strength=0.7)
        curv = skinmod.curvature_proxy(field, verts, normals, p.stature)
        curv = skinmod.smooth_scalar(curv, faces, iterations=4, strength=0.7)
        colours = skinmod.skin_colours(verts, normals, ao, curv, p, p.derived,
                                       tone=args.tone, redness=args.redness,
                                       translucency=args.translucency)
        rough = skinmod.roughness_for(ao, curv)
        uv = skinmod.cylindrical_uv(verts, p)
        rough_mean = float(rough.mean())
        print(f"  tone {args.tone}  AO mean {ao.mean():.3f} "
              f"(min {ao.min():.3f})  roughness mean {rough_mean:.3f}")

    print("\nVerifying")
    rep = audit(p, gains, voxel=max(voxel, 0.8), hand_detail=args.hands)
    topo = verify(mesh, p)["topology"]
    rep["topology"] = topo
    for key in ("chest", "waist", "hip", "stature"):
        r = rep[key]
        if r["measured"] is None:
            print(f"  {key:<9} target {r['target']:7.2f}   not measurable in this pose")
        else:
            print(f"  {key:<9} target {r['target']:7.2f}  measured {r['measured']:7.2f}"
                  f"   error {r['error_pct']:+.2f} %")
    topo["fragments_removed"] = clean_info["removed"]
    rep["topology"] = topo
    print(f"  topology  watertight={topo['watertight']} genus={topo['genus']} "
          f"components={topo['components']} volume={topo['volume_litres']} L"
          + (f"  (removed {clean_info['removed']} fragment(s))"
             if clean_info["removed"] else ""))

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
    written = []
    if "obj" in fmts:
        written.append(export.write_obj(f"{stem}.obj", posed, faces, normals))
    tri = [f for f in fmts if f in ("stl", "glb", "ply")]
    if tri:
        out, _ = export.write_trimesh_formats(stem, posed, faces, tri,
                                              colours=colours, uv=uv,
                                              roughness=rough_mean)
        written.extend(out.values())
    if colours is not None:
        written.append(export.write_skin_sidecar(f"{stem}_skin.npz", colours,
                                                 ao, rough, uv, args.tone))
    written.append(export.write_rig_json(f"{stem}_rig.json", joints, idx, w, names,
                                         p.to_dict(), rep))
    written.append(export.write_blender_script(stem.parent / "import_pxhuman.py"))
    with open(f"{stem}_report.json", "w") as fh:
        json.dump({"parameters": p.to_dict(), "gains": gains,
                   "verification": rep,
                   "mesh": {"vertices": int(len(verts)), "faces": int(len(faces)),
                            "voxel_cm": voxel, "pose": args.pose}}, fh, indent=2)
    written.append(Path(f"{stem}_report.json"))

    print(f"\nWrote in {time.time()-t0:.1f}s")
    for pth in written:
        print(f"  {pth}  ({Path(pth).stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
