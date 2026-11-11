"""Mesh and rig exporters."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List

from .sdf import HIERARCHY


def write_obj(path, verts: np.ndarray, faces: np.ndarray,
              normals: np.ndarray | None = None, name: str = "synthetic_body"):
    path = Path(path)
    with open(path, "w") as f:
        f.write("# parametric synthetic human - measurement driven\n")
        f.write("# no real person's likeness is represented\n")
        f.write(f"o {name}\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            for t in faces + 1:
                f.write(f"f {t[0]}//{t[0]} {t[1]}//{t[1]} {t[2]}//{t[2]}\n")
        else:
            for t in faces + 1:
                f.write(f"f {t[0]} {t[1]} {t[2]}\n")
    return path


def write_rig_json(path, joints: Dict[str, np.ndarray], idx: np.ndarray,
                   w: np.ndarray, names: List[str], params: dict,
                   verification: dict):
    path = Path(path)
    data = {
        "format": "pxhuman-rig/1.0",
        "units": "centimetres",
        "up_axis": "Y",
        "note": "Synthetic parametric body. Generated from numeric measurements "
                "only; not derived from any photograph or real individual.",
        "parameters": params,
        "verification": verification,
        "joint_order": names,
        "hierarchy": {k: v for k, v in HIERARCHY.items() if k in joints},
        "bind_pose": {k: [round(float(x), 5) for x in v] for k, v in joints.items()},
        "skinning": {
            "type": "linear_blend",
            "max_influences": int(idx.shape[1]),
            "joint_indices": idx.tolist(),
            "weights": np.round(w, 5).tolist(),
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return path


def write_trimesh_formats(path_stem, verts, faces, formats=("stl", "glb", "ply"),
                          colours=None, uv=None, roughness: float = 0.62):
    """Export. When per-vertex colours are supplied the GLB and PLY carry them,
    and the GLB gets a skin-appropriate PBR material."""
    import trimesh
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    if colours is not None:
        rgba = np.empty((len(verts), 4), dtype=np.uint8)
        srgb = np.clip(colours, 0.0, 1.0) ** (1.0 / 2.2)
        rgba[:, :3] = np.round(srgb * 255).astype(np.uint8)
        rgba[:, 3] = 255
        m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=rgba)

    out = {}
    for ext in formats:
        p = Path(f"{path_stem}.{ext}")
        if ext == "glb" and colours is not None:
            g = m.copy()
            mat = trimesh.visual.material.PBRMaterial(
                name="skin",
                baseColorFactor=[255, 255, 255, 255],
                metallicFactor=0.0,
                roughnessFactor=float(roughness),
                doubleSided=False,
            )
            vis = trimesh.visual.TextureVisuals(material=mat)
            if uv is not None:
                vis.uv = uv
            vis.vertex_attributes = {}
            g.visual = m.visual          # keep colours; material rides along
            try:
                g.visual.material = mat
            except Exception:
                pass
            g.export(p)
        else:
            m.export(p)
        out[ext] = p
    return out, m


def write_skin_sidecar(path, colours, ao, roughness, uv, tone: str):
    """Per-vertex skin channels, for anyone who wants to rebuild the look."""
    np.savez_compressed(
        path,
        colour_linear=np.asarray(colours, np.float32),
        ambient_occlusion=np.asarray(ao, np.float32),
        roughness=np.asarray(roughness, np.float32),
        uv=np.asarray(uv, np.float32),
        tone=np.array([tone]),
    )
    return Path(path)


BLENDER_SCRIPT = '''"""
Import a pxhuman mesh + rig into Blender.

    blender --python import_pxhuman.py -- body.obj body_rig.json

Builds the armature from the rig JSON, parents the mesh to it and writes the
linear-blend skinning weights into vertex groups.
"""
import bpy, json, sys, math
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(argv) < 2:
    raise SystemExit("usage: blender --python import_pxhuman.py -- mesh.obj rig.json")
mesh_path, rig_path = argv[0], argv[1]

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

try:
    bpy.ops.wm.obj_import(filepath=mesh_path)
except AttributeError:
    bpy.ops.import_scene.obj(filepath=mesh_path)
obj = bpy.context.selected_objects[0]
obj.name = "SyntheticBody"

rig = json.load(open(rig_path))
joints = rig["bind_pose"]
hierarchy = rig["hierarchy"]
order = rig["joint_order"]
S = 0.01  # centimetres -> metres

bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
arm = bpy.context.object
arm.name = "SyntheticRig"
eb = arm.data.edit_bones
for b in list(eb):
    eb.remove(b)

children = {}
for child, parent in hierarchy.items():
    children.setdefault(parent, []).append(child)

created = {}
for name in order:
    x, y, z = joints[name]
    head = Vector((x * S, -z * S, y * S))
    kids = children.get(name, [])
    if kids:
        kx, ky, kz = joints[kids[0]]
        tail = Vector((kx * S, -kz * S, ky * S))
    else:
        tail = head + Vector((0, 0, 0.05))
    if (tail - head).length < 1e-4:
        tail = head + Vector((0, 0, 0.05))
    bone = eb.new(name)
    bone.head, bone.tail = head, tail
    created[name] = bone

for name in order:
    par = hierarchy.get(name)
    if par and par in created:
        created[name].parent = created[par]

bpy.ops.object.mode_set(mode="OBJECT")

for name in order:
    obj.vertex_groups.new(name=name)

idx = rig["skinning"]["joint_indices"]
wts = rig["skinning"]["weights"]
groups = [obj.vertex_groups[n] for n in order]
for vi, (ii, ww) in enumerate(zip(idx, wts)):
    for j, w in zip(ii, ww):
        if w > 1e-4:
            groups[j].add([vi], float(w), "REPLACE")

obj.scale = (S, S, S)
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.transform_apply(scale=True)
obj.rotation_euler = (math.radians(90), 0, 0)
bpy.ops.object.transform_apply(rotation=True)

obj.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE")
print("pxhuman: imported mesh + %d bones" % len(order))
'''


def write_blender_script(path):
    path = Path(path)
    path.write_text(BLENDER_SCRIPT)
    return path
