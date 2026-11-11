"""
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
