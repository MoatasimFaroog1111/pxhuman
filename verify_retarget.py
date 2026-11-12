#!/usr/bin/env python3
"""
Check body-clip retargeting against rigs whose correct answer is known.

    python verify_retarget.py avatar/model.glb [out/retarget_check.png]

Three source rigs are built from the avatar's own joint positions, each
carrying the same motion (arms lowered, a walking leg swing, a spine twist,
a hip bob):

  same rig     the avatar's own nodes. Retargeting onto itself must return
               the clip's local rotations unchanged.
  identity cm  same joints, but every rest rotation is identity, units are
               centimetres under a 0.01 root. This is the shape of rig that
               turned the legs upside down when local rotations were copied.
  A-pose cm    as above with the arms resting 40 degrees down.

For every bone with one matched child, the bone's world direction on the
avatar must equal the source's at every frame. The posed avatar is skinned
and checked upright (feet below hips, head above) with the hips displaced
exactly as the clip's hips, scaled. Writing a file proves nothing; this
measures the pose that a viewer would show.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pxhuman.gltf_anim import Glb                                  # noqa: E402
from pxhuman.skeleton import (Skeleton, mat_to_quat, quat_to_mat,  # noqa: E402
                              angle_deg, animation_overrides, skinned_vertices)
from pxhuman.bodyanim import BodyClip, Retarget, canonical         # noqa: E402

FPS, DUR = 30, 2.0


def rot(axis, deg):
    a = np.radians(deg)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def world_motion(t):
    """World-space rotation deltas per bone at time t (avatar faces +Z)."""
    s = np.sin(2 * np.pi * t / DUR)
    return {
        "Hips": rot([0, 1, 0], 10 * s),
        "Spine1": rot([0, 1, 0], 8 * s),
        "LeftArm": rot([0, 0, 1], -70 + 10 * s),     # left arm is +X: down
        "LeftForeArm": rot([0, 0, 1], -70 + 10 * s) @ rot([0, 1, 0], -30),
        "RightArm": rot([0, 0, 1], 70 - 10 * s),
        "RightForeArm": rot([0, 0, 1], 70 - 10 * s) @ rot([0, 1, 0], 30),
        "LeftUpLeg": rot([1, 0, 0], -25 * s),
        "LeftLeg": rot([1, 0, 0], -25 * s) @ rot([1, 0, 0], 20 * max(s, 0)),
        "RightUpLeg": rot([1, 0, 0], 25 * s),
        "RightLeg": rot([1, 0, 0], 25 * s) @ rot([1, 0, 0], 20 * max(-s, 0)),
    }


def hip_bob(t):
    return np.array([0.0, 0.03 * np.sin(4 * np.pi * t / DUR), 0.0])


# ---------------------------------------------------------------------------
def build_identity_rig(tgt: Skeleton, path: Path, apose: float = 0.0,
                       units: float = 100.0):
    """Avatar joints with identity rests, in centimetres under a 0.01 root."""
    _, P0, _ = tgt.world()
    P0 = P0[0].copy()
    skin_j = set(tgt.g.json["skins"][0]["joints"])
    names = {i: canonical(tgt.names[i]) for i in skin_j}
    # optional A-pose: swing each arm chain down about its shoulder joint
    if apose:
        for side, sgn in (("Left", -1), ("Right", 1)):
            root = next(i for i in skin_j if names[i] == f"{side}Arm")
            R = rot([0, 0, 1], sgn * apose)
            stack = [root]
            while stack:
                i = stack.pop()
                if i != root:
                    P0[i] = P0[root] + R @ (P0[i] - P0[root])
                stack += [c for c in tgt.children(i) if c in skin_j]
    hips = next(i for i in skin_j if names[i] == "Hips")
    order = [i for i in tgt.order if i in skin_j]
    new_index = {old: k + 1 for k, old in enumerate(order)}
    nodes = [{"name": "Armature", "scale": [1 / units] * 3, "children": [1]}]
    for old in order:
        p = tgt.parent[old]
        base = P0[p] if (old != hips and p in skin_j) else np.zeros(3)
        nd = {"name": "mixamorig:" + names[old],
              "translation": [float(x) for x in (P0[old] - base) * units]}
        kids = [new_index[c] for c in tgt.children(old) if c in skin_j]
        if kids:
            nd["children"] = kids
        nodes.append(nd)
    g = Glb.new(nodes, [0])
    times = np.arange(0, DUR + 1e-6, 1 / FPS)
    # identity rests: a bone's world rotation is the motion itself
    tracks = []
    # bones the motion does not name follow their parent, as on a real rig
    W = {}
    for old in order:
        nm = names[old]
        if nm in world_motion(0):
            W[nm] = np.array([world_motion(t)[nm] for t in times])
        else:
            p = tgt.parent[old]
            W[nm] = W[names[p]] if (p in skin_j and old != hips) else \
                np.tile(np.eye(3), (len(times), 1, 1))
    for old in order:
        nm = names[old]
        p = tgt.parent[old]
        Wp = W[names[p]] if (p in skin_j and old != hips) else np.eye(3)
        L = np.swapaxes(Wp, -1, -2) @ W[nm] if np.ndim(Wp) == 3 else W[nm]
        if angle_deg(L).max() < 1e-6:
            continue
        tracks.append(dict(node=new_index[old], path="rotation", times=times,
                           values=mat_to_quat(L).astype(np.float32)))
    hp = np.array([P0[hips] + hip_bob(t) for t in times]) * units
    tracks.append(dict(node=new_index[hips], path="translation", times=times,
                       values=hp.astype(np.float32)))
    g.add_animation("clip", tracks)
    g.save(path)


def build_same_rig(avatar: Path, path: Path):
    """The avatar's own file with the motion applied in its own bone frames."""
    g = Glb(avatar)
    sk = Skeleton(g)
    times = np.arange(0, DUR + 1e-6, 1 / FPS)
    _, P0, _ = sk.world()
    skin_j = set(g.json["skins"][0]["joints"])
    names = {i: canonical(sk.names[i]) for i in skin_j}
    # posed world = motion * rest world, in hierarchy order
    RW0, _, _ = sk.world()
    Wposed = {}
    tracks = []
    for i in sk.order:
        p = sk.parent[i]
        if i not in skin_j:
            continue
        D = np.array([world_motion(t).get(names[i], None) if names[i] in
                      world_motion(0) else np.eye(3) for t in times])
        # bones the motion does not name keep their pose relative to parent
        if names[i] in world_motion(0):
            Wi = D @ RW0[0, i]
        else:
            Wp = Wposed.get(p, np.broadcast_to(RW0[0, p], (len(times), 3, 3)))
            Wi = Wp @ sk.R[i]
        Wposed[i] = Wi
        Wp = Wposed.get(p, np.broadcast_to(RW0[0, p], (len(times), 3, 3)))
        L = np.swapaxes(Wp, -1, -2) @ Wi
        if angle_deg(np.swapaxes(L, 1, 2) @ sk.R[i]).max() > 1e-4:
            tracks.append(dict(node=i, path="rotation", times=times,
                               values=mat_to_quat(L).astype(np.float32)))
    hips = next(i for i in skin_j if names[i] == "Hips")
    hp = np.array([sk.t[hips] + hip_bob(t) for t in times])
    tracks.append(dict(node=hips, path="translation", times=times,
                       values=hp.astype(np.float32)))
    g.json.pop("animations", None)
    g.add_animation("clip", tracks)
    g.save(path)
    return {sk.names[t["node"]]: t["values"] for t in tracks
            if t["path"] == "rotation"}


# ---------------------------------------------------------------------------
def naive(clip, rt, times):
    """The old transfer: local rotations and hip translation copied by name."""
    rq = {}
    for b in clip.rot:
        if b in rt.t_idx and rt.t_idx[b] in rt.pairs:
            rq[b] = clip.sample_rotation(b, times, loop=False)
    tr = {}
    if "Hips" in clip.trans:
        tr["Hips"] = clip.sample_translation("Hips", times, loop=False)
    return rq, tr


def check(label, clip_path, avatar_g, expect_local=None, old=False):
    clip = BodyClip(clip_path)
    rt = Retarget(clip, avatar_g)
    times = np.arange(0, DUR + 1e-6, 1 / FPS).astype(np.float32)
    if old:
        rq, tr = naive(clip, rt, times)
    else:
        rq, tr = rt.solve(times, loop=False, root_motion=True)

    tgt, src = rt.tgt, rt.src
    rot_t = {tgt.names.index(b) if b in tgt.names else rt.t_idx[b]:
             quat_to_mat(q) for b, q in rq.items()}
    tr_t = {rt.t_idx["Hips"]: tr["Hips"]} if "Hips" in tr else {}
    RT, PT, _ = tgt.world(rot_t, tr_t, len(times))
    ov_r, ov_t = animation_overrides(clip.src, src, times)
    RS, PS, _ = src.world(ov_r, ov_t, len(times))

    # bone direction agreement
    worst, worst_b = 0.0, ""
    for ti, si in rt.pairs.items():
        kids = [c for c in tgt.children(ti) if c in rt.pairs]
        if len(kids) != 1:
            continue
        c = kids[0]
        dt = PT[:, c] - PT[:, ti]
        ds = PS[:, rt.pairs[c]] - PS[:, si]
        cos = (dt * ds).sum(1) / (np.linalg.norm(dt, axis=1) *
                                  np.linalg.norm(ds, axis=1) + 1e-12)
        e = float(np.degrees(np.arccos(np.clip(cos, -1, 1))).max())
        if e > worst:
            worst, worst_b = e, canonical(tgt.names[ti])

    # hips displacement, scaled
    h_t, h_s = rt.t_idx["Hips"], rt.s_idx["Hips"]
    dp_t = PT[:, h_t] - rt.PT0[h_t]
    dp_s = (PS[:, h_s] - rt.PS0[h_s]) * rt.scale
    hip_err = float(np.abs(dp_t - dp_s).max())

    # skinned: upright, on the floor
    V = skinned_vertices(avatar_g, tgt, rot_t, tr_t, len(times), stride=7)
    V0 = skinned_vertices(avatar_g, tgt, stride=7)[0]
    def y(b):
        return PT[:, rt.t_idx[b], 1]
    upright = bool(np.all(y("LeftFoot") < y("Hips")) and
                   np.all(y("RightFoot") < y("Hips")) and
                   np.all(y("Head") > y("Hips")))
    height = V[:, :, 1].max(1)
    floor = V[:, :, 1].min(1)

    ok = worst < 0.5 and hip_err < 1e-3 and upright and \
        height.min() > 0.85 * V0[:, 1].max() and floor.min() > -0.06
    line = (f"{label:12s} dir err {worst:6.3f}deg ({worst_b:12s}) "
            f"hips err {hip_err*1000:6.3f}mm  upright {upright}  "
            f"height {height.min():.2f}-{height.max():.2f}m  "
            f"floor {floor.min():+.3f}m  rest: {rt.report()}")
    if expect_local is not None:
        lerr = 0.0
        for b, q in expect_local.items():
            cb = canonical(b)
            if cb in rq:
                lerr = max(lerr, float(angle_deg(
                    np.swapaxes(quat_to_mat(rq[cb]), 1, 2) @ quat_to_mat(q)).max()))
        line += f"  local err {lerr:.4f}deg"
        ok = ok and lerr < 0.05
    print(("PASS " if ok else "FAIL ") + line)
    return ok, V[::max(1, len(times) // 4)]


def main():
    avatar = Path(sys.argv[1] if len(sys.argv) > 1 else "avatar/model.glb")
    png = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    ag = Glb(avatar)
    tgt = Skeleton(ag)
    tmp = Path(tempfile.mkdtemp())
    results, sheets = [], []

    expect = build_same_rig(avatar, tmp / "same.glb")
    ok, v = check("same rig", tmp / "same.glb", ag, expect)
    results.append(ok); sheets.append(v)
    build_identity_rig(tgt, tmp / "ident.glb")
    ok, v = check("identity cm", tmp / "ident.glb", ag)
    results.append(ok); sheets.append(v)
    build_identity_rig(tgt, tmp / "apose.glb", apose=40)
    ok, v = check("A-pose cm", tmp / "apose.glb", ag)
    # the arms are the only place the two rests differ; anything else
    # flagged means the alignment invented a difference (spurious twist)
    rt = Retarget(BodyClip(tmp / "apose.glb"), ag)
    flagged = set(rt.mismatch)
    clean = flagged == {"LeftArm", "RightArm"} and all(
        abs(d - 40) < 0.5 for d in rt.mismatch.values())
    print(("PASS " if clean else "FAIL ") +
          f"A-pose rest  mismatch found only at {sorted(flagged)}")
    results.append(ok and clean); sheets.append(v)

    # negative control: the old copy-by-name transfer must be caught
    print("\nnegative control (old transfer, must FAIL):")
    bad, v = check("old/ident", tmp / "ident.glb", ag, old=True)
    results.append(not bad); sheets.append(v)

    if png:
        render(sheets, png)
        print(f"wrote {png}")
    print("\nALL PASS" if all(results) else "\nFAILED")
    return 0 if all(results) else 1


def render(sheets, path, S=260):
    from PIL import Image
    cols = sum(len(s) for s in sheets)
    img = np.full((S, cols * S, 3), 248, np.uint8)
    k = 0
    for s in sheets:
        for V in s:
            sc = S / 2.0
            x = (V[:, 0] * sc + S / 2).astype(int)
            yy = (S * 0.95 - V[:, 1] * sc).astype(int)
            m = (x >= 0) & (x < S) & (yy >= 0) & (yy < S)
            img[yy[m], k * S + x[m]] = (60, 60, 70)
            img[int(S * 0.95), k * S:(k + 1) * S] = (200, 60, 60)
            k += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


if __name__ == "__main__":
    raise SystemExit(main())
