#!/usr/bin/env python3
"""
Make an avatar speak.

    python talk.py avatar/model.glb "Good morning. How can I help you today?" \
        --out out/talking.glb

Builds a viseme timeline from the text, derives jaw, lip rounding, blinks and
idle head motion from it, and writes the result into the GLB as a glTF
animation. The avatar's own geometry, skin, materials and morph targets are
untouched.

The viseme track is the seam: swap this generator for Audio2Face, which emits
weights over the same viseme names, and nothing downstream changes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pxhuman.lipsync import (VISEMES, build_timeline, viseme_track,   # noqa: E402
                             jaw_and_round, blink_track, idle_head)
from pxhuman.gltf_anim import Glb, quat_from_euler, quat_mul          # noqa: E402

# meshes that carry mouth shapes, and the extra ARKit keys we drive on top
MOUTH_MESHES = ["AvatarHead", "AvatarTeethLower"]
BLINK_MESHES = ["AvatarHead", "AvatarEyelashes"]
HEAD_CHAIN = [("Head", 1.0), ("Neck2", 0.45), ("Neck1", 0.30), ("Neck", 0.20)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a talking animation")
    ap.add_argument("glb")
    ap.add_argument("text")
    ap.add_argument("--out", default="out/talking.glb")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--rate", type=float, default=1.0, help="speech rate")
    ap.add_argument("--lang", choices=["en", "ar", "auto"], default="auto",
                    help="auto detects Arabic script")
    ap.add_argument("--a2f", help="Audio2Face blendshape weights (CSV or JSON); "
                                  "drives the mouth from audio instead of text")
    ap.add_argument("--coarticulation", type=float, default=0.55)
    ap.add_argument("--jaw-gain", type=float, default=0.20,
                    help="extra jawOpen on top of the viseme shapes")
    ap.add_argument("--round-gain", type=float, default=0.15,
                    help="extra mouthPucker/Funnel on rounded visemes")
    ap.add_argument("--body", help="glTF/GLB containing a skeletal clip "
                                   "(e.g. a Mixamo animation converted to GLB)")
    ap.add_argument("--body-clip", default="0",
                    help="animation index or name inside --body")
    ap.add_argument("--body-loop", action="store_true", default=True)
    ap.add_argument("--root-motion", action="store_true",
                    help="also move the hips: the clip's offset from its own "
                         "rest, scaled by leg length")
    ap.add_argument("--no-rest-align", action="store_true",
                    help="do not align differing rest poses (A- vs T-pose)")
    ap.add_argument("--head-motion", type=float, default=1.0)
    ap.add_argument("--no-blink", action="store_true")
    ap.add_argument("--name", default="Speech")
    args = ap.parse_args()

    g = Glb(args.glb)

    a2f_w = None
    if args.a2f:
        # Audio2Face drives the mouth directly through ARKit keys. The avatar
        # carries the full ARKit set, so no viseme conversion sits in between.
        from pxhuman import a2f as a2fmod
        src_t, raw = a2fmod.load_weights(args.a2f)
        # a column of 0,1,2,... is frame indices, not seconds
        if len(src_t) > 1 and np.allclose(src_t, np.arange(len(src_t))):
            src_t = src_t / float(args.fps)
        total = float(src_t[-1])
        times = np.arange(0.0, total + 1e-6, 1.0 / args.fps, dtype=np.float32)
        a2f_w = a2fmod.normalise_names(a2fmod.resample(src_t, raw, times))
        W = np.zeros((len(times), len(VISEMES)), np.float32)
        jaw = a2f_w.get("jawOpen", np.zeros(len(times)))
        rnd = np.zeros(len(times), np.float32)
        seq = []
        print(f"audio2face: {args.a2f}")
        print(f"            {len(raw)} blendshapes, {len(src_t)} source frames"
              f" -> {len(times)} at {args.fps}fps, {total:.2f}s")
    else:
        lang = args.lang
        if lang == "auto":
            lang = "ar" if re.search(r"[؀-ۿ]", args.text) else "en"
        if lang == "ar":
            from pxhuman.arabic import build_timeline_ar
            seq, total = build_timeline_ar(args.text, rate=args.rate)
        else:
            seq, total = build_timeline(args.text, rate=args.rate)
        W, times = viseme_track(seq, total, fps=args.fps,
                                coarticulation=args.coarticulation)
        jaw, rnd = jaw_and_round(W)
        words = len([p for p in seq if p.arpa not in ("SIL",)])
        print(f"text      : {args.text[:72]}{'...' if len(args.text)>72 else ''}")
        print(f"phonemes  : {words}   duration {times[-1]:.2f}s   "
              f"{len(times)} frames @ {args.fps}fps  [{lang}]")

    blink = np.zeros(len(times)) if args.no_blink else blink_track(times)
    n = len(times)

    tracks = []

    # ---- mouth ---------------------------------------------------------
    for mesh_name in MOUTH_MESHES:
        mi = g.mesh_index_by_name(mesh_name)
        if mi is None:
            print(f"  ! {mesh_name}: not found, skipped")
            continue
        node = g.node_by_name_with_mesh(mesh_name) or g.node_for_mesh(mi)
        names = g.target_names(mi)
        idx = {nm: i for i, nm in enumerate(names)}
        k = g.num_targets(mi)
        vals = np.zeros((n, k), np.float32)

        hit = 0
        if a2f_w is not None:
            # Audio2Face path: copy its ARKit weights straight onto the
            # matching targets. Anything the avatar does not carry is ignored
            # rather than guessed at.
            for key, series in a2f_w.items():
                j = idx.get(key)
                if j is not None:
                    vals[:, j] = np.clip(series, 0.0, 1.0)
                    hit += 1
        else:
            for vi, vname in enumerate(VISEMES):
                if vname == "sil":
                    continue
                j = idx.get(vname)
                if j is not None:
                    vals[:, j] = W[:, vi]
                    hit += 1
            for key, src, gain in (("jawOpen", jaw, args.jaw_gain),
                                   ("mouthPucker", rnd, args.round_gain),
                                   ("mouthFunnel", rnd, args.round_gain * 0.6)):
                j = idx.get(key)
                if j is not None and gain > 0:
                    vals[:, j] = np.clip(vals[:, j] + src * gain, 0, 1)
        tracks.append(dict(node=node, path="weights", times=times,
                           values=vals.reshape(-1)))
        print(f"  {mesh_name:18s} node {node:3d}  {k:3d} targets  "
              f"{hit} visemes driven")

    # ---- blinks --------------------------------------------------------
    if not args.no_blink:
        for mesh_name in BLINK_MESHES:
            mi = g.mesh_index_by_name(mesh_name)
            if mi is None:
                continue
            node = g.node_by_name_with_mesh(mesh_name) or g.node_for_mesh(mi)
            names = g.target_names(mi)
            idx = {nm: i for i, nm in enumerate(names)}
            k = g.num_targets(mi)
            existing = next((t for t in tracks if t["node"] == node), None)
            vals = (existing["values"].reshape(n, k).copy() if existing
                    else np.zeros((n, k), np.float32))
            for key in ("eyeBlinkLeft", "eyeBlinkRight"):
                j = idx.get(key)
                if j is not None:
                    vals[:, j] = np.clip(vals[:, j] + blink, 0, 1)
            if existing:
                existing["values"] = vals.reshape(-1)
            else:
                tracks.append(dict(node=node, path="weights", times=times,
                                   values=vals.reshape(-1)))
                print(f"  {mesh_name:18s} node {node:3d}  blink only")

    # ---- body clip ------------------------------------------------------
    # Both the body clip and the speech want the neck and head. Rather than
    # letting one silently overwrite the other, the body pose is taken as the
    # base and the speech motion is composed on top of it as a delta.
    clip = None
    body_rot: dict[str, np.ndarray] = {}
    body_trans: dict[str, np.ndarray] = {}
    if args.body:
        from pxhuman.bodyanim import BodyClip, Retarget, match_report
        try:
            which = int(args.body_clip)
        except ValueError:
            which = args.body_clip
        clip = BodyClip(args.body, which)
        matched, missing = match_report(clip, g)
        print(f"\nbody clip '{clip.name}': {clip.duration:.2f}s, "
              f"{len(clip.bones())} bones -> {len(matched)} matched")
        if missing:
            print(f"  unmatched ({len(missing)}): {', '.join(missing[:8])}"
                  f"{' ...' if len(missing) > 8 else ''}")
        # world-space transfer: the clip's local numbers are only meaningful
        # in its own rig's bone frames (see pxhuman/bodyanim.py)
        rt = Retarget(clip, g, align=not args.no_rest_align)
        print(f"  rest mismatch     {rt.report()}")
        if rt.src.rebound:
            print(f"  rest from bind    {rt.src.rebound} clip bones "
                  f"(node TRS was not the bind pose)")
        body_rot, body_trans = rt.solve(times, loop=args.body_loop,
                                        root_motion=args.root_motion)
        if args.root_motion:
            print(f"  root scale        x{rt.scale:.3f} (leg length ratio)")

    head_names = {b for b, _ in HEAD_CHAIN}

    # ---- head motion, composed over the body pose -----------------------
    pitch = yaw = roll = None
    if args.head_motion > 0:
        pitch, yaw, roll = idle_head(times, amount=args.head_motion)
        pitch = pitch - jaw * np.radians(1.2)   # emphasis on the open vowels

    written = 0
    for bone, share in HEAD_CHAIN:
        ni = g.node_index_by_name(bone)
        if ni is None:
            continue
        if bone in body_rot:
            b = body_rot[bone]
        else:
            rest = g.json["nodes"][ni].get("rotation", [0, 0, 0, 1])
            b = np.tile(np.asarray(rest, np.float32), (n, 1))
        if pitch is not None:
            b = quat_mul(b, quat_from_euler(pitch * share, yaw * share,
                                            roll * share))
        tracks.append(dict(node=ni, path="rotation", times=times, values=b))
        written += 1
    if written:
        src = "body + speech" if body_rot else "speech"
        print(f"  head chain        {written} bones ({src})")

    # ---- the rest of the body -------------------------------------------
    if body_rot:
        count = 0
        for bone, q in body_rot.items():
            if bone in head_names:
                continue
            ni = g.node_index_by_name(bone)
            if ni is None:
                from pxhuman.bodyanim import canonical
                ni = next((i for i, nd in enumerate(g.json["nodes"])
                           if canonical(nd.get("name", "")) == bone), None)
            if ni is None:
                continue
            tracks.append(dict(node=ni, path="rotation", times=times, values=q))
            count += 1
        print(f"  body bones        {count} rotation tracks")

    if args.root_motion and clip is not None:
        t = body_trans.get("Hips")
        ni = g.node_index_by_name("Hips")
        if t is not None and ni is not None:
            tracks.append(dict(node=ni, path="translation", times=times, values=t))
            print("  root motion       Hips offset from rest, scaled")
        else:
            print("  root motion       no hip translation in the clip")

    if not tracks:
        print("no drivable targets found - is this a MetaPerson avatar?")
        return 1

    g.add_animation(args.name, tracks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.save(out)

    size = out.stat().st_size / 1024 / 1024
    print(f"\nwrote {out}  ({size:.1f} MB)  animation '{args.name}', "
          f"{len(tracks)} tracks, {times[-1]:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
