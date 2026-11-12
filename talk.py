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
    ap.add_argument("--coarticulation", type=float, default=0.55)
    ap.add_argument("--jaw-gain", type=float, default=0.20,
                    help="extra jawOpen on top of the viseme shapes")
    ap.add_argument("--round-gain", type=float, default=0.15,
                    help="extra mouthPucker/Funnel on rounded visemes")
    ap.add_argument("--head-motion", type=float, default=1.0)
    ap.add_argument("--no-blink", action="store_true")
    ap.add_argument("--name", default="Speech")
    args = ap.parse_args()

    g = Glb(args.glb)

    seq, total = build_timeline(args.text, rate=args.rate)
    W, times = viseme_track(seq, total, fps=args.fps,
                            coarticulation=args.coarticulation)
    jaw, rnd = jaw_and_round(W)
    blink = np.zeros(len(times)) if args.no_blink else blink_track(times)
    n = len(times)

    words = len([p for p in seq if p.arpa not in ("SIL",)])
    print(f"text      : {args.text[:72]}{'...' if len(args.text)>72 else ''}")
    print(f"phonemes  : {words}   duration {times[-1]:.2f}s   {n} frames @ {args.fps}fps")

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

    # ---- head motion ----------------------------------------------------
    if args.head_motion > 0:
        pitch, yaw, roll = idle_head(times, amount=args.head_motion)
        # speaking adds a little emphasis on the open vowels
        pitch = pitch - jaw * np.radians(1.2)
        for bone, share in HEAD_CHAIN:
            ni = g.node_index_by_name(bone)
            if ni is None:
                continue
            base = g.json["nodes"][ni].get("rotation", [0, 0, 0, 1])
            b = np.tile(np.asarray(base, np.float32), (n, 1))
            d = quat_from_euler(pitch * share, yaw * share, roll * share)
            tracks.append(dict(node=ni, path="rotation", times=times,
                               values=quat_mul(b, d)))
        print(f"  head chain        {len([b for b,_ in HEAD_CHAIN])} bones")

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
