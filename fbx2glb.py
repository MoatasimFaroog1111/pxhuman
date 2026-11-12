#!/usr/bin/env python3
"""
Convert a Mixamo (or any) FBX animation to GLB for `talk.py --body`.

    python fbx2glb.py Angry.fbx                  -> Angry.glb next to it
    python fbx2glb.py Angry.fbx -o clips/angry.glb
    python fbx2glb.py clips/*.fbx                -> one GLB per file

FBX can only be read reliably by Blender, so this needs Blender installed,
but it does not need Blender on PATH: run it with ordinary Python and it
finds Blender itself (--blender, then $BLENDER, then PATH, then the usual
install folders on Windows, macOS and Linux) and re-runs itself inside it.
With the `bpy` module installed it converts in-process instead.

Meshes are dropped by default: a clip only needs its skeleton and motion,
and Mixamo's "With Skin" download carries a full character. --keep-mesh
keeps them.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fbx", nargs="+", help="FBX file(s)")
    ap.add_argument("-o", "--out", help="output .glb (single input only)")
    ap.add_argument("--keep-mesh", action="store_true",
                    help="keep meshes instead of exporting the skeleton only")
    ap.add_argument("--blender", help="path to the Blender executable")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# outside Blender: find it and hand over

def _version_key(path: str):
    digits = "".join(c if c.isdigit() or c == "." else " " for c in path)
    return [tuple(int(x) for x in part.split(".") if x)
            for part in digits.split() if part.strip(".")]


def find_blender(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if Path(explicit).is_file() else None
    env = os.environ.get("BLENDER")
    if env and Path(env).is_file():
        return env
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    patterns = []
    if sys.platform.startswith("win"):
        for root in {os.environ.get("ProgramFiles", r"C:\Program Files"),
                     os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")}:
            patterns.append(os.path.join(root, "Blender Foundation", "*",
                                         "blender.exe"))
        local = os.environ.get("LOCALAPPDATA")
        if local:   # Microsoft Store / winget user installs
            patterns.append(os.path.join(local, "Programs", "Blender*",
                                         "blender.exe"))
    elif sys.platform == "darwin":
        patterns += ["/Applications/Blender*.app/Contents/MacOS/Blender",
                     os.path.expanduser(
                         "~/Applications/Blender*.app/Contents/MacOS/Blender")]
    else:
        patterns += ["/snap/bin/blender", "/usr/bin/blender",
                     "/usr/local/bin/blender", "/opt/blender*/blender",
                     os.path.expanduser("~/blender*/blender")]
    found = [p for pat in patterns for p in glob.glob(pat) if Path(p).is_file()]
    return max(found, key=_version_key) if found else None


INSTALL_HINT = {
    "win": "winget install BlenderFoundation.Blender",
    "darwin": "brew install --cask blender",
    "linux": "sudo snap install blender --classic",
}


def run_in_blender(args, argv) -> int:
    exe = find_blender(args.blender)
    if exe is None:
        key = ("win" if sys.platform.startswith("win") else
               "darwin" if sys.platform == "darwin" else "linux")
        print("Blender was not found. Install it from https://www.blender.org"
              f" or with:\n    {INSTALL_HINT[key]}\n"
              "or point to it: python fbx2glb.py clip.fbx --blender "
              "\"C:\\path\\to\\blender.exe\"", file=sys.stderr)
        return 2
    print(f"using Blender: {exe}")
    # absolute paths: Blender's working directory is not ours to assume
    fwd = [str(Path(a).resolve()) if a in args.fbx or a == args.out else a
           for a in argv]
    cmd = [exe, "-b", "--factory-startup", "--python",
           str(Path(__file__).resolve()), "--"] + fwd
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    # Blender is chatty; pass through only this script's own lines and errors
    keep = ("fbx2glb", "error", "Error", "Traceback", "  File")
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.startswith(keep) or line.startswith("    "):
            print(line)
    return proc.returncode


# ---------------------------------------------------------------------------
# inside Blender (or with the bpy module)

def convert(bpy, src: Path, dst: Path, keep_mesh: bool) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(src))
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    acts = list(bpy.data.actions)
    if not arms:
        raise RuntimeError(f"{src.name}: no skeleton in the file")
    if not acts:
        raise RuntimeError(f"{src.name}: no animation in the file")
    if not keep_mesh:
        for o in [o for o in bpy.data.objects if o.type == "MESH"]:
            bpy.data.objects.remove(o, do_unlink=True)
    fps = bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
    dst.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(dst), export_format="GLB",
                              export_animations=True, export_skins=True,
                              export_force_sampling=True)
    a = acts[0]
    f0, f1 = a.frame_range
    secs = (f1 - f0) / fps
    print(f"fbx2glb: {src.name} -> {dst}  "
          f"{len(arms[0].data.bones)} bones, {len(acts)} animation(s), "
          f"'{a.name}' {int(f1 - f0 + 1)} frames @ {fps:g} fps = {secs:.2f}s")


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv \
        else sys.argv[1:]
    args = parse(argv)
    if args.out and len(args.fbx) > 1:
        print("fbx2glb: -o takes a single input", file=sys.stderr)
        return 2
    try:
        import bpy                                   # noqa: F401
    except ImportError:
        return run_in_blender(args, argv)

    failed = 0
    for f in args.fbx:
        src = Path(f)
        dst = Path(args.out) if args.out else src.with_suffix(".glb")
        if not src.is_file():
            print(f"fbx2glb: error: {src} not found")
            failed += 1
            continue
        try:
            convert(bpy, src, dst, args.keep_mesh)
        except Exception as e:                       # keep going on a batch
            print(f"fbx2glb: error: {e}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
