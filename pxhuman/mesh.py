"""Volume sampling, surface extraction and measurement verification."""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

from .anthro import BodyParams
from .sdf import BodyField


def body_bounds(p: BodyParams, pad: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    d = p.derived
    H = p.stature
    reach = d["shoulder_width"] * 0.46 + (
        d["arm_length"] + d["hand_length"]) * np.sin(np.radians(d["arm_abduction"]))
    x = reach + d["upperarm_girth"] / 6.0 + pad
    zf = max(d["chest"] * 0.22, d["foot_length"] * 0.75) + pad
    zb = d["hip"] * 0.26 + pad
    lo = np.array([-x, -pad * 0.4, -zb])
    hi = np.array([x, H + pad, zf])
    return lo, hi


def sample_volume(field: BodyField, lo, hi, voxel: float,
                  slab: int = 24, verbose: bool = True):
    nx, ny, nz = [max(8, int(np.ceil((hi[i] - lo[i]) / voxel)) + 1) for i in range(3)]
    xs = lo[0] + np.arange(nx) * voxel
    ys = lo[1] + np.arange(ny) * voxel
    zs = lo[2] + np.arange(nz) * voxel
    vol = np.empty((nx, ny, nz), dtype=np.float32)

    X, Z = np.meshgrid(xs, zs, indexing="ij")
    base = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], axis=1)

    for j0 in range(0, ny, slab):
        j1 = min(j0 + slab, ny)
        chunk = np.repeat(base[None, :, :], j1 - j0, axis=0).reshape(-1, 3)
        chunk[:, 1] = np.repeat(ys[j0:j1], base.shape[0])
        vals = field(chunk).astype(np.float32)
        vol[:, j0:j1, :] = vals.reshape(j1 - j0, nx, nz).transpose(1, 0, 2)
        if verbose and (j0 // slab) % 4 == 0:
            print(f"    sampling {100.0*j1/ny:5.1f}%", end="\r", flush=True)
    if verbose:
        print(f"    sampling 100.0%   grid={nx}x{ny}x{nz} ({nx*ny*nz/1e6:.1f}M)")
    return vol, (xs, ys, zs)


def extract_surface(vol: np.ndarray, lo, voxel: float):
    from skimage import measure
    verts, faces, normals, _ = measure.marching_cubes(
        vol, level=0.0, spacing=(voxel, voxel, voxel), allow_degenerate=False)
    verts = verts + np.asarray(lo, float)
    return verts, faces, -normals


def build_mesh(p: BodyParams, voxel: float = 0.60, verbose: bool = True,
               gains: Dict[str, float] | None = None, hand_detail: str = "mitt"):
    field = BodyField(p, gains=gains, hand_detail=hand_detail)
    lo, hi = body_bounds(p)
    if verbose:
        print(f"  bounds  {np.round(lo,1)} -> {np.round(hi,1)} cm   voxel={voxel} cm")
    vol, axes = sample_volume(field, lo, hi, voxel, verbose=verbose)
    verts, faces, normals = extract_surface(vol, lo, voxel)
    if verbose:
        print(f"  surface {len(verts):,} vertices  {len(faces):,} triangles")
    return verts, faces, normals, field


MEASURE_ABDUCTION = 75.0   # arms clear of the torso, as anthropometry requires


def measurement_pose(p: BodyParams) -> BodyParams:
    """A copy of the parameters posed so horizontal girth slices are valid.

    Torso girth is a property of the torso, but with the arms lowered the
    slice plane cuts through them and the perimeter is meaningless. Girths are
    therefore solved and audited in a standard pose, then the body is built in
    whatever pose the caller asked for.
    """
    import copy
    q = copy.deepcopy(p)
    q.arm_abduction = max(p.arm_abduction, MEASURE_ABDUCTION)
    return q.resolve()


def calibrate(p: BodyParams, voxel: float = 0.9, iters: int = 3,
              tol_pct: float = 0.35, hand_detail: str = "mitt",
              verbose: bool = True) -> Dict[str, float]:
    """Solve multiplicative profile gains so the *measured* surface girths hit
    the requested measurements.

    Smooth-union blending with the limbs perturbs the torso girth by a few
    percent, and the amount depends on the parameter combination. Rather than
    assume it away, the generator measures its own output and corrects.
    """
    import trimesh
    q = measurement_pose(p)
    d = q.derived
    heights = {"chest": d["chest_h"], "waist": d["waist_h"], "hip": d["hip_h"]}
    targets = {"chest": d["chest"], "waist": d["waist"], "hip": d["hip"]}
    caps = {k: v * 1.55 for k, v in targets.items()}
    gains = {k: 1.0 for k in targets}

    for it in range(iters):
        v, f, _, _ = build_mesh(q, voxel=voxel, verbose=False,
                                gains=gains, hand_detail=hand_detail)
        m = trimesh.Trimesh(vertices=v, faces=f, process=True)
        worst = 0.0
        line = []
        for key, tgt in targets.items():
            got = measure_circumference(m, heights[key], max_perimeter=caps[key])
            if not got:
                line.append(f"{key} n/a")
                continue
            err = (got - tgt) / tgt
            worst = max(worst, abs(err) * 100.0)
            gains[key] *= tgt / got
            line.append(f"{key} {err*100:+.2f}%")
        if verbose:
            print(f"  calibrate pass {it+1}: " + "  ".join(line))
        if worst <= tol_pct:
            break
    return gains


def audit(p: BodyParams, gains: Dict[str, float], voxel: float = 0.7,
          hand_detail: str = "mitt") -> Dict[str, dict]:
    """Independent verification pass: rebuild in the measurement pose with the
    solved gains and measure the result."""
    import trimesh
    q = measurement_pose(p)
    v, f, _, _ = build_mesh(q, voxel=voxel, verbose=False,
                            gains=gains, hand_detail=hand_detail)
    m = trimesh.Trimesh(vertices=v, faces=f, process=True)
    return verify(m, q)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def cross_section_loops(mesh, height: float):
    """Closed cross-section polylines at y=height, without shapely/rtree.

    Returns a list of (perimeter, centroid_x, n_points) for each closed loop.
    """
    sec = mesh.section(plane_origin=[0, height, 0], plane_normal=[0, 1, 0])
    if sec is None:
        return []
    loops = []
    for poly in sec.discrete:
        pts = np.asarray(poly, float)
        if len(pts) < 4:
            continue
        if np.linalg.norm(pts[0] - pts[-1]) > 1e-6:
            pts = np.vstack([pts, pts[0]])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        loops.append(dict(perimeter=float(seg.sum()),
                          cx=float(pts[:, 0].mean()),
                          xmin=float(pts[:, 0].min()),
                          xmax=float(pts[:, 0].max()),
                          n=len(pts)))
    return loops


def measure_circumference(mesh, height: float, max_perimeter: float | None = None
                          ) -> float | None:
    """Perimeter of the torso cross-section at `height`.

    The torso is the loop that straddles the centreline, not merely the one
    whose centroid is nearest it: a detached fragment or a limb section can sit
    closer to x=0 than the torso and silently become the measurement. Among
    straddling loops the largest wins, so an interior void is never mistaken
    for the outline.

    Returns None if the loop has merged with the arms, which makes the slice
    unmeasurable - the same reason real anthropometry specifies a pose.
    """
    loops = cross_section_loops(mesh, height)
    if not loops:
        return None
    straddling = [L for L in loops if L["xmin"] <= 0.0 <= L["xmax"]]
    if not straddling:
        return None
    per = max(L["perimeter"] for L in straddling)
    if max_perimeter is not None and per > max_perimeter:
        return None
    return per


def verify(mesh, p: BodyParams) -> Dict[str, dict]:
    d = p.derived
    checks = {
        "chest": (d["chest_h"], d["chest"]),
        "waist": (d["waist_h"], d["waist"]),
        "hip": (d["hip_h"], d["hip"]),
    }
    out: Dict[str, dict] = {}
    for name, (h, target) in checks.items():
        got = measure_circumference(mesh, h, max_perimeter=target * 1.55)
        if got is None:
            out[name] = dict(target=target, measured=None, error_pct=None)
            continue
        out[name] = dict(target=round(target, 2), measured=round(got, 2),
                         error_pct=round(100.0 * (got - target) / target, 2))
    # stature from the bounding box
    lo, hi = mesh.bounds
    out["stature"] = dict(target=round(p.stature, 2),
                          measured=round(float(hi[1] - lo[1]), 2),
                          error_pct=round(100.0 * ((hi[1] - lo[1]) - p.stature) / p.stature, 2))
    out["shoulder_width"] = dict(target=round(d["shoulder_width"], 2),
                                 measured=None, error_pct=None)
    lo_b, hi_b = mesh.bounds
    out["foot_length"] = dict(target=round(d["foot_length"], 2),
                              measured=round(float(hi_b[2] - lo_b[2]), 2),
                              error_pct=round(100.0 * ((hi_b[2] - lo_b[2])
                                              - d["foot_length"]) / d["foot_length"], 2))
    out["topology"] = dict(watertight=bool(mesh.is_watertight),
                           euler=int(mesh.euler_number),
                           genus=int((2 - mesh.euler_number) // 2),
                           components=int(mesh.body_count),
                           volume_litres=round(float(mesh.volume) / 1000.0, 2))
    return out
