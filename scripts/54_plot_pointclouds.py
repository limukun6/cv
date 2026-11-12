#!/usr/bin/env python3
"""Render the two point clouds we actually have locally, side by side.

  data/dl3dv-2/colmap/sparse/0/points3D.ply       the seed: 7,860 triangulated points
  outputs/runs/dl3dv-2__colmap__full__3dgs__20261030-0952/
      point_cloud/iteration_30000/point_cloud.ply  the result: 507,766 Gaussians

Both come from the same run -- the run's input.ply is byte-for-byte the same
point set as the COLMAP model (verified: identical bbox and vertex count), so
the pair is a legitimate before/after, not two different experiments.

Why this figure exists: it answers "what does the pipeline actually output?".
It does NOT support the track-length argument -- that needs the VGGT cloud,
which was never pulled off the instance. Keep the caption honest about it.

No GPU: points are projected with a plain pinhole and drawn back-to-front.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SH_C0 = 0.28209479177387814
BG = "#12161F"
INK = "#1A1F2B"
MUTED = "#5A6473"
FAINT = "#8B94A3"


def read_ply(path: Path) -> dict:
    """Minimal PLY reader: ascii or binary_little_endian, scalar props."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise SystemExit(f"not a ply: {path}")
        fmt, n, props, in_vertex = None, 0, [], False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n = int(line.split()[2])
                in_vertex = True
            elif line.startswith("element"):
                in_vertex = False
            elif line.startswith("property") and in_vertex:
                parts = line.split()
                props.append((parts[-1], parts[1]))
            elif line == "end_header":
                break
        names = [p[0] for p in props]
        types = [p[1] for p in props]
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n, dtype=np.float64)
            return {k: data[:, i] for i, k in enumerate(names)}
        dt = {"float": "<f4", "float32": "<f4", "double": "<f8",
              "uchar": "u1", "uint8": "u1", "int": "<i4"}
        array = np.frombuffer(f.read(n * np.dtype([(a, dt[b]) for a, b in zip(names, types)]).itemsize),
                              dtype=np.dtype([(a, dt[b]) for a, b in zip(names, types)]), count=n)
        return {k: array[k] for k in names}


def load_camera(run_dir: Path, which: str = "middle"):
    cams = json.load(open(run_dir / "cameras.json", encoding="utf-8-sig"))
    pos = np.array([c["position"] for c in cams])
    centre = pos.mean(0)
    # A camera near the middle of the rig sees the scene from a representative
    # angle instead of the first frame's edge-on view.
    i = int(np.argmin(np.linalg.norm(pos - centre, axis=1)))
    c = cams[i]
    R_c2w = np.array(c["rotation"], dtype=np.float64)
    fx, fy = float(c["fx"]), float(c["fy"])
    W, H = int(c["width"]), int(c["height"])
    return dict(R_c2w=R_c2w, C=pos[i], fx=fx, fy=fy, W=W, H=H, n=len(cams))


def project(xyz, cam):
    R_wc = cam["R_c2w"].T
    p = (xyz - cam["C"]) @ R_wc.T          # world -> camera
    z = p[:, 2]
    ok = z > 1e-6
    u = cam["fx"] * p[:, 0] / np.where(ok, z, 1.0) + cam["W"] * 0.5
    v = cam["fy"] * p[:, 1] / np.where(ok, z, 1.0) + cam["H"] * 0.5
    return u, v, z, ok


def panel(ax, xyz, rgb, cam, title):
    u, v, z, ok = project(xyz, cam)
    u, v, z, rgb = u[ok], v[ok], z[ok], rgb[ok]
    # Draw far points first so near ones land on top (matplotlib paints in order).
    order = np.argsort(-z)
    ax.scatter(u[order], v[order], c=rgb[order], s=1.2, marker="o",
               linewidths=0, alpha=0.95, rasterized=True)
    ax.set_xlim(0, cam["W"])
    ax.set_ylim(cam["H"], 0)
    # Without this the square image is stretched to fill a wide axes box and the
    # scene comes out horizontally smeared.
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#2A3140")
    ax.set_title(title, fontsize=11.5, color=INK, fontweight="bold", pad=9)
    return len(u)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="dl3dv-2")
    ap.add_argument("--run", default="dl3dv-2__colmap__full__3dgs__20261030-0952")
    ap.add_argument("--out", default="docs/figures/p1_pointcloud.png")
    ap.add_argument("--max-points", type=int, default=150_000,
                    help="random subsample of the Gaussian cloud for drawing speed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(".")
    seed_ply = root / "data" / args.scene / "colmap" / "sparse" / "0" / "points3D.ply"
    run_dir = root / "outputs" / "runs" / args.run
    gauss_ply = run_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"

    s = read_ply(seed_ply)
    seed_xyz = np.stack([s["x"], s["y"], s["z"]], 1).astype(np.float64)
    seed_rgb = np.stack([s["red"], s["green"], s["blue"]], 1).astype(np.float64) / 255.0

    g = read_ply(gauss_ply)
    xyz = np.stack([g["x"], g["y"], g["z"]], 1).astype(np.float64)
    dc = np.stack([g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]], 1).astype(np.float64)
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    alpha = 1.0 / (1.0 + np.exp(-np.asarray(g["opacity"], dtype=np.float64)))
    keep = alpha > 0.05                     # drop essentially invisible floaters
    xyz, rgb = xyz[keep], rgb[keep]
    n_visible = len(xyz)
    if n_visible > args.max_points:
        rs = np.random.default_rng(args.seed)
        sel = rs.choice(n_visible, args.max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]

    cam = load_camera(run_dir)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.45))
    # Explicit margins: set_aspect("equal") re-shrinks the axes inside whatever
    # box it is given, and tight_layout does not account for that, which clipped
    # the titles and stacked the captions on top of each other.
    fig.subplots_adjust(left=0.015, right=0.985, top=0.905, bottom=0.155, wspace=0.05)

    n1 = panel(axes[0], seed_xyz, seed_rgb, cam,
               f"COLMAP seed \u00b7 {len(seed_xyz):,} triangulated points")
    n2 = panel(axes[1], xyz, rgb, cam,
               f"after 30 k iterations \u00b7 {507_766:,} Gaussians")

    fig.text(0.262, 0.075, "mean track length 8.19  \u00b7  what 3DGS starts from",
             ha="center", va="center", fontsize=9, color=MUTED)
    fig.text(0.738, 0.075,
             f"{n_visible:,} above opacity 0.05  \u00b7  {len(xyz):,} drawn",
             ha="center", va="center", fontsize=9, color=MUTED)
    fig.text(0.5, 0.028,
             "DL3DV-2  \u00b7  same run, same camera (one of 306 training views), same projection  "
             "\u00b7  Gaussian colour = SH DC term, sizes not to scale",
             ha="center", va="center", fontsize=8.0, color=FAINT, style="italic")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"{out}  {out.stat().st_size/1024:.0f} KB   seed={n1}  gauss_drawn={n2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
