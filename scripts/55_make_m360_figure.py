#!/usr/bin/env python3
"""Build the Mip-NeRF 360 qualitative grid: GT | COLMAP | VGGT-Omega.

The three-way script (48_make_3way_figure.py) hardcodes three inits, and the
Mip-NeRF 360 rerun only has two (we never re-ran VGGT-1B on the verified
splits). This builds a two-backend grid for two scenes so the bonus section has
a picture instead of only a table.

Per-view PSNR is recomputed here from the GT/render pairs rather than read from
per_view.json, so the corner labels cannot drift out of sync with the images.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

INK = "#1A1F2B"
MUTED = "#5A6473"
FAINT = "#8B94A3"
BLUE = "#1f77b4"
GREEN = "#2ca02c"


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    # Normalise to [0, 1] first: these are uint8 arrays, so a raw MSE would be
    # in units of 255^2 and the formula would return a negative "dB".
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return 10.0 * math.log10(1.0 / mse) if mse > 0 else 99.0


def load_pair(run: Path, name: str):
    gt = np.asarray(Image.open(run / "test/ours_30000/gt" / name).convert("RGB"))
    rd = np.asarray(Image.open(run / "test/ours_30000/renders" / name).convert("RGB"))
    if gt.shape != rd.shape:                     # 41_eval resizes GT to the render size
        gt = np.asarray(Image.fromarray(gt).resize((rd.shape[1], rd.shape[0]), Image.LANCZOS))
    return gt, rd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="scene:colmap_run:omega_run (one per scene)")
    ap.add_argument("--scene-labels", nargs="+", default=None)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--label-mode", choices=["psnr", "index"], default="psnr")
    ap.add_argument("--out", default="docs/figures/p1_qual_m360.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    blocks = []
    for spec in args.pairs:
        scene, cm, om = spec.split(":")
        blocks.append((scene, Path(cm), Path(om)))
    labels = args.scene_labels or [b[0] for b in blocks]

    # Pick evenly spaced test views, identical indices across scenes.
    sample = sorted(blocks[0][1].glob("test/ours_30000/gt/*"))
    names_all = [p.name for p in sample]
    if not names_all:
        raise SystemExit("no gt images found")
    idx = [round(i * (len(names_all) - 1) / max(1, args.views - 1)) for i in range(args.views)]
    names = [names_all[i] for i in sorted(set(idx))]

    nrow = len(blocks) * len(names)
    fig = plt.figure(figsize=(10.6, 0.95 + 2.05 * nrow))
    gs = fig.add_gridspec(nrow, 4, width_ratios=[0.42, 1, 1, 1],
                          wspace=0.035, hspace=0.10,
                          left=0.004, right=0.996, top=0.945, bottom=0.035)

    col_psnr = {0: [], 1: [], 2: []}
    row = 0
    for bi, (scene, cm_run, om_run) in enumerate(blocks):
        for vi, name in enumerate(names):
            gt, rd_cm = load_pair(cm_run, name)
            _, rd_om = load_pair(om_run, name)
            for col, (img, tag) in enumerate(((gt, "gt"), (rd_cm, "colmap"), (rd_om, "omega"))):
                ax = fig.add_subplot(gs[row, col + 1])
                ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#D8DEE8"); sp.set_linewidth(0.7)
                if tag != "gt":
                    p = psnr(gt, img)
                    col_psnr[col].append(p)
                    txt = f"{p:.1f} dB" if args.label_mode == "psnr" else name
                    ax.text(0.985, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=7.6, color="white",
                            bbox=dict(boxstyle="round,pad=0.18", fc="#000000", ec="none", alpha=0.55))
                elif args.label_mode == "index":
                    ax.text(0.985, 0.03, name, transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=7.6, color="white",
                            bbox=dict(boxstyle="round,pad=0.18", fc="#000000", ec="none", alpha=0.55))
                if row == 0:
                    head = {"gt": "Ground truth", "colmap": "COLMAP",
                            "omega": "VGGT-\u03a9"}[tag]
                    ax.set_title(head, fontsize=11.5, color=INK, fontweight="bold", pad=9)
            if vi == 0:
                axl = fig.add_subplot(gs[row:row + len(names), 0])
                axl.axis("off")
                axl.text(0.92, 0.5, labels[bi], ha="right", va="center", rotation=0,
                         fontsize=10.5, color=BLUE, fontweight="bold", transform=axl.transAxes)
            row += 1

    mean_line = " / ".join(
        f"{nm} {np.mean(col_psnr[c]):.2f}" for c, nm in
        ((1, "COLMAP"), (2, "VGGT-\u03a9")))
    fig.text(0.5, 0.006,
             f"Mip-NeRF 360, identical test views  \u00b7  means {mean_line} dB  "
             f"\u00b7  corners are per-view",
             ha="center", va="center", fontsize=7.4, color=FAINT, style="italic")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"{out}  {out.stat().st_size/1024:.0f} KB")
    for c, nm in ((1, "COLMAP"), (2, "VGGT-\u03a9")):
        print(f"  shown-view mean {nm}: {np.mean(col_psnr[c]):.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
