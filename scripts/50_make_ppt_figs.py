#!/usr/bin/env python3
"""Generate the two deck-specific charts for the Part 1 slide deck.

These do NOT replace the report figures -- they are presentation-optimised
redrawings of the same numbers:

  ppt_headline.png  grouped bars, PSNR at N=160, with the COLMAP-VGGT gaps
  ppt_runtime.png   initialization wall-clock, log axis, to show COLMAP's
                    resolution sensitivity against the flat feed-forward cost

Every value is copied from docs/PART1_REPORT.md sections 3.1 and 3.3.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Palette shared with 47_plot_scaling.py / 45_plot_p1_curves.py so the deck and
# the report read as one artefact.
STYLE = {
    "colmap": dict(color="#1f77b4", label="COLMAP"),
    "vggt": dict(color="#d62728", label="VGGT-1B"),
    "omega": dict(color="#2ca02c", label="VGGT-\u03a9"),
}
ORDER = ["colmap", "vggt", "omega"]
INK = "#1A1F2B"
MUTED = "#5A6473"
GRID = "#DCE2EC"

# --- results.csv / PART1_REPORT.md 3.1 -------------------------------------
SCENES = ["DL3DV-2", "RealEstate10K-1", "Waymo-405841"]
PSNR = {
    "colmap": [32.17, 34.30, 32.71],
    "vggt":   [29.08, 27.61, 23.43],
    "omega":  [28.40, 32.54, 24.72],
}
STD = {
    "colmap": [0.54, 0.03, 1.40],
    "vggt":   [0.07, 0.14, 0.23],
    "omega":  [0.04, 0.03, 0.07],
}

# --- PART1_REPORT.md 3.3 ---------------------------------------------------
# Ordered by input resolution, not by scene name: the point of the chart is that
# COLMAP's cost tracks pixel count. kitchen is a bonus-phase run (85 views, not
# ~190), which is labelled on the axis because the frame counts differ -- fewer
# views and 5.9x the wall-clock makes the resolution argument stronger, not weaker.
RUNTIME_SCENES = ["DL3DV-2\n(256\u00b2, 199 views)",
                  "RealEstate10K-1\n(256\u00b2, 195 views)",
                  "Mip-NeRF 360 kitchen\n(1559\u00d71039, 85 views)",
                  "Waymo-405841\n(1920\u00d71280, 185 views)"]
RUNTIME = {
    "colmap": [133.0, 270.0, 782.0, 1950.0],
    "vggt":   [162.0, 222.0, None, 156.0],
    "omega":  [96.0, 102.0, None, 120.0],
}
APPROX = {"colmap": [False, False, False, False],
          "vggt":   [False, False, False, False],
          "omega":  [False, False, False, True]}   # Waymo \u03a9 is ~120 s


def _style_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def headline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.15))
    x = np.arange(len(SCENES))
    width = 0.255

    for i, key in enumerate(ORDER):
        off = (i - 1) * width
        vals = PSNR[key]
        errs = STD[key]
        bars = ax.bar(x + off, vals, width * 0.92, yerr=errs,
                      color=STYLE[key]["color"], label=STYLE[key]["label"],
                      capsize=3, error_kw=dict(elinewidth=1.1, ecolor=MUTED, alpha=0.75),
                      zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.85, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9.5,
                    color=STYLE[key]["color"], fontweight="bold")

    # One gap callout per group, stacked above the bars: COLMAP's margin over
    # each feed-forward model, coloured to match the series it refers to.
    for j, scene in enumerate(SCENES):
        d1 = PSNR["colmap"][j] - PSNR["vggt"][j]
        dw = PSNR["colmap"][j] - PSNR["omega"][j]
        ax.text(x[j], 38.4, f"+{d1:.1f} dB vs VGGT-1B", ha="center", va="bottom",
                fontsize=9.5, color=STYLE["vggt"]["color"], fontweight="bold")
        ax.text(x[j], 39.9, f"+{dw:.1f} dB vs VGGT-\u03a9", ha="center", va="bottom",
                fontsize=9.5, color=STYLE["omega"]["color"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(SCENES, fontsize=11, color=INK)
    ax.set_ylabel("test PSNR (dB)", fontsize=10, color=MUTED)
    ax.set_ylim(20, 41.2)
    ax.set_yticks(np.arange(20, 37, 4))
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left", ncol=3,
              bbox_to_anchor=(0.0, -0.30), labelcolor=INK)
    ax.text(1.0, -0.34, "y-axis truncated at 20 dB  \u00b7  error bars = run-to-run std (2\u20133 seeds)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def runtime(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.75))
    y = np.arange(len(RUNTIME_SCENES))
    height = 0.255

    for i, key in enumerate(ORDER):
        off = (i - 1) * height
        vals = RUNTIME[key]
        # None = this backend was not timed on that scene (kitchen, bonus phase)
        idx = [j for j, v in enumerate(vals) if v is not None]
        bars = ax.barh([y[j] + off for j in idx], [vals[j] for j in idx],
                       height * 0.92, color=STYLE[key]["color"],
                       label=STYLE[key]["label"], zorder=3)
        for b, j in zip(bars, idx):
            v = vals[j]
            txt = (f"~{v:.0f} s" if APPROX[key][j] else f"{v:.0f} s")
            if v >= 1000 and not APPROX[key][j]:
                txt = f"{v/60:.1f} min"
            ax.text(v * 1.10, b.get_y() + b.get_height() / 2, txt,
                    va="center", ha="left", fontsize=9.5,
                    color=STYLE[key]["color"], fontweight="bold")

    # kitchen has no feed-forward timing; say so instead of leaving a hole.
    k = RUNTIME_SCENES.index("Mip-NeRF 360 kitchen\n(1559\u00d71039, 85 views)")
    ax.text(RUNTIME["colmap"][k] * 1.10, y[k] + height * 1.35,
            "feed-forward not timed on this scene", va="center", ha="left",
            fontsize=7.8, color=MUTED, style="italic")

    ax.set_xscale("log")
    ax.set_xlim(60, 9000)
    ax.set_yticks(y)
    ax.set_yticklabels(RUNTIME_SCENES, fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("initialization wall-clock (s, log scale)", fontsize=10, color=MUTED)
    _style_axes(ax, grid_axis="x")
    ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="lower right", labelcolor=INK)
    ax.text(0.0, 1.02, "COLMAP's cost tracks input resolution; the feed-forward backends stay flat",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/figures")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    headline(outdir / "ppt_headline.png")
    runtime(outdir / "ppt_runtime.png")
    for name in ("ppt_headline.png", "ppt_runtime.png"):
        p = outdir / name
        print(f"{p}  {p.stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
