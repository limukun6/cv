#!/usr/bin/env python3
"""Build the 3x3 contact sheet of the nine demo videos for the slide deck.

Every clip is an 8 s / 30 fps flythrough, rendered by the same evaluation pass
that produced the PSNR tables. This sheet exists because a static deck cannot
show motion: the grid proves coverage (three scenes x two view budgets), and the
playable slide next to it carries the actual motion.

Output: docs/figures/ppt_videos.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1A1F2B"
MUTED = "#5A6473"
FAINT = "#8B94A3"
ACCENT = "#1f77b4"

# (file, caption, row) -- row groups the clips by experimental condition.
CLIPS = [
    ("dl3dv-2_dense160.mp4",          "DL3DV-2  \u00b7  N = 160",         0),
    ("re10k-1_dense160.mp4",          "RealEstate10K-1  \u00b7  N = 160",  0),
    ("waymo_dense160.mp4",            "Waymo-405841  \u00b7  N = 160",     0),
    ("m360-counter_dense160.mp4",     "M360 counter  \u00b7  N = 160",     1),
    ("m360-bonsai_dense160.mp4",      "M360 bonsai  \u00b7  N = 160",      1),
    ("m360-flowers_dense160.mp4",     "M360 flowers  \u00b7  N = 160",     1),
    ("dl3dv-2.mp4",                   "DL3DV-2  \u00b7  N = 50",           2),
    ("m360-counter.mp4",              "M360 counter  \u00b7  N = 50",      2),
    ("m360-kitchen.mp4",              "M360 kitchen  \u00b7  N = 50",      2),
]
ROW_LABELS = ["dense \u00b7\nmandatory", "dense \u00b7\nMip-NeRF 360", "sparse \u00b7\nN = 50"]

# The clip embedded (and playable) on the following appendix slide. Waymo is the
# most legible flythrough and only 1.1 MB, so the deck stays small.
EMBED_CLIP = "waymo_dense160.mp4"


def grab(path: Path, frame_no: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {frame_no} of {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def letterbox(img, aspect=1.5):
    """Pad to a fixed aspect on white so every cell in the grid is identical.

    The clips range from 1:1 (256x256) to 1.52:1, so without this the grid rows
    end up at different heights and the whole sheet looks ragged.
    """
    import numpy as np
    h, w = img.shape[:2]
    if w / h > aspect:                      # too wide -> pad height
        nh, nw = int(round(w / aspect)), w
    else:                                   # too tall -> pad width
        nh, nw = h, int(round(h * aspect))
    # Light grey pad, not white: it reads as letterboxing rather than as part of
    # the frame (the 1:1 clips would otherwise look like they had white borders).
    canvas = np.empty((nh, nw, 3), dtype=np.uint8)
    canvas[:, :] = (0xEE, 0xF1, 0xF6)
    top, left = (nh - h) // 2, (nw - w) // 2
    canvas[top:top + h, left:left + w] = img
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="docs/videos")
    ap.add_argument("--out", default="docs/figures/ppt_videos.png")
    ap.add_argument("--frame", type=int, default=120, help="frame index to sample")
    args = ap.parse_args()

    vdir = Path(args.videos)
    fig = plt.figure(figsize=(11.6, 4.30))
    gs = fig.add_gridspec(3, 4, width_ratios=[0.50, 1, 1, 1],
                          wspace=0.05, hspace=0.22,
                          left=0.004, right=0.996, top=0.945, bottom=0.075)

    for i, (name, caption, row) in enumerate(CLIPS):
        col = i % 3
        ax = fig.add_subplot(gs[row, col + 1])
        img = letterbox(grab(vdir / name, args.frame))
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#D8DEE8"); sp.set_linewidth(0.8)
        ax.set_title(caption, fontsize=8.6, color=INK, pad=3.0)

    for r, lab in enumerate(ROW_LABELS):
        ax = fig.add_subplot(gs[r, 0])
        ax.axis("off")
        ax.text(0.97, 0.5, lab, ha="right", va="center", fontsize=8.4,
                color=ACCENT, fontweight="bold", transform=ax.transAxes)

    fig.text(0.5, 0.022,
             "Each clip: 8 s flythrough at 30 fps, rendered from the same evaluation pass that produced the tables. "
             "Frame 120 shown.",
             ha="center", va="center", fontsize=7.6, color=FAINT, style="italic")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"{out}  {out.stat().st_size/1024:.0f} KB")

    # Poster frame for the appendix slide that embeds a playable clip.
    poster = out.with_name("ppt_video_poster.png")
    cv2.imwrite(str(poster),
                cv2.cvtColor(grab(vdir / EMBED_CLIP, 0), cv2.COLOR_RGB2BGR))
    print(f"{poster}  {poster.stat().st_size/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
