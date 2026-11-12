#!/usr/bin/env python3
"""Convert the demo flythroughs to animated GIFs for the slide deck.

Why GIFs: an embedded mp4 needs a click and a working codec/decoder on the
presenting machine, and it renders as a dead still in PDF. An animated GIF
autoplays and loops in PowerPoint and WPS slideshow with no interaction, so the
motion is visible even if media embedding is blocked.

Size is the whole problem -- 8 s at 30 fps is 240 frames. Two knobs are exposed:
  --fps     frame rate of the GIF (drops frames from the source)
  --width   pixel width (height follows the source aspect)
  --colors  palette size (fewer colours shrink the file but band gradients)

Output: docs/gifs/<name>.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image


def convert(src: Path, dst: Path, *, width: int, fps: float, colors: int) -> Path:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Never upscale: a 256px source stretched to 320 is just a bigger file that
    # looks no better in a 2.25-inch cell.
    width = min(width, native_w or width)
    step = max(1, round(src_fps / fps))

    frames: list[Image.Image] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            nh = max(2, int(round(h * width / w)))
            frame = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).convert("P", palette=Image.Palette.ADAPTIVE,
                                              colors=colors)
            frames.append(img)
        i += 1
    cap.release()

    if not frames:
        raise SystemExit(f"no frames decoded from {src}")

    # Duration must follow the frames we actually kept, not the requested rate.
    # step is an integer, so 30 fps asked for 12 fps yields step=2 (15 fps); using
    # 1/12 s per frame there would stretch an 8 s clip to 10 s.
    per_frame_ms = int(round(1000.0 * step / src_fps))

    dst.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(dst),
        save_all=True,
        append_images=frames[1:],
        duration=per_frame_ms,
        loop=0,                 # loop forever: a presenter should not have to restart it
        optimize=True,
        disposal=2,             # clear each frame, otherwise motion smears
    )
    return dst


ALL_CLIPS = [
    "dl3dv-2_dense160.mp4", "re10k-1_dense160.mp4", "waymo_dense160.mp4",
    "m360-counter_dense160.mp4", "m360-bonsai_dense160.mp4", "m360-flowers_dense160.mp4",
    "dl3dv-2.mp4", "m360-counter.mp4", "m360-kitchen.mp4",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="docs/videos")
    ap.add_argument("--outdir", default="docs/gifs")
    ap.add_argument("--clip", default="waymo_dense160.mp4")
    ap.add_argument("--out", default=None,
                    help="explicit output path for a single clip (e.g. the big playback GIF)")
    ap.add_argument("--all", action="store_true",
                    help="convert all nine clips (used by the 3x3 live grid slide)")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--colors", type=int, default=192)
    args = ap.parse_args()

    clips = ALL_CLIPS if args.all else [args.clip]
    total = 0
    for clip in clips:
        if args.out and not args.all:
            out = Path(args.out)
        else:
            out = Path(args.outdir) / (Path(clip).stem + ".gif")
        convert(Path(args.videos) / clip, out,
                width=args.width, fps=args.fps, colors=args.colors)
        with Image.open(out) as im:
            n = getattr(im, "n_frames", 1)
        kb = out.stat().st_size / 1024
        total += kb
        print(f"  {out.name:32s} {kb:7.0f} KB  {n:3d} frames")
    print(f"  {'TOTAL':32s} {total/1024:7.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
