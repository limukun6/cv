"""
44_plot_sparsity.py — 画「训练帧数 → PSNR」的退化曲线，以及过拟合差。

为什么这是 Part 2 / Part 3 的动机图
---------------------------------
原始数据（Plan A 上界，位姿由全序列 COLMAP 提供）：

    dl3dv-2   267 帧 → 33.41 dB   |   23 帧 → 20.60 dB   |   8 帧 → 16.38 dB
    re10k-1   244 帧 → 34.34 dB   |   21 帧 → 29.67 dB   |   7 帧 → 22.55 dB

同时 train PSNR 一路涨到 42~50 dB：
    → 模型把训练视角"背下来"了，但在没见过的视角上崩掉
    → 过拟合差从 0.7 dB 放大到 34 dB

这张图一眼说明三件事：
    1. 视角一稀疏，3DGS 就断崖式退化（不是渐变，是断崖）
    2. 退化幅度远超噪声地板（dl3dv 掉 12.8 dB vs 门槛 0.70 dB）
    3. 两个场景对稀疏的敏感度差 3 倍 → 场景尺度/类型影响极大

这正是 Part 3 要解决的问题：**观测不够，就用生成先验补。**

用法:
    python scripts/44_plot_sparsity.py
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(ROOT, "results", "results.csv")


def load():
    rows = []
    with open(CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("psnr") or not r.get("train_frames"):
                continue
            try:
                r["psnr"] = float(r["psnr"])
                r["train_frames"] = int(r["train_frames"])
                r["train_psnr"] = float(r["train_psnr"]) if r.get("train_psnr") else float("nan")
                r["lpips"] = float(r["lpips"]) if r.get("lpips") else float("nan")
            except ValueError:
                continue
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/sparsity_curve.png")
    args = ap.parse_args()

    rows = load()
    # 同一 (scene, split) 可能有多条（多种子），取均值/标准差
    g = defaultdict(list)
    for r in rows:
        g[(r["scene"], r["split"])].append(r)

    scenes = sorted({r["scene"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"dl3dv-2": "#1f77b4", "re10k-1": "#ff7f0e", "waymo-405841": "#2ca02c"}

    for scene in scenes:
        pts = []
        for (s, split), rs in g.items():
            if s != scene:
                continue
            n = int(np.mean([r["train_frames"] for r in rs]))
            p = np.array([r["psnr"] for r in rs])
            l = np.array([r["lpips"] for r in rs], dtype=float)
            tr = np.array([r["train_psnr"] for r in rs], dtype=float)
            pts.append((n, p.mean(), p.std(), np.nanmean(l), np.nanmean(tr)))
        if not pts:
            continue
        pts.sort()
        ns = [p[0] for p in pts]
        ps = [p[1] for p in pts]
        sd = [p[2] for p in pts]
        c = colors.get(scene, None)

        axes[0].errorbar(ns, ps, yerr=sd, marker="o", capsize=4, lw=2, ms=7,
                         color=c, label=scene, alpha=0.9)
        for n, p, _, l, _ in pts:
            axes[0].annotate(f"{n}f\n{p:.1f}", (n, p), textcoords="offset points",
                             xytext=(0, -24), ha="center", fontsize=8, color=c)
        # 过拟合差 = train - test（右图）
        gaps = [t - p for (_, p, _, _, t) in pts]
        axes[1].plot(ns, gaps, marker="s", lw=2, ms=7, color=c, label=scene)

    for ax, title, ylab in zip(
            axes,
            ["Novel-view quality vs. number of training views",
             "Overfitting gap (train PSNR - test PSNR)"],
            ["test PSNR (dB)", "gap (dB)"]):
        ax.set_xlabel("number of training frames (log)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks([10, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(230, 6)        # 从"多"到"少"，符合"越来越稀疏"的直觉

    fig.suptitle("Sparse-view degradation with ORACLE poses (COLMAP) — the Part-3 motivation",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"写出 {out}")

    print(f"\n{'场景':<14} {'训练帧':>6} {'test':>8} {'train':>8} {'过拟合差':>9} {'LPIPS':>8}")
    for (s, split), rs in sorted(g.items()):
        n = int(np.mean([r["train_frames"] for r in rs]))
        p = np.mean([r["psnr"] for r in rs])
        t = np.nanmean([r["train_psnr"] for r in rs])
        l = np.nanmean([r["lpips"] for r in rs])
        print(f"{s:<14} {n:>6} {p:>8.3f} {t:>8.3f} {t-p:>+9.2f} {l:>8.4f}")


if __name__ == "__main__":
    main()
