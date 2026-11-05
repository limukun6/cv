"""
47_plot_scaling.py — Part 1 的核心图：帧数预算如何改变三种初始化的相对优劣。

画什么
------
左列：渲染质量（PSNR）随训练帧数变化 —— 50 / 160 / 全量，三个场景各一个子图
右列：位姿精度（ATE）随训练帧数变化 —— 50 / 160

为什么这张图重要
--------------
原文 Part 1 说 "study how data-driven scaling affects reconstruction"（PDF L11），
所以图上必须能直接看出"给更多观测时，谁上升得快、谁几乎不动"。

三个可以一眼看出的结论：
  1. COLMAP 的曲线最陡（帧数越多优势越大）
  2. VGGT 和 Ω 的曲线明显更平
  3. 只有 dl3dv 上 Ω 会从第一名掉到最后 —— 因为它上升最慢

数据来源
--------
全部来自 results/results.csv（split=n50 / n160 / full）与 results/ate.csv。
全量条件只有 COLMAP（VGGT 在 300 帧级别会超显存），所以右端用空心点区分。

用法:
    python scripts/47_plot_scaling.py
    python scripts/47_plot_scaling.py --out docs/figures/p1_scaling.png
"""
import argparse
import csv
import math
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results", "results.csv")
ATE = os.path.join(ROOT, "results", "ate.csv")

# 作废的 run：相机发散修复前的两次 + 手工探针
VOID = ("20261102-2207", "20261102-2225", "probe_", "fin_colmap")

STYLE = {
    "colmap":     dict(color="#1f77b4", marker="o", label="COLMAP"),
    "vggt":       dict(color="#d62728", marker="s", label="VGGT-1B"),
    "vggt_omega": dict(color="#2ca02c", marker="^", label="VGGT-Ω"),
}
# 画图用的样式副本（不含 label，label 只在需要时单独传一次）
DRAW = {k: {kk: vv for kk, vv in v.items() if kk != "label"} for k, v in STYLE.items()}
SCENES = [("dl3dv-2", "DL3DV-2"), ("re10k-1", "RealEstate10K-1"), ("waymo-405841", "Waymo-405841")]
# 各条件的训练帧数（用于横轴）
X = {"n50": 50, "n160": 160, "full": 267}


def _legend_handles():
    """手工构造图例句柄。

    不能依赖画线时传 label：同一种初始化会画多段（实线 + 全量虚线），
    直接 legend() 会出现重复条目（第一版就是这样）。
    """
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=v["color"], marker=v["marker"], linewidth=2,
                   markersize=7, label=v["label"]) for v in STYLE.values()]


def load_render():
    """{(scene, init, split): (mean, std, n)}"""
    g = defaultdict(list)
    with open(RES, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not r.get("psnr") or any(m in (r.get("run_id") or "") for m in VOID):
                continue
            try:
                g[(r["scene"], r["source"], r["split"])].append(float(r["psnr"]))
            except ValueError:
                continue
    out = {}
    for k, v in g.items():
        out[k] = (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))
    return out


def load_ate():
    out = {}
    if not os.path.isfile(ATE):
        return out
    with open(ATE, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[(r["scene"], r["init"])] = float(r["ate_rmse_pct"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "p1_scaling.png"))
    args = ap.parse_args()

    render = load_render()
    ate = load_ate()

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    fig.suptitle("Part 1 — how the view budget changes the ranking of initialisation backends",
                 fontsize=13.5, fontweight="bold")

    for col, (sc, name) in enumerate(SCENES):
        ax = axes[0][col]
        for init in ("colmap", "vggt", "vggt_omega"):
            xs, ys, es = [], [], []
            for split, x in X.items():
                d = render.get((sc, init, split))
                if d:
                    xs.append(x); ys.append(d[0]); es.append(d[1])
            if not xs:
                continue
            stl = dict(DRAW[init])
            # 全量只有 COLMAP 有，用空心标记那一段
            solid_n = 0 if (sc, init, "full") not in render else len(xs) - 1
            if solid_n > 0:
                ax.errorbar(xs[:solid_n + 1], ys[:solid_n + 1], yerr=es[:solid_n + 1],
                            capsize=3, linewidth=2, markersize=7, **stl)
            if solid_n < len(xs) - 1 or (sc, init, "full") in render:
                stl_open = dict(stl); stl_open["markerfacecolor"] = "none"
                ax.plot(xs, ys, linewidth=1.2, linestyle=":", markersize=8, **stl_open)
            # 标注每个点和标准差
            for x, y, e in zip(xs, ys, es):
                ax.annotate(f"{y:.1f}" + (f"±{e:.1f}" if e > 0.01 else ""),
                            xy=(x, y), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=7.5, color=stl["color"])
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("training views", fontsize=10)
        ax.set_ylabel("Test PSNR (dB)", fontsize=10)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks([50, 160, 267])
        ax.set_xticklabels(["50", "160", "267\n(full)"])
        ax.grid(alpha=0.3, linestyle="--")
        if col == 0:
            ax.legend(handles=_legend_handles(), fontsize=8.5, loc="lower right",
                      framealpha=0.93)

        # ---- 下排：ATE
        ax2 = axes[1][col]
        for init in ("colmap", "vggt", "vggt_omega"):
            pts = []
            for split, x in (("n50", 50), ("n160", 160)):
                key = (sc, init) if split == "n50" else (sc, f"{init}_n160")
                if key in ate:
                    pts.append((x, ate[key]))
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            ax2.plot(xs, ys, linewidth=2, markersize=7, **DRAW[init])
            for x, y in pts:
                ax2.annotate(f"{y:.2f}%", xy=(x, y), xytext=(0, 7),
                             textcoords="offset points", ha="center",
                             fontsize=8, color=STYLE[init]["color"])
        ax2.set_xlabel("training views", fontsize=10)
        ax2.set_ylabel("ATE (% of diagonal) ↓", fontsize=10)
        ax2.set_xscale("log")
        ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax2.set_xticks([50, 160])
        ax2.set_xticklabels(["50", "160"])
        ax2.grid(alpha=0.3, linestyle="--")
        ax2.set_title(f"{name} — pose accuracy", fontsize=10)
        if col == 0:
            ax2.legend(handles=_legend_handles(), fontsize=8.5, loc="upper right",
                       framealpha=0.93)

    fig.text(0.5, 0.005,
             "Top: rendering quality vs. view budget (error bars = run-to-run std; dotted = full-sequence baseline, COLMAP only). "
             "Bottom: pose accuracy; lower is better.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"写出 {args.out}")

    # 顺便把图里的关键趋势打成表，便于写作时引用
    print(f"\n{'场景':<14}{'初始化':<12}{'50帧':>8}{'160帧':>8}{'提升':>8}")
    for sc, name in SCENES:
        for init in ("colmap", "vggt", "vggt_omega"):
            a = render.get((sc, init, "n50")); b = render.get((sc, init, "n160"))
            if a and b:
                print(f"{name:<14}{init:<12}{a[0]:>8.2f}{b[0]:>8.2f}{b[0]-a[0]:>+8.2f}")


if __name__ == "__main__":
    main()
