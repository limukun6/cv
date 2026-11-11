"""
49_plot_convergence_n160.py —— 稠密设定（N=160）的收敛曲线。

为什么单独写
------------
`45_plot_p1_curves.py` 画的是 50 帧那批（含 Ω 的缺项）。原文 L186-188 要求
"visualize and compare: Part 1: Reconstruction quality **and convergence** among
[COLMAP+3DGS], [VGGT+3DGS], and [VGGT-Ω+3DGS]" —— 三路都要有，而且应该在
Part 1 的稠密主设定下给出。

数据来源
--------
各 run 的 train.log 里 `[ITER N] Evaluating test: ... PSNR X` 行，
检查点是 1k/5k/10k/15k/20k/25k/30k（矩阵脚本传了 --test_iterations）。

除了画图，还会把表写进 results/p1_curves_n160.csv，便于报告直接引用。
"""
import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "outputs", "runs")

INITS = [("colmap", "COLMAP", "#1f77b4", "o"),
         ("vggt", "VGGT-1B", "#d62728", "s"),
         ("vggt_omega", "VGGT-Ω", "#2ca02c", "^")]
SCENES = ["dl3dv-2", "re10k-1", "waymo-405841"]

# [ITER 1000] Evaluating test: L1 0.067 PSNR 19.076 [11/11 15:05:02]
RE_TEST = re.compile(r"\[ITER (\d+)\] Evaluating test:.*?PSNR ([0-9.]+)")


def find_run(scene, init):
    pat = os.path.join(RUNS, f"{scene}__{init}__n160__3dgs__*")
    cands = [d for d in glob.glob(pat) if os.path.isdir(d) and "__seed" not in d]
    if not cands:
        return None
    # 必须有 results.json（排除失败的 run）
    ok = [d for d in cands if os.path.isfile(os.path.join(d, "results.json"))]
    pool = ok or cands
    return sorted(pool, key=os.path.getmtime)[-1]


def read_curve(run_dir):
    """返回 [(iter, test_psnr), ...]，按迭代排序。"""
    p = os.path.join(run_dir, "train.log")
    if not os.path.isfile(p):
        p = run_dir + ".train.log"
    if not os.path.isfile(p):
        return []
    txt = open(p, encoding="utf-8", errors="replace").read().replace("\r", "\n")
    pts = {}
    for m in RE_TEST.finditer(txt):
        pts[int(m.group(1))] = float(m.group(2))
    return sorted(pts.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures",
                                                  "p1_convergence_n160.png"))
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "p1_curves_n160.csv"))
    args = ap.parse_args()

    data = {}
    for sc in SCENES:
        for key, _, _, _ in INITS:
            r = find_run(sc, key)
            if r:
                c = read_curve(r)
                if c:
                    data[(sc, key)] = c

    if not data:
        print("✗ 没找到任何训练日志")
        return 1

    # ---- 写 CSV ----
    iters = sorted({i for c in data.values() for i, _ in c})
    with open(args.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "init", "run_id"] + [f"iter_{i}" for i in iters] + ["peak_iter", "peak_psnr"])
        for (sc, key), c in sorted(data.items()):
            d = dict(c)
            pk = max(c, key=lambda x: x[1])
            w.writerow([sc, key, os.path.basename(find_run(sc, key))]
                       + [f"{d[i]:.4f}" if i in d else "" for i in iters]
                       + [pk[0], f"{pk[1]:.4f}"])
    print(f"写出 {args.csv}")

    # ---- 画图 ----
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    fig.suptitle("Part 1 — 3DGS convergence at the dense setting (N = 160), test PSNR",
                 fontsize=13.5, fontweight="bold")
    for ax, sc in zip(axes, SCENES):
        for key, name, color, marker in INITS:
            c = data.get((sc, key))
            if not c:
                continue
            xs = [i / 1000 for i, _ in c]
            ys = [v for _, v in c]
            ax.plot(xs, ys, color=color, marker=marker, markersize=5.5,
                    linewidth=2, label=name)
            pk = max(c, key=lambda x: x[1])
            last = c[-1]
            # 峰值不在最后一个检查点时特别标出来（"练久了反而差"）
            if pk[0] != last[0] and pk[1] - last[1] > 0.05:
                ax.scatter([pk[0] / 1000], [pk[1]], s=140, facecolors="none",
                           edgecolors=color, linewidths=2, zorder=5)
                ax.annotate(f"peak {pk[1]:.2f} @ {pk[0]//1000}k\n→ {last[1]:.2f} @ 30k",
                            xy=(pk[0] / 1000, pk[1]), xytext=(6, -22),
                            textcoords="offset points", fontsize=7.6, color=color)
            ax.annotate(f"{ys[-1]:.2f}", xy=(xs[-1], ys[-1]), xytext=(5, -3),
                        textcoords="offset points", fontsize=8, color=color)
        ax.set_title(sc, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("training iteration (k)", fontsize=10)
        ax.set_ylabel("test PSNR (dB)", fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_xlim(0, 32.5)
    axes[0].legend(fontsize=9, loc="lower right", framealpha=0.93)
    fig.text(0.5, 0.005,
             "Hollow circles mark configurations whose peak occurs before 30 k and then decays; "
             "the two numbers give peak and final test PSNR.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"写出 {args.out}")

    # ---- 终端汇总 ----
    print(f"\n{'场景':<14}{'初始化':<10}" + "".join(f"{i//1000:>7}k" for i in iters) + f"{'峰值':>14}")
    for sc in SCENES:
        for key, name, _, _ in INITS:
            c = data.get((sc, key))
            if not c:
                continue
            d = dict(c)
            pk = max(c, key=lambda x: x[1])
            row = "".join(f"{d[i]:7.2f}" if i in d else "      -" for i in iters)
            flag = "" if pk[0] == c[-1][0] else f"  ← {pk[0]//1000}k 后下降"
            print(f"{sc:<14}{name:<10}{row}{pk[0]//1000:>6}k/{pk[1]:.2f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
