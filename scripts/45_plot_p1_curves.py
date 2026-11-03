"""
45_plot_p1_curves.py — Part 1 的 3DGS 收敛曲线（COLMAP vs VGGT）。

数据来源
--------
`results/p1_curves.csv`，是从租用机器上每次 30k 训练的 train.log 里
逐行解析出来的 `[ITER n] Evaluating test: ... PSNR x` 记录（test_iterations 设为
1000/5000/.../30000）。

画什么
------
每张子图一个场景，两条线（COLMAP / VGGT），横轴迭代次数（log），纵轴测试集 PSNR。
标注峰值位置——这一步是关键，因为 dl3dv-2 的 COLMAP 曲线是**早峰后降**的，
如果只看 30k 的终点会得出"训练越久越好"的错误结论。

用法:
    python scripts/45_plot_p1_curves.py
    python scripts/45_plot_p1_curves.py --csv results/p1_curves.csv --out docs/figures/p1_convergence.png
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

STYLE = {
    "colmap": dict(color="#1f77b4", marker="o", label="COLMAP"),
    "vggt":   dict(color="#d62728", marker="s", label="VGGT"),
}
SCENE_TITLE = {
    "dl3dv-2": "DL3DV-2 (39 test views)",
    "re10k-1": "RealEstate10K-1 (35 test views)",
    "waymo-405841": "Waymo-405841 (25 test views)",
}


def load(csv_path):
    data = defaultdict(lambda: defaultdict(list))   # scene -> init -> [(iter, psnr)]
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            data[r["scene"]][r["init"]].append((int(r["iter"]), float(r["psnr"])))
    for s in data:
        for i in data[s]:
            data[s][i].sort()
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "p1_curves.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "p1_convergence.png"))
    args = ap.parse_args()

    data = load(args.csv)
    scenes = sorted(data.keys())
    fig, axes = plt.subplots(1, len(scenes), figsize=(5.2 * len(scenes), 4.2), squeeze=False)

    for ax, scene in zip(axes[0], scenes):
        all_y = [p[1] for init in ("colmap", "vggt") if init in data[scene]
                 for p in data[scene][init]]
        ylo, yhi = min(all_y), max(all_y)
        span = max(yhi - ylo, 1e-6)
        # 上下各留 18% 余量，保证峰值标注不会跑出画框
        ax.set_ylim(ylo - 0.18 * span, yhi + 0.18 * span)

        for init in ("colmap", "vggt"):
            if init not in data[scene]:
                continue
            xs = [p[0] for p in data[scene][init]]
            ys = [p[1] for p in data[scene][init]]
            st = STYLE[init]
            ax.plot(xs, ys, linewidth=2, markersize=6, **st)

            # 标出峰值；文字位置按峰值在纵轴上的高低自动上下翻转，避免出框
            imax = max(range(len(ys)), key=lambda k: ys[k])
            ax.scatter([xs[imax]], [ys[imax]], s=150, facecolors="none",
                       edgecolors=st["color"], linewidths=2, zorder=5)
            near_top = (ys[imax] - ylo) > 0.72 * span
            dy = -16 if near_top else 13
            va = "top" if near_top else "bottom"
            # 峰值在最后一帧时，文字右对齐往左挪一点，避免超出画框
            if xs[imax] >= max(xs):
                dx, ha = -10, "right"
            else:
                dx, ha = 0, "center"
            ax.annotate(f"peak {ys[imax]:.2f} @ {xs[imax]//1000}k",
                        xy=(xs[imax], ys[imax]), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va=va,
                        fontsize=8.5, color=st["color"])

        ax.set_xscale("log")
        ax.set_xticks([1000, 5000, 10000, 30000])
        ax.set_xticklabels(["1k", "5k", "10k", "30k"])
        ax.set_xlabel("Training iterations", fontsize=11)
        ax.set_ylabel("Test PSNR (dB)", fontsize=11)
        ax.set_title(SCENE_TITLE.get(scene, scene), fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=9.5, loc="best", framealpha=0.92)

    fig.suptitle("Part 1 — 3DGS convergence from COLMAP vs VGGT initialization "
                 "(50 training views, 30k iterations)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"写出 {args.out}")

    # 顺便打印一张峰值表，方便写报告
    print(f"\n{'场景':<14} {'初始化':<8} {'30k PSNR':>9} {'峰值':>8} {'峰值位置':>10} {'早峰?':>6}")
    for scene in scenes:
        for init in ("colmap", "vggt"):
            if init not in data[scene]:
                continue
            ys = [p[1] for p in data[scene][init]]
            xs = [p[0] for p in data[scene][init]]
            imax = max(range(len(ys)), key=lambda k: ys[k])
            early = "是" if xs[imax] < 30000 and ys[imax] > ys[-1] + 0.05 else "否"
            print(f"{scene:<14} {init:<8} {ys[-1]:>9.3f} {ys[imax]:>8.3f} "
                  f"{str(xs[imax]//1000)+'k':>10} {early:>6}")


if __name__ == "__main__":
    main()
