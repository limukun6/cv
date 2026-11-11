"""
48_make_3way_figure.py —— Part 1 要求的定性对比图（三路初始化并排）。

原文 L186-188：
    "You must visualize and compare: Part 1: Reconstruction quality and convergence
     among [COLMAP + 3DGS], [VGGT + 3DGS], and [VGGT-Ω + 3DGS]."

也就是说这张图不是可选项。我们之前的 9 个视频全是 COLMAP 一家的，凑不出三路对比，
所以单独写这个脚本。

图的样子
--------
        真值        COLMAP      VGGT-1B     VGGT-Ω
 视角 A  [gt]   [32.4 dB]    [29.9 dB]   [28.1 dB]
 视角 B  ...
 视角 C  ...

每个面板右下角标该视角的 PSNR（单视角），列标题标该初始化的整场景平均 PSNR。

用法
----
    # 在实例上（9 个 run 都在 outputs/runs 下）
    python scripts/48_make_3way_figure.py --scene dl3dv-2

    # 指定 run（名字含糊时）
    python scripts/48_make_3way_figure.py --scene dl3dv-2 \
        --runs outputs/runs/x__colmap__n160__3dgs__20261111-1010 \
               outputs/runs/x__vggt__n160__3dgs__20261111-1020 \
               outputs/runs/x__vggt_omega__n160__3dgs__20261111-1030

为什么不用视频抽帧
------------------
视频是沿一条合成轨迹飞，三个 run 的轨迹虽然构造方式相同，但取样点在各自模型上
是对齐到不同的训练相机——不如直接用**同一批测试视角**来得严格。
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = os.path.join(ROOT, "outputs", "runs")

# 报告里用的整场景均值（N=160），用来给列标题做交叉检查。
# 找不到 results.json 时退回这些数字；找到就用实测值覆盖。
KNOWN = {
    ("dl3dv-2", "colmap"): 32.166, ("dl3dv-2", "vggt"): 29.077,
    ("dl3dv-2", "vggt_omega"): 28.404,
    ("re10k-1", "colmap"): 34.303, ("re10k-1", "vggt"): 27.613,
    ("re10k-1", "vggt_omega"): 32.540,
    ("waymo-405841", "colmap"): 32.712, ("waymo-405841", "vggt"): 23.430,
    ("waymo-405841", "vggt_omega"): 24.722,
}
INITS = [("colmap", "COLMAP"), ("vggt", "VGGT-1B"), ("vggt_omega", "VGGT-Ω")]
SPLIT = "n160"


def find_run(scene, init):
    """按 (scene, init, n160) 找最新的一个 run 目录。"""
    pat = os.path.join(RUNS, f"{scene}__{init}__{SPLIT}__3dgs__*")
    cands = [d for d in glob.glob(pat) if os.path.isdir(d)]
    # 排除 seed 重复（名字里带 seed 的）
    base = [d for d in cands if "__seed" not in d]
    pool = base or cands
    if not pool:
        return None
    return sorted(pool, key=os.path.getmtime)[-1]


def load_psnr(run_dir):
    """从 results.json 读整场景 PSNR；没有就返回 None。"""
    p = os.path.join(run_dir, "results.json")
    if os.path.isfile(p):
        try:
            return json.load(open(p))["test_ours_30000"]["PSNR"]
        except Exception:
            pass
    return None


def per_view_psnr(a, b):
    """单视角 PSNR（渲染 vs 真值），用来标在每个面板角上。"""
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    if a.shape != b.shape:
        # 与 41_eval.py 保持一致：把真值缩放到渲染尺寸再比。
        # waymo 的测试图是原始 1066×1600，而 3DGS 用 -r 4 渲染成 480×320，
        # 不处理会直接 broadcast 报错（第一次跑就崩在这）。
        b = np.asarray(
            Image.fromarray((b * 255).astype(np.uint8)).resize((a.shape[1], a.shape[0])),
            dtype=np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="场景名，如 dl3dv-2")
    ap.add_argument("--runs", nargs=3, default=None,
                    help="三个 run 目录（colmap vggt omega 顺序）；不给就自动找")
    ap.add_argument("--views", type=int, default=3, help="展示几个测试视角（默认 3）")
    ap.add_argument("--label-mode", choices=["none", "psnr", "index"], default="psnr",
                    help="面板角标内容")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = args.runs or [find_run(args.scene, k) for k, _ in INITS]
    if any(r is None for r in runs):
        print("✗ 找不到完整的三个 run：")
        for (k, name), r in zip(INITS, runs):
            print(f"   {name:<8} {'✓' if r else '✗ 缺失'}  {r or ''}")
        return 1

    # 用 COLMAP 那一路的测试集作为视角来源（三者测试集相同，取谁都行）
    ref_run = runs[0]
    gt_dir = os.path.join(ref_run, "test", "ours_30000", "gt")
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    if not gt_files:
        print(f"✗ {gt_dir} 里没有真值图")
        return 1

    # 均匀抽 views 个
    n = len(gt_files)
    idx = [int(round(i * (n - 1) / max(args.views - 1, 1))) for i in range(args.views)] \
        if args.views > 1 else [n // 2]
    idx = sorted(set(idx))
    picked = [gt_files[i] for i in idx]

    # 整场景均值：优先读 results.json
    col_psnr = []
    for r, (k, _) in zip(runs, INITS):
        v = load_psnr(r)
        col_psnr.append(v if v is not None else KNOWN.get((args.scene, k)))

    fig, axes = plt.subplots(len(picked), 4,
                             figsize=(4.0 * 4, 3.2 * len(picked) + 0.6),
                             squeeze=False)
    fig.suptitle(f"{args.scene} — qualitative comparison at the same test views "
                 f"(N = 160 training views)", fontsize=13.5, fontweight="bold")

    headers = ["Ground truth"] + [
        f"{name}\n{('%.2f dB' % p) if p else 'n/a'}"
        for (_, name), p in zip(INITS, col_psnr)
    ]

    for row, gt_path in enumerate(picked):
        name = os.path.basename(gt_path)
        gt = np.array(Image.open(gt_path).convert("RGB"))

        # 第一列的渲染图尺寸（waymo 是 480×320），真值图可能是原始分辨率。
        # 显示时把真值缩到同一尺寸，否则网格里各列大小不一致、看不出对应关系。
        ref_path = os.path.join(runs[0], "test", "ours_30000", "renders", name)
        gt_disp = gt
        if os.path.isfile(ref_path):
            rw, rh = Image.open(ref_path).size
            if (rw, rh) != (gt.shape[1], gt.shape[0]):
                gt_disp = np.array(Image.fromarray(gt).resize((rw, rh)))

        ax = axes[row][0]
        ax.imshow(gt_disp)
        ax.set_xticks([]); ax.set_yticks([])
        if args.label_mode == "psnr":
            ax.set_xlabel(name, fontsize=8, color="#444")
        elif args.label_mode == "index":
            ax.set_xlabel(f"view {idx[row]}", fontsize=8, color="#444")

        for col, (run_dir, (k, _)) in enumerate(zip(runs, INITS), start=1):
            img_path = os.path.join(run_dir, "test", "ours_30000", "renders", name)
            ax = axes[row][col]
            if not os.path.isfile(img_path):
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            img = np.array(Image.open(img_path).convert("RGB"))
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            p = per_view_psnr(img, gt)
            ax.text(0.985, 0.02, f"{p:.1f} dB", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.5, color="white",
                    bbox=dict(boxstyle="round,pad=0.22", fc="black", alpha=0.55))

        if row == 0:
            for col in range(4):
                axes[row][col].set_title(headers[col], fontsize=10.5,
                                         fontweight="bold" if col else "normal")

    fig.text(0.5, 0.004,
             "Column headers give the scene-level mean test PSNR (mean over all test views, "
             "2–3 seeds where available); corner labels give that single view's PSNR.\n"
             "Note: the official VGGT exporter caps the point cloud at 100 000 points chosen by a "
             "random unseeded subsample (vggt/utils/helper.py::randomly_limit_trues), so VGGT-1B "
             "figures carry an export-level spread far larger than its training-seed spread — "
             "see the report's limitations.",
             ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])

    out = args.out or os.path.join(ROOT, "docs", "figures", f"p1_qual_{args.scene}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"写出 {out}")

    print(f"\n{args.scene} 列均值:")
    for (k, nm), p, r in zip(INITS, col_psnr, runs):
        print(f"  {nm:<9} {('%.3f' % p) if p else 'n/a':>8}  {os.path.basename(r)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
