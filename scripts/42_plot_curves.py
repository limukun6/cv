"""
42_plot_curves.py — 从 train.log 里抽出收敛曲线并作图。

为什么这是 P1 的核心图
---------------------
P1 要求对比四项：初始化耗时、位姿精度、**3DGS 收敛速度**、最终渲染质量。

前两项和最后一项都只是一个数字，只有"收敛速度"需要一张曲线才能说清：
    · 如果两条曲线在 5k 迭代就重合 → 初始化质量的差别在训练中被"吸收"了
    · 如果终值差 1 dB 但曲线形状一样 → 差别是系统性的，训练无法弥补
    · 如果 A 早早就到 30 dB 而 B 要到 20k → 初始化质量直接决定训练成本

这张图直接回答 P1 的核心问题："初始化到底值多少 dB，值多少时间？"

用法:
    python scripts/42_plot_curves.py --runs outputs/runs/dl3dv-2__... outputs/runs/re10k-1__...
    python scripts/42_plot_curves.py --auto          # 自动找所有有 train.log 的运行
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 匹配形如：
# [ITER 7000] Evaluating test: L1 0.0230... PSNR 26.9619... [29/10 18:42:46]
PAT = re.compile(r"\[ITER (\d+)\] Evaluating (test|train): .*?PSNR\s+([\d.]+)")


def read_text(path):
    """自动识别编码。

    坑：PowerShell 的 Tee-Object 在这台机器上默认写 **UTF-16LE**（BOM ff fe），
    用 utf-8 读会全是空字符，正则一条都匹配不上，现象是"日志里明明有、脚本却读不到"。
    """
    with open(path, "rb") as f:
        head = f.read(4)
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        enc = "utf-16"
    elif head[:3] == b"\xef\xbb\xbf":
        enc = "utf-8-sig"
    else:
        enc = "utf-8"
    with open(path, encoding=enc, errors="replace") as f:
        return f.read()


def parse_log(path):
    """返回 {'test': [(iter, psnr), ...], 'train': [...]}"""
    out = {"test": [], "train": []}
    for line in read_text(path).splitlines():
        for m in PAT.finditer(line):
            it, split, psnr = int(m.group(1)), m.group(2), float(m.group(3))
            out[split].append((it, psnr))
    for k in out:
        out[k].sort()
    return out


def label_of(run_dir):
    """从 run_id 里取一个短标签：scene/source/split"""
    name = os.path.basename(run_dir.rstrip("/\\"))
    parts = name.split("__")
    if len(parts) >= 4:
        return f"{parts[0]} / {parts[1]} / {parts[2]}"
    return name[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--out", default="outputs/convergence.png")
    args = ap.parse_args()

    runs = args.runs or []
    if args.auto or not runs:
        runs = sorted({os.path.dirname(p) for p in
                       glob.glob(os.path.join("outputs", "runs", "*", "train.log"))})
    runs = [r for r in runs if os.path.isfile(os.path.join(r, "train.log"))]
    if not runs:
        raise SystemExit("没有找到任何 train.log")

    data = {}
    for r in runs:
        d = parse_log(os.path.join(r, "train.log"))
        if d["test"]:
            data[label_of(r)] = d
    if not data:
        raise SystemExit("日志里没有 [ITER ...] Evaluating 记录")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.tab10.colors

    for i, (lab, d) in enumerate(sorted(data.items())):
        c = colors[i % len(colors)]
        xs = [p[0] for p in d["test"]]
        ys = [p[1] for p in d["test"]]
        axes[0].plot(xs, ys, "o-", color=c, label=f"{lab} (test)", lw=2, ms=5)
        if d["train"]:
            axes[1].plot([p[0] for p in d["train"]], [p[1] for p in d["train"]],
                         "s--", color=c, label=f"{lab} (train)", lw=1.5, ms=4)

    for ax, title in zip(axes, ["TEST PSNR (novel view)", "TRAIN PSNR (overfit check)"]):
        ax.set_xlabel("iteration")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    # 标出"最后 20% 迭代带来的增益"，这是判断是否收敛的关键
    for lab, d in sorted(data.items()):
        if len(d["test"]) >= 2:
            xs = [p[0] for p in d["test"]]
            ys = [p[1] for p in d["test"]]
            if xs[-1] >= 30000:
                gain = ys[-1] - ys[max(0, len(ys) - 3)]
                print(f"{lab}: 最终 {ys[-1]:.2f} dB, 最后两段增益 {gain:+.2f} dB")

    fig.suptitle("3DGS convergence — COLMAP initialization (Plan A)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\n写出 {args.out}")

    # 同时打印一张表，方便直接抄进报告
    print(f"\n{'run':<40} {'iter':>7} {'test':>8} {'train':>8}")
    for lab, d in sorted(data.items()):
        tr = dict(d["train"])
        for it, p in d["test"]:
            print(f"{lab:<40} {it:>7} {p:>8.2f} {tr.get(it, float('nan')):>8.2f}")


if __name__ == "__main__":
    main()
