"""
43_stats.py — 从 results.csv 里算每个场景的"噪声地板"，用来判断两组结果是否真的不同。

为什么必须做这件事
------------------
我们用完全相同的配置、只换随机种子，把每个场景跑了 3–5 遍，得到：

    dl3dv-2 : 33.89 / 33.41 / 33.17 / 33.00 / 33.57   → 极差 0.89 dB
    re10k-1 : 34.35 / 34.34 / 34.27 / 34.39           → 极差 0.12 dB

**两个场景的抖动差了 7 倍。** 所以不存在一个"全场景通用"的显著性阈值：
    · 在 dl3dv-2 上，0.5 dB 的差距是噪声
    · 在 re10k-1 上，0.5 dB 的差距是真实差异

本脚本算出每个 (scene, source, split) 组合的 mean / std / 极差，
并提供 `--test` 用来检验"某个新方法是否显著优于基线"。

用法:
    python scripts/43_stats.py
    python scripts/43_stats.py --test dl3dv-2 vggt full 33.0
"""
import argparse
import csv
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV = os.path.join(ROOT, "results", "results.csv")
SKIP_NOTES = ("aborted", "smoke")


def load():
    rows = []
    with open(CSV, encoding="utf-8-sig") as f:  # tolerance for BOM
        for r in csv.DictReader(f):
            if not r.get("psnr"):
                continue
            if any(s in (r.get("notes") or "") for s in SKIP_NOTES):
                continue
            try:
                r["psnr"] = float(r["psnr"])
                r["ssim"] = float(r.get("ssim") or "nan")
                r["lpips"] = float(r.get("lpips") or "nan")
            except ValueError:
                continue
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs=4, metavar=("SCENE", "SOURCE", "SPLIT", "PSNR"),
                    help="检验一个新结果是否超出该场景的噪声范围")
    args = ap.parse_args()

    rows = load()
    if not rows:
        raise SystemExit("results.csv 里没有可用数据")

    groups = defaultdict(list)
    for r in rows:
        groups[(r["scene"], r["source"], r["split"])].append(r)

    print(f"{'场景':<14} {'初始化':<8} {'split':<8} {'n':>3} "
          f"{'PSNR 均值':>10} {'标准差':>8} {'极差':>7} {'SSIM':>8} {'LPIPS':>8}")
    print("-" * 92)
    floors = {}
    for (scene, source, split), rs in sorted(groups.items()):
        p = [r["psnr"] for r in rs]
        mean = st.mean(p)
        sd = st.stdev(p) if len(p) > 1 else 0.0
        rng = max(p) - min(p)
        ssim = st.mean([r["ssim"] for r in rs if r["ssim"] == r["ssim"]])
        lip = st.mean([r["lpips"] for r in rs if r["lpips"] == r["lpips"]])
        print(f"{scene:<14} {source:<8} {split:<8} {len(p):>3} "
              f"{mean:>10.3f} {sd:>8.3f} {rng:>7.3f} {ssim:>8.4f} {lip:>8.4f}")
        floors[(scene, source, split)] = (mean, sd, rng, len(p))

    # 显著性门槛
    print(f"\n{'='*92}\n显著性门槛（2 倍标准差，约 95% 置信）\n{'='*92}")
    print(f"{'场景':<14} {'split':<8} {'噪声 2σ':>10} {'极差':>7}  含义")
    for (scene, source, split), (mean, sd, rng, n) in sorted(floors.items()):
        if n < 2:
            continue
        thr = 2 * sd
        print(f"{scene:<14} {split:<8} {thr:>9.3f} {rng:>7.3f}  "
              f"差距 < {thr:.2f} dB 视为噪声")

    # 可选：检验一个新结果
    if args.test:
        scene, source, split, val = args.test[0], args.test[1], args.test[2], float(args.test[3])
        key = (scene, source, split)
        # 找同场景同 split 的任意 source 作为基线参考
        base = None
        for (s, so, sp), v in floors.items():
            if s == scene and sp == split and v[3] >= 2:
                base = (so, v)
                break
        if not base:
            print(f"\n没有 {scene}/{split} 的基线可用于比较")
            return
        bsource, (mean, sd, rng, n) = base
        diff = val - mean
        thr = 2 * sd
        print(f"\n{'='*92}\n检验: {source} 在 {scene}/{split} 上 PSNR = {val:.3f}\n{'='*92}")
        print(f"  基线 {bsource}: {mean:.3f} ± {sd:.3f} (n={n}, 极差 {rng:.3f})")
        print(f"  差值: {diff:+.3f} dB")
        print(f"  门槛: ±{thr:.3f} dB (2σ)")
        if abs(diff) < thr:
            print(f"  → 结论：**无法区分**。{abs(diff):.2f} dB 落在这条噪声带内。")
        elif diff > 0:
            print(f"  → 结论：**显著优于**基线（超出噪声带 {abs(diff) - thr:.2f} dB）")
        else:
            print(f"  → 结论：**显著劣于**基线（超出噪声带 {abs(diff) - thr:.2f} dB）")


if __name__ == "__main__":
    main()
