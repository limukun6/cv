"""
52_final_tables.py — 从 results.csv / ate.csv 生成报告用的最终表格。

为什么单独写一个
--------------
results.csv 里混了三类行，直接聚合会算错：
  1. **笔记本上的全帧实验**（split=full，267/244/154 帧）—— 和租用机的 N=50 不可比
  2. **已作废的 run**：相机跑飞那两次（dl3dv colmap 17.602 / 15.178）
     以及若干手工探针（probe_*、fin_*）
  3. 正常的 N=50 实验 + 种子重复

本脚本把 1、2 显式排除，输出：
  · 表 A：渲染质量（必做场景 + 额外数据集，含噪声 std 与 t 值）
  · 表 B：初始化耗时
  · 表 C：位姿精度（来自 ate.csv）

用法:
    python scripts/52_final_tables.py
    python scripts/52_final_tables.py --markdown > docs/TABLES.md
"""
import argparse
import csv
import math
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results", "results.csv")
ATE = os.path.join(ROOT, "results", "ate.csv")

# 作废的 run：相机发散修复前的两次，以及手工探针
VOID_MARKERS = ("20261102-2207", "20261102-2225", "probe_", "fin_colmap")

MAIN_SCENES = ["dl3dv-2", "re10k-1", "waymo-405841"]
EXTRA_SCENES = ["m360-counter", "m360-kitchen", "m360-bonsai"]


def is_void(rid):
    return any(m in (rid or "") for m in VOID_MARKERS)


def load_results():
    """{(scene, init): {"main": p, "seeds": [...]}}，只保留 N=50 的有效行。"""
    data = defaultdict(lambda: {"main": None, "seeds": []})
    with open(RESULTS, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("split") or "") != "n50":
                continue
            rid = r.get("run_id") or ""
            if is_void(rid):
                continue
            try:
                v = float(r["psnr"])
            except (TypeError, ValueError):
                continue
            k = (r["scene"], r["source"])
            if "seed" in rid:
                data[k]["seeds"].append(v)
            elif data[k]["main"] is None:
                data[k]["main"] = v
    return data


def load_ate():
    out = {}
    if not os.path.isfile(ATE):
        return out
    with open(ATE, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[(r["scene"], r["init"])] = r
    return out


def agg(entry):
    vals = ([] if entry["main"] is None else [entry["main"]]) + list(entry["seeds"])
    if not vals:
        return None
    mu = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else float("nan")
    return dict(mean=mu, std=sd, n=len(vals))


INITS = [("colmap", "COLMAP"), ("vggt", "VGGT"), ("vggt_omega", "VGGT-Ω")]


def table_main(data, md):
    lines = []
    for scenes, no, title in ((MAIN_SCENES, 1, "必做场景"),
                              (EXTRA_SCENES, 2, "额外数据集（加分项）")):
        lines.append(f"## 表 A{no} — 渲染质量（{title}，N=50 训练视角，30k 迭代）")
        lines.append("")
        if md:
            lines.append("| 场景 | COLMAP | VGGT | VGGT-Ω | CM−VGGT | CM−Ω |")
            lines.append("|---|---|---|---|---|---|")
        for sc in scenes:
            cells, vals = [], {}
            for key, label in INITS:
                a = agg(data.get((sc, key), {"main": None, "seeds": []}))
                vals[key] = a
                if not a:
                    cells.append("—")
                else:
                    sd = "" if math.isnan(a["std"]) else f" ± {a['std']:.3f}"
                    cells.append(f"{a['mean']:.3f}{sd} (n={a['n']})")
            if md:
                def d(k):
                    a, b = vals.get("colmap"), vals.get(k)
                    # 用带符号格式：CM−Ω 可能是负数（Ω 更好），不能硬加 "+"
                    return f"{a['mean']-b['mean']:+.2f}" if a and b else "—"
                lines.append(f"| {sc} | **{cells[0]}** | {cells[1]} | {cells[2]} "
                             f"| {d('vggt')} | {d('vggt_omega')} |")
            else:
                lines.append(f"  {sc:<14} " + "  ".join(
                    f"{lab:<7} {c}" for (k, lab), c in zip(INITS, cells)))
        lines.append("")
    return lines


def init_times():
    """初始化耗时：优先读 data/<scene>/colmap_n50/colmap_time.txt，
    没有就回退到 results.csv 里的 t_init_s（复用已有模型时不会写 time 文件）。"""
    out = {}
    for sc in MAIN_SCENES + EXTRA_SCENES:
        ct = os.path.join(ROOT, "data", sc, "colmap_n50", "colmap_time.txt")
        if os.path.isfile(ct):
            out[sc] = float(open(ct).read().strip())
    # 回退
    if os.path.isfile(RESULTS):
        with open(RESULTS, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if (r.get("split") or "") != "n50" or r.get("source") != "colmap":
                    continue
                sc = r["scene"]
                if sc in out:
                    continue
                try:
                    t = float(r.get("t_init_s") or 0)
                except ValueError:
                    continue
                if t > 0:
                    out[sc] = t
    return out


def table_time(md):
    lines = ["## 表 B — 初始化耗时（COLMAP，实测 wall time，同一批帧）", ""]
    if md:
        lines.append("| 场景 | 帧数 | COLMAP | 说明 |")
        lines.append("|---|---|---|---|")
    times = init_times()
    for sc in MAIN_SCENES + EXTRA_SCENES:
        if sc not in times:
            continue
        t = times[sc]
        if md:
            note = ""
            if sc == "m360-counter":
                note = "手工运行，观测值"
            elif not os.path.isfile(os.path.join(ROOT, "data", sc, "colmap_n50", "colmap_time.txt")):
                note = "来自 results.csv 的 t_init_s"
            lines.append(f"| {sc} | 50 | **{t:.0f} s** | {note} |")
        else:
            lines.append(f"  {sc:<16} COLMAP {t:6.0f} s")
    lines.append("")
    lines.append("  （VGGT 的耗时见结果文档：不随场景分辨率变化，稳定在 70–100 s）")
    lines.append("")
    return lines


def table_ate(ate, md):
    lines = ["## 表 C — 位姿精度（N=50 子集模型 vs 数据集真值）", ""]
    if md:
        lines.append("| 场景 | 初始化 | 旋转误差（中位） | 相邻帧相对旋转 | ATE（占对角） |")
        lines.append("|---|---|---|---|---|")
    for sc in MAIN_SCENES:
        for key, label in INITS:
            r = ate.get((sc, key))
            if not r:
                continue
            rot = float(r["rot_median_deg"])
            rel = float(r["rel_rot_median_deg"])
            ate_pct = float(r["ate_rmse_pct"])
            if md:
                lines.append(f"| {sc} | {label} | {rot:.3f}° | {rel:.4f}° | {ate_pct:.3f}% |")
            else:
                lines.append(f"  {sc:<14} {label:<8} 旋转 {rot:6.3f}°   "
                             f"相对 {rel:7.4f}°   ATE {ate_pct:6.3f}%")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    md = args.markdown

    data = load_results()
    ate = load_ate()
    lines = table_main(data, md) + table_time(md) + table_ate(ate, md)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
