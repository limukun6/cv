#!/bin/bash
# run_part1_qual.sh —— 一次性跑完 Part 1 定性对比图所需的全部产物。
#
# 为什么要重跑：定性对比图（原文 6.3 节 "must"）需要三路初始化各自的渲染结果，
# 而之前的机器已释放，所有 run 目录都没了。
#
# 产出：
#   outputs/runs/<scene>__{colmap,vggt,vggt_omega}__n160__3dgs__*/test/ours_30000/{renders,gt}
#   docs/figures/p1_qual_<scene>.png   ← 三路并排图
#
# 顺序安排（总约 2.6 小时）：
#   1. COLMAP × 3      39 min（waymo 一个人吃 32.5 min，它原生 1920×1280）
#   2. VGGT   × 3       9 min
#   3. Ω      × 3       6 min 推理 + 33 min 训练
#   4. 3DGS   × 9      99 min
#   5. 拼图             10 min
#
# 训练没有并行：虽然 48 GB 装得下 3 个，但 CPU 争抢会让单个训练慢 2–6 倍
# （上一轮实测：COLMAP mapper 跑着的时候训练从 90 it/s 掉到 14 it/s），得不偿失。
set -u
cd /root/autodl-tmp/cv || exit 1
export PATH=/usr/local/cuda/bin:/root/miniconda3/bin:$PATH
export COLMAP_BIN=/usr/bin/colmap
export PYTHONIOENCODING=utf-8

SCENES="dl3dv-2 re10k-1 waymo-405841"
N=160
LOGDIR=/root/autodl-tmp/p1qual
mkdir -p "$LOGDIR"

say() { echo ""; echo "########## $(date '+%H:%M:%S')  $* ##########"; }

# ---------------- 0. 解压数据 ----------------
# ⚠️ 不能用 `[ ! -d data/dl3dv-2 ]` 判断：git 仓库里已经带了
#    data/<scene>/meta.yaml 和 splits/，所以那个目录总是存在——上一轮就是这样
#    把解压跳过了，结果 COLMAP/VGGT 全部因为找不到图片秒失败（rc=1）。
#    必须检查**图片目录**。
need_extract=0
for sc in $SCENES; do
    if [ ! -d "data/$sc/rgb" ] && [ ! -d "data/$sc/images" ]; then
        need_extract=1
    fi
done
if [ "$need_extract" -eq 1 ]; then
    say "解压数据包（缺图片目录）"
    tar xzf /root/autodl-tmp/p1_bundle.tar.gz || { echo "解压失败"; exit 1; }
fi
# 解压后再确认一次，图片数量对不上就直接退出，别浪费几小时
for sc in $SCENES; do
    n=0
    for d in rgb images; do
        [ -d "data/$sc/$d" ] && n=$((n + $(ls "data/$sc/$d" | wc -l)))
    done
    echo "  $sc 图片 $n 张"
    if [ "$n" -lt 100 ]; then echo "✗ $sc 图片不足（$n 张），中止"; exit 1; fi
done
say "数据就绪: $(du -sh data | cut -f1)"

# ---------------- 1+2. COLMAP 与 VGGT ----------------
say "阶段 1-2：COLMAP + VGGT（3 场景 × N=$N）"
.venv/bin/python -u scripts/35_run_p1_matrix.py --stage main \
    --scenes $SCENES --inits colmap vggt --frames $N --iterations 30000 \
    2>&1 | tee "$LOGDIR/matrix.log"
echo "matrix rc=$?"

# ---------------- 3. VGGT-Ω ----------------
for sc in $SCENES; do
    say "阶段 3：Ω $sc"
    .venv/bin/python -u scripts/14_run_omega_pipeline.py --scene $sc --frames $N \
        2>&1 | tee "$LOGDIR/omega_$sc.log" | tail -12
    echo "omega $sc rc=$?"
done

# ---------------- 4. 三路对比图 ----------------
say "阶段 4：生成三路定性对比图"
mkdir -p docs/figures
for sc in $SCENES; do
    .venv/bin/python scripts/48_make_3way_figure.py --scene "$sc" --views 3 \
        2>&1 | tee "$LOGDIR/fig_$sc.log"
done

# ---------------- 5. 汇总 ----------------
say "汇总"
.venv/bin/python - <<'PY'
import csv, os
p = "results/results.csv"
if not os.path.isfile(p):
    print("  没有 results.csv"); raise SystemExit
rows = [r for r in csv.DictReader(open(p, encoding="utf-8-sig"))
        if r.get("split") == "n160" and r.get("psnr")]
scenes = ["dl3dv-2", "re10k-1", "waymo-405841"]
print(f"  {'scene':<16}{'COLMAP':>8}{'VGGT':>8}{'Ω':>8}{'CM-VGGT':>9}{'CM-Ω':>8}")
for sc in scenes:
    d = {}
    for r in rows:
        if r["scene"] == sc and "__seed" not in (r.get("run_id") or ""):
            d[r["source"]] = float(r["psnr"])
    f = lambda k: f"{d[k]:8.2f}" if k in d else "       -"
    g = lambda a, b: f"{d[a]-d[b]:+9.2f}" if a in d and b in d else "        -"
    print(f"  {sc:<16}{f('colmap')}{f('vggt')}{f('vggt_omega')}"
          f"{g('colmap','vggt')}{g('colmap','vggt_omega')}")
print()
for sc in scenes:
    p = f"docs/figures/p1_qual_{sc}.png"
    print(f"  {'✓' if os.path.isfile(p) else '✗'} {p}")
PY

say "全部完成"
