"""
32_sparse_sweep.py — 稀疏视角实验：只用少量帧训练，仍在该场景固定的测试视角上评测。

实验定位（必须在报告里写清）
--------------------------
这是 **"真位姿上界"**：给稀疏视角，但位姿由全序列 COLMAP 提供（准的），
且初始点云也来自全序列 COLMAP。

它回答的问题是：**如果位姿完全正确，稀疏视角的重建能到多好？**
它是 Part 2 的对照组 —— Part 2 用无位姿方法（i3dgs 等），两者的差值
就是"位姿估计"这件事本身付出的代价。

⚠ 已知的"信息泄漏"，报告里必须声明：
   初始高斯是从**全序列** COLMAP 的点云撒下去的（不仅来自稀疏帧），
   所以这个数偏乐观，是**上界**而不是严格稀疏初始化。

为什么需要 --train-list/--test-list
----------------------------------
3DGS 的 --eval 固定按「排序后每 8 张取 1」划分，无法指定训练子集；
eval=False 又完全没有测试集。所以用 run_with_guard 的划分覆盖功能。

用法:
    python scripts/32_sparse_sweep.py --scenes dl3dv-2 re10k-1 --splits sparse10 sparse30
    python scripts/32_sparse_sweep.py --scenes waymo-405841 --splits sparse10 sparse30 --resolution 4
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
GUARD = os.path.join(ROOT, "scripts", "run_with_guard.py")
TRAIN = os.path.join(ROOT, "third_party", "gaussian-splatting", "train.py")
RENDER = os.path.join(ROOT, "third_party", "gaussian-splatting", "render.py")
EVAL = os.path.join(ROOT, "scripts", "41_eval.py")
RESULTS = os.path.join(ROOT, "results", "results.csv")
HEADER = ["run_id", "scene", "part", "source", "split", "train_frames", "test_frames",
          "psnr", "ssim", "lpips", "train_psnr", "ate_rmse", "t_init_s", "t_train_s",
          "t_eval_s", "gpu", "git_commit", "notes"]


def nlines(p):
    with open(p, encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip())


def run(cmd, log):
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    return rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["dl3dv-2", "re10k-1"])
    ap.add_argument("--splits", nargs="+", default=["sparse10", "sparse30"])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--resolution", type=int, default=None)
    args = ap.parse_args()

    summary = []
    for scene in args.scenes:
        src = os.path.join(ROOT, "data", scene, "colmap")
        test_list = os.path.join(ROOT, "data", scene, "splits", "test.txt")
        if not os.path.isdir(src) or not os.path.isfile(test_list):
            print(f"[跳过] {scene}: 缺 COLMAP 结果或 test.txt")
            continue
        n_test = nlines(test_list)

        for split in args.splits:
            train_list = os.path.join(ROOT, "data", scene, "splits", f"{split}.txt")
            if not os.path.isfile(train_list):
                print(f"[跳过] {scene}/{split}: 没有 {split}.txt")
                continue
            n_train = nlines(train_list)

            ts = time.strftime("%Y%m%d-%H%M")
            run_id = f"{scene}__colmap__{split}__3dgs__{ts}"
            out = os.path.join(ROOT, "outputs", "runs", run_id)
            os.makedirs(out, exist_ok=True)
            print(f"\n{'='*72}\n{run_id}   训练 {n_train} 帧 / 测试 {n_test} 帧\n{'='*72}")

            cmd = [PY, GUARD,
                   "--train-list", train_list, "--test-list", test_list, "--",
                   TRAIN, "-s", src, "-m", out, "--eval", "--disable_viewer",
                   "--iterations", str(args.iterations)]
            if args.resolution:
                cmd += ["-r", str(args.resolution)]
            rc, dt = run(cmd, os.path.join(out, "train.log"))
            print(f"  训练 rc={rc}  {dt/60:.1f} 分钟")
            if rc != 0:
                print("  ✗ 训练失败")
                continue

            rc2, _ = run([PY, GUARD, "--", RENDER, "-m", out, "--skip_train", "--quiet"],
                         os.path.join(out, "render.log"))
            if rc2 != 0:
                print("  ✗ 渲染失败")
                continue

            rc3, et = run([PY, EVAL, "--run", out], os.path.join(out, "eval.log"))
            if rc3 != 0:
                print("  ✗ 评测失败")
                continue

            with open(os.path.join(out, "results.json"), encoding="utf-8") as f:
                m = list(json.load(f).values())[0]

            train_psnr = ""
            for line in open(os.path.join(out, "train.log"), encoding="utf-8", errors="replace"):
                mm = re.search(r"\[ITER \d+\] Evaluating train: .*?PSNR\s+([\d.]+)", line)
                if mm:
                    train_psnr = round(float(mm.group(1)), 4)

            row = dict(run_id=run_id, scene=scene, part="P2", source="colmap(gt-pose)",
                       split=split, train_frames=n_train, test_frames=n_test,
                       psnr=round(m["PSNR"], 4), ssim=round(m["SSIM"], 4),
                       lpips=round(m["LPIPS"], 4), train_psnr=train_psnr, ate_rmse="",
                       t_init_s="0", t_train_s=round(dt, 1), t_eval_s=round(et, 1),
                       gpu="RTX 5070 Laptop", git_commit="sparse-sweep",
                       notes=f"oracle-pose upper bound; init pts from full COLMAP; "
                             f"LPIPS=vgg/0-1" + (f"; -r {args.resolution}" if args.resolution else ""))
            summary.append(row)
            gap = (train_psnr - row["psnr"]) if train_psnr != "" else float("nan")
            print(f"  ✓ PSNR {row['psnr']:.4f}  SSIM {row['ssim']:.4f}  LPIPS {row['lpips']:.4f}  "
                  f"过拟合差 {gap:+.2f} dB")

    if summary:
        with open(RESULTS, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            for r in summary:
                w.writerow(r)

    print(f"\n{'='*72}\n稀疏结果汇总（含 full 基线做对比）\n{'='*72}")
    print(f"{'场景':<14} {'split':<9} {'训练帧':>6} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} "
          f"{'train':>8} {'过拟合差':>9}")
    for r in summary:
        gap = (r["train_psnr"] - r["psnr"]) if r["train_psnr"] != "" else float("nan")
        print(f"{r['scene']:<14} {r['split']:<9} {r['train_frames']:>6} {r['psnr']:>8.3f} "
              f"{r['ssim']:>8.4f} {r['lpips']:>8.4f} {r['train_psnr']:>8.3f} {gap:>+9.2f}")


if __name__ == "__main__":
    main()
