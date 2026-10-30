"""
30_seed_sweep.py — 多随机种子重复实验，测出"噪声地板"。

为什么必须先做这件事
------------------
我们用完全相同的配置把 dl3dv-2 跑了两遍，得到 33.89 和 33.41 dB —— **差 0.48 dB**。
3DGS 的 safe_state() 里写死了 torch.manual_seed(0)，所以这个差异**不是种子造成的**，
而是 CUDA 非确定性（光栅化反向传播里的原子加、相机采样顺序等）。

这意味着：**任何小于 ~0.5 dB 的差距都不能当结论。**
在不知道噪声地板之前，P1/P2/P3 的所有对比表都是拿噪声当信号。

本脚本对每个场景用 3 个不同种子各跑一次，输出 mean ± std。

用法:
    python scripts/30_seed_sweep.py --scenes dl3dv-2 re10k-1 --seeds 1 2 3
"""
import argparse
import csv
import json
import os
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


def count_lines(p):
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f if _.strip())


def run(cmd, log_path):
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    return p.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["dl3dv-2", "re10k-1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--iterations", type=int, default=30000)
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, "outputs", "runs"), exist_ok=True)
    summary = []

    for scene in args.scenes:
        src = os.path.join(ROOT, "data", scene, "colmap")
        if not os.path.isdir(src):
            print(f"[跳过] {scene}: 没有 COLMAP 结果")
            continue
        n_train = count_lines(os.path.join(ROOT, "data", scene, "splits", "full.txt"))
        n_test = count_lines(os.path.join(ROOT, "data", scene, "splits", "test.txt"))

        for seed in args.seeds:
            ts = time.strftime("%Y%m%d-%H%M")
            run_id = f"{scene}__colmap__full__3dgs__seed{seed}__{ts}"
            out = os.path.join(ROOT, "outputs", "runs", run_id)
            os.makedirs(out, exist_ok=True)
            print(f"\n{'='*70}\n{run_id}\n{'='*70}")

            rc, dt = run([PY, GUARD, "--seed", str(seed), "--", TRAIN,
                          "-s", src, "-m", out, "--eval", "--disable_viewer",
                          "--iterations", str(args.iterations)],
                         os.path.join(out, "train.log"))
            print(f"  训练 rc={rc}  {dt/60:.1f} 分钟")
            if rc != 0:
                print("  ✗ 训练失败，跳过")
                continue

            rc2, _ = run([PY, GUARD, "--", RENDER, "-m", out, "--skip_train", "--quiet"],
                         os.path.join(out, "render.log"))
            if rc2 != 0:
                print("  ✗ 渲染失败，跳过")
                continue

            rc3, et = run([PY, EVAL, "--run", out], os.path.join(out, "eval.log"))
            if rc3 != 0:
                print("  ✗ 评测失败，跳过")
                continue

            with open(os.path.join(out, "results.json"), encoding="utf-8") as f:
                m = list(json.load(f).values())[0]

            # 从训练日志里取最终 train PSNR
            train_psnr = ""
            try:
                import re
                for line in open(os.path.join(out, "train.log"), encoding="utf-8", errors="replace"):
                    mm = re.search(r"\[ITER \d+\] Evaluating train: .*?PSNR\s+([\d.]+)", line)
                    if mm:
                        train_psnr = round(float(mm.group(1)), 4)
            except Exception:
                pass

            row = dict(run_id=run_id, scene=scene, part="P1", source="colmap", split="full",
                       train_frames=n_train, test_frames=n_test,
                       psnr=round(m["PSNR"], 4), ssim=round(m["SSIM"], 4),
                       lpips=round(m["LPIPS"], 4), train_psnr=train_psnr, ate_rmse="",
                       t_init_s="", t_train_s=round(dt, 1), t_eval_s=round(et, 1),
                       gpu="RTX 5070 Laptop", git_commit="seed-sweep",
                       notes=f"seed={seed}; noise-floor experiment; LPIPS=vgg/0-1")
            summary.append(row)
            print(f"  ✓ PSNR {row['psnr']:.4f}  SSIM {row['ssim']:.4f}  LPIPS {row['lpips']:.4f}")

    # 追加到 results.csv
    if summary:
        with open(RESULTS, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HEADER)
            for r in summary:
                w.writerow(r)

    # 汇总
    print(f"\n{'='*70}\n噪声地板汇总\n{'='*70}")
    import statistics as st
    for scene in args.scenes:
        vals = [r["psnr"] for r in summary if r["scene"] == scene]
        lip = [r["lpips"] for r in summary if r["scene"] == scene]
        if len(vals) >= 2:
            print(f"{scene:<14} n={len(vals)}  PSNR {st.mean(vals):.3f} ± {st.stdev(vals):.3f} dB"
                  f"   LPIPS {st.mean(lip):.4f} ± {st.stdev(lip):.4f}")
            print(f"{'':<14} 取值范围 {min(vals):.3f} ~ {max(vals):.3f}"
                  f"  (极差 {max(vals)-min(vals):.3f} dB)")
        elif vals:
            print(f"{scene:<14} 只有一个成功样本: {vals[0]:.3f}")


if __name__ == "__main__":
    main()
