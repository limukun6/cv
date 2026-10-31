"""
35_run_p1_matrix.py — 一条命令跑完 Part 1 的全部实验矩阵。

设计目标
--------
租来的机器按小时计费，**不能让人坐在那里一条条敲命令**。
所以整个矩阵写成一个脚本：上线后 `python scripts/35_run_p1_matrix.py --stage all`，
然后就可以去干别的。

实验矩阵
--------
主实验（controlled，三边统一 N 帧）：
    每个场景 × 每个初始化方法（colmap / vggt / vggt_ba）→ 训练 + 评测
帧数扩展（scaling）：
    指定场景 × N ∈ {20, 50, 100, 200} × 三方法
噪声重复：
    主实验里指定的组再跑 extra 个种子

关键设计：**三边看到同一批图片**
    COLMAP 和 VGGT 都在「N 个训练帧 ∪ 全部测试帧」上估计位姿，
    训练时只用 N 个训练帧（由 run_with_guard 的 --train-list 控制）。
    这样三边的输入完全一致，训练预算也一致，只差"初始化来源"这一个变量。

⚠ 必须声明的局限：位姿估计阶段看到了测试帧（和稠密基线一致）。
   严格的无泄漏做法是只用训练帧估位姿、测试帧用真值位姿；我们在报告里说明这个选择。

用法:
    # 全部（主实验 + 帧数扩展 + 噪声）
    python scripts/35_run_p1_matrix.py --stage all --frames 50

    # 只跑主实验
    python scripts/35_run_p1_matrix.py --stage main --frames 50

    # 只跑帧数扩展
    python scripts/35_run_p1_matrix.py --stage scale --scale-scene dl3dv-2

    # 先在一个场景上冒烟
    python scripts/35_run_p1_matrix.py --stage main --scenes dl3dv-2 --scenes-only-first
"""
import argparse
import csv
import json
import os
import random
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
COLMAP_RUN = os.path.join(ROOT, "scripts", "10_run_colmap.py")
VGGT_RUN = os.path.join(ROOT, "scripts", "11_run_vggt.py")
RESULTS = os.path.join(ROOT, "results", "results.csv")

HEADER = ["run_id", "scene", "part", "source", "split", "train_frames", "test_frames",
          "psnr", "ssim", "lpips", "train_psnr", "ate_rmse", "t_init_s", "t_train_s",
          "t_eval_s", "gpu", "git_commit", "notes"]

TEST_ITER = [1000, 5000, 10000, 15000, 20000, 25000, 30000]


# ---------------------------------------------------------------- helpers

def sh(cmd, log):
    t0 = time.time()
    os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
    with open(log, "a", encoding="utf-8", errors="replace") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    return rc, time.time() - t0


def read_list(p):
    with open(p, encoding="utf-8-sig") as f:
        return [ln.strip() for ln in f if ln.strip()]


def write_list(p, names):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(names) + "\n")


def make_n_frame_split(scene, n, seed=0):
    """从该场景的 full 训练集里均匀抽 n 帧，并和 test 取并集做位姿估计的图片列表。"""
    sp = os.path.join(ROOT, "data", scene, "splits")
    full = read_list(os.path.join(sp, "full.txt"))
    test = read_list(os.path.join(sp, "test.txt"))
    if n >= len(full):
        chosen = list(full)
    else:
        # 均匀抽样而不是随机，保证沿轨迹覆盖均匀（随机可能扎堆）
        step = len(full) / n
        chosen = [full[min(int(i * step), len(full) - 1)] for i in range(n)]
        chosen = sorted(set(chosen))
    train_p = os.path.join(sp, f"p1_n{n}.txt")
    pose_p = os.path.join(sp, f"p1_n{n}_pose.txt")
    write_list(train_p, chosen)
    write_list(pose_p, sorted(set(chosen) | set(test)))
    return train_p, pose_p, len(chosen), len(test)


def append_result(row):
    exists = os.path.isfile(RESULTS)
    with open(RESULTS, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if not exists or os.path.getsize(RESULTS) == 0:
            w.writeheader()
        w.writerow(row)


def count_png(d):
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if f.lower().endswith((".png", ".jpg")))


# ---------------------------------------------------------------- stages

def ensure_poses(scene, n, src_kind, pose_list, force=False):
    """准备某个初始化来源的场景目录，返回 (scene_dir, t_init_seconds)。"""
    sd = os.path.join(ROOT, "data", scene)
    tag = f"n{n}" if n else "full"

    if src_kind == "colmap":
        out_name = f"colmap_{tag}"
        ws = os.path.join(sd, out_name)
        tfile = os.path.join(ws, "colmap_time.txt")
        if os.path.isfile(os.path.join(ws, "sparse", "0", "images.bin")) and not force:
            t = float(open(tfile).read().strip()) if os.path.isfile(tfile) else 0.0
            print(f"  [colmap] 已有 {ws}，复用（初始化耗时 {t:.0f}s）")
            return ws, t
        cmd = [PY, "-u", COLMAP_RUN, "--data", "data", "--scene", scene,
               "--out-name", out_name, "--image-list", pose_list, "--force"]
        rc, dt = sh(cmd, os.path.join(ws, "run.log"))
        print(f"  [colmap] rc={rc} {dt/60:.1f} min")
        if rc != 0:
            return None, 0
        with open(tfile, "w") as f:
            f.write(f"{dt:.1f}\n")
        return ws, dt

    else:  # vggt / vggt_ba
        out_name = f"vggt{'_ba' if src_kind == 'vggt_ba' else ''}_{tag}"
        ws = os.path.join(sd, out_name)
        tfile = os.path.join(ws, "vggt_time.txt")
        if os.path.isdir(os.path.join(ws, "sparse")) and not force:
            t = float(open(tfile).read().strip()) if os.path.isfile(tfile) else 0.0
            print(f"  [{src_kind}] 已有 {ws}，复用（{t/60:.1f} min）")
            return ws, t
        cmd = [PY, "-u", VGGT_RUN, "--data", "data", "--scene", scene,
               "--image-list", pose_list, "--out-name", out_name]
        if src_kind == "vggt_ba":
            cmd.append("--use-ba")
        rc, dt = sh(cmd, os.path.join(ws, "run.log"))
        print(f"  [{src_kind}] rc={rc} {dt/60:.1f} min")
        if rc != 0:
            return None, 0
        return ws, dt


def train_and_eval(scene, init, scene_dir, train_list, test_list, n_train, n_test,
                   t_init, iterations, seed=None, tag="p1", resolution=None,
                   test_iterations=None):
    ts = time.strftime("%Y%m%d-%H%M")
    run_id = f"{scene}__{init}__{tag}__3dgs__{ts}"
    if seed is not None:
        run_id = f"{scene}__{init}__{tag}__seed{seed}__{ts}"
    out = os.path.join(ROOT, "outputs", "runs", run_id)
    os.makedirs(out, exist_ok=True)
    print(f"\n  → {run_id}")

    cmd = [PY, GUARD]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    cmd += ["--train-list", train_list, "--test-list", test_list, "--",
            TRAIN, "-s", scene_dir, "-m", out, "--eval", "--disable_viewer",
            "--iterations", str(iterations)]
    if test_iterations:
        cmd += ["--test_iterations"] + [str(i) for i in test_iterations]
    if resolution:
        cmd += ["-r", str(resolution)]
    rc, dt_train = sh(cmd, os.path.join(out, "train.log"))
    print(f"    训练 rc={rc} {dt_train/60:.1f} min")
    if rc != 0:
        return None

    cmd = [PY, GUARD, "--test-list", test_list, "--",
           RENDER, "-m", out, "--skip_train", "--quiet"]
    rc2, _ = sh(cmd, os.path.join(out, "render.log"))
    if rc2 != 0:
        print("    渲染失败")
        return None

    rc3, dt_eval = sh([PY, EVAL, "--run", out], os.path.join(out, "eval.log"))
    if rc3 != 0:
        print("    评测失败")
        return None

    with open(os.path.join(out, "results.json"), encoding="utf-8") as f:
        m = list(json.load(f).values())[0]

    train_psnr = ""
    for line in open(os.path.join(out, "train.log"), encoding="utf-8", errors="replace"):
        mm = re.search(r"\[ITER \d+\] Evaluating train: .*?PSNR\s+([\d.]+)", line)
        if mm:
            train_psnr = round(float(mm.group(1)), 4)

    row = dict(run_id=run_id, scene=scene, part="P1", source=init, split=tag,
               train_frames=n_train, test_frames=n_test,
               psnr=round(m["PSNR"], 4), ssim=round(m["SSIM"], 4),
               lpips=round(m["LPIPS"], 4), train_psnr=train_psnr, ate_rmse="",
               t_init_s=round(t_init, 1), t_train_s=round(dt_train, 1),
               t_eval_s=round(dt_eval, 1), gpu=os.environ.get("GPU_NAME", "rented"),
               git_commit="p1-matrix",
               notes=f"N={n_train} train / {n_test} test; poses from {init}; "
                     f"LPIPS=vgg/0-1" + (f"; seed={seed}" if seed else "")
                     + (f"; -r {resolution}" if resolution else ""))
    append_result(row)
    print(f"    ✓ PSNR {row['psnr']:.3f}  SSIM {row['ssim']:.4f}  LPIPS {row['lpips']:.4f}")
    return row


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main",
                    choices=["main", "scale", "noise", "all"])
    ap.add_argument("--scenes", nargs="+",
                    default=["dl3dv-2", "re10k-1", "waymo-405841"])
    ap.add_argument("--inits", nargs="+", default=["colmap", "vggt", "vggt_ba"])
    ap.add_argument("--frames", type=int, default=50, help="主实验统一的训练帧数 N")
    ap.add_argument("--scale-scene", default="dl3dv-2")
    ap.add_argument("--scale-frames", type=int, nargs="+", default=[20, 50, 100, 200])
    ap.add_argument("--noise-scene", default="dl3dv-2")
    ap.add_argument("--noise-seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--resolution", type=int, default=None,
                    help="Waymo 必须给 -r 4（1920x1280 在 24GB 上也该降）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # 每个场景的分辨率：Waymo 默认降采样
    def res_for(scene):
        if args.resolution:
            return args.resolution
        return 4 if scene == "waymo-405841" else None

    print(f"{'='*72}\nPart 1 实验矩阵\n{'='*72}")
    print(f"  阶段: {args.stage}")
    print(f"  场景: {args.scenes}")
    print(f"  初始化: {args.inits}")
    print(f"  主实验 N={args.frames}，迭代 {args.iterations}")

    todo = []

    if args.stage in ("main", "all"):
        for scene in args.scenes:
            train_p, pose_p, n_tr, n_te = make_n_frame_split(scene, args.frames)
            print(f"\n[{scene}] 抽 {n_tr} 训练帧 / {n_te} 测试帧 → {os.path.basename(train_p)}")
            for init in args.inits:
                todo.append(("main", scene, init, args.frames, train_p, pose_p, n_tr, n_te, None))

    if args.stage in ("scale", "all"):
        for n in args.scale_frames:
            train_p, pose_p, n_tr, n_te = make_n_frame_split(args.scale_scene, n)
            print(f"\n[{args.scale_scene}] 帧数扩展 N={n_tr}")
            for init in args.inits:
                todo.append(("scale", args.scale_scene, init, n, train_p, pose_p, n_tr, n_te, None))

    if args.stage in ("noise", "all"):
        for seed in args.noise_seeds:
            train_p, pose_p, n_tr, n_te = make_n_frame_split(args.noise_scene, args.frames)
            print(f"\n[{args.noise_scene}] 噪声重复 seed={seed} N={n_tr}")
            for init in args.inits:
                todo.append(("noise", args.noise_scene, init, args.frames, train_p, pose_p,
                             n_tr, n_te, seed))

    print(f"\n共 {len(todo)} 个训练任务（位姿估计会按需复用）\n")

    done, failed = [], []
    t_start = time.time()
    for i, (stage, scene, init, n, train_p, pose_p, n_tr, n_te, seed) in enumerate(todo, 1):
        print(f"\n{'#'*72}\n[{i}/{len(todo)}] {stage} | {scene} | {init} | N={n}"
              f"{' | seed=' + str(seed) if seed else ''}\n{'#'*72}")
        scene_dir, t_init = ensure_poses(scene, n, init, pose_p, force=args.force)
        if scene_dir is None:
            print("  位姿估计失败，跳过")
            failed.append((stage, scene, init, n, seed))
            continue
        tag = f"n{n}" if n != len(read_list(os.path.join(ROOT, "data", scene, "splits", "full.txt"))) else "full"
        row = train_and_eval(scene, init, scene_dir, train_p, pose_p.replace("_pose", ""),
                             n_tr, n_te, t_init, args.iterations, seed=seed, tag=tag,
                             resolution=res_for(scene),
                             test_iterations=TEST_ITER if args.iterations >= 30000 else None)
        if row:
            done.append(row)
        else:
            failed.append((stage, scene, init, n, seed))
        print(f"\n  累计用时 {(time.time()-t_start)/3600:.2f} h，完成 {len(done)}，失败 {len(failed)}")

    print(f"\n{'='*72}\n汇总\n{'='*72}")
    print(f"{'场景':<14} {'初始化':<10} {'N':>5} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    for r in done:
        print(f"{r['scene']:<14} {r['source']:<10} {r['train_frames']:>5} "
              f"{r['psnr']:>8.3f} {r['ssim']:>8.4f} {r['lpips']:>8.4f}")
    if failed:
        print(f"\n失败 {len(failed)} 个：{failed}")
    print(f"\n总用时 {(time.time()-t_start)/3600:.2f} h")
    print(f"结果已追加到 {RESULTS}")


if __name__ == "__main__":
    main()
