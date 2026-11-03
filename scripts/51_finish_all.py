"""
51_finish_all.py — 一条命令跑完 Part 1 剩下的全部工作。

为什么要这个脚本
--------------
租来的机器按小时计费，人不能一直盯着。这个脚本把四个阶段串起来，
每个阶段独立容错（一个失败不影响后面），全部结果追加到
`results/results.csv`，最后打印汇总表。

阶段
----
  noise  噪声重复：每个 (场景, 初始化) 再跑 2 个种子 → 得到标准差
  ate    位姿精度：对 n50 的子集模型算 ATE / 旋转误差 → results/ate.csv
  extra  额外数据集（加分项）：下载 T&T + Deep Blending → 4 个场景 × 2 种初始化
  video  Demo 视频（Bonus）：给每个场景渲染一段 240 帧的相机轨迹视频

用法
----
    python scripts/51_finish_all.py --stage all
    python scripts/51_finish_all.py --stage noise ate
    python scripts/51_finish_all.py --stage video --video-runs outputs/runs/<id>
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
RESULTS = os.path.join(ROOT, "results", "results.csv")
ATE_CSV = os.path.join(ROOT, "results", "ate.csv")

HEADER = ["run_id", "scene", "part", "source", "split", "train_frames", "test_frames",
          "psnr", "ssim", "lpips", "train_psnr", "ate_rmse", "t_init_s", "t_train_s",
          "t_eval_s", "gpu", "git_commit", "notes"]

TANDT_URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
EXTRA_SCENES = ["tt-truck", "tt-train", "db-drjohnson", "db-playroom"]
EXTRA_SRC = {                       # 解压后的目录名 → 我们的场景名
    "truck": "tt-truck",
    "train": "tt-train",
    "drjohnson": "db-drjohnson",
    "playroom": "db-playroom",
}


def log(msg):
    print(msg, flush=True)


def sh(cmd, logpath=None, timeout=None):
    """跑子进程；给了 logpath 就把输出写文件（便于事后排查）。"""
    t0 = time.time()
    if logpath:
        os.makedirs(os.path.dirname(logpath), exist_ok=True)
        with open(logpath, "a", encoding="utf-8", errors="replace") as f:
            f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
            f.flush()
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
    else:
        rc = subprocess.run(cmd, cwd=ROOT, timeout=timeout).returncode
    return rc, time.time() - t0


def append_result(row):
    exists = os.path.isfile(RESULTS) and os.path.getsize(RESULTS) > 0
    with open(RESULTS, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in HEADER})


def read_list(p):
    with open(p, encoding="utf-8-sig") as f:
        return [ln.strip() for ln in f if ln.strip()]


# ---------------------------------------------------------------- noise

def stage_noise(args):
    """噪声重复：同一配置换随机种子重跑。

    为什么必须做：没有标准差，两个初始化之间 2 dB 的差距就不能当结论——
    我们实测同一份配置换种子能差 0.3 dB 量级，而 main 表里有的差距只有 0.05 dB。
    """
    log("\n" + "=" * 72 + "\n阶段：噪声重复\n" + "=" * 72)
    matrix = os.path.join(ROOT, "scripts", "35_run_p1_matrix.py")
    for scene in args.scenes:
        for seed in args.seeds:
            cmd = [PY, "-u", matrix, "--stage", "noise",
                   "--noise-scene", scene, "--noise-seeds", str(seed),
                   "--inits", *args.inits, "--frames", str(args.frames),
                   "--iterations", str(args.iterations)]
            rc, dt = sh(cmd, os.path.join(ROOT, "outputs", "logs",
                                          f"noise_{scene}_seed{seed}.log"))
            log(f"  {scene} seed={seed}: rc={rc}  {dt/60:.1f} min")
            if rc != 0:
                log(f"  ⚠ {scene} seed={seed} 失败，继续后面的")


# ---------------------------------------------------------------- ate

def stage_ate(args):
    """位姿精度：把我们估的位姿和数据集真值对比。

    只有 dl3dv-2 / re10k-1 有 cameras.json 真值；waymo 用 gt/ 目录。
    额外数据集（T&T）自带的模型就是参考，所以不和自己做 ATE。
    """
    log("\n" + "=" * 72 + "\n阶段：位姿精度（ATE）\n" + "=" * 72)
    ate_script = os.path.join(ROOT, "scripts", "46_ate.py")
    rows = []
    for scene in args.scenes:
        for init, sub in (("colmap", f"colmap_n{args.frames}"),
                          ("vggt", f"vggt_n{args.frames}")):
            model = os.path.join(ROOT, "data", scene, sub, "sparse", "0")
            if not os.path.isfile(os.path.join(model, "images.bin")):
                log(f"  {scene}/{init}: 模型不存在，跳过（{model}）")
                continue
            rc, dt = sh([PY, ate_script, "--scene", scene, "--model", model, "--json"],
                        os.path.join(ROOT, "outputs", "logs", f"ate_{scene}_{init}.log"))
            if rc != 0:
                log(f"  {scene}/{init}: 失败")
                continue
            # 重新单独跑一次拿 JSON（sh 把输出写进了日志文件）
            out = subprocess.run([PY, ate_script, "--scene", scene, "--model", model, "--json"],
                                 cwd=ROOT, capture_output=True, text=True)
            try:
                d = json.loads(out.stdout)
            except Exception:
                log(f"  {scene}/{init}: 解析 JSON 失败")
                continue
            d["init"] = init
            rows.append(d)
            log(f"  {scene:<14} {init:<7} 旋转 {d['rot_median_deg']:.3f}°  "
                f"ATE {d['ate_rmse_pct']:.3f}%  (n={d['n_frames']})")

    if rows:
        os.makedirs(os.path.dirname(ATE_CSV), exist_ok=True)
        keys = ["scene", "init", "n_frames", "rot_median_deg", "rot_mean_deg",
                "rot_p95_deg", "rel_rot_median_deg", "rot_align_delta_deg",
                "ate_rmse", "ate_rmse_pct", "scale", "reference"]
        with open(ATE_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for d in rows:
                w.writerow(d)
        log(f"\n  → 写出 {ATE_CSV}")


# ---------------------------------------------------------------- extra

def stage_extra(args):
    """额外数据集实验（课程加分项：要在“额外数据”上报告结果）。

    数据源：INRIA 官方源，国内直连可达（Google Storage 上的 Mip-NeRF 360 被墙）。
    这批数据自带 COLMAP 模型，所以 'colmap' 初始化不用重跑 SfM。
    """
    log("\n" + "=" * 72 + "\n阶段：额外数据集（Tanks&Temples + Deep Blending）\n" + "=" * 72)
    dl_dir = os.path.join(ROOT, "_dl")
    zip_path = os.path.join(dl_dir, "tandt_db.zip")
    os.makedirs(dl_dir, exist_ok=True)

    if not os.path.isdir(os.path.join(dl_dir, "tandt_db")):
        if not os.path.isfile(zip_path) or os.path.getsize(zip_path) < 600 * 1024 * 1024:
            log(f"  下载 {TANDT_URL}")
            rc, dt = sh(["curl", "-L", "--retry", "3", "-C", "-", "-o", zip_path, TANDT_URL])
            log(f"  curl rc={rc}  {dt/60:.1f} min  "
                f"{os.path.getsize(zip_path)/1024**2:.0f} MB" if os.path.isfile(zip_path) else "")
        log("  解压")
        rc, _ = sh(["unzip", "-q", "-o", zip_path, "-d", dl_dir])
        if rc != 0:
            log("  ✘ 解压失败，跳过 extra 阶段")
            return

    for src_name, our_name in EXTRA_SRC.items():
        src = os.path.join(dl_dir, "tandt_db", src_name)
        if not os.path.isdir(src):
            log(f"  跳过 {src_name}（源不存在）")
            continue
        # 1) 导入
        rc, _ = sh([PY, os.path.join(ROOT, "scripts", "00d_import_3dgs_dataset.py"),
                    "--src", src, "--name", our_name, "--resolution", str(args.extra_res)],
                   os.path.join(ROOT, "outputs", "logs", f"import_{our_name}.log"))
        if rc != 0:
            log(f"  ✘ 导入 {our_name} 失败")
            continue
        # 2) 划分
        rc, _ = sh([PY, os.path.join(ROOT, "scripts", "01_make_splits.py"),
                    "--scene", our_name],
                   os.path.join(ROOT, "outputs", "logs", f"splits_{our_name}.log"))
        if rc != 0:
            log(f"  ✘ 划分 {our_name} 失败")
            continue
        log(f"  {our_name} 准备完成")

    scenes = [s for s in EXTRA_SCENES
              if os.path.isdir(os.path.join(ROOT, "data", s, "splits"))]
    if not scenes:
        log("  没有可用的额外场景，跳过训练")
        return

    matrix = os.path.join(ROOT, "scripts", "35_run_p1_matrix.py")
    cmd = [PY, "-u", matrix, "--stage", "main", "--scenes", *scenes,
           "--inits", "colmap", "vggt", "--frames", str(args.frames),
           "--iterations", str(args.iterations)]
    rc, dt = sh(cmd, os.path.join(ROOT, "outputs", "logs", "extra_train.log"))
    log(f"  额外数据集训练 rc={rc}  {dt/60:.1f} min")


# ---------------------------------------------------------------- video

def stage_video(args):
    """Demo 视频（课程原文里唯一明确写 Bonus 的一项）。"""
    log("\n" + "=" * 72 + "\n阶段：Demo 视频\n" + "=" * 72)
    vscript = os.path.join(ROOT, "scripts", "50_make_video.py")

    runs = args.video_runs or []
    if not runs:
        # 没指定就自动挑：每个场景取 PSNR 最高的那次
        best = {}
        if os.path.isfile(RESULTS):
            with open(RESULTS, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    try:
                        p = float(r["psnr"])
                    except Exception:
                        continue
                    if r["scene"] not in best or p > best[r["scene"]][0]:
                        best[r["scene"]] = (p, r["run_id"])
        runs = [os.path.join(ROOT, "outputs", "runs", rid) for _, rid in best.values()]
        log(f"  自动挑选 {len(runs)} 个最好的 run")

    for rp in runs:
        if not os.path.isdir(rp):
            log(f"  跳过（不存在）{rp}")
            continue
        rc, dt = sh([PY, vscript, "--model", rp, "--frames", str(args.video_frames),
                     "--fps", str(args.video_fps)],
                    os.path.join(rp, "video.log"))
        log(f"  {os.path.basename(rp)}: rc={rc}  {dt/60:.1f} min")


# ---------------------------------------------------------------- 汇总

def summary():
    log("\n" + "=" * 72 + "\n结果汇总\n" + "=" * 72)
    if not os.path.isfile(RESULTS):
        log("没有 results.csv")
        return
    rows = []
    with open(RESULTS, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                r["_p"] = float(r["psnr"])
            except Exception:
                continue
            rows.append(r)

    # 每个 (场景, 初始化) 聚合出 均值 ± 标准差
    from collections import defaultdict
    import statistics as st
    agg = defaultdict(list)
    for r in rows:
        agg[(r["scene"], r["source"])].append(r["_p"])

    log(f"{'场景':<16} {'初始化':<8} {'n':>3} {'均值 PSNR':>10} {'标准差':>8} {'最小':>8} {'最大':>8}")
    for (scene, init), ps in sorted(agg.items()):
        if len(ps) == 0:
            continue
        sd = st.stdev(ps) if len(ps) > 1 else float("nan")
        log(f"{scene:<16} {init:<8} {len(ps):>3} {st.mean(ps):>10.3f} {sd:>8.3f} "
            f"{min(ps):>8.3f} {max(ps):>8.3f}")
    log("\n（标准差是 nan 表示该配置只跑了一次，不能当噪声门槛用）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["all"],
                    choices=["all", "noise", "ate", "extra", "video"])
    ap.add_argument("--scenes", nargs="+", default=["dl3dv-2", "re10k-1", "waymo-405841"])
    ap.add_argument("--inits", nargs="+", default=["colmap", "vggt"])
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--extra-res", type=int, default=2,
                    help="额外数据集的降采样倍数（T&T 是 1920x1080，24GB 卡用 2）")
    ap.add_argument("--video-frames", type=int, default=240)
    ap.add_argument("--video-fps", type=int, default=30)
    ap.add_argument("--video-runs", nargs="+", default=None)
    args = ap.parse_args()

    stages = ["noise", "ate", "extra", "video"] if "all" in args.stage else args.stage
    log(f"阶段: {stages}   场景: {args.scenes}")
    t0 = time.time()
    for s in stages:
        try:
            {"noise": stage_noise, "ate": stage_ate,
             "extra": stage_extra, "video": stage_video}[s](args)
        except Exception as e:
            log(f"✘ 阶段 {s} 抛异常: {type(e).__name__}: {e}")
        log(f"  （累计 {(time.time()-t0)/3600:.2f} h）")

    summary()
    log(f"\n全部完成，总用时 {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
