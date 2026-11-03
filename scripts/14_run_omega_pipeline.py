"""
14_run_omega_pipeline.py — 一键跑 VGGT-Ω 初始化 + 3DGS 训练 + 评测（单个场景）。

为什么单独写
-----------
`35_run_p1_matrix.py` 的 `ensure_poses()` 只认 colmap / vggt 两种来源。
VGGT-Ω 需要一个额外步骤（先推理导出 COLMAP 格式，再训练），所以单独封装。

流程
----
  1. 用 `11_run_vggt.py` 的同款方式建 `data/<scene>/vggt_omega_n<N>/images/`
  2. `13_export_omega_colmap.py` 推理 + 导出 COLMAP 格式
  3. `12_clean_colmap_model.py` 剔除跑飞的相机（相机发散会污染 3DGS 的尺度归一化）
  4. `run_with_guard.py` + 3DGS 训练
  5. render + `41_eval.py`
  6. 结果追加进 `results/results.csv`

用法:
    python scripts/14_run_omega_pipeline.py --scene dl3dv-2 --frames 50
    python scripts/14_run_omega_pipeline.py --scene re10k-1 --frames 50
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
GUARD = os.path.join(ROOT, "scripts", "run_with_guard.py")
TRAIN = os.path.join(ROOT, "third_party", "gaussian-splatting", "train.py")
RENDER = os.path.join(ROOT, "third_party", "gaussian-splatting", "render.py")
EVAL = os.path.join(ROOT, "scripts", "41_eval.py")
EXPORT = os.path.join(ROOT, "scripts", "13_export_omega_colmap.py")
CLEAN = os.path.join(ROOT, "scripts", "12_clean_colmap_model.py")
RESULTS = os.path.join(ROOT, "results", "results.csv")

HEADER = ["run_id", "scene", "part", "source", "split", "train_frames", "test_frames",
          "psnr", "ssim", "lpips", "train_psnr", "ate_rmse", "t_init_s", "t_train_s",
          "t_eval_s", "gpu", "git_commit", "notes"]

TEST_ITER = [1000, 5000, 10000, 15000, 20000, 25000, 30000]

# 归一化 fx 不再硬编码：直接从数据集的 cameras.json 读。
# （踩过的坑：一开始两个场景都写死 dl3dv 的 0.797，导致 re10k 报"偏差 7.56%"，
#   而 re10k 的真值是 0.8573，Ω 给的 438.95/512=0.8573 其实完全正确。）
def dataset_fx_norm(scene_dir):
    cj = os.path.join(scene_dir, "cameras.json")
    if not os.path.isfile(cj):
        return None
    try:
        with open(cj, encoding="utf-8-sig") as f:
            d = json.load(f)
        return float(d[0]["fx"])
    except Exception:
        return None


def sh(cmd, log):
    os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
    with open(log, "a", encoding="utf-8", errors="replace") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc


def read_list(p):
    with open(p, encoding="utf-8-sig") as f:
        return [ln.strip() for ln in f if ln.strip()]


def append_result(row):
    exists = os.path.isfile(RESULTS) and os.path.getsize(RESULTS) > 0
    with open(RESULTS, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in HEADER})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--ckpt",
                    default="third_party/weights/vggt_omega/vggt_omega_1b_416_reproduce.pt")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--force", action="store_true", help="重新推理导出")
    args = ap.parse_args()

    scene = args.scene
    sd = os.path.join(ROOT, "data", scene)
    train_list = os.path.join(sd, "splits", f"p1_n{args.frames}.txt")
    test_list = os.path.join(sd, "splits", "test.txt")
    for p in (train_list, test_list):
        if not os.path.isfile(p):
            raise SystemExit(f"缺少 {p}，先跑 scripts/35_run_p1_matrix.py --stage main")

    ws = os.path.join(sd, f"vggt_omega_n{args.frames}")
    log_base = os.path.join(ROOT, "outputs", "logs")

    # ---------- 1. 准备 image 目录（用训练帧 ∪ 测试帧，和其他两种初始化一致）
    if not os.path.isdir(os.path.join(ws, "images")):
        src = os.path.join(sd, "images") if os.path.isdir(os.path.join(sd, "images")) \
            else os.path.join(sd, "rgb")
        names = sorted(set(read_list(train_list)) | set(read_list(test_list)))
        os.makedirs(os.path.join(ws, "images"), exist_ok=True)
        for n in names:
            dst = os.path.join(ws, "images", n)
            if not os.path.exists(dst):
                try:
                    os.link(os.path.join(src, n), dst)
                except OSError:
                    shutil.copy2(os.path.join(src, n), dst)
        print(f"[{scene}] 图片 {len(names)} 张 → {ws}/images")

    # ---------- 2. 推理 + 导出
    if args.force or not os.path.isfile(os.path.join(ws, "sparse", "0", "images.bin")):
        t0 = time.time()
        cmd = [PY, EXPORT, "--scene-dir", ws, "--ckpt", args.ckpt,
               "--resolution", str(args.resolution)]
        fx_ref = dataset_fx_norm(sd)
        if fx_ref:
            cmd += ["--expect-fx-norm", str(fx_ref)]
        rc = sh(cmd, os.path.join(ws, "export.log"))
        t_init = time.time() - t0
        print(f"[{scene}] 导出 rc={rc}  {t_init/60:.1f} min")
        if rc != 0:
            raise SystemExit("导出失败，看 export.log")
    else:
        t_init = 0.0
        print(f"[{scene}] 已有导出结果，复用")

    # ---------- 3. 清理跑飞的相机
    sh([PY, CLEAN, "--model", os.path.join(ws, "sparse", "0"),
        "--inplace", "--max-factor", "10"], os.path.join(ws, "clean.log"))

    # ---------- 4. 训练
    run_id = f"{scene}__vggt_omega__n{args.frames}__3dgs__{time.strftime('%Y%m%d-%H%M')}"
    out = os.path.join(ROOT, "outputs", "runs", run_id)
    os.makedirs(out, exist_ok=True)
    print(f"[{scene}] 训练 → {run_id}")
    cmd = [PY, GUARD, "--train-list", train_list, "--test-list", test_list, "--",
           TRAIN, "-s", ws, "-m", out, "--eval", "--disable_viewer",
           "--iterations", str(args.iterations),
           "--test_iterations"] + [str(i) for i in TEST_ITER]
    # 分辨率：跟其他初始化保持一致（从 meta.yaml 读）
    meta = os.path.join(sd, "meta.yaml")
    if os.path.isfile(meta):
        for line in open(meta, encoding="utf-8-sig"):
            if line.strip().startswith("resolution:"):
                rv = line.split(":", 1)[1].strip()
                if rv.lstrip("-").isdigit() and int(rv) > 0:
                    cmd += ["-r", rv]
    t0 = time.time()
    rc = sh(cmd, os.path.join(out, "train.log"))
    t_train = time.time() - t0
    print(f"[{scene}] 训练 rc={rc}  {t_train/60:.1f} min")
    if rc != 0:
        raise SystemExit("训练失败，看 train.log")

    # ---------- 5. 渲染 + 评测
    sh([PY, GUARD, "--test-list", test_list, "--",
        RENDER, "-m", out, "--skip_train", "--quiet"], os.path.join(out, "render.log"))
    t0 = time.time()
    rc = sh([PY, EVAL, "--run", out], os.path.join(out, "eval.log"))
    t_eval = time.time() - t0
    if rc != 0:
        raise SystemExit("评测失败，看 eval.log")

    with open(os.path.join(out, "results.json"), encoding="utf-8") as f:
        m = list(json.load(f).values())[0]

    n_tr = len([n for n in read_list(train_list)
                if os.path.isfile(os.path.join(ws, "sparse", "0", "images.bin"))])
    row = dict(run_id=run_id, scene=scene, part="P1", source="vggt_omega",
               split=f"n{args.frames}", train_frames=args.frames,
               test_frames=len(read_list(test_list)),
               psnr=round(m["PSNR"], 4), ssim=round(m["SSIM"], 4),
               lpips=round(m["LPIPS"], 4), ate_rmse="",
               t_init_s=round(t_init, 1), t_train_s=round(t_train, 1),
               t_eval_s=round(t_eval, 1), gpu="RTX 4090",
               git_commit="omega-pipeline",
               notes=f"N={args.frames}; VGGT-Omega 416-reproduce; "
                     f"export+inference time; LPIPS=vgg/0-1")
    append_result(row)
    print(f"[{scene}] ✓ PSNR {row['psnr']:.3f}  SSIM {row['ssim']:.4f}  "
          f"LPIPS {row['lpips']:.4f}")


if __name__ == "__main__":
    main()
