"""
10_run_colmap.py — 用 COLMAP 跑经典 SfM，得到相机位姿 + 稀疏点云。

这是 Part 1 的 Plan A（传统路线）。

设计要点
--------
1. **对全部帧跑 COLMAP**，不区分 train/test。
   原因：NVS 评测的通行做法就是「所有图都参与 SfM，训练时只用 train 子集」。
   3DGS 官方 `--eval` 用的是「排序后每 8 张留 1 张作测试」，和我们契约里的
   `pool[::8]` **完全一致**，所以两边能对上。

2. **用硬链接而不是复制**。306 张图复制一份要几百 MB，硬链接不占额外空间。

3. **相机模型用 PINHOLE**。因为数据自带的 cameras.json 只给 fx/fy/cx/cy、
   没有畸变参数，用 PINHOLE 才能和 Ground Truth 做干净的交叉验证。

4. 输出目录就是 3DGS 原生的输入格式：
      data/<scene>/colmap/
      ├── images/          (硬链接)
      ├── database.db
      └── sparse/0/{cameras.bin, images.bin, points3D.bin}
   这样 3DGS 可以直接 `-s data/<scene>/colmap --eval` 训练，不用再转换。

用法:
    python scripts/10_run_colmap.py --data data --scene dl3dv-2
    python scripts/10_run_colmap.py --data data --scene dl3dv-2 --use-gpu 1
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

COLMAP_EXE = r"D:\conda\Anaconda\envs\colmap\Library\bin\colmap.exe"
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def image_subdir(scene_dir):
    meta = os.path.join(scene_dir, "meta.yaml")
    if os.path.isfile(meta):
        for line in open(meta, encoding="utf-8"):
            if line.startswith("image_subdir:"):
                return line.split(":", 1)[1].strip()
    for cand in ("rgb", "images"):
        if os.path.isdir(os.path.join(scene_dir, cand)):
            return cand
    sys.exit("找不到图片目录")


def link_images(src_dir, dst_dir):
    """用硬链接把图片挂进 COLMAP 工作区；跨卷或不支持时退回复制。"""
    os.makedirs(dst_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_EXTS))
    linked, copied = 0, 0
    for n in names:
        s, d = os.path.join(src_dir, n), os.path.join(dst_dir, n)
        if os.path.exists(d):
            continue
        try:
            os.link(s, d)
            linked += 1
        except OSError:
            shutil.copy2(s, d)
            copied += 1
    return len(names), linked, copied


def run(cmd, log_path):
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8", errors="replace") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--use-gpu", type=int, default=0, help="1=SIFT 用 GPU（需 CUDA 版本匹配）")
    ap.add_argument("--overlap", type=int, default=10, help="sequential_matcher 的帧间隔")
    ap.add_argument("--max-image-size", type=int, default=1600)
    ap.add_argument("--force", action="store_true", help="覆盖已有结果")
    args = ap.parse_args()

    if not os.path.isfile(COLMAP_EXE):
        sys.exit(f"找不到 colmap: {COLMAP_EXE}")

    scene_dir = os.path.join(args.data, args.scene)
    src = os.path.join(scene_dir, image_subdir(scene_dir))
    ws = os.path.join(scene_dir, "colmap")
    sparse = os.path.join(ws, "sparse")
    db = os.path.join(ws, "database.db")
    log = os.path.join(ws, "colmap.log")

    if os.path.isdir(sparse) and os.listdir(sparse) and not args.force:
        print(f"{sparse} 已有结果，跳过（加 --force 重跑）")
        return

    os.makedirs(ws, exist_ok=True)
    if args.force:
        for p in (sparse, db, log):
            if os.path.isdir(p):
                shutil.rmtree(p)
            elif os.path.isfile(p):
                os.remove(p)

    print(f"=== COLMAP on {args.scene} ===")
    n, nl, nc = link_images(src, os.path.join(ws, "images"))
    print(f"  图片 {n} 张（硬链接 {nl}，复制 {nc}）")

    common = ["--database_path", db, "--image_path", os.path.join(ws, "images")]

    # 1) 特征提取
    rc, dt = run([COLMAP_EXE, "feature_extractor", *common,
                  "--ImageReader.single_camera", "1",
                  "--ImageReader.camera_model", "PINHOLE",
                  "--FeatureExtraction.use_gpu", str(args.use_gpu),
                  "--FeatureExtraction.max_image_size", str(args.max_image_size),
                  "--SiftExtraction.max_num_features", "8192"], log)
    print(f"  [1/3] feature_extractor  {dt:6.1f}s  rc={rc}")
    if rc != 0:
        sys.exit(f"特征提取失败，看 {log}")

    # 2) 顺序匹配（我们的数据是视频序列，用 sequential 比 exhaustive 快得多）
    rc, dt = run([COLMAP_EXE, "sequential_matcher", "--database_path", db,
                  "--SequentialMatching.overlap", str(args.overlap),
                  "--SequentialMatching.quadratic_overlap", "1",
                  "--SequentialMatching.loop_detection", "0",
                  "--FeatureMatching.use_gpu", str(args.use_gpu)], log)
    print(f"  [2/3] sequential_matcher {dt:6.1f}s  rc={rc}")
    if rc != 0:
        sys.exit(f"匹配失败，看 {log}")

    # 3) 增量式重建
    #    注意：COLMAP 的 --output_path 必须是**已存在**的目录，否则报
    #    "output_path is not a directory"（这个报错不直观，容易卡人）
    os.makedirs(sparse, exist_ok=True)
    rc, dt = run([COLMAP_EXE, "mapper", *common,
                  "--output_path", sparse,
                  "--Mapper.min_num_matches", "15",
                  "--Mapper.ba_global_function_tolerance", "1e-6"], log)
    print(f"  [3/3] mapper             {dt:6.1f}s  rc={rc}")
    if rc != 0:
        sys.exit(f"重建失败，看 {log}")

    # 结果统计
    models = [d for d in os.listdir(sparse) if os.path.isdir(os.path.join(sparse, d))] \
        if os.path.isdir(sparse) else []
    if not models:
        sys.exit("COLMAP 没有产出任何模型（所有帧都注册失败）")

    print(f"\n重建出 {len(models)} 个模型: {sorted(models)}")
    for m in sorted(models):
        rc, _ = run([COLMAP_EXE, "model_analyzer", "--path", os.path.join(sparse, m)], log)
    with open(log, encoding="utf-8", errors="replace") as f:
        print("\n".join(f.read().splitlines()[-12:]))

    print(f"\n下一步: python scripts/22_compare_poses.py --scene {args.scene}")


if __name__ == "__main__":
    main()
