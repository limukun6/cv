"""
11_run_vggt.py — 用 VGGT 估位姿，并直接导出成 COLMAP 格式。

为什么能做到"直接导出"
--------------------
VGGT 官方仓库自带 `demo_colmap.py`，它把预测的相机参数和三维点
**直接写成 COLMAP 的二进制格式**（cameras.bin / images.bin / points3D.bin）。
所以不需要我们写转换器 —— 省掉一整天的工作量。

目录约定（和 3DGS 完全兼容）：
    data/<scene>/vggt[_n50]/
    ├── images/          ← 硬链接过去的帧
    └── sparse/0/        ← VGGT 写出来的模型
之后 3DGS 直接 `-s data/<scene>/vggt_n50` 就能训练。

用法:
    python scripts/11_run_vggt.py --data data --scene dl3dv-2 --image-list data/dl3dv-2/splits/p1_n50.txt
    python scripts/11_run_vggt.py --data data --scene dl3dv-2 --use-ba
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VGGT_DEMO = os.path.join(ROOT, "third_party", "vggt", "demo_colmap.py")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def image_subdir(scene_dir):
    meta = os.path.join(scene_dir, "meta.yaml")
    if os.path.isfile(meta):
        for line in open(meta, encoding="utf-8-sig"):
            if line.startswith("image_subdir:"):
                return line.split(":", 1)[1].strip()
    for cand in ("rgb", "images"):
        if os.path.isdir(os.path.join(scene_dir, cand)):
            return cand
    sys.exit("找不到图片目录")


def link(src, dst_dir, only=None):
    os.makedirs(dst_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(src) if f.lower().endswith(IMAGE_EXTS))
    if only is not None:
        names = [n for n in names if n in only]
    n = 0
    for name in names:
        d = os.path.join(dst_dir, name)
        if os.path.exists(d):
            n += 1
            continue
        try:
            os.link(os.path.join(src, name), d)
        except OSError:
            shutil.copy2(os.path.join(src, name), d)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--image-list", default=None, help="只处理这些图片（一行一个文件名）")
    ap.add_argument("--out-name", default=None, help="输出目录名，默认 vggt / vggt_n{N}")
    ap.add_argument("--use-ba", action="store_true",
                    help="用 bundle adjustment 精修（官方 --use_ba）。"
                         "如果 VGGT-Ω 没开源，这就是 Plan C 的替代方案")
    ap.add_argument("--camera-type", default="PINHOLE",
                    choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "OPENCV"])
    ap.add_argument("--shared-camera", action="store_true", default=True,
                    help="所有帧共用一个相机（我们的数据都是单相机，默认开）")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    if not os.path.isfile(VGGT_DEMO):
        sys.exit(f"找不到 {VGGT_DEMO}。先跑 env/setup_linux.sh，或 git clone "
                 f"https://github.com/facebookresearch/vggt third_party/vggt")

    scene_dir = os.path.join(args.data, args.scene)
    src = os.path.join(scene_dir, image_subdir(scene_dir))

    only = None
    if args.image_list:
        with open(args.image_list, encoding="utf-8-sig") as f:
            only = set(ln.strip() for ln in f if ln.strip())

    name = args.out_name
    if name is None:
        name = "vggt_ba" if args.use_ba else "vggt"
        if only is not None:
            name += f"_n{len(only)}"
    ws = os.path.join(scene_dir, name)

    if os.path.isdir(os.path.join(ws, "sparse")) and not args.force:
        print(f"{ws}/sparse 已存在，跳过（加 --force 重跑）")
        return

    if args.force and os.path.isdir(ws):
        shutil.rmtree(ws)

    n = link(src, os.path.join(ws, "images"), only=only)
    if n == 0:
        sys.exit("没有可处理的图片")
    print(f"VGGT 输入: {n} 张图 → {ws}")

    cmd = [sys.executable, "-u", VGGT_DEMO, f"--scene_dir={ws}",
           f"--camera_type={args.camera_type}"]
    if args.use_ba:
        cmd.append("--use_ba")
    if args.shared_camera:
        cmd.append("--shared_camera")

    print("运行:", " ".join(cmd))
    t0 = time.time()
    log = os.path.join(ws, "vggt.log")
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    dt = time.time() - t0
    print(f"rc={rc}  用时 {dt/60:.1f} 分钟  日志: {log}")
    if rc != 0:
        print("✗ VGGT 失败。常见原因：显存不足（减少图片数）、权重没下（跑 00b 脚本）")
        sys.exit(rc)

    sparse = os.path.join(ws, "sparse")
    sub = [d for d in os.listdir(sparse)] if os.path.isdir(sparse) else []
    print(f"输出: {sparse}  子目录 {sub}")

    # 写一行耗时记录，方便后面填 results.csv
    with open(os.path.join(ws, "vggt_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"{dt:.1f}\n")
    print(f"\n下一步: 3DGS 训练 -s {ws}")


if __name__ == "__main__":
    main()
