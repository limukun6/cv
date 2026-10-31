"""
00c_make_bundle.py — 把上租用机器需要的最小数据集打包成一个 tar.gz。

传什么 / 不传什么
----------------
传：
    data/<scene>/rgb 或 images/     原始帧（3 场景共 0.84 GB）
    data/<scene>/splits/            冻结的划分（可复现的关键，几 KB）
    data/<scene>/meta.yaml          场景元信息
    data/<scene>/cameras.json       数据集自带的位姿（用于 ATE 评测）
    data/<scene>/intrinsics.json
    data/<scene>/colmap/sparse/     已算好的 COLMAP 模型（几 MB，省得在租用时间里重跑）
    data/<scene>/calib, gt          Waymo 的标定与位姿（用于 ATE）

不传：
    outputs/   运行产物（几十 GB）
    Project4.zip / 原始下载包
    third_party/weights  VGGT 权重 4.68 GB —— 在租用机器上用 00b 脚本重新下（那边网快）

为什么要把 COLMAP 模型传上去
--------------------------
租用时间按小时计。COLMAP 三场景一共跑了 ~19 分钟（本机 CPU），
传一个几 MB 的模型比在那边重跑划算得多。而且**用同一份模型**才能保证
位姿和本机的基线完全一致。

用法:
    python scripts/00c_make_bundle.py
    python scripts/00c_make_bundle.py --out p1_bundle.tar.gz
"""
import argparse
import os
import sys
import tarfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 每个场景里要带走的相对路径（不存在就跳过）
WANTED = [
    "meta.yaml",
    "cameras.json",
    "intrinsics.json",
    "splits",           # 目录
    "colmap/sparse",    # 目录
    "calib",            # Waymo
    "gt",               # Waymo
]
# 图片目录名候选
IMG_DIRS = ["rgb", "images"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="p1_bundle.tar.gz")
    ap.add_argument("--scenes", nargs="+",
                    default=["dl3dv-2", "re10k-1", "waymo-405841"])
    args = ap.parse_args()

    data_root = os.path.join(ROOT, args.data)
    if not os.path.isdir(data_root):
        sys.exit(f"找不到 {data_root}")

    out = os.path.join(ROOT, args.out)
    total = 0
    counts = {}

    with tarfile.open(out, "w:gz") as tar:
        for scene in args.scenes:
            sd = os.path.join(data_root, scene)
            if not os.path.isdir(sd):
                print(f"[跳过] {scene}: 目录不存在")
                continue

            # 图片目录
            for d in IMG_DIRS:
                p = os.path.join(sd, d)
                if os.path.isdir(p):
                    n = sum(1 for f in os.listdir(p) if f.lower().endswith((".png", ".jpg")))
                    print(f"  {scene}/{d}: {n} 张图")
                    tar.add(p, arcname=os.path.join(args.data, scene, d))
                    counts[scene] = n

            # 其他小文件/目录
            for rel in WANTED:
                p = os.path.join(sd, rel)
                if os.path.isfile(p):
                    tar.add(p, arcname=os.path.join(args.data, scene, rel))
                elif os.path.isdir(p):
                    # 过滤掉 .ipynb_checkpoints 之类的垃圾
                    for f in os.listdir(p):
                        if f.startswith(".ipynb"):
                            continue
                        fp = os.path.join(p, f)
                        tar.add(fp, arcname=os.path.join(args.data, scene, rel, f))

    total = os.path.getsize(out)
    print(f"\n打包完成: {out}  ({total / 1024 ** 3:.2f} GB)")
    for s, n in counts.items():
        print(f"  {s}: {n} 张图")

    # 附一份说明，免得对方不知道该怎么解
    print("\n对方拿到后：")
    print(f"  tar xzf {os.path.basename(out)}    # 解到仓库根目录，得到 data/")
    print("  python scripts/00b_fetch_vggt_weights.py")


if __name__ == "__main__":
    main()
