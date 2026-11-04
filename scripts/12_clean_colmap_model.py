"""
12_clean_colmap_model.py — 剔除跑飞的相机，修复场景尺度被污染的问题。

为什么需要这个脚本（踩过的坑）
----------------------------
COLMAP 在稀疏帧子集上做增量重建时，BA 偶尔会让**个别相机的位置发散**。
实测 dl3dv-2 的 89 帧子集：

    colmap_n50:  相机中心包围盒对角 = 7.2e9
                 相邻间距中位数 0.300，而最大值 7.2e9   ← 只有 1 个坏相机

后果非常隐蔽而且严重：
  3DGS 的 `getNerfppNorm()` 用**所有训练相机的中心**算一个半径 radius，
  然后用它做 `spatial_lr_scale`、位置学习率缩放、以及致密化阈值。
  一个跑飞到 7e9 的相机，会把 radius 从 ~5 放大到 ~1e9，
  于是位置学习率和各种阈值全部失效 —— 训练完全练废，
  PSNR 卡在 17 dB 上不去，而**日志里没有任何报错**。

同一批数据在 59 帧时没有发散，所以"帧数多的那次反而更差"，
这个反直觉现象就是这么来的。

用法:
    python scripts/12_clean_colmap_model.py --model data/dl3dv-2/colmap_n50/sparse/0
    # 原地清理（先备份到 sparse/0_raw）
    python scripts/12_clean_colmap_model.py --model <dir> --inplace
"""
import argparse
import os
import shutil
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="COLMAP 模型目录（含 cameras.bin 等）")
    ap.add_argument("--out", default=None, help="输出目录，默认 <model>_clean")
    ap.add_argument("--inplace", action="store_true",
                    help="原地替换（原模型备份为 <model>_raw）")
    ap.add_argument("--max-factor", type=float, default=10.0,
                    help="中心离群判定：到中位中心的距离 > max_factor × p95 就剔除")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import pycolmap

    rec = pycolmap.Reconstruction(args.model)
    ids = list(rec.images.keys())
    if len(ids) < 3:
        print("相机太少，跳过")
        return

    centers = []
    for i in ids:
        img = rec.images[i]
        # cam_from_world 是 world→camera；取逆得到相机中心。
        # ⚠️ 不同 pycolmap 版本里它可能是**属性**也可能是**方法**：
        #    属性: img.cam_from_world      （4.2 上是这个）
        #    方法: img.cam_from_world()    （其它版本）
        # 不做兼容会报 'builtin_function_or_method' object has no attribute 'inverse'
        cfw = img.cam_from_world
        if callable(cfw):
            cfw = cfw()
        c = cfw.inverse().translation
        centers.append(np.array([c[0], c[1], c[2]]))
    centers = np.array(centers)

    med = np.median(centers, axis=0)
    d = np.linalg.norm(centers - med, axis=1)
    p95 = np.percentile(d, 95)
    if p95 <= 0:
        p95 = np.median(d) + 1e-9
    thresh = args.max_factor * p95

    bad = np.where(d > thresh)[0]
    ext_before = np.linalg.norm(centers.max(0) - centers.min(0))
    print(f"模型 {args.model}")
    print(f"  相机 {len(ids)} 个，中心包围盒对角 {ext_before:.3f}")
    print(f"  到中位中心距离: 中位 {np.median(d):.3f}  p95 {p95:.3f}  最大 {d.max():.3e}")
    print(f"  判定阈值 {thresh:.3f} → 离群 {len(bad)} 个")

    if len(bad) == 0:
        print("  ✓ 没有离群相机，无需处理")
        return

    for i in bad:
        print(f"    ✗ 剔除 {rec.images[ids[i]].name}  距离 {d[i]:.3e}")

    if args.dry_run:
        return

    keep = [ids[i] for i in range(len(ids)) if i not in set(bad)]
    names_keep = {rec.images[i].name for i in keep}

    # 用 pycolmap 自带的 deregister_image：它会同时删掉该相机的观测，
    # 并清理因此只剩 <2 个观测的 3D 点。
    # （不要自己重建 Reconstruction —— 逐个 add_point3D 会因为 point2D 已经
    #   带着 point3D_id 而触发 "HasPoint3D" 断言失败。）
    for i in bad:
        rec.deregister_image(ids[i])

    out = args.out or (args.model.rstrip("/") + "_clean")
    if args.inplace:
        raw = args.model.rstrip("/") + "_raw"
        if not os.path.isdir(raw):
            shutil.copytree(args.model, raw)
            print(f"  原模型已备份到 {raw}")
        tmp = args.model.rstrip("/") + "_tmp"
        os.makedirs(tmp, exist_ok=True)
        rec.write(tmp)
        shutil.rmtree(args.model)
        shutil.move(tmp, args.model)
    else:
        os.makedirs(out, exist_ok=True)
        rec.write(out)

    c2 = np.array([[rec.images[i].cam_from_world.inverse().translation[k] for k in range(3)]
                   for i in keep])
    ext_after = np.linalg.norm(c2.max(0) - c2.min(0))
    print(f"  清理后: {len(keep)} 个相机，包围盒对角 {ext_after:.3f}")
    print(f"  输出: {args.model if args.inplace else out}")


if __name__ == "__main__":
    main()
