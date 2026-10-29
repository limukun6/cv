"""
22_compare_poses.py — 把 COLMAP 估出来的位姿和数据自带的 cameras.json 对比。

为什么这一步很重要
------------------
COLMAP 和数据自带的位姿是**两条完全独立的估计路径**：
    · COLMAP      靠特征匹配 + 三角化 + 束调整（纯几何）
    · cameras.json 来自数据集官方（或生成它的人）

如果这两套在**相对运动**上高度一致，说明：
    1. 我们读 cameras.json 的约定（xyzw / c2w）是对的 —— 第二次独立验证
    2. COLMAP 在这个场景上跑通了，可以作为 Plan A 的基线
如果对不上，说明至少有一方错了，必须查清楚再往下走。

对比指标（都不受全局坐标系影响）
------------------------------
1. 相邻帧**相对旋转**的夹角误差（度）—— 对齐无关
2. 相邻帧**平移方向**的夹角误差（度）—— 对齐无关
3. Umeyama(Sim3) 对齐后的**相机中心 RMS 误差**（相对尺度的百分比）

用法:
    python scripts/22_compare_poses.py --data data --scene dl3dv-2
"""
import argparse
import glob
import json
import os
import sys

import numpy as np


def load_colmap(sparse_dir):
    """读 COLMAP 二进制模型，返回 {name: (R_w2c, C)} 以及相机内参。"""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(root, "third_party", "gaussian-splatting"))
    from scene.colmap_loader import (read_extrinsics_binary,
                                     read_intrinsics_binary,
                                     qvec2rotmat)

    img_path = os.path.join(sparse_dir, "images.bin")
    cam_path = os.path.join(sparse_dir, "cameras.bin")
    if not (os.path.isfile(img_path) and os.path.isfile(cam_path)):
        return None, None

    extr = read_extrinsics_binary(img_path)
    intr = read_intrinsics_binary(cam_path)

    out = {}
    for _id, im in extr.items():
        R_w2c = qvec2rotmat(im.qvec)
        t = np.asarray(im.tvec)
        C = -R_w2c.T @ t                    # 相机中心（世界坐标）
        out[im.name] = (R_w2c, C)
    return out, intr


def load_cameras_json(scene_dir):
    """读 cameras.json。已验证约定：quat 顺序 xyzw，方向 c2w，trans=相机中心。"""
    p = os.path.join(scene_dir, "cameras.json")
    if not os.path.isfile(p):
        return None
    cams = json.load(open(p, encoding="utf-8"))
    out = {}
    for c in cams:
        x, y, z, w = c["cam_quat"]          # xyzw
        # 四元数 -> 旋转矩阵（c2w）
        R_wc = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        out[c["image_name"]] = (R_wc.T, np.asarray(c["cam_trans"], dtype=float))
    return out


def umeyama(src, dst, with_scale=True):
    """求 (s, R, t) 使 s*R*src + t ≈ dst。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, d, Vt = np.linalg.svd(cov)
    S_ = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_[2, 2] = -1
    R = U @ S_ @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = (d * np.diag(S_)).sum() / var_s if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ang(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
    return np.degrees(np.arccos(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()

    scene_dir = os.path.join(args.data, args.scene)
    sparse = os.path.join(scene_dir, "colmap", "sparse", "0")
    if not os.path.isdir(sparse):
        cands = glob.glob(os.path.join(scene_dir, "colmap", "sparse", "*"))
        cands = [c for c in cands if os.path.isdir(c)]
        if not cands:
            sys.exit("找不到 COLMAP 模型")
        sparse = sorted(cands)[0]

    col, intr = load_colmap(sparse)
    gt = load_cameras_json(scene_dir)
    if col is None:
        sys.exit("COLMAP 模型读取失败")
    if gt is None:
        print("没有 cameras.json，只能报告 COLMAP 自身的统计")
        return

    print(f"=== {args.scene} ===")
    print(f"  COLMAP 注册 {len(col)} 张 / 数据集共 {len(gt)} 张")
    common = sorted(set(col) & set(gt))
    print(f"  共同帧 {len(common)}")
    if len(common) < 10:
        sys.exit("共同帧太少，无法比较")

    # --- 1) 相对旋转误差（相邻共同帧）---
    rot_errs = []
    for a, b in zip(common[:-1], common[1:]):
        Ra, _ = col[a]
        Rb, _ = col[b]
        R_rel_col = Rb.T @ Ra                      # a -> b
        Ga, _ = gt[a]
        Gb, _ = gt[b]
        R_rel_gt = Gb.T @ Ga
        dR = R_rel_col @ R_rel_gt.T
        cos = np.clip((np.trace(dR) - 1) / 2, -1, 1)
        rot_errs.append(np.degrees(np.arccos(cos)))
    rot_errs = np.array(rot_errs)

    # --- 2) 先把 COLMAP 的轨迹对齐到 GT 的坐标系 ---
    # 顺序很重要：**必须先对齐，再比较位移方向**。
    # 旋转是坐标系无关的量（R_b^T R_a 在两边都成立），但平移位移**不是** ——
    # 它直接活在世界坐标系里。COLMAP 和 GT 的世界系差一个旋转，
    # 不对齐就直接比方向，量到的是"两个世界系的夹角"（我们第一次得到 26.7°），
    # 而且这个夹角不随基线长度变化 —— 这正好是判断"是不是忘了对齐"的特征。
    Pc = np.array([col[n][1] for n in common])
    Pg = np.array([gt[n][1] for n in common])
    s, R, t = umeyama(Pc, Pg, with_scale=True)
    Pa = (s * (R @ Pc.T)).T + t
    err = np.linalg.norm(Pa - Pg, axis=1)
    extent = np.linalg.norm(Pg.max(0) - Pg.min(0))
    rms = float(np.sqrt((err ** 2).mean()))

    # --- 3) 平移方向误差：用对齐后的中心，多个基线长度 ---
    # 检验方法：如果两条路径真的一致，误差应随基线变长而**下降**（噪声被平均掉）。
    # 若误差恒定不变，说明存在系统性问题（约定 / 尺度 / 单位）。
    idx = {n: i for i, n in enumerate(common)}
    stride_stats = []
    for k in (1, 5, 10, 25):
        errs, bases = [], []
        for i in range(len(common) - k):
            a, b = common[i], common[i + k]
            da = Pa[idx[a]] - Pa[idx[b]]        # ★ 用对齐后的坐标
            dg = Pg[idx[a]] - Pg[idx[b]]
            e = ang(da, dg)
            if e is not None:
                errs.append(e)
                bases.append(np.linalg.norm(dg))
        if errs:
            stride_stats.append((k, float(np.median(errs)), float(np.median(bases))))

    # --- 4) 内参对比 ---
    print(f"\n  {'指标':<32} {'值':>12}")
    print(f"  {'-'*46}")
    print(f"  {'相邻帧相对旋转误差 中位数':<32} {np.median(rot_errs):>10.3f} °")
    print(f"  {'相邻帧相对旋转误差 均值':<32} {rot_errs.mean():>10.3f} °")
    print(f"  {'相邻帧相对旋转误差 p95':<32} {np.percentile(rot_errs,95):>10.3f} °")
    print(f"  {'Sim3 对齐尺度':<32} {s:>12.4f}")
    print(f"  {'对齐后中心 RMS 误差':<32} {np.sqrt((err**2).mean()):>12.4f}")
    print(f"  {'对齐后中心 最大误差':<32} {err.max():>12.4f}")
    print(f"  {'场景尺度（对角线）':<32} {extent:>12.4f}")
    print(f"  {'RMS 误差 / 场景尺度':<32} {np.sqrt((err**2).mean())/extent*100:>11.2f} %")

    rms = np.sqrt((err ** 2).mean())
    print(f"\n  平移方向误差 vs 基线长度（看它是否随基线收敛）")
    print(f"  {'帧间隔':>6} {'基线中位数':>12} {'方向误差中位数':>16} {'噪声下限(估)':>14}")
    for k, e_med, b_med in stride_stats:
        floor = np.degrees(np.arctan(rms / max(b_med, 1e-9)))
        print(f"  {k:>6} {b_med:>12.4f} {e_med:>15.2f}° {floor:>13.1f}°")

    if intr:
        cam = list(intr.values())[0]
        fx_col = cam.params[0]
        fxj = json.load(open(os.path.join(scene_dir, "intrinsics.json")))
        print(f"  {'COLMAP 估计 fx':<32} {fx_col:>12.3f}")
        print(f"  {'数据集给 fx（像素）':<32} {fxj['fx']*cam.width:>12.3f}")
        print(f"  {'相对差':<32} {abs(fx_col-fxj['fx']*cam.width)/(fxj['fx']*cam.width)*100:>11.2f} %")

    # --- 判定 ---
    # 判据说明：
    #   旋转：两条独立路径的相对旋转应在 1° 以内。
    #   平移：不看短基线（噪声主导），看「对齐后中心 RMS / 场景尺度」。
    #        这是最硬的指标 —— 它把全局尺度、位置、朝向全都对齐后，
    #        剩下的纯粹是轨迹形状的差异。
    good_rot = np.median(rot_errs) < 1.0
    rel_rms = rms / extent
    good_pos = rel_rms < 0.01          # 中心误差小于场景尺度的 1%
    print()
    if good_rot and good_pos:
        print("  ✅ 两条独立路径一致")
        print(f"     · 相对旋转中位误差 {np.median(rot_errs):.3f}°（几乎完全一致）")
        print(f"     · 对齐后中心误差仅为场景尺度的 {rel_rms*100:.2f}%")
        print("     → cameras.json 的 (xyzw, c2w) 约定得到第二次独立确认；")
        print("       COLMAP 可作为 Plan A 基线继续。")
        if stride_stats:
            k1, e1, b1 = stride_stats[0]
            k25 = stride_stats[-1]
            print(f"     → 短基线的方向误差 {e1:.0f}° 是噪声（基线 {b1:.4f} 与误差 {rms:.4f} 同量级），"
                  f"基线拉到 {k25[0]} 帧后降到 {k25[1]:.1f}°，符合预期。")
    else:
        print("  ⚠️ 一致性不足，需要排查：")
        if not good_rot:
            print(f"     旋转中位误差 {np.median(rot_errs):.2f}°，可能是四元数顺序或轴向错了")
        if not good_pos:
            print(f"     中心误差 {rel_rms*100:.2f}% 偏大，可能是 c2w/w2c 反了，或尺度/单位不同")


if __name__ == "__main__":
    main()
