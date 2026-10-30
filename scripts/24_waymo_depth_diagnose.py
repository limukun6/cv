"""
24_waymo_depth_diagnose.py — 不解位姿，先审深度图本身。

之前的推理线（重要）
------------------
现象：无论用哪种位姿假设（gt 直接当 c2w / 复合 Tr_velo_to_cam / 取逆），
      深度比值 z1/d1 都稳定在 0.49 左右，离散度 ~0.08。

关键推理：**这个比值对深度的整体缩放是不变的**（z 和 d 同时按比例变）。
         只有平移项不随缩放变：ratio = (R·ray·d0)_z/d1 + t_z/(k·d1)。
         当平移远小于深度时（3.4 m vs 10000 单位），平移项可忽略，
         于是 ratio ≈ d0/d1 —— **即两帧在对应像素上的深度值之比**。

比值 0.49 意味着 d0 ≈ 0.49·d1。帧间运动 2 秒，静止场景的深度变化应该只有几个
百分点，绝不可能差 2 倍。**所以问题不在位姿，在深度图本身**（编码/缩放/配对）。

本脚本按顺序验三件事：
  1. 每帧深度图的统计量 → 如果每帧的 max 都不一样，说明做了"逐帧归一化"
  2. 同一像素上相邻帧的深度比 → 如果稳定在 0.49，确认是全局尺度问题而非几何
  3. 把 rgb 和 depth 并排存成图 → 直接看深度图是不是真的深度

用法:
    python scripts/24_waymo_depth_diagnose.py --data data --scene waymo-405841
"""
import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="waymo-405841")
    ap.add_argument("--out", default="docs/figures/waymo_depth_check.png")
    args = ap.parse_args()

    sd = os.path.join(args.data, args.scene)
    rgb_dir, dep_dir = os.path.join(sd, "rgb"), os.path.join(sd, "depth")
    names = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    print(f"共 {len(names)} 帧\n")

    # ---------- 1. 每帧深度图统计 ----------
    print("=" * 78)
    print("1. 每帧深度图统计（如果 max 每帧都不同 → 逐帧归一化）")
    print("=" * 78)
    print(f"{'帧':>6} {'min':>7} {'p50':>8} {'p95':>8} {'max':>8} {'mean':>9} {'非零占比':>9}")
    rows = []
    idxs = [0, 1, 5, 20, 50, 100, 150, 198]
    for i in idxs:
        if i >= len(names):
            continue
        d = cv2.imread(os.path.join(dep_dir, names[i]), cv2.IMREAD_UNCHANGED)
        nz = d[d > 0]
        rows.append((i, nz.min(), np.percentile(nz, 50), np.percentile(nz, 95),
                     nz.max(), nz.mean(), (d > 0).mean()))
        print(f"{i:>6} {nz.min():>7.0f} {np.percentile(nz,50):>8.0f} "
              f"{np.percentile(nz,95):>8.0f} {nz.max():>8.0f} {nz.mean():>9.0f} "
              f"{(d>0).mean():>8.1%}")

    maxes = [r[4] for r in rows]
    if max(maxes) / max(min(maxes), 1) > 1.2:
        print(f"\n  ⚠ 各帧 max 相差 {max(maxes)/max(min(maxes),1):.2f} 倍 → 可能是逐帧归一化")
    else:
        print(f"\n  ✓ 各帧 max 基本一致（{min(maxes):.0f}~{max(maxes):.0f}）→ 全局同一缩放")

    # ---------- 2. 同一像素的帧间深度比（完全不涉及位姿）----------
    print("\n" + "=" * 78)
    print("2. 同一像素上的帧间深度比（不涉及任何位姿假设）")
    print("=" * 78)
    print(f"{'帧对':>12} {'公共像素':>10} {'d_i/d_j 中位数':>16} {'log 离散度':>12}")
    d0 = cv2.imread(os.path.join(dep_dir, names[0]), cv2.IMREAD_UNCHANGED).astype(np.float64)
    for j in [1, 2, 5, 10, 20, 50, 100]:
        if j >= len(names):
            continue
        dj = cv2.imread(os.path.join(dep_dir, names[j]), cv2.IMREAD_UNCHANGED).astype(np.float64)
        m = (d0 > 0) & (dj > 0)
        if m.sum() < 1000:
            continue
        r = d0[m] / dj[m]
        med = float(np.median(r))
        spread = float(np.median(np.abs(np.log(r / med))))
        print(f"{'0 vs ' + str(j):>12} {int(m.sum()):>10} {med:>16.3f} {spread:>12.4f}")

    print("\n  判读：静止场景下，帧间深度比应接近 1.00（只随相机前进略微变大）。")
    print("        如果这个比值本身就远离 1，那问题与位姿无关，是深度图自身的问题。")

    # ---------- 3. 可视化 ----------
    print("\n" + "=" * 78)
    print("3. 目视检查：rgb vs depth")
    print("=" * 78)
    rgb = cv2.imread(os.path.join(rgb_dir, names[0]))
    dep = cv2.imread(os.path.join(dep_dir, names[0]), cv2.IMREAD_UNCHANGED).astype(np.float32)

    # 深度图转伪彩色（近=暖色，远=冷色），无效像素标黑
    valid = dep > 0
    dn = np.zeros_like(dep)
    if valid.any():
        lo, hi = np.percentile(dep[valid], 1), np.percentile(dep[valid], 99)
        dn = np.clip((dep - lo) / max(hi - lo, 1e-6), 0, 1)
    dep_vis = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dep_vis[~valid] = (0, 0, 0)

    # 无效像素掩码图
    mask_vis = np.zeros((*valid.shape, 3), np.uint8)
    mask_vis[valid] = (255, 255, 255)

    h = 480
    def rs(img):
        return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))

    panel = np.hstack([rs(rgb), rs(dep_vis), rs(mask_vis)])
    cv2.putText(panel, "RGB", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    cv2.putText(panel, "DEPTH (near=red/far=blue)", (rs(rgb).shape[1] + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(panel, "VALID MASK", (rs(rgb).shape[1] * 2 + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, panel)
    print(f"写出 {args.out}  ({panel.shape[1]}x{panel.shape[0]})")


if __name__ == "__main__":
    main()
