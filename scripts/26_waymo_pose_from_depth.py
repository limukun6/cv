"""
26_waymo_pose_from_depth.py — 用两帧的深度 + 特征匹配，直接"反推"真实相对位姿，
                              再和 gt 给的位姿对比。这是不依赖任何约定假设的判决性实验。

思路
----
1. 在两帧 rgb 上做 SIFT 匹配（比率测试筛选）
2. 只保留两帧深度都有效的匹配点
3. 各自反投影成三维点 X0、X1（都在各自相机坐标系下）
4. 用 Umeyama 求 (s, R, t) 使 s·R·X0 + t ≈ X1
   —— 这里的 s 是"两组深度之间的相对尺度"，正确解码时应该 ≈ 1
   —— R 是真值相对旋转（cam0 → cam1），可以直接和 gt 比
5. 输出：
      · 尺度 s        → 判断深度单位是否有系数错
      · 旋转夹角      → 判断 gt 的旋转约定对不对
      · 平移方向夹角  → 判断 c2w / w2c
      · 平移模长      → 直接读出真实基线（米），用来反推 gt 平移的单位

这是最强的判据：它不猜任何约定，答案从几何里自己长出来。

用法:
    python scripts/26_waymo_pose_from_depth.py --data data --scene waymo-405841 --stride 20
"""
import argparse
import os
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, d, Vt = np.linalg.svd(cov)
    Sm = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sm[2, 2] = -1
    R = U @ Sm @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = (d * np.diag(Sm)).sum() / var_s if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def angle_between_rot(Ra, Rb):
    dR = Ra @ Rb.T
    c = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def angle_between_vec(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="waymo-405841")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--max-pts", type=int, default=4000)
    args = ap.parse_args()

    sd = os.path.join(args.data, args.scene)
    rgb_dir, dep_dir = os.path.join(sd, "rgb"), os.path.join(sd, "depth")
    names = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    i, j = 0, min(args.stride, len(names) - 1)

    txt = open(os.path.join(sd, "calib", "000000.txt"), encoding="utf-8").read()
    g = lambda k: float(re.search(rf"\b{k}\s*:\s*(-?[\d.eE+]+)", txt).group(1))
    fx, fy, cx, cy = g("fx"), g("fy"), g("cx"), g("cy")
    print(f"内参 fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    im0 = cv2.imread(os.path.join(rgb_dir, names[i]))
    im1 = cv2.imread(os.path.join(rgb_dir, names[j]))
    d0 = cv2.imread(os.path.join(dep_dir, names[i]), cv2.IMREAD_UNCHANGED).astype(np.float64)
    d1 = cv2.imread(os.path.join(dep_dir, names[j]), cv2.IMREAD_UNCHANGED).astype(np.float64)

    # --- 特征匹配 ---
    sift = cv2.SIFT_create(nfeatures=args.max_pts)
    g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    k0, des0 = sift.detectAndCompute(g0, None)
    k1, des1 = sift.detectAndCompute(g1, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des0, des1, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    print(f"SIFT 匹配: {len(good)} 对（比率测试后）")

    p0 = np.float32([k0[m.queryIdx].pt for m in good])
    p1 = np.float32([k1[m.trainIdx].pt for m in good])

    def sample(d, p):
        H, W = d.shape
        u = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
        return d[v, u], u, v

    z0, u0, v0 = sample(d0, p0)
    z1, u1, v1 = sample(d1, p1)
    ok = (z0 > 0) & (z1 > 0)
    print(f"两帧深度都有效的匹配: {ok.sum()}")

    z0, z1 = z0[ok], z1[ok]
    p0, p1 = p0[ok], p1[ok]

    # 反投影（深度按 mm→m）
    X0 = np.stack([(p0[:, 0] - cx) / fx * z0, (p0[:, 1] - cy) / fy * z0, z0], 1) * 0.001
    X1 = np.stack([(p1[:, 0] - cx) / fx * z1, (p1[:, 1] - cy) / fy * z1, z1], 1) * 0.001

    # --- 粗剔除：深度差太大的匹配（错配或遮挡）---
    keep = np.abs(np.linalg.norm(X0, axis=1) - np.linalg.norm(X1, axis=1)) < \
           0.3 * np.linalg.norm(X0, axis=1)
    X0, X1 = X0[keep], X1[keep]
    print(f"深度一致性粗筛后: {len(X0)}")

    # --- Umeyama ---
    s, R_est, t_est = umeyama(X0, X1, with_scale=True)
    P = (s * (R_est @ X0.T)).T + t_est
    rms = float(np.sqrt(((P - X1) ** 2).sum(1).mean()))
    print(f"\nUmeyama 结果: 尺度 s = {s:.4f}, 残差 RMS = {rms:.4f} m")

    # --- gt 的候选解释 ---
    M0 = np.loadtxt(os.path.join(sd, "gt", names[i].replace(".png", ".txt")))
    M1 = np.loadtxt(os.path.join(sd, "gt", names[j].replace(".png", ".txt")))
    C0, C1 = M0[:3, 3], M1[:3, 3]
    Rw0, Rw1 = M0[:3, :3], M1[:3, :3]
    # c2w 解释下的相对位姿 cam0 -> cam1
    R_rel_c2w = Rw1.T @ Rw0
    t_rel_c2w = Rw1.T @ (C0 - C1)

    print(f"\n{'='*70}\n与 gt 的对比\n{'='*70}")
    print(f"gt 相对旋转 vs 从深度估计的旋转: "
          f"夹角 {angle_between_rot(R_est, R_rel_c2w):.2f}°")
    print(f"gt 相对平移方向 vs 估计的平移: "
          f"夹角 {angle_between_vec(t_est, t_rel_c2w):.2f}°")
    print(f"\ngt 平移模长      = {np.linalg.norm(C1 - C0):.4f}")
    print(f"深度估计的平移模长 = {np.linalg.norm(t_est):.4f} m")
    if np.linalg.norm(t_est) > 1e-6:
        ratio = np.linalg.norm(C1 - C0) / np.linalg.norm(t_est)
        print(f"两者之比 (gt / 估计) = {ratio:.4f}")
        print(f"  → 如果 gt 平移单位是米，这个比值应该 ≈ 1.0")
        for unit, k in [("米", 1.0), ("分米", 10.0), ("厘米", 100.0), ("毫米", 1000.0)]:
            print(f"     假设 gt 单位 = {unit:<3} → 换算后位移 = {np.linalg.norm(C1-C0)/k*1.0:.4f} m "
                  f"({'✔ 接近估计值' if abs(np.linalg.norm(C1-C0)/k - np.linalg.norm(t_est)) < 0.5*np.linalg.norm(t_est) else ''})")

    # --- 判决 ---
    print(f"\n{'='*70}\n判决\n{'='*70}")
    dR = angle_between_rot(R_est, R_rel_c2w)
    if dR < 5:
        print(f"✅ 旋转约定正确（c2w，误差 {dR:.1f}°）")
    else:
        print(f"❌ 旋转不一致（{dR:.1f}°）→ gt 的轴/四元数约定需要改")
    if 0.8 < s < 1.25:
        print(f"✅ 深度尺度自洽（s={s:.3f}）")
    else:
        print(f"⚠ 两组深度的尺度比 s={s:.3f} 明显偏离 1")


if __name__ == "__main__":
    main()
