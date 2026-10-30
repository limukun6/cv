"""
27_waymo_depth_model_test.py — 判定 depth/ 里的值到底是"线性深度"还是"视差"。

为什么怀疑是视差
--------------
证据 1：同一匹配点在帧 0 和帧 20 里，三维距离差 2 倍（723 个匹配筛到只剩 5 个）
证据 2：帧 0 vs 帧 2 的深度比 0.718，但 log 离散度只有 0.0093（极紧）
        → 两帧几乎只差一个**全局系数**，真实几何运动不可能这么均匀
证据 3：各帧的 mean 从 8743 变到 27650（3 倍），所以不是逐帧归一化

如果存的是视差 v ∝ 1/z，那么：
    · 相邻帧 1/z 近似等比 → 比值紧（证据 2 ✓）
    · 跨大基线时 1/z 非线性变化 → 离散度增大（实测 0.14 ✓）
    · 用"线性深度"解码 → 三维距离系统性差 2 倍（证据 1 ✓）

判定方法
--------
对每种解码方式，做 3D-3D Umeyama（带 RANSAC 式剔除外点），看：
    · inlier 比例   —— 几何一致的点占比
    · 残差 RMS      —— 配准精度
    · 相对旋转与 gt 的夹角
正确解码方式应该：inlier 高、RMS 小、旋转和 gt 接近。

用法:
    python scripts/27_waymo_depth_model_test.py --data data --scene waymo-405841 --stride 20
"""
import argparse
import os
import re

import cv2
import numpy as np


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


def rot_angle(Ra, Rb):
    dR = Ra @ Rb.T
    return float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))


def robust_umeyama(X0, X1, iters=30, thresh_k=2.0):
    """简单 RANSAC 式迭代：拟合 → 算残差 → 取内点 → 重拟合。"""
    idx = np.arange(len(X0))
    best = None
    for _ in range(iters):
        s, R, t = umeyama(X0[idx], X1[idx], with_scale=True)
        P = (s * (R @ X0.T)).T + t
        r = np.linalg.norm(P - X1, axis=1)          # 对全部点算残差
        sig = np.median(r[idx])
        new_idx = np.where(r < thresh_k * max(sig, 1e-6))[0]
        if len(new_idx) < 50:
            break
        if best is not None and len(new_idx) == len(idx):
            break
        idx = new_idx
        best = (s, R, t, len(idx), float(np.sqrt((r[idx] ** 2).mean())))
    if best is None:
        return None
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="waymo-405841")
    ap.add_argument("--stride", type=int, default=20)
    args = ap.parse_args()

    sd = os.path.join(args.data, args.scene)
    rgb_dir, dep_dir = os.path.join(sd, "rgb"), os.path.join(sd, "depth")
    names = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    i, j = 0, min(args.stride, len(names) - 1)

    txt = open(os.path.join(sd, "calib", "000000.txt"), encoding="utf-8").read()
    g = lambda k: float(re.search(rf"\b{k}\s*:\s*(-?[\d.eE+]+)", txt).group(1))
    fx, fy, cx, cy = g("fx"), g("fy"), g("cx"), g("cy")

    im0 = cv2.imread(os.path.join(rgb_dir, names[i]))
    im1 = cv2.imread(os.path.join(rgb_dir, names[j]))
    raw0 = cv2.imread(os.path.join(dep_dir, names[i]), cv2.IMREAD_UNCHANGED).astype(np.float64)
    raw1 = cv2.imread(os.path.join(dep_dir, names[j]), cv2.IMREAD_UNCHANGED).astype(np.float64)

    sift = cv2.SIFT_create(nfeatures=4000)
    k0, d0 = sift.detectAndCompute(cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY), None)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [m for pair in bf.knnMatch(d0, d1, k=2) if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]
    p0 = np.float32([k0[m.queryIdx].pt for m in good])
    p1 = np.float32([k1[m.trainIdx].pt for m in good])

    def samp(d, p):
        H, W = d.shape
        u = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1)
        v = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
        return d[v, u]

    z0r, z1r = samp(raw0, p0), samp(raw1, p1)
    ok = (z0r > 0) & (z1r > 0)
    p0, p1, z0r, z1r = p0[ok], p1[ok], z0r[ok], z1r[ok]
    print(f"匹配点（两帧深度均有效）: {len(p0)}")

    # gt 的相对旋转（c2w 解释）
    M0 = np.loadtxt(os.path.join(sd, "gt", names[i].replace(".png", ".txt")))
    M1 = np.loadtxt(os.path.join(sd, "gt", names[j].replace(".png", ".txt")))
    Rrel_gt = M1[:3, :3].T @ M0[:3, :3]

    def unproject(p, z):
        return np.stack([(p[:, 0] - cx) / fx * z, (p[:, 1] - cy) / fy * z, z], 1)

    models = {
        "线性 mm→m  (z = v*0.001)": lambda v: v * 0.001,
        "线性 m     (z = v)":       lambda v: v,
        "视差 A=65535 (z=65535/v)": lambda v: 65535.0 / np.maximum(v, 1),
        "视差 A=100000":            lambda v: 100000.0 / np.maximum(v, 1),
        "视差 A=200000":            lambda v: 200000.0 / np.maximum(v, 1),
    }

    print(f"\n{'解码假设':<30} {'内点':>6} {'内点率':>7} {'尺度s':>7} {'RMS(m)':>8} {'旋转vs gt':>10}")
    print("-" * 82)
    best = None
    for name, fn in models.items():
        Z0, Z1 = fn(z0r), fn(z1r)
        X0, X1 = unproject(p0, Z0), unproject(p1, Z1)
        r = robust_umeyama(X0, X1)
        if r is None:
            print(f"{name:<30} {'失败':>6}")
            continue
        s, R, t, n, rms = r
        ang = rot_angle(R, Rrel_gt)
        print(f"{name:<30} {n:>6} {n/len(p0):>6.1%} {s:>7.3f} {rms:>8.3f} {ang:>9.2f}°")
        score = (1 - n / len(p0)) + rms + ang / 90.0
        if best is None or score < best[1]:
            best = (name, score, n / len(p0), rms, ang, s)

    if best:
        print(f"\n最符合数据的解码方式: **{best[0]}**")
        print(f"  内点率 {best[2]:.1%}，RMS {best[3]:.3f} m，旋转与 gt 差 {best[4]:.2f}°，尺度 {best[5]:.3f}")
        if best[4] < 10 and best[2] > 0.5:
            print("  → 该解码方式下，gt 的旋转与数据一致。位姿可用。")
        else:
            print("  → 即便最优解码，旋转/内点率仍不理想，说明位姿本身可能也有问题。")


if __name__ == "__main__":
    main()
