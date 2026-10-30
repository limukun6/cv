"""
25_waymo_scale_sweep.py — 搜索 gt 位姿的平移单位（决定 Waymo 位姿悬案的最后一环）。

已知事实链
----------
1. 深度图本身正常（天空远、路面近，帧间视差符合车辆运动）
2. 无论用哪种位姿假设，warp 后的「深度比」都稳定在 ~0.49
3. 而**同一像素**上帧 0 vs 帧 20 的深度比是 **0.506**，离散度 0.14

第 2、3 条几乎相等 → 说明 warp 时**平移几乎没有起作用**，
点几乎还在原地。也就是说网格位移为零，等价于把位姿当成了"只有旋转"。

对纯旋转（平移被忽略）的情形：
    ratio = z1 / d1 ≈ d0 / d1    ← 正好就是"同一像素帧间比"

所以嫌疑指向：**gt 里那个 3.365 的平移，单位不是米**。
如果它是厘米，3.365 cm = 0.034 m，相对 10 m 的深度就是 0.3%，等于没有。

本脚本扫一组平移缩放系数，看哪一个能让 warp 一致性变好：
    · 正确尺度 → 深度比趋近 1.0，且离散度显著下降
    · 错误尺度 → 比值远离 1 或离散度变大

用法:
    python scripts/25_waymo_scale_sweep.py --data data --scene waymo-405841 --stride 20
"""
import argparse
import os

import cv2
import numpy as np


def warp_ratio(dep0, dep1, K, T10, step=8):
    """把 cam0 的点按 T10 变换到 cam1，返回 (落点数, 深度比中位数, log 离散度)。"""
    H, W = dep0.shape
    fx, fy, cx, cy = K
    R, t = T10[:3, :3], T10[:3, 3]
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    d0 = dep0[ys, xs]
    ok0 = d0 > 0
    xs, ys, d0 = xs[ok0], ys[ok0], d0[ok0]
    X = np.stack([(xs - cx) / fx * d0, (ys - cy) / fy * d0, d0], 0)
    X1 = R @ X + t.reshape(3, 1)
    z = X1[2]
    f = z > 1e-6
    u = (fx * X1[0, f] / z[f] + cx).astype(np.int32)
    v = (fy * X1[1, f] / z[f] + cy).astype(np.int32)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[inb], v[inb], z[f][inb]
    if u.size < 500:
        return 0, np.nan, np.nan
    d1 = dep1[v, u]
    m = d1 > 0
    if m.sum() < 500:
        return int(m.sum()), np.nan, np.nan
    r = z[m] / d1[m]
    med = float(np.median(r))
    spread = float(np.median(np.abs(np.log(r / med))))
    return int(m.sum()), med, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="waymo-405841")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.0, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0])
    args = ap.parse_args()

    sd = os.path.join(args.data, args.scene)
    rgb_dir, dep_dir = os.path.join(sd, "rgb"), os.path.join(sd, "depth")
    names = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))

    # 内参
    import re
    txt = open(os.path.join(sd, "calib", "000000.txt"), encoding="utf-8").read()
    g = lambda k: float(re.search(rf"\b{k}\s*:\s*(-?[\d.eE+]+)", txt).group(1))
    K = (g("fx"), g("fy"), g("cx"), g("cy"))

    i, j = 0, min(args.stride, len(names) - 1)
    # 深度按 mm → m 换算
    dep0 = cv2.imread(os.path.join(dep_dir, names[i]), cv2.IMREAD_UNCHANGED).astype(np.float64) * 0.001
    dep1 = cv2.imread(os.path.join(dep_dir, names[j]), cv2.IMREAD_UNCHANGED).astype(np.float64) * 0.001

    M0 = np.loadtxt(os.path.join(sd, "gt", names[i].replace(".png", ".txt")))
    M1 = np.loadtxt(os.path.join(sd, "gt", names[j].replace(".png", ".txt")))

    print(f"帧 {i} vs {j}（间隔 {j} 帧）")
    print(f"gt 原始平移: 帧0 {M0[:3,3]}  帧{j} {M1[:3,3]}")
    print(f"原始位移范数: {np.linalg.norm(M1[:3,3]-M0[:3,3]):.4f}")
    print(f"深度中位数: 帧0 {np.median(dep0[dep0>0]):.2f} m  帧{j} {np.median(dep1[dep1>0]):.2f} m\n")

    print(f"{'平移缩放':>10} {'等效位移(m)':>13} {'落点':>7} {'深度比':>8} {'离散度':>8} {'得分':>8}")
    print("-" * 66)
    best = None
    for s in args.scales:
        T0 = np.eye(4); R0 = M0[:3, :3]; C0 = M0[:3, 3]
        T1 = np.eye(4); R1 = M1[:3, :3]; C1 = M1[:3, 3]
        # 相对位姿 cam0 -> cam1，平移按 s 缩放
        Rrel = R1.T @ R0
        trel = s * (R1.T @ (C0 - C1))
        T10 = np.eye(4); T10[:3, :3] = Rrel; T10[:3, 3] = trel
        n, med, spread = warp_ratio(dep0, dep1, K, T10)
        if not np.isfinite(med):
            print(f"{s:>10.3f} {np.linalg.norm(trel):>13.2f} {n:>7} {'-':>8} {'-':>8} {'-':>8}")
            continue
        score = abs(np.log(med)) + spread
        flag = ""
        if abs(med - 1.0) < 0.05 and spread < 0.06:
            flag = "  ★"
        print(f"{s:>10.3f} {np.linalg.norm(trel):>13.2f} {n:>7} {med:>8.3f} {spread:>8.4f} {score:>8.4f}{flag}")
        if best is None or score < best[1]:
            best = (s, score, med, spread)

    if best:
        print(f"\n最优平移缩放 = {best[0]}  (深度比 {best[2]:.3f}, 离散度 {best[3]:.4f})")
        print(f"等效位移 = {np.linalg.norm(M1[:3,3]-M0[:3,3]) * best[0]:.3f} 米")
        if best[0] == 0.0:
            print("  → 最优解是「纯旋转」：说明目前的 gt 平移和深度尺度严重不匹配")


if __name__ == "__main__":
    main()
