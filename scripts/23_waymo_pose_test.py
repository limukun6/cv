"""
23_waymo_pose_test.py — 判定 Waymo 的 gt/ 到底是「相机位姿」还是「车体位姿」。

背景
----
405841/FRONT/gt/*.txt 是 4×4 矩阵，每帧一个。
405841/FRONT/calib/*.txt 里有一个 Tr_velo_to_cam（车体/激光雷达 → 相机）的 3×4 矩阵。

Waymo 官方给的位姿通常是**车体（ego）在全局坐标下的位姿**，相机位姿需要再复合一个
固定的外参（相机相对车体的安装位置）。直接拿车体位姿当相机位姿用，会因为
「力臂」（相机装在车上，距车体原点约 2 米）引入随转角放大的误差。

判定方法（纯几何，不用 GPU）
--------------------------
把 A 帧的深度反投影成三维点，用候选位姿变换到 B 帧，投影回去和 B 的深度比较：
    · 如果位姿对：z1 / depth_B ≈ 1.0，且离散度小
    · 如果位姿错：比值偏离 1，或者离散度大（说明几何对不上）

测试 4 个假设：
    H1  gt 直接当作 相机 c2w
    H2  相机 c2w = gt @ inv(Tr_velo_to_cam)
    H3  相机 c2w = gt @ Tr_velo_to_cam
    H4  gt 当作 w2c（相机 ← 世界），直接取逆

用法:
    python scripts/23_waymo_pose_test.py --data data --scene waymo-405841 --stride 20
"""
import argparse
import os
import re

import cv2
import numpy as np


def read_calib(path):
    txt = open(path, encoding="utf-8").read()

    def grab(name):
        m = re.search(rf"\b{name}\s*:\s*(-?[\d.eE+]+)", txt)
        return float(m.group(1)) if m else None

    fx, fy, cx, cy = grab("fx"), grab("fy"), grab("cx"), grab("cy")
    m = re.search(r"Tr_velo_to_cam:\s*([-\d.eE+\s]+)", txt)
    T = None
    if m:
        v = [float(x) for x in m.group(1).split()]
        if len(v) == 12:
            T = np.eye(4)
            T[:3, :4] = np.array(v).reshape(3, 4)
    return dict(fx=fx, fy=fy, cx=cx, cy=cy, T_velo2cam=T)


def warp_consistency(dep0, dep1, K, T_cam0_w, T_cam1_w, step=8):
    """
    把 cam0 的深度反投影，按相对位姿变换到 cam1，比较预测深度与 dep1。
    返回 (落点数, 比值中位数, log 离散度)
    """
    H, W = dep0.shape
    fx, fy, cx, cy = K
    # 相对位姿 cam0 -> cam1
    T_10 = np.linalg.inv(T_cam1_w) @ T_cam0_w
    R, t = T_10[:3, :3], T_10[:3, 3]

    ys, xs = np.mgrid[0:H:step, 0:W:step]
    d0 = dep0[ys, xs].astype(np.float64)
    valid = d0 > 0
    xs, ys, d0 = xs[valid], ys[valid], d0[valid]
    if d0.size < 100:
        return 0, np.nan, np.nan

    X = np.stack([(xs - cx) / fx * d0, (ys - cy) / fy * d0, d0], axis=0)  # cam0 系，单位同深度
    X1 = R @ X + t.reshape(3, 1)
    z = X1[2]
    ok = z > 1e-6
    u = (fx * X1[0, ok] / z[ok] + cx).astype(np.int32)
    v = (fy * X1[1, ok] / z[ok] + cy).astype(np.int32)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[inb], v[inb], z[ok][inb]
    if u.size < 200:
        return int(u.size), np.nan, np.nan

    d1 = dep1[v, u].astype(np.float64)
    m = d1 > 0
    if m.sum() < 200:
        return int(m.sum()), np.nan, np.nan

    ratio = z[m] / d1[m]
    med = float(np.median(ratio))
    spread = float(np.median(np.abs(np.log(ratio / (med + 1e-12)))))
    return int(m.sum()), med, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="waymo-405841")
    ap.add_argument("--stride", type=int, default=20)
    args = ap.parse_args()

    sd = os.path.join(args.data, args.scene)
    rgb_dir, dep_dir = os.path.join(sd, "rgb"), os.path.join(sd, "depth")
    calib = read_calib(os.path.join(sd, "calib", "000000.txt"))
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    Tvc = calib["T_velo2cam"]
    print(f"calib: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"Tr_velo_to_cam 平移部分: {Tvc[:3, 3] if Tvc is not None else '缺失'}")
    print(f"  相机相对车体原点的距离: {np.linalg.norm(Tvc[:3, 3]) if Tvc is not None else 0:.3f} 米")
    print("  （这就是「力臂」——它决定了拿车体位姿当相机位姿会错多少）\n")

    names = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    i, j = 0, min(args.stride, len(names) - 1)
    n0, n1 = names[i], names[j]
    dep0 = cv2.imread(os.path.join(dep_dir, n0), cv2.IMREAD_UNCHANGED)
    dep1 = cv2.imread(os.path.join(dep_dir, n1), cv2.IMREAD_UNCHANGED)

    # 深度单位：Waymo 深度为毫米，位姿平移为米 → 统一到米
    dep0 = dep0.astype(np.float64) * 0.001
    dep1 = dep1.astype(np.float64) * 0.001

    M0 = np.loadtxt(os.path.join(sd, "gt", n0.replace(".png", ".txt")))
    M1 = np.loadtxt(os.path.join(sd, "gt", n1.replace(".png", ".txt")))
    print(f"比较帧 {n0} 与 {n1}（间隔 {j} 帧）")
    print(f"  车体位移范数 = {np.linalg.norm(M1[:3,3]-M0[:3,3]):.3f} 米\n")

    K = (fx, fy, cx, cy)

    def as_c2w(M):
        T = np.eye(4)
        T[:3, :4] = M[:3, :4]
        return T

    hypotheses = {}
    hypotheses["H1  gt 直接当相机 c2w"] = (as_c2w(M0), as_c2w(M1))
    if Tvc is not None:
        hypotheses["H2  gt @ inv(Tr_velo_to_cam)"] = (
            as_c2w(M0) @ np.linalg.inv(Tvc), as_c2w(M1) @ np.linalg.inv(Tvc))
        hypotheses["H3  gt @ Tr_velo_to_cam"] = (
            as_c2w(M0) @ Tvc, as_c2w(M1) @ Tvc)
    hypotheses["H4  gt 当 w2c（取逆）"] = (
        np.linalg.inv(as_c2w(M0)), np.linalg.inv(as_c2w(M1)))

    print(f"{'假设':<34} {'落点':>7} {'深度比中位数':>13} {'log离散度':>11}  判定")
    print("-" * 82)
    best = None
    for name, (A0, A1) in hypotheses.items():
        n, med, spread = warp_consistency(dep0, dep1, K, A0, A1)
        if not np.isfinite(med):
            print(f"{name:<34} {n:>7} {'-':>13} {'-':>11}  无法判定")
            continue
        score = abs(np.log(med)) + spread
        flag = ""
        if abs(med - 1.0) < 0.02 and spread < 0.05:
            flag = "★ 一致"
        print(f"{name:<34} {n:>7} {med:>13.3f} {spread:>11.4f}  {flag}")
        if best is None or score < best[1]:
            best = (name, score, med, spread)

    print()
    if best:
        print(f"最可能的解释: {best[0]}")
        print(f"  深度比 {best[2]:.3f}，离散度 {best[3]:.4f}")
        if abs(best[2] - 1.0) > 0.05:
            print("  ⚠ 比值仍偏离 1 —— 可能还有尺度/单位问题，或者深度图与 rgb 不同源")


if __name__ == "__main__":
    main()
