"""
20_verify_poses.py — 用几何一致性的硬证据来判定位姿文件的坐标约定。

为什么必须做这件事：
    三个场景的位姿格式完全不同，而且相机 json 里只写了 cam_quat / cam_trans，
    没有任何字段说明「这是 cam->world 还是 world->cam」「四元数是 wxyz 还是 xyzw」。
    猜错的后果是：整份数据的重建全废，而且症状是「PSNR 很低」这种极难归因的表现。

判定方法（不靠猜，靠几何）：
    A. 有 features 的场景（DL3DV / Re10k）：
       两帧匹配特征点 → 按每种假设做三角化 → 算重投影误差 + 是否在相机前方。
       正确答案的误差应该是「几个像素」，错误答案会是「几百像素」或全部在相机后方。

    B. 有深度的场景（Waymo）：
       把 A 帧的深度反投影成 3D，用假设的相对位姿变换到 B 帧，投影回去和 B 的深度比较。
       正确答案的相对深度误差小；错误答案误差巨大。

用法:
    python scripts/20_verify_poses.py --all
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------- 工具

def quat_to_R(q, order):
    """order: 'wxyz' 或 'xyzw' -> 3x3 旋转矩阵"""
    q = np.asarray(q, dtype=np.float64)
    if order == "wxyz":
        w, x, y, z = q
        xyz, w_ = np.array([x, y, z]), w
    else:
        x, y, z, w = q
        xyz, w_ = np.array([x, y, z]), w
    return R.from_quat(np.r_[xyz, w_]).as_matrix()


def build_w2c(quat, trans, order, direction):
    """
    返回 (R_w2c, t_w2c, C)
      direction='c2w'  : quat/trans 描述 cam->world，C 就是 trans
      direction='w2c'  : quat/trans 描述 world->cam，t 就是 trans，C = -R^T t
    """
    Rq = quat_to_R(quat, order)
    if direction == "c2w":
        R_w2c = Rq.T
        C = np.asarray(trans, dtype=np.float64)
    else:
        R_w2c = Rq
        t = np.asarray(trans, dtype=np.float64)
        C = -R_w2c.T @ t
    t_w2c = -R_w2c @ C
    return R_w2c, t_w2c, C


def reproj_error(K, R1, t1, R2, t2, pts1, pts2):
    """三角化 + 重投影误差（像素），返回 (median, in_front_ratio)"""
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)          # 4xN
    X /= X[3]
    X3 = X[:3].T

    # 两个相机前方检验
    z1 = (R1 @ X3.T + t1.reshape(3, 1))[2]
    z2 = (R2 @ X3.T + t2.reshape(3, 1))[2]
    in_front = float(np.mean((z1 > 0) & (z2 > 0)))

    def proj(Rm, tm):
        p = (K @ (Rm @ X3.T + tm.reshape(3, 1)))
        return (p[:2] / p[2]).T

    e = np.concatenate([np.linalg.norm(proj(R1, t1) - pts1, axis=1),
                        np.linalg.norm(proj(R2, t2) - pts2, axis=1)])
    return float(np.median(e)), in_front


# ---------------------------------------------------------------- A. cameras.json

def match_features(img1, img2, max_pts=2000):
    sift = cv2.SIFT_create(nfeatures=max_pts)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return None, None
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 30:
        return None, None
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    return p1, p2


def verify_cameras_json(scene_dir, stride=10):
    cj = os.path.join(scene_dir, "cameras.json")
    if not os.path.isfile(cj):
        print(f"  {scene_dir}: 没有 cameras.json，跳过")
        return
    cams = json.load(open(cj, encoding="utf-8"))
    imgdir = os.path.join(scene_dir, "rgb")
    if not os.path.isdir(imgdir):
        imgdir = os.path.join(scene_dir, "images")
    imgs = sorted(glob.glob(os.path.join(imgdir, "*.png")))
    print(f"\n=== {os.path.basename(scene_dir)}: {len(cams)} 帧, {len(imgs)} 图 ===")

    i, j = 0, min(stride, len(imgs) - 1)
    a = cv2.imread(imgs[i])
    b = cv2.imread(imgs[j])
    if a is None or b is None:
        print("  读图失败")
        return
    H, W = a.shape[:2]
    print(f"  图像 {W}x{H}, 测试帧对 ({i}, {j})")

    p1, p2 = match_features(a, b)
    if p1 is None:
        print("  匹配点太少，跳过")
        return
    print(f"  匹配点: {len(p1)}")

    fx = cams[i]["fx"] * W
    fy = cams[i]["fy"] * H
    cx = cams[i]["cx"] * W
    cy = cams[i]["cy"] * H
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    print(f"  内参: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    print(f"  {'假设':<22} {'重投影误差(px)':>16} {'前方点占比':>12}")
    results = []
    for order in ("wxyz", "xyzw"):
        for direction in ("c2w", "w2c"):
            R1, t1, _ = build_w2c(cams[i]["cam_quat"], cams[i]["cam_trans"], order, direction)
            R2, t2, _ = build_w2c(cams[j]["cam_quat"], cams[j]["cam_trans"], order, direction)
            med, front = reproj_error(K, R1, t1, R2, t2, p1, p2)
            tag = f"quat={order}, {direction}"
            print(f"  {tag:<22} {med:>16.2f} {front * 100:>11.1f}%")
            results.append((med, front, tag))
    best = min(results, key=lambda r: r[0] if r[1] > 0.5 else 1e9)
    print(f"  >>> 判定: {best[2]}  (误差 {best[0]:.2f} px, 前方 {best[1] * 100:.0f}%)")


# ---------------------------------------------------------------- B. Waymo depth

def verify_waymo(scene_dir, stride=5):
    rgb_dir = os.path.join(scene_dir, "rgb")
    dep_dir = os.path.join(scene_dir, "depth")
    calib_f = os.path.join(scene_dir, "calib", "000000.txt")
    if not os.path.isdir(dep_dir):
        print(f"  {scene_dir}: 没有 depth，跳过")
        return

    # 读 calib（注意：fx fy cx cy 全写在同一行，不能用 split(':') 逐行解析）
    import re
    txt = open(calib_f, encoding="utf-8").read()

    def grab(name):
        m = re.search(rf"\b{name}\s*:\s*(-?[\d.eE+]+)", txt)
        if not m:
            sys.exit(f"calib 里找不到 {name}: {calib_f}")
        return float(m.group(1))

    fx, fy, cx, cy = grab("fx"), grab("fy"), grab("cx"), grab("cy")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    imgs = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    print(f"\n=== {os.path.basename(scene_dir)}: {len(imgs)} 帧, 有深度 ===")
    print(f"  calib: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    dep0 = cv2.imread(os.path.join(dep_dir, os.path.basename(imgs[0])), cv2.IMREAD_UNCHANGED)
    print(f"  深度图 dtype={dep0.dtype} 形状={dep0.shape}")
    nz = dep0[dep0 > 0]
    if nz.size:
        print(f"  深度取值: min={nz.min()} p50={np.percentile(nz,50):.0f} "
              f"p95={np.percentile(nz,95):.0f} max={nz.max()}")

    # ★ 单位归一：位姿平移以「米」为单位（Waymo 惯例）。
    #   深度若是 uint16 且中位数在数千量级，说明单位是毫米，必须换算成米，
    #   否则平移项会被缩放到可忽略，导致 cam2world / world2cam 两种假设都「看似一致」。
    depth_scale = 1.0
    if nz.size and np.percentile(nz, 50) > 1000:
        depth_scale = 0.001
        print(f"  → 判定深度单位为毫米，统一换算为米 (×{depth_scale})")
    else:
        print("  → 深度单位视为米")

    for sub in ("gt", "gt_old"):
        f0 = os.path.join(scene_dir, sub, "000000.txt")
        f1 = os.path.join(scene_dir, sub, os.path.basename(imgs[stride]).replace(".png", ".txt"))
        if not os.path.isfile(f1):
            print(f"  跳过 {sub}: 缺文件")
            continue
        M0 = np.loadtxt(f0)
        M1 = np.loadtxt(f1)
        print(f"\n  -- {sub} --")
        print(f"     平移0={M0[:3,3]}")
        print(f"     平移1={M1[:3,3]}   相对位移范数={np.linalg.norm(M1[:3,3]-M0[:3,3]):.3f}")

        dep1 = cv2.imread(os.path.join(dep_dir, os.path.basename(imgs[stride])), cv2.IMREAD_UNCHANGED)
        img1 = cv2.imread(imgs[stride])
        H, W = dep0.shape

        # 假设 M 是 c2w 或 w2c
        for direction in ("c2w", "w2c"):
            if direction == "c2w":
                C0, Rw0 = M0[:3, 3], M0[:3, :3]
                C1, Rw1 = M1[:3, 3], M1[:3, :3]
            else:
                Rw0, t0 = M0[:3, :3], M0[:3, 3]
                Rw1, t1v = M1[:3, :3], M1[:3, 3]
                C0, C1 = -Rw0.T @ t0, -Rw1.T @ t1v
                Rw0, Rw1 = Rw0.T, Rw1.T
            # 相对位姿: cam0 -> cam1
            Rrel = Rw1.T @ Rw0
            trel = Rw1.T @ (C0 - C1)

            # 反投影 cam0 的深度（已换算为米）
            step = 8
            ys, xs = np.mgrid[0:H:step, 0:W:step]
            d0 = dep0[ys, xs].astype(np.float64) * depth_scale
            valid = d0 > 0
            xs, ys, d0 = xs[valid], ys[valid], d0[valid]
            ray = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)], axis=0)
            Xc0 = ray * d0  # 米

            Xc1 = Rrel @ Xc0 + trel.reshape(3, 1)
            z1 = Xc1[2]
            ok = z1 > 1e-6
            u = (fx * Xc1[0, ok] / z1[ok] + cx).astype(np.int32)
            v = (fy * Xc1[1, ok] / z1[ok] + cy).astype(np.int32)
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v, z1ok = u[inb], v[inb], z1[ok][inb]
            if u.size < 100:
                print(f"     {direction:<4} 有效投影过少({u.size})")
                continue
            d1 = dep1[v, u].astype(np.float64) * depth_scale
            m = d1 > 0
            if m.sum() < 100:
                print(f"     {direction:<4} 落点无深度({m.sum()})")
                continue
            ratio = z1ok[m] / d1[m]
            med = float(np.median(ratio))
            spread = float(np.median(np.abs(np.log(ratio / (med + 1e-9)))))
            # 正确答案应同时满足：比值≈1 且离散度小
            score = abs(np.log(med + 1e-9)) + spread
            print(f"     {direction:<4} 落点 {int(m.sum()):6d}  深度比中位数={med:8.3f}  "
                  f"log 离散度={spread:.4f}  综合得分={score:.4f}"
                  f"  {'★ 一致' if score < 0.05 else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    verify_cameras_json(os.path.join(args.data, "dl3dv-2"))
    verify_cameras_json(os.path.join(args.data, "re10k-1"))
    # Waymo 用大一点的步长，让基线足够长，才能区分 c2w / w2c
    verify_waymo(os.path.join(args.data, "waymo-405841"), stride=20)


if __name__ == "__main__":
    main()
