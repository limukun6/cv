"""
46_ate.py — 计算位姿精度（ATE）：把我们估的位姿和数据集真值比。

为什么 Part 1 必须要这个
---------------------
原文要求对比四个维度，其中"pose accuracy"是其中之一。渲染质量（PSNR）只能
间接反映位姿好坏——位姿差会被 3DGS 的过拟合部分掩盖（我们实测 dl3dv 上
train PSNR 和 test PSNR 只差 2 dB，说明模型在硬拟合）。所以必须直接量位姿误差。

指标
----
1. **旋转误差（度）**  —— 每帧 c2w 旋转与真值的夹角，取中位数
2. **平移误差（%）**   —— Umeyama 对齐后，相机中心误差的 RMS，除以场景包围盒对角
3. **ATE RMSE**       —— 对齐后每帧中心位移的均方根（以"占对角百分比"表示，尺度无关）

为什么要 Umeyama 对齐
--------------------
SfM / 前馈方法重建出来的世界坐标系是任意的：原点、朝向、**尺度**都不固定。
直接减两个中心点没有意义（我们早期就踩过这个坑，看到恒定 26.7° 的假误差，
原因是没有对齐就比方向）。Umeyama 求最优的 (s, R, t) 把估计映射到真值坐标系，
之后剩下的才是真正的误差。

约定（已在本项目验证过）
----------------------
- 数据集 `cameras.json`：quat 顺序 **xyzw**，方向是 **c2w**，`cam_trans` 是相机中心
- COLMAP：`images.bin` 存的是 **w2c**，相机中心 = −Rᵀt

用法:
    python scripts/46_ate.py --scene dl3dv-2 --model data/dl3dv-2/colmap_n50/sparse/0
    python scripts/46_ate.py --scene dl3dv-2 --model data/dl3dv-2/vggt_n50/sparse/0 --json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------- 读模型

def load_colmap_model(model_dir):
    """返回 {image_name: (R_w2c, tvec)}，即 COLMAP 的 world→camera。"""
    sys.path.insert(0, os.path.join(ROOT, "third_party", "gaussian-splatting"))
    from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat
    ex = read_extrinsics_binary(os.path.join(model_dir, "images.bin"))
    return {v.name: (qvec2rotmat(v.qvec), np.array(v.tvec, dtype=float))
            for v in ex.values()}


def cam_centers(extr):
    """c2w 的相机中心 = −Rᵀ t。"""
    return {k: (-R.T @ t) for k, (R, t) in extr.items()}


def load_reference(scene_dir):
    """读数据集真值位姿。

    支持两种来源：
      1. cameras.json（DL3DV / Re10k 原生格式，已验证 quat=xyzw、方向=c2w）
      2. gt/ 目录下的 COLMAP 模型（Waymo 用）
    """
    cj = os.path.join(scene_dir, "cameras.json")
    if os.path.isfile(cj):
        with open(cj, encoding="utf-8-sig") as f:
            data = json.load(f)
        out = {}
        for c in data:
            # 实际字段名是 cam_quat / cam_trans / image_name（不是 rotation/img_name）
            q = np.array(c["cam_quat"], dtype=float)          # xyzw
            x, y, z, w = q
            # quat(xyzw) → R
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            name = os.path.basename(c["image_name"])
            out[name] = (R, np.array(c["cam_trans"], dtype=float))   # c2w 的 R 和相机中心
        return out, "cameras.json"

    for cand in ("gt", "gt_old"):
        p = os.path.join(scene_dir, cand, "sparse", "0")
        if os.path.isdir(p):
            extr = load_colmap_model(p)
            return {k: (R.T, -R.T @ t) for k, (R, t) in extr.items()}, f"{cand}/sparse/0"
    raise SystemExit(f"{scene_dir} 下既没有 cameras.json 也没有 gt/")


# ---------------------------------------------------------------- 对齐与评估

def umeyama(X, Y):
    """求 s, R, t 使 Y ≈ s·R·X + t。"""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    C = Yc.T @ Xc / len(X)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / Xc.var(0).sum()
    return s, R, my - s * R @ mx


def rotation_error_deg(R_est_c2w, R_gt_c2w):
    """两帧 c2w 旋转的夹角。注意必须比较 **相机系内** 的相对量，
    或者先把估计的旋转用全局旋转对齐，否则会看到恒定的假误差。"""
    dR = R_est_c2w @ R_gt_c2w.T
    c = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def align_rotation(R_est_list, R_gt_list):
    """求全局旋转 R_g，使 R_g·R_est ≈ R_gt（旋转平均，SVD 解）。

    ★ 为什么不能像平移那样用 Umeyama 一起解：
      Umeyama 用**相机中心**的协方差估计旋转。当轨迹接近平面时
      （RealEstate10K 这类前视相机小位移数据，我们的实测尺度只有 0.33），
      中心点几乎共面 → 绕平面法向的旋转估不出来 → 旋转误差被虚高。
      我们实测 re10k 用中心点对齐得到 1.53°，而真实值只有 0.07°。
      所以旋转必须单独用旋转本身来平均。
    """
    M = sum(Rg @ Re.T for Re, Rg in zip(R_est_list, R_gt_list))
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def evaluate(model_dir, scene_dir, verbose=True):
    est = load_colmap_model(model_dir)
    est_c = cam_centers(est)
    ref, ref_src = load_reference(scene_dir)

    common = sorted(set(est_c) & set(ref))
    if len(common) < 3:
        raise SystemExit(f"共同帧太少（{len(common)}），无法评估")

    # ---- 平移：标准 Sim(3)/Umeyama 对齐（ATE 的标准定义）
    X = np.array([est_c[k] for k in common])
    Y = np.array([ref[k][1] for k in common])
    s, R_um, t_um = umeyama(X, Y)
    X_al = (s * (R_um @ X.T)).T + t_um

    err = np.linalg.norm(X_al - Y, axis=1)
    diag = np.linalg.norm(Y.max(0) - Y.min(0))
    ate_rmse = float(np.sqrt((err ** 2).mean()))
    ate_pct = 100.0 * ate_rmse / diag

    # ---- 旋转：用旋转平均单独求全局旋转（与上面的 R_um 是两个独立的最优解）
    #   两个指标各自用最适合自己的对齐，这是 standard practice；
    #   混用同一个变换反而会把两个数都搞错（实测 ATE 会从 0.29% 虚高到 0.81%）。
    R_g = align_rotation([est[k][0].T for k in common], [ref[k][0] for k in common])
    rerrs = np.array([rotation_error_deg(R_g @ est[k][0].T, ref[k][0]) for k in common])
    rot_align_delta = rotation_error_deg(R_g, R_um)   # 两种全局旋转相差多少

    # ---- 附加：相邻帧相对旋转误差（完全不需要对齐，最纯粹的位姿一致性指标）
    rel = []
    for a, b in zip(common[:-1], common[1:]):
        rel_gt = ref[b][0].T @ ref[a][0]
        rel_est = (est[b][0].T).T @ (est[a][0].T)
        d = rel_est @ rel_gt.T
        rel.append(np.degrees(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))))
    rel = np.array(rel)

    out = dict(
        model=os.path.relpath(model_dir, ROOT),
        scene=os.path.basename(scene_dir.rstrip("/")),
        reference=ref_src,
        n_frames=len(common),
        scale=float(s),
        rot_median_deg=float(np.median(rerrs)),
        rot_mean_deg=float(rerrs.mean()),
        rot_p95_deg=float(np.percentile(rerrs, 95)),
        rel_rot_median_deg=float(np.median(rel)),
        rot_align_delta_deg=float(rot_align_delta),
        trans_rms_pct=float(100 * err.std() / diag),
        ate_rmse=ate_rmse,
        ate_rmse_pct=ate_pct,
        scene_diag=float(diag),
    )

    if verbose:
        print(f"模型      : {out['model']}")
        print(f"真值来源  : {ref_src}   共同帧 {len(common)}")
        print(f"对齐尺度  : {s:.4f}")
        print(f"旋转误差  : 中位 {out['rot_median_deg']:.3f}°  均值 {out['rot_mean_deg']:.3f}°  "
              f"p95 {out['rot_p95_deg']:.3f}°")
        print(f"相邻帧相对旋转误差（无需对齐）: 中位 {out['rel_rot_median_deg']:.4f}°")
        print(f"两种全局旋转的差异: {out['rot_align_delta_deg']:.3f}°"
              f"（差异大 → 轨迹接近退化，旋转/平移两个指标要对齐各自的自由度）")
        print(f"ATE RMSE  : {ate_rmse:.4f}（= 场景对角的 {ate_pct:.3f}%）")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="data/ 下的场景名")
    ap.add_argument("--model", required=True, help="COLMAP 模型目录（含 images.bin）")
    ap.add_argument("--data", default="data")
    ap.add_argument("--json", action="store_true", help="输出 JSON，方便写进表格")
    args = ap.parse_args()

    scene_dir = os.path.join(ROOT, args.data, args.scene)
    model_dir = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    out = evaluate(model_dir, scene_dir, verbose=not args.json)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
