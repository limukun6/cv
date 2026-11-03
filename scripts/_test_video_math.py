"""
_test_video_math.py — 在 CPU 上验证 50_make_video.py 的轨迹数学（不碰 GPU）。

为什么只测数学
------------
渲染那一步必须用 GPU，而"造轨迹"是最容易写错、且完全可以离线验证的部分：
  · 四元数符号连续性（q 和 −q 是同一个旋转，搞错会让相机突然翻 180°）
  · slerp 是否真的走最短路径
  · Catmull-Rom 是否一阶连续（速度不突变 —— 否则视频一顿一顿）
  · 是否有 NaN / 非正交矩阵

用合成的相机序列跑，检查这三件事。全部通过再去开卡渲染。
"""
import importlib.util
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
spec = importlib.util.spec_from_file_location(
    "vid", os.path.join(ROOT, "scripts", "50_make_video.py"))
vid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vid)


def make_fake_cams(n, radius=3.0, height=1.0):
    """造一条螺旋线上的一组相机（c2w），模拟真实数据集的运动。"""
    cams = []
    for i in range(n):
        a = 2 * np.pi * i / n
        C = np.array([radius * np.cos(a), height, radius * np.sin(a)])
        # 朝向圆心（y 轴向上，OpenCV 约定 z 朝前，y 朝下 → 用 y 轴向上构造再看是否自洽）
        fwd = -C / np.linalg.norm(C)
        up = np.array([0.0, -1.0, 0.0])        # OpenCV: y 朝下
        right = np.cross(up, fwd)
        right /= np.linalg.norm(right)
        up2 = np.cross(fwd, right)
        R_c2w = np.stack([right, up2, fwd], axis=1)   # 列向量 = 相机坐标轴在世界系
        # 3DGS 的 Camera 存的是 w2c 的 R 和 T；R_w2c = R_c2wᵀ, T = −R_w2c·C
        Rw2c = R_c2w.T
        Tw2c = -Rw2c @ C
        cams.append(SimpleNamespace(R=Rw2c, T=Tw2c, FoVx=0.8, FoVy=0.8,
                                    image_width=256, image_height=256))
    return cams


def main():
    ok = True
    cams = make_fake_cams(12)
    N = 240
    path = vid.build_path(cams, N, loop=True)

    # ---- 1. 数量与形状
    assert len(path) == N, f"帧数不对: {len(path)}"
    print(f"[1] 帧数 {len(path)} ✔")

    # ---- 2. 旋转矩阵必须是正交且 det=+1
    worst_orth = 0.0
    for R, C, fx, fy in path:
        e = np.abs(R @ R.T - np.eye(3)).max()
        worst_orth = max(worst_orth, e)
        if np.linalg.det(R) < 0:
            ok = False
            print("[2] ✘ 出现了 det=-1 的反射矩阵")
    print(f"[2] 最大正交误差 {worst_orth:.2e} {'✔' if worst_orth < 1e-9 else '✘'}")
    ok &= worst_orth < 1e-9

    # ---- 3. 位置/旋转的相邻帧变化要平滑（无跳变）
    Cs = np.array([p[1] for p in path])
    dpos = np.linalg.norm(np.diff(Cs, axis=0), axis=1)
    # 相邻帧的旋转夹角
    drot = []
    for i in range(len(path) - 1):
        dR = path[i + 1][0] @ path[i][0].T
        drot.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    drot = np.array(drot)

    # 判据：最大步长不应超过中位步长的 5 倍（跳变会远超）
    r_pos = dpos.max() / max(np.median(dpos), 1e-12)
    r_rot = drot.max() / max(np.median(drot), 1e-12)
    print(f"[3] 位置步长 中位 {np.median(dpos):.5f} 最大 {dpos.max():.5f} (×{r_pos:.2f})")
    print(f"    旋转步长 中位 {np.median(drot):.4f}° 最大 {drot.max():.4f}° (×{r_rot:.2f})")
    smooth = r_pos < 5.0 and r_rot < 5.0
    print(f"    {'✔ 平滑' if smooth else '✘ 有跳变（四元数符号或样条有问题）'}")
    ok &= smooth

    # ---- 4. 四元数插值的正确性：两个已知旋转的中点
    from math import cos, sin, radians
    qa = np.array([0, 0, 0, 1.0])                     # 单位
    qb = np.array([0, 0, sin(radians(45)), cos(radians(45))])  # 绕 z 转 90°
    qm = vid.slerp_quat(qa, qb, 0.5)
    ang = np.degrees(2 * np.arccos(np.clip(abs(qm[3]), -1, 1)))
    print(f"[4] 0° 与 90° 的 slerp 中点 = {ang:.3f}° {'✔' if abs(ang - 45) < 0.01 else '✘'}")
    ok &= abs(ang - 45) < 0.01

    # ---- 5. 反向四元数（符号翻转）也必须走同一路径
    qm2 = vid.slerp_quat(qa, -qb, 0.5)
    ang2 = np.degrees(2 * np.arccos(np.clip(abs(qm2[3]), -1, 1)))
    print(f"[5] 反向四元数 -q 的中点 = {ang2:.3f}° {'✔' if abs(ang2 - 45) < 0.01 else '✘'}")
    ok &= abs(ang2 - 45) < 0.01

    # ---- 6. 首尾是否闭合（loop=True 时应接近）
    gap = np.linalg.norm(Cs[-1] - Cs[0])
    print(f"[6] 首尾位置距离 {gap:.5f}（循环播放的接缝）")

    print("\n" + ("全部通过 ✔ 可以去 GPU 上试渲染了" if ok else "有失败项 ✘"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
