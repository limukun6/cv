"""
50_make_video.py — 给训练好的 3DGS 模型合成一段相机轨迹视频（课程 Demo Video 加分项）。

为什么需要自己写脚本
------------------
3DGS 自带的 `render.py` 只能渲染**数据集里已有的相机**。想要视频就得有
连续的、数据集里没有的视角，所以必须自己造一条相机轨迹。

轨迹怎么造
----------
把训练相机按图像名排序，得到一条"参观路线"，然后在这条路线上
**做平滑插值**（位置用三次样条、旋转用四元数 slerp），
在路线之间插入 `--frames` 个中间视角。

为什么位置用三次样条而不是线性插值
--------------------------------
线性插值在每一段相机的交界处速度会突变，视频看起来一顿一顿的。
三次样条（Catmull-Rom 风格）能保证一阶连续，运动才平滑。

为什么旋转要用 slerp
-------------------
直接对旋转矩阵做元素插值会得到非正交矩阵，渲染会扭曲。
四元数 slerp 是单位球面上的最短路径插值，是正确的做法。
还要注意**四元数符号连续性**：q 和 −q 表示同一个旋转，但直接插值会走远路，
表现为相机在某一帧突然翻转 180°。

用法:
    # 256² 的场景，800 帧，30fps
    python scripts/50_make_video.py --model outputs/runs/<run_id> --frames 240
    # 只出图不编码
    python scripts/50_make_video.py --model <run> --frames 120 --no-encode
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GS = os.path.join(ROOT, "third_party", "gaussian-splatting")
sys.path.insert(0, GS)


def load_run(model_path):
    """读 3DGS 训练时保存的 cfg_args。

    文件内容是 `Namespace(source_path=..., sh_degree=..., ...)` 这样一行文本，
    所以 eval 时必须把 `Namespace` 喂进 globals，否则报
    `NameError: name 'Namespace' is not defined`（已踩过）。
    """
    from argparse import Namespace
    cfg = os.path.join(model_path, "cfg_args")
    with open(cfg, "r") as f:
        return eval(f.read(), {"Namespace": Namespace})


def build_scene_and_gaussians(model_path, args):
    """复刻 render.py 的初始化流程，保证和训练时一致。

    ★ 不用 3DGS 的 `get_combined_args()`：它内部读 `sys.argv`，而本脚本有
      自己的命令行参数（--model/--frames/...），会被它当成未知参数报错。
      直接从 cfg_args 构造 Namespace 更干净。
    """
    import torch
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams
    from scene import Scene
    from gaussian_renderer import GaussianModel

    ns = load_run(model_path)
    mp_parser, pp_parser = ArgumentParser(), ArgumentParser()
    mp = ModelParams(mp_parser)
    pp = PipelineParams(pp_parser)
    dataset = mp.extract(ns)
    pipeline = pp.extract(ns)

    # ★ `extract()` 只拷贝 **cfg_args 里存在的键**。老模型（或不同版本的
    #   train.py 保存的 cfg_args）可能缺少 `debug` / `antialiasing` 这类新参数，
    #   于是 GroupParams 上没有这些属性，render() 里 `pipe.debug` 就会报
    #   AttributeError: 'GroupParams' object has no attribute 'debug'（已踩过）。
    #   这里用各自类的默认值补齐。
    for obj, tmpl in ((pipeline, pp), (dataset, mp)):
        for k, v in vars(tmpl).items():
            if not hasattr(obj, k):
                setattr(obj, k, v)
                print(f"  [补齐] {k} = {v}")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")
    return scene, gaussians, dataset, pipeline, background


# ---------------------------------------------------------------- 轨迹

def slerp_quat(q0, q1, t):
    """四元数球面插值（xyzw）。处理符号翻转，避免走远路。"""
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:                       # 几乎相同，退化成线性插值再归一化
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(d, -1, 1))
    s0 = np.sin((1 - t) * theta) / np.sin(theta)
    s1 = np.sin(t * theta) / np.sin(theta)
    return s0 * q0 + s1 * q1


def rot_to_quat_xyzw(R):
    """旋转矩阵 → 四元数（xyzw 顺序，和 datasets 的 cameras.json 一致）。"""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def catmull_rom(p0, p1, p2, p3, t):
    """一维三次 Catmull-Rom，保证段间一阶连续（速度不突变）。"""
    return 0.5 * ((2 * p1) +
                  (-p0 + p2) * t +
                  (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                  (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t)


def build_path(cams, n_frames, loop):
    """在训练相机的路线上平滑插值出 n_frames 个相机位姿。

    返回 [(R_c2w, center, FoVx, FoVy), ...]

    loop=True 时按**环形**取样：最后一帧之后接回第 0 个相机，
    Catmull-Rom 的邻居也用模 n 索引，这样视频首尾能无缝循环。
    非环形时首帧=第 0 个相机、末帧=最后一个相机。
    """
    # 3DGS 的 Camera 存的是 R(w2c) 和 T，c2w 的旋转是 Rᵀ，中心是 −RᵀT
    Rc, C, fovx, fovy = [], [], [], []
    for c in cams:
        Rc.append(c.R.T)
        C.append(-c.R.T @ np.asarray(c.T, dtype=float).reshape(3))
        fovx.append(c.FoVx)
        fovy.append(c.FoVy)
    Rc, C = np.array(Rc), np.array(C)
    fovx, fovy = np.array(fovx), np.array(fovy)

    # 四元数化并修正符号连续性
    Q = np.array([rot_to_quat_xyzw(R) for R in Rc])
    for i in range(1, len(Q)):
        if np.dot(Q[i - 1], Q[i]) < 0:
            Q[i] = -Q[i]

    n = len(C)
    if loop:
        # 覆盖 [0, n)，共 n 段；取 n_frames 个点，最后一帧之后回到起点
        idx = np.arange(n_frames) * (n / n_frames)
        def g(k):
            return k % n
    else:
        idx = np.linspace(0, n - 1, n_frames)
        def g(k):
            return int(np.clip(k, 0, n - 1))

    out = []
    for u in idx:
        i = int(np.floor(u))
        t = u - i
        i0, i1, i2, i3 = g(i - 1), g(i), g(i + 1), g(i + 2)
        p = np.array([catmull_rom(C[i0][k], C[i1][k], C[i2][k], C[i3][k], t)
                      for k in range(3)])
        q = slerp_quat(Q[i1], Q[i2], t)
        R = quat_to_rot(q)
        fx = fovx[i1] * (1 - t) + fovx[i2] * t
        fy = fovy[i1] * (1 - t) + fovy[i2] * t
        out.append((R, p, fx, fy))
    return out


# ---------------------------------------------------------------- 渲染

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="训练输出目录（含 cfg_args 和 point_cloud/）")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--iteration", type=int, default=-1)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="渲染分辨率缩放（0.5 = 面积降到 1/4，明显更快）")
    ap.add_argument("--out", default=None, help="输出目录，默认 <model>/video")
    ap.add_argument("--no-encode", action="store_true", help="只出 PNG，不做 mp4")
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from scene.cameras import Camera
    from gaussian_renderer import render
    from utils.general_utils import safe_state

    model_path = os.path.abspath(args.model)
    out_dir = args.out or os.path.join(model_path, "video")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    safe_state(True)
    scene, gaussians, dataset, pipeline, background = build_scene_and_gaussians(model_path, args)

    cams = scene.getTrainCameras()
    if len(cams) < 3:
        raise SystemExit(f"训练相机只有 {len(cams)} 个，造不出轨迹")
    print(f"模型: {model_path}")
    print(f"训练相机 {len(cams)} 个，iter={scene.loaded_iter}")

    W = int(cams[0].image_width * args.scale)
    H = int(cams[0].image_height * args.scale)
    W, H = W // 8 * 8, H // 8 * 8          # 光栅化要求 8 的倍数
    print(f"渲染尺寸 {W}x{H}，{args.frames} 帧")

    path = build_path(cams, args.frames, loop=True)
    dummy = Image.new("RGB", (W, H))       # 只用于确定尺寸，内容不参与渲染

    with torch.no_grad():
        for i, (R, center, fx, fy) in enumerate(path):
            # 3DGS 内部要的是 w2c 的 R 和 T
            Rw2c = R.T
            Tw2c = -Rw2c @ center
            cam = Camera(resolution=(W, H), colmap_id=-1,
                         R=Rw2c, T=Tw2c, FoVx=fx, FoVy=fy,
                         depth_params=None, image=dummy, invdepthmap=None,
                         image_name=f"video_{i:05d}", uid=i,
                         data_device="cuda")
            img = render(cam, gaussians, pipeline, background,
                         separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
            arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(frames_dir, f"{i:05d}.png"))
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{args.frames}")

    print(f"帧已写出: {frames_dir}")

    if not args.no_encode:
        mp4 = os.path.join(out_dir, os.path.splitext(os.path.basename(model_path))[0] + ".mp4")
        ff = shutil.which("ffmpeg")
        if not ff:
            print("⚠ 没找到 ffmpeg，跳过编码。装一下： apt-get install -y ffmpeg")
            print(f"  之后手动： ffmpeg -y -framerate {args.fps} -i {frames_dir}/%05d.png "
                  f"-c:v libx264 -pix_fmt yuv420p -crf 18 {mp4}")
        else:
            cmd = [ff, "-y", "-framerate", str(args.fps), "-i",
                   os.path.join(frames_dir, "%05d.png"),
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", mp4]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"视频: {mp4}  ({os.path.getsize(mp4)/1024**2:.1f} MB)")
            if not args.keep_frames:
                shutil.rmtree(frames_dir)
                print("已清理中间帧（加 --keep-frames 可保留）")


SPARSE_ADAM_AVAILABLE = False
try:
    import diff_gaussian_rasterization  # noqa: F401
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    pass


if __name__ == "__main__":
    main()
