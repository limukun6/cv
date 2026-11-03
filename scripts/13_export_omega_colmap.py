"""
13_export_omega_colmap.py — 把 VGGT-Ω 的输出导出成 COLMAP 格式，供 3DGS 使用。

为什么需要自己写
--------------
VGGT v1 官方带 `demo_colmap.py`，能直接吐出 cameras.bin / images.bin / points3D.bin。
**VGGT-Ω 官方没有这个工具**（只有一个 Gradio 可视化 demo，输出 GLB）。
所以要接 3DGS，必须自己写转换。

复用而不是重写
------------
VGGT v1 的这套工具是通用的，直接复用，避免重造轮子：
  · `load_and_preprocess_images_square`  加载大图 + 补成正方形，并记下原图坐标
  · `unproject_depth_map_to_point_map`   深度图 + 位姿 → 三维点
  · `batch_np_matrix_to_pycolmap_wo_track`   numpy → pycolmap Reconstruction
  · `create_pixel_coordinate_grid` / `randomly_limit_trues`
  · `rename_colmap_recons_and_rescale_camera`（本文件内复制，因为它是 demo_colmap.py 里的私有函数）

内参缩放这件事最容易错
--------------------
预处理会把图放大/补边（我们的图 256² → 补到 1024² → 再缩到模型输入 512²），
**内参是在这个处理后的空间里的**。如果直接把处理空间的内参配上原图，
渲染会整体变形。v1 的做法是：
  1. 用 `img_load_resolution / model_resolution` 把内参放大到 1024 空间
  2. 导出后用 `rename_colmap_recons_and_rescale_camera` 按 `original_coords` 缩回原图分辨率
本脚本完全照抄这两步。

内参的两种约定
------------
VGGT v1 的内参是**归一化**的（fx∈[0,1]，乘以宽度才是像素）。
VGGT-Ω 的 `encoding_to_camera` 返回哪一种官方文档没写清。
所以本脚本会自动判别（看 fx 的量级），并在 `--dry-run` 时打印出来，
避免"用错约定 → 渲染全糊"这种最难查的问题。

用法:
    # 先干跑一次，看清形状和内参约定（不写文件）
    python scripts/13_export_omega_colmap.py --scene-dir data/dl3dv-2/vggt_omega_n50 \
        --ckpt third_party/weights/vggt_omega/vggt_omega_1b_416_reproduce.pt --dry-run

    # 正式导出
    python scripts/13_export_omega_colmap.py --scene-dir data/dl3dv-2/vggt_omega_n50 \
        --ckpt third_party/weights/vggt_omega/vggt_omega_1b_416_reproduce.pt
"""
import argparse
import copy
import glob
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "third_party", "vggt"))         # v1 的通用工具
sys.path.insert(0, os.path.join(ROOT, "third_party", "vggt_omega"))   # Ω 的模型


# ---------------------------------------------------------------- 复制自 VGGT v1 demo_colmap.py
def rename_colmap_recons_and_rescale_camera(reconstruction, image_paths, original_coords,
                                            img_size, shift_point2d_to_original_res=False,
                                            shared_camera=False):
    """把模型空间的内参缩回**原图分辨率**，并把图像名改回原始文件名。

    （原函数定义在 VGGT v1 的 demo_colmap.py 里，不在包内，所以这里复制一份。）
    """
    rescale_camera = True
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            pred_params[-2:] = real_image_size / 2      # 主点放到图心
            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False
    return reconstruction


# ---------------------------------------------------------------- 模型

def load_omega(ckpt, device):
    """加载 VGGT-Ω。checkpoint 可能是纯 state_dict，也可能被包了一层。"""
    from vggt_omega.models import VGGTOmega
    import torch

    print(f"加载 VGGT-Ω: {ckpt}")
    model = VGGTOmega().to(device).eval()
    sd = torch.load(ckpt, map_location="cpu")
    for key in ("model", "state_dict", "module"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            print(f"  checkpoint 包了一层，取 '{key}'")
            sd = sd[key]
            break
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  ⚠ 缺失权重 {len(missing)} 个（前 3 个: {missing[:3]}）")
    if unexpected:
        print(f"  ⚠ 多余权重 {len(unexpected)} 个（前 3 个: {unexpected[:3]}）")
    if not missing and not unexpected:
        print("  权重完全匹配 ✓")
    return model


def run_omega(model, images, resolution, device, dtype):
    """跑一次推理，返回 (extrinsic, intrinsic, depth, depth_conf)。

    extrinsic 是 w2c（OpenCV 约定），intrinsic 可能是归一化或像素——由调用方判别。
    """
    import torch
    import torch.nn.functional as F
    from vggt_omega.utils.pose_enc import encoding_to_camera

    x = F.interpolate(images, size=(resolution, resolution),
                      mode="bilinear", align_corners=False)
    print(f"  模型输入: {tuple(x.shape)}")

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        preds = model(x)

    print(f"  输出键: {list(preds.keys())}")
    for k in ("pose_enc", "depth", "depth_conf"):
        if k in preds:
            print(f"    {k}: {tuple(preds[k].shape)}")

    # pose_enc 可能带 batch 维（[1,S,9]）也可能没有（[S,9]）
    pose_enc = preds["pose_enc"]
    imgs = preds["images"] if "images" in preds else x
    if pose_enc.dim() == 2:
        pose_enc = pose_enc[None]
        squeeze = True
    else:
        squeeze = False

    extr, intr = encoding_to_camera(pose_enc, imgs.shape[-2:])

    def sq(t):
        if not hasattr(t, "dim"):
            return t
        if t.dim() >= 3 and t.shape[0] == 1 and squeeze:
            return t[0]
        return t

    extr, intr = sq(extr), sq(intr)
    depth = sq(preds["depth"])
    depth_conf = sq(preds["depth_conf"])

    to_np = lambda t: t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return to_np(extr), to_np(intr), to_np(depth), to_np(depth_conf)


def normalize_intrinsics(intr):
    """把内参统一成**归一化**形式（fx∈[0,1]），并返回判定的原始约定。

    判别依据：归一化的 fx 通常在 0.3~3 之间；像素的 fx 与图像宽度同量级。
    用一个与图像尺寸无关的阈值（fx > 4 视为像素坐标）。
    """
    fx = float(intr[0, 0])
    if fx > 4.0:
        # 像素坐标：需要知道宽度才能归一化，调用方会再乘回去，这里先按图像宽度归一
        return intr, "pixel"
    return intr, "normalized"


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True,
                    help="含 images/ 的目录；结果写到 <scene-dir>/sparse/0")
    ap.add_argument("--ckpt", required=True, help="VGGT-Ω 权重路径")
    ap.add_argument("--resolution", type=int, default=512, help="模型输入分辨率（512 或 256）")
    ap.add_argument("--img-load-resolution", type=int, default=1024,
                    help="加载后补齐的边长，和 v1 保持一致")
    ap.add_argument("--conf-thres", type=float, default=5.0, help="深度置信度阈值")
    ap.add_argument("--max-points", type=int, default=150000, help="写进 COLMAP 的点数上限")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--dry-run", action="store_true", help="只打印诊断信息，不写文件")
    args = ap.parse_args()

    import torch
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track

    device = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    scene_dir = os.path.abspath(args.scene_dir)
    image_paths = sorted(glob.glob(os.path.join(scene_dir, "images", "*")))
    if not image_paths:
        raise SystemExit(f"{scene_dir}/images 下没有图片")
    base_names = [os.path.basename(p) for p in image_paths]
    print(f"图片 {len(image_paths)} 张")

    model = load_omega(args.ckpt, device)

    images, original_coords = load_and_preprocess_images_square(
        image_paths, args.img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"预处理后: {tuple(images.shape)}  original_coords: {tuple(original_coords.shape)}")

    extr, intr, depth, conf = run_omega(model, images, args.resolution, device, dtype)
    print(f"extrinsic {extr.shape}  intrinsic {intr.shape}  "
          f"depth {depth.shape}  conf {conf.shape}")

    intr_n, conv = normalize_intrinsics(intr[0])
    print(f"\n内参约定判定: {conv}")
    print(f"  第 0 帧 K = \n{np.round(intr[0], 4)}")
    if conv == "pixel":
        # 像素坐标 → 归一化（除以模型输入边长）
        intr = intr / float(args.resolution)
        print(f"  已按模型输入边长 {args.resolution} 归一化")

    # 深度反投影成三维点（v1 的 intrinsic 就是归一化的，所以必须先归一化）
    points_3d = unproject_depth_map_to_point_map(depth, extr, intr)
    print(f"points_3d {points_3d.shape}")

    if args.dry_run:
        print("\n[dry-run] 诊断完成，未写文件。")
        print(f"  每帧点云形状: {points_3d.shape[1]}x{points_3d.shape[2]}")
        print(f"  深度范围: [{np.nanmin(depth):.3f}, {np.nanmax(depth):.3f}]")
        print(f"  置信度范围: [{np.nanmin(conf):.3f}, {np.nanmax(conf):.3f}]")
        return

    num_frames, height, width, _ = points_3d.shape
    points_rgb = (images.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    # points_rgb 已经是模型输入分辨率，与 points_3d 的 H/W 一致
    if points_rgb.shape[1] != height or points_rgb.shape[2] != width:
        import torch.nn.functional as F
        r = F.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)
        points_rgb = (r.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
    conf_mask = conf >= args.conf_thres
    keep = int(conf_mask.sum())
    print(f"置信度 >= {args.conf_thres} 的点: {keep} / {conf_mask.size}")
    conf_mask = randomly_limit_trues(conf_mask, args.max_points)

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]
    print(f"最终写入 {len(points_3d)} 个点")

    image_size = np.array([args.resolution, args.resolution])
    print("转换到 COLMAP 格式")
    recon = batch_np_matrix_to_pycolmap_wo_track(
        points_3d, points_xyf, points_rgb, extr, intr, image_size,
        shared_camera=False, camera_type="PINHOLE")
    if recon is None:
        raise SystemExit("pycolmap 构建失败（可能是点全部无效）")

    recon = rename_colmap_recons_and_rescale_camera(
        recon, base_names, original_coords.cpu().numpy(),
        img_size=args.resolution, shift_point2d_to_original_res=True,
        shared_camera=False)

    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    recon.write(sparse_dir)
    print(f"写出: {sparse_dir}  {os.listdir(sparse_dir)}")

    import trimesh
    trimesh.PointCloud(points_3d, colors=points_rgb).export(
        os.path.join(scene_dir, "sparse", "points.ply"))
    print(f"下一步: 检查相机是否跑飞 —— "
          f"python scripts/12_clean_colmap_model.py --model {sparse_dir} --dry-run")


if __name__ == "__main__":
    main()
