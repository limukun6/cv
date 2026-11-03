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

    extr, intr = encoding_to_camera(pose_enc, imgs.shape[-2:])

    def drop_batch(t):
        """去掉前导的单例维度。

        实测 Ω 的输出是 (1,89,9) / (1,89,3,4) / (1,89,512,512,1) 这种带 batch 的，
        而 v1 的下游工具（unproject / pycolmap）要的是去掉 batch 的 (S,...)。
        """
        if not hasattr(t, "dim"):
            return t
        while t.dim() > 2 and t.shape[0] == 1:
            t = t[0]
        return t

    extr, intr = drop_batch(extr), drop_batch(intr)
    depth = drop_batch(preds["depth"])
    depth_conf = drop_batch(preds["depth_conf"])

    to_np = lambda t: t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return to_np(extr), to_np(intr), to_np(depth), to_np(depth_conf)




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
    ap.add_argument("--expect-fx-norm", type=float, default=None,
                    help="数据集真值的归一化 fx（如 dl3dv 0.797），用来核对内参是否离谱")
    args = ap.parse_args()

    import torch
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track

    expect_fx_norm = args.expect_fx_norm
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

    # ---- 内参单位：必须是**像素**
    # ★ 这里踩过一个致命的坑：我原先以为 VGGT 的内参是归一化的，把 Ω 的
    #   fx=407（512 输入下的像素值）除以 512 变成 0.795，于是导出模型的
    #   fx 变成 0.4，3DGS 渲染相当于从 500 倍远处看场景 → PSNR 只有 9.1 dB。
    #
    #   事实是：VGGT v1 的 `pose_encoding_to_extri_intri(pose_enc, image_size)`
    #   返回的就是**像素**内参（v1 在 BA 分支里做 `intrinsic[:, :2, :] *= scale`
    #   把 518 空间放大到 1024 空间，也只有像素单位才需要这么放缩）。
    #   "归一化"是**数据集 cameras.json** 的约定，两者被我混了。
    #
    #   所以：像素 → 直接用；归一化 → 乘回去。
    fx0 = float(intr[0, 0, 0])
    if fx0 < 4.0:
        print(f"  ⚠ 内参看起来是归一化的（fx={fx0:.4f}），按 {args.resolution} 还原成像素")
        intr = intr * float(args.resolution)
    else:
        print(f"  内参已是像素单位（fx={fx0:.2f} @ {args.resolution} 输入）")
    exp_fx = None
    if expect_fx_norm:
        exp_fx = expect_fx_norm * args.resolution
        print(f"  参考：若真值归一化 fx={expect_fx_norm:.4f}，"
              f"对应 {args.resolution} 输入下应为 {exp_fx:.1f}")
        print(f"  实际 {float(intr[0,0,0]):.1f}，偏差 {abs(float(intr[0,0,0])-exp_fx)/exp_fx*100:.2f}%")
    print(f"  第 0 帧 K =\n{np.round(intr[0], 2)}")

    # 深度反投影成三维点（和 v1 一致：传像素内参）
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
