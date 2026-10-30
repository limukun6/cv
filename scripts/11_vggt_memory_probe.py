"""
11_vggt_memory_probe.py — 摸清 VGGT 在本机上能吃多少帧。

为什么先做这件事
----------------
VGGT 是 1.2B 参数的 Transformer，跨帧注意力让显存大致随帧数**平方**增长。
我们需要知道在本机 8 GB 显存上：
    · 上限是几帧？
    · 每帧的边际显存成本是多少？
    · 这个上限够不够做题（P1 需要三条路线用**相同帧数**才公平）

这张表会同时回答三件事：机器行不行、P1 统一用几帧、要不要租显卡。

用法:
    python scripts/11_vggt_memory_probe.py --scene data/dl3dv-2 --frames 5 10 15 20 26 32
"""
import argparse
import gc
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/dl3dv-2")
    ap.add_argument("--frames", type=int, nargs="+", default=[5, 10, 15, 20, 26, 32])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--model-dtype", choices=["fp32", "bf16"], default="bf16",
                    help="权重存精度。官方 README 用 fp32（1B 参数占 4.7GB），"
                         "转 bf16 可省 2.3GB —— 对 8GB 卡是决定性的")
    ap.add_argument("--heads", choices=["all", "camera", "camera_depth"], default="all",
                    help="跑哪些预测头。track_head 最贵，而我们做 P1 根本不需要它")
    ap.add_argument("--max-budget-gb", type=float, default=7.2,
                    help="超过这个占用就停止（留出系统/驱动余量）")
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(ROOT, "third_party", "vggt"))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    # --- 收集图片 ---
    img_dir = os.path.join(args.scene, "rgb")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(args.scene, "images")
    imgs = sorted(os.path.join(img_dir, f) for f in os.listdir(img_dir)
                  if f.lower().endswith((".png", ".jpg")))
    print(f"场景 {args.scene}: 可用图片 {len(imgs)} 张")

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"精度: {dtype}")

    # --- 载入模型 ---
    t0 = time.time()
    print("载入 VGGT-1B ...")
    if args.weights and os.path.isfile(args.weights):
        model = VGGT()
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    else:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device).eval()
    if args.model_dtype == "bf16":
        model = model.to(dtype)
    load_s = time.time() - t0
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"模型载入完成 {load_s:.1f}s，权重占用 {base:.2f} GB (dtype={args.model_dtype})")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"显卡总显存 {total_vram:.2f} GB，留给激活值的余量约 {total_vram - base:.2f} GB\n")

    print(f"{'帧数':>5} {'输入尺寸':>12} {'峰值显存':>10} {'增量/帧':>10} {'耗时':>9}  状态")
    print("-" * 70)
    prev_mem, prev_n = None, None
    results = []

    for n in args.frames:
        if n > len(imgs):
            print(f"{n:>5}  {'-':>12} {'-':>10} {'-':>10} {'-':>9}  图片不够")
            continue
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        try:
            images = load_and_preprocess_images(imgs[:n]).to(device)
            h, w = images.shape[-2:]
            t1 = time.time()
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    if args.heads == "all":
                        _ = model(images)
                    else:
                        # 只跑需要的头：省掉 track_head 和 point_head 的大量中间激活
                        imgs_b = images[None]
                        agg, ps_idx = model.aggregator(imgs_b)
                        _ = model.camera_head(agg)[-1]          # 位姿
                        if args.heads == "camera_depth":
                            _ = model.depth_head(agg, imgs_b, ps_idx)
            torch.cuda.synchronize()
            dt = time.time() - t1

            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            delta = (peak - prev_mem) / (n - prev_n) if prev_mem and n > prev_n else float("nan")
            results.append((n, w, h, peak, dt, "OK"))
            print(f"{n:>5} {w:>5}x{h:<6} {peak:>8.2f}GB {delta:>8.3f} {dt:>8.2f}s  OK")
            prev_mem, prev_n = peak, n

            del images, _
            torch.cuda.empty_cache()

            if peak > args.max_budget_gb:
                print(f"\n>>> 峰值 {peak:.2f} GB 已超过预算 {args.max_budget_gb} GB，停止扫描")
                break
        except torch.cuda.OutOfMemoryError as e:
            print(f"{n:>5}  {'-':>12} {'OOM':>10} {'-':>10} {'-':>9}  ★ 显存不足")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"{n:>5}  {'-':>12} {'-':>10} {'-':>10} {'-':>9}  错误: {type(e).__name__}: {str(e)[:50]}")
            torch.cuda.empty_cache()
            break

    if results:
        best = results[-1]
        print(f"\n结论：本机可安全运行到约 {best[0]} 帧（峰值 {best[3]:.2f} GB）")
        print(f"      输入分辨率 {best[1]}x{best[2]}，单次前向 {best[4]:.2f}s")
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"      显卡总显存 {total:.1f} GB，余量 {total - best[3]:.2f} GB")


if __name__ == "__main__":
    main()
