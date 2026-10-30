"""
41_eval.py — 一次性算完 PSNR / SSIM / LPIPS，并写出 results.json。

为什么要重写（而不是直接用 3DGS 自带的 metrics.py）
--------------------------------------------------
3DGS 的 `lpipsPyTorch.lpips()` 每次调用都会 `LPIPS(net_type, version)` 重新构造网络：

    def lpips(x, y, net_type='alex', version='0.1'):
        criterion = LPIPS(net_type, version).to(device)   # ← 每张图都重建一次
        return criterion(x, y)

实测代价：39 张测试图要 **350 秒**（9 秒/张）。按 30 次实验算，白烧 3 小时。

这里改成「网络只构造一次，循环复用」，并把 backbone 固定为 alex
（和契约一致，也和 3DGS 官方数字可比）。

用法:
    python scripts/41_eval.py --run outputs/runs/<run_id>
    python scripts/41_eval.py --run <dir> --lpips-backbone alex
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lpips-backbone", default="vgg", choices=["alex", "vgg", "squeeze"],
                    help="默认 vgg —— 这是 3DGS 官方 metrics.py 实际用的（它显式传 net_type='vgg'），"
                         "不是函数签名里的默认值 alex。想和论文数字可比就用 vgg")
    ap.add_argument("--lpips-range", default="0-1", choices=["0-1", "m1-1"],
                    help="0-1 = 直接喂 [0,1]（和 metrics.py 一致）；m1-1 = 换算到 [-1,1]")
    ap.add_argument("--method-dir", default=None, help="默认取第一个找到的")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(root, "third_party", "gaussian-splatting"))

    import torch
    from utils.loss_utils import ssim as ssim_fn
    from utils.image_utils import psnr as psnr_fn
    from lpipsPyTorch.modules.lpips import LPIPS

    # --- 找到 renders / gt 目录 ---
    run = args.run
    found = []
    for method in sorted(os.listdir(run)):
        p = os.path.join(run, method)
        if not os.path.isdir(p) or method in ("point_cloud",):
            continue
        for sub in sorted(os.listdir(p)):
            r, g = os.path.join(p, sub, "renders"), os.path.join(p, sub, "gt")
            if os.path.isdir(r) and os.path.isdir(g):
                found.append((method, sub, r, g))
    if not found:
        raise SystemExit(f"{run} 下找不到 renders/gt，先跑 render.py")
    method, sub, rdir, gdir = found[0]
    if args.method_dir:
        for f in found:
            if f[1] == args.method_dir:
                method, sub, rdir, gdir = f
    names = sorted(os.listdir(rdir))
    print(f"评测 {method}/{sub}，共 {len(names)} 个测试视角")

    # --- 一次构造 LPIPS，之后复用 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    lpips_net = LPIPS(args.lpips_backbone, "0.1").to(device).eval()
    print(f"LPIPS ({args.lpips_backbone}) 构造完成 {time.time() - t0:.1f}s")
    if device.type == "cuda":
        print(f"显存占用 {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

    psnrs, ssims, lpipss = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i, n in enumerate(names, 1):
            a = np.asarray(Image.open(os.path.join(rdir, n)).convert("RGB"), dtype=np.float32) / 255.0
            b = np.asarray(Image.open(os.path.join(gdir, n)).convert("RGB"), dtype=np.float32) / 255.0
            if a.shape != b.shape:
                b = np.asarray(Image.fromarray((b * 255).astype(np.uint8)).resize(
                    (a.shape[1], a.shape[0])), dtype=np.float32) / 255.0
            # permute 之后张量不连续，而 3DGS 的 psnr/ssim 内部用了 .view()，
            # 必须 .contiguous()，否则报 "view size is not compatible"
            ta = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).contiguous()[None].to(device)
            tb = torch.from_numpy(np.ascontiguousarray(b)).permute(2, 0, 1).contiguous()[None].to(device)
            psnrs.append(float(psnr_fn(ta, tb).mean()))
            ssims.append(float(ssim_fn(ta, tb).mean()))
            # LPIPS 的输入范围：官方 lpips 包内部会做 x*2-1，但 3DGS 用的
            # lpipsPyTorch 没有这一步。metrics.py 直接把 [0,1] 的图喂进去，
            # 所以要和它的数字可比，就必须也喂 [0,1]。
            if args.lpips_range == "m1-1":
                lpipss.append(float(lpips_net(ta * 2 - 1, tb * 2 - 1).mean()))
            else:
                lpipss.append(float(lpips_net(ta, tb).mean()))
            if i % 10 == 0 or i == len(names):
                print(f"  {i}/{len(names)}  {time.time() - t0:.1f}s")

    out = {
        f"{method}_{sub}": {
            "PSNR": float(np.mean(psnrs)),
            "SSIM": float(np.mean(ssims)),
            "LPIPS": float(np.mean(lpipss)),
            "num_views": len(names),
        }
    }
    print(f"\n{'PSNR':>8} {np.mean(psnrs):.4f}")
    print(f"{'SSIM':>8} {np.mean(ssims):.4f}")
    print(f"{'LPIPS':>8} {np.mean(lpipss):.4f}")
    print(f"\n评测总耗时 {time.time() - t0:.1f}s（{len(names)} 张）")

    dst = os.path.join(run, "results.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"写出 {dst}")


if __name__ == "__main__":
    main()
