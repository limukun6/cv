"""
40_make_compare.py — 把「渲染结果 / 真值 / 误差图」拼成一张对比图，用于目视检查。

为什么要目视检查：指标会骗人。
    · PSNR 高但图糊成一片（因为相邻帧太像，插值就能拿到高分）
    · LPIPS 异常低（同样原因）
    · 评测读错了目录（gt 和 render 拿反了）
这些错误在数字上看不出来，但眼睛一看就知道。

用法:
    python scripts/40_make_compare.py --run <model_dir> --n 3
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image


def find_dirs(model_dir):
    """找到 render.py 产出的 renders / gt 目录。"""
    cands = []
    for method in os.listdir(model_dir):
        p = os.path.join(model_dir, method)
        if not os.path.isdir(p):
            continue
        for sub in os.listdir(p):
            r = os.path.join(p, sub, "renders")
            g = os.path.join(p, sub, "gt")
            if os.path.isdir(r) and os.path.isdir(g):
                cands.append((method, sub, r, g))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--scale", type=int, default=2, help="放大倍数，方便看清细节")
    args = ap.parse_args()

    cands = find_dirs(args.run)
    if not cands:
        raise SystemExit(f"{args.run} 下找不到 renders/gt，先跑 render.py")

    method, sub, rdir, gdir = cands[0]
    names = sorted(os.listdir(rdir))[: args.n]
    print(f"用 {method}/{sub}，共 {len(os.listdir(rdir))} 个测试视角，取前 {len(names)} 个")

    rows = []
    for n in names:
        A = np.asarray(Image.open(os.path.join(rdir, n)).convert("RGB"), dtype=np.float32) / 255.0
        B = np.asarray(Image.open(os.path.join(gdir, n)).convert("RGB"), dtype=np.float32) / 255.0
        if A.shape != B.shape:
            B = np.asarray(Image.fromarray((B * 255).astype(np.uint8)).resize(
                (A.shape[1], A.shape[0])), dtype=np.float32) / 255.0
        D = np.abs(A - B).mean(axis=2)
        D3 = np.clip(D * 4.0, 0, 1)                      # 误差放大 4 倍才看得见
        D3 = np.stack([D3, D3, D3], axis=2)
        rows.append(np.concatenate([A, B, D3], axis=1))
        mse = ((A - B) ** 2).mean()
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        print(f"  {n}: 单图 PSNR = {psnr:.2f} dB, 平均绝对误差 = {D.mean():.4f}")

    grid = np.concatenate(rows, axis=0)
    img = Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8))
    if args.scale > 1:
        img = img.resize((img.width * args.scale, img.height * args.scale), Image.NEAREST)
    out = os.path.join(args.run, "compare_grid.png")
    img.save(out)
    print(f"\n写出 {out}  ({img.width}x{img.height})")
    print("三列依次为：渲染 | 真值 | 误差(×4)")


if __name__ == "__main__":
    main()
