"""
00d_import_3dgs_dataset.py — 把标准 3DGS 基准数据集接入我们的流水线目录约定。

为什么需要这个脚本
----------------
课程的加分项要求"在额外数据上报告结果"，而课程给的压缩包里只有 3 个必做场景。
我们改用**标准 3DGS 基准数据集**（Tanks & Temples / Deep Blending / Mip-NeRF 360），
理由是它们：
  1. 好下载（T&T 的官方源在 INRIA，国内直连可达，实测 HTTP 200）
  2. **自带做好的 COLMAP 稀疏模型**，不用我们再跑一遍 SfM
  3. 有大量已发表的 3DGS 数字，方便横向对比

它们的原始结构与我们的约定不同：

    3DGS 基准数据集                我们的流水线
    ─────────────────              ──────────────
    truck/images/*.jpg      →      data/tt-truck/images/
    truck/sparse/0/*.bin    →      data/tt-truck/colmap/sparse/0/
    （没有划分文件）          →      data/tt-truck/splits/（用 01_make_splits.py 生成）

用法:
    python scripts/00d_import_3dgs_dataset.py --src _dl/tandt_db/truck --name tt-truck
    python scripts/00d_import_3dgs_dataset.py --src _dl/tandt_db/train --name tt-train --resolution 2
"""
import argparse
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_EXTS = (".jpg", ".jpeg", ".png")


def find_images(src):
    """基准数据集的图片都在 images/ 下；兼容少数放在根目录的情况。"""
    d = os.path.join(src, "images")
    if os.path.isdir(d):
        return d
    if any(f.lower().endswith(IMG_EXTS) for f in os.listdir(src)):
        return src
    for sub in os.listdir(src):
        p = os.path.join(src, sub)
        if os.path.isdir(p) and any(f.lower().endswith(IMG_EXTS) for f in os.listdir(p)):
            return p
    raise SystemExit(f"在 {src} 下找不到图片目录")


def find_sparse(src):
    """找 COLMAP 模型：优先 sparse/0。"""
    s = os.path.join(src, "sparse", "0")
    if os.path.isdir(s) and os.path.isfile(os.path.join(s, "images.bin")):
        return s
    s = os.path.join(src, "sparse")           # 有的数据集直接把 bin 放在 sparse/
    if os.path.isdir(s) and os.path.isfile(os.path.join(s, "images.bin")):
        return s
    return None


def link_images(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(src_dir)):
        if not f.lower().endswith(IMG_EXTS):
            continue
        d = os.path.join(dst_dir, f)
        if os.path.exists(d):
            n += 1
            continue
        try:
            os.link(os.path.join(src_dir, f), d)      # 同盘硬链接，几乎不占空间
        except OSError:
            shutil.copy2(os.path.join(src_dir, f), d)  # 跨盘退回复制
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="基准数据集里某个场景的目录")
    ap.add_argument("--name", required=True, help="我们这边的场景名，如 tt-truck")
    ap.add_argument("--data", default="data")
    ap.add_argument("--resolution", type=int, default=None,
                    help="训练时的降采样倍数（写进 meta.yaml）。"
                         "1920x1080 的场景在 24GB 卡上建议 2；256² 的不用填")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    if not os.path.isdir(src):
        raise SystemExit(f"源目录不存在: {src}")

    dst = os.path.join(ROOT, args.data, args.name)
    os.makedirs(dst, exist_ok=True)

    print(f"导入 {src}\n  → {dst}")

    # 1) 图片
    img_dir = find_images(src)
    n = link_images(img_dir, os.path.join(dst, "images"))
    print(f"  图片 {n} 张（来自 {os.path.relpath(img_dir, ROOT)}）")

    # 2) COLMAP 模型
    sp = find_sparse(src)
    if sp:
        out_sp = os.path.join(dst, "colmap", "sparse", "0")
        os.makedirs(out_sp, exist_ok=True)
        for f in os.listdir(sp):
            if f.endswith(".bin"):
                shutil.copy2(os.path.join(sp, f), os.path.join(out_sp, f))
        print(f"  COLMAP 模型已复制: {os.listdir(out_sp)}")
    else:
        print("  ⚠ 源里没有 COLMAP 模型，需要自己跑 scripts/10_run_colmap.py")

    # 3) meta.yaml
    meta = [
        f"scene: {args.name}",
        "source: 3dgs-benchmark",
        "image_subdir: images",
    ]
    if args.resolution:
        meta.append(f"resolution: {args.resolution}")
    with open(os.path.join(dst, "meta.yaml"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(meta) + "\n")
    print(f"  meta.yaml: {meta}")

    # 4) 图片尺寸
    try:
        from PIL import Image
        first = sorted(os.listdir(os.path.join(dst, "images")))[0]
        with Image.open(os.path.join(dst, "images", first)) as im:
            print(f"  图片尺寸: {im.size[0]}x{im.size[1]}")
            if im.size[0] * im.size[1] > 512 * 512 and not args.resolution:
                print(f"  ⚠ 分辨率较高，24GB 卡训练建议加 --resolution 2（本次没加）")
    except Exception:
        pass

    print(f"\n下一步（生成训练/测试划分）:")
    print(f"  python scripts/01_make_splits.py --scene {args.name}")


if __name__ == "__main__":
    main()
