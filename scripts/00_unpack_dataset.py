"""
00_unpack_dataset.py — 把课程发的 Project4.zip 解成本项目的标准目录结构。

用法:
    python scripts/00_unpack_dataset.py --zip Project4.zip --out data

设计要点（对应 README 第三节「场景契约」）:
  1. 三个场景的原始格式完全不同，这里统一成 data/<scene>/ 的骨架。
  2. 过滤 Jupyter 检查点垃圾文件（DL3DV-2 里有 12 张 `*-checkpoint.png`）。
  3. 位姿文件原样保留（不做任何转换），转换留给 20_to_canonical.py。
     这样做的好处：原始数据永远可回溯，出错了能重跑。
  4. 写出 data/<scene>/meta.yaml，供后续脚本读取。
"""
import argparse
import os
import sys
import zipfile

try:
    from PIL import Image
except ImportError:
    Image = None

# 源 zip 里的顶层目录名 -> 我们的 scene id
SCENE_MAP = {
    "405841": "waymo-405841",
    "DL3DV-2": "dl3dv-2",
    "Re10k-1": "re10k-1",
}

JUNK_PATTERNS = (".ipynb_checkpoints", "-checkpoint", "checkpoint.png")


def is_junk(name: str) -> bool:
    low = name.lower()
    return any(p in low for p in JUNK_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="Project4.zip")
    ap.add_argument("--out", default="data")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.zip):
        sys.exit(f"找不到 {args.zip}")

    z = zipfile.ZipFile(args.zip)
    names = [n for n in z.namelist() if not n.endswith("/")]
    print(f"zip 内文件数: {len(names)}")

    stats = {}
    for top, scene in SCENE_MAP.items():
        members = [n for n in names if n.startswith(top + "/")]
        keep = [n for n in members if not is_junk(n)]
        dropped = len(members) - len(keep)
        stats[scene] = (len(keep), dropped)
        print(f"  {top:<12} -> {scene:<15} 保留 {len(keep):4d}  丢弃垃圾 {dropped:3d}")

    if args.dry_run:
        print("\n[dry-run] 不写盘")
        return

    for top, scene in SCENE_MAP.items():
        dest = os.path.join(args.out, scene)
        members = [n for n in names if n.startswith(top + "/") and not is_junk(n)]
        print(f"\n解压 {top} -> {dest}  ({len(members)} 个文件)")
        for i, m in enumerate(members):
            rel = m[len(top) + 1:]  # 去掉顶层目录名
            # 统一重命名：405841/FRONT/xxx -> waymo/xxx 结构更扁平
            if rel.startswith("FRONT/"):
                rel = rel[len("FRONT/"):]
            target = os.path.join(dest, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target) and os.path.getsize(target) == z.getinfo(m).file_size:
                continue
            with z.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{len(members)}")

    # 写 meta.yaml
    write_meta(args.out, stats)
    print("\n完成。下一步: python scripts/01_make_splits.py --scene <scene>")


def write_meta(out_root, stats):
    meta = {
        "waymo-405841": dict(scene="waymo-405841", source_dataset="waymo",
                             scene_type="outdoor", raw_format="frames",
                             image_subdir="rgb", has_gt_pose=True,
                             has_depth=True, notes="FRONT camera only; 199 frames; 1920x1280"),
        "dl3dv-2": dict(scene="dl3dv-2", source_dataset="dl3dv",
                        scene_type="outdoor", raw_format="frames",
                        image_subdir="rgb", has_gt_pose=True,
                        has_depth=False, notes="306 frames after removing 12 jupyter checkpoint files; 256x256"),
        "re10k-1": dict(scene="re10k-1", source_dataset="re10k",
                        scene_type="indoor", raw_format="frames",
                        image_subdir="images", has_gt_pose=True,
                        has_depth=False, notes="279 frames; 256x256; stray video.mp4 present"),
    }
    for scene, d in meta.items():
        path = os.path.join(out_root, scene, "meta.yaml")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for k, v in d.items():
                f.write(f"{k}: {v}\n")
            f.write(f"files_kept: {stats[scene][0]}\n")
            f.write(f"files_dropped: {stats[scene][1]}\n")
        print("  写出", path)


if __name__ == "__main__":
    main()
