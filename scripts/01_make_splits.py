"""
01_make_splits.py — 按场景契约生成 train/test 划分。

三条硬规则（见 README 第三节 3.3）：
  1. test 一旦生成就冻结，此后任何实验不得修改。改 test = 作废之前所有数字。
  2. 稀疏集先在完整帧池上按 k 抽，再减掉 test —— 保留「原序列 1/k」的语义。
  3. sparse30 减完 test 若 < 5 帧，写进 _index.csv 的 warning 列。

输出格式：纯文本，一行一个文件名，UTF-8 无 BOM。

用法:
    python scripts/01_make_splits.py --data data --scene all --test-every 8
"""
import argparse
import csv
import os
import sys

IMAGE_EXTS = (".png", ".jpg", ".jpeg")

SCENES = ["waymo-405841", "dl3dv-2", "re10k-1"]
SPARSE_STEPS = {"sparse10": 10, "sparse30": 30}


def image_subdir(scene_dir):
    """不同场景的图片目录名不一样，从 meta.yaml 读更稳，这里做简单探测。"""
    meta = os.path.join(scene_dir, "meta.yaml")
    if os.path.isfile(meta):
        for line in open(meta, encoding="utf-8"):
            if line.startswith("image_subdir:"):
                return line.split(":", 1)[1].strip()
    for cand in ("rgb", "images"):
        if os.path.isdir(os.path.join(scene_dir, cand)):
            return cand
    sys.exit(f"{scene_dir} 下找不到 rgb/ 或 images/")


def list_images(scene_dir):
    sub = image_subdir(scene_dir)
    d = os.path.join(scene_dir, sub)
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS))
    return sub, files


def write_split(path, names):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(names) + ("\n" if names else ""))


def process(scene, data_root, test_every, dry_run=False):
    scene_dir = os.path.join(data_root, scene)
    if not os.path.isdir(scene_dir):
        print(f"[跳过] {scene}: 目录不存在")
        return None

    sub, pool = list_images(scene_dir)
    if not pool:
        print(f"[跳过] {scene}: {sub}/ 里没有图片")
        return None

    # ★ 核心三行
    test = pool[::test_every]
    test_set = set(test)
    full = [n for n in pool if n not in test_set]

    splits = {"test": test, "full": full}
    warnings = {}
    for name, k in SPARSE_STEPS.items():
        sparse = [n for n in pool[::k] if n not in test_set]
        splits[name] = sparse
        if name == "sparse30" and len(sparse) < 5:
            warnings[name] = f"only {len(sparse)} frames after removing test"
            print(f"  ⚠ {scene}: {name} 只剩 {len(sparse)} 帧")

    print(f"\n=== {scene} ===")
    print(f"  图片目录 {sub}/, 帧池 {len(pool)}")
    print(f"  test     {len(splits['test']):4d}  (>0: 每 {test_every} 帧留 1)")
    print(f"  full     {len(splits['full']):4d}  (= 帧池 - test)")
    print(f"  sparse10 {len(splits['sparse10']):4d}  (>0: 池[::10] - test)")
    print(f"  sparse30 {len(splits['sparse30']):4d}  (>0: 池[::30] - test)")

    if dry_run:
        return splits

    out_dir = os.path.join(scene_dir, "splits")
    os.makedirs(out_dir, exist_ok=True)
    for name, names in splits.items():
        write_split(os.path.join(out_dir, f"{name}.txt"), names)

    # _index.csv 追加（保留已有行，避免重复运行污染）
    index = os.path.join(out_dir, "_index.csv")
    rows = []
    if os.path.isfile(index):
        with open(index, encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("scene") != scene]
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    for name in splits:
        rows.append(dict(scene=scene, split_name=name, frames=len(splits[name]),
                         test_every=test_every if name == "test" else "-",
                         source_pool=sub, created_at=now,
                         warning=warnings.get(name, "")))
    with open(index, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "split_name", "frames",
                                          "test_every", "source_pool",
                                          "created_at", "warning"])
        w.writeheader()
        w.writerows(rows)
    print(f"  写出 {out_dir}/*.txt 和 _index.csv")
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--scene", default="all")
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scenes = SCENES if args.scene == "all" else [args.scene]
    for s in scenes:
        process(s, args.data, args.test_every, args.dry_run)

    print("\n完成。下一步: 跑 COLMAP 或 VGGT 得到位姿，再转到 canonical/")


if __name__ == "__main__":
    main()
