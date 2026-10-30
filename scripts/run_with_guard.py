"""
run_with_guard.py — 给任何 GPU 脚本套上显存硬上限，然后照原样运行它。

为什么需要这个
--------------
2026-10-29 我在 8 GB 笔记本上做「扫到爆」的显存探测，把显存吃到 7.47 GB，
导致 Windows 开始把显存页换到内存（WDDM 分页），整机卡死并强制关机。

教训：**"我记得要留余量"不是护栏。** 真正的护栏必须由运行时强制执行。

torch.cuda.set_per_process_memory_fraction(f) 会告诉 PyTorch 的缓存分配器：
本进程最多只能用显存总量的 f 倍；一旦超了就直接抛
torch.cuda.OutOfMemoryError，**不会去抢系统最后一点余量**，也就不会拖死系统。

为什么写成"包装器"而不是改脚本
------------------------------
因为 3DGS 的 train.py 是第三方代码（third_party/），不 ours 改；
即使是我们自己的脚本，散落着改也容易漏。
统一用一个入口包住，最可靠。

用法
----
    # 默认上限 55%（8 GB 卡 → 约 4.5 GB）
    python scripts/run_with_guard.py -- third_party/gaussian-splatting/train.py -s ... -m ...

    # 更保守
    python scripts/run_with_guard.py --fraction 0.45 -- train.py ...

    # 看当前卡和默认预算
    python scripts/run_with_guard.py --info
"""
import argparse
import os
import runpy
import sys

# 笔记本 8 GB 默认只给 55%：显示合成器 + 浏览器（DSH 界面本身就在吃显存）
DEFAULT_FRACTION = 0.55


def info():
    import torch
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return
    p = torch.cuda.get_device_properties(0)
    total = p.total_memory / 1024 ** 3
    print(f"GPU        : {p.name}")
    print(f"总显存     : {total:.2f} GB")
    print(f"默认上限   : {DEFAULT_FRACTION:.0%} → {total * DEFAULT_FRACTION:.2f} GB")
    print(f"留给系统   : {total * (1 - DEFAULT_FRACTION):.2f} GB（显示/浏览器/驱动）")
    print(f"当前占用   : {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--hard-cap-gb", type=float, default=None,
                    help="直接指定 GB 上限，覆盖 --fraction")
    known, rest = ap.parse_known_args()

    if known.info:
        info()
        return

    if rest and rest[0] == "--":
        rest = rest[1:]
    if not rest:
        print("用法: python scripts/run_with_guard.py -- <script.py> [args...]", file=sys.stderr)
        sys.exit(2)

    target = rest[0]
    if not os.path.isfile(target):
        print(f"找不到脚本: {target}", file=sys.stderr)
        sys.exit(2)

    import torch
    if not torch.cuda.is_available():
        print("[guard] CUDA 不可用，直接运行", file=sys.stderr)
    else:
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if known.hard_cap_gb is not None:
            frac = min(1.0, known.hard_cap_gb / total)
        else:
            frac = known.fraction
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"[guard] 显存硬上限 {frac:.0%} = {total * frac:.2f} GB "
              f"(总共 {total:.2f} GB，留给系统 {total * (1 - frac):.2f} GB)", file=sys.stderr)
        print(f"[guard] 超过上限会直接抛 OOM 异常，不会拖垮系统", file=sys.stderr)

    # 让被包装的脚本看到正确的 argv 和 __name__
    # 注意：runpy.run_path 不会把脚本所在目录加入 sys.path，
    # 而第三方脚本（如 3DGS 的 train.py）用的是 `from scene import Scene` 这类
    # 依赖自身目录的绝对导入。不补这一行，包装器一用就 ImportError。
    target_dir = os.path.dirname(os.path.abspath(target))
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)

    sys.argv = rest
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
