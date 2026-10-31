"""
00b_fetch_vggt_weights.py — 下载 VGGT-1B 权重，多源自动回退。

为什么需要这个脚本
----------------
1. **huggingface.co 在国内经常连不上**（本机实测直接 `HTTP 000`）
2. `hf-mirror.com` 能连，但速度极不稳定（实测 79 KB/s ~ 300 KB/s）
3. **ModelScope（阿里魔搭）镜像同一个仓库，实测 5–15 MB/s**（快 60 倍）

所以顺序是：ModelScope → hf-mirror → huggingface.co，哪个成功用哪个。
权重约 **4.68 GB**，下载时间差别是"10 分钟"和"16 小时"的区别。

用法:
    python scripts/00b_fetch_vggt_weights.py
    python scripts/00b_fetch_vggt_weights.py --dest third_party/weights --force
"""
import argparse
import os
import sys
import time

REPO = "facebook/VGGT-1B"
FILENAME = "model.pt"
SIZE = 5026874952          # 官方文件大小，用于完整性校验
MS_API = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={fn}"
MS_FILES = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files"
HF_MIRROR = "https://hf-mirror.com"
HF_OFFICIAL = "https://huggingface.co"


def human(n):
    return f"{n / 1024 ** 3:.2f} GB"


def try_modelscope(dest, force=False):
    import subprocess
    url = MS_API.format(repo=REPO, fn=FILENAME)
    print(f"[1/3] ModelScope: {url}")
    cmd = ["curl", "-L", "--retry", "3", "--retry-delay", "3", "-o", dest, url]
    if not force:
        # curl 的断点续传必须是 `-C -`（一个短横），单独写 -C 会吃掉后面的参数
        cmd[1:1] = ["-C", "-"]
    t0 = time.time()
    p = subprocess.run(cmd)
    if p.returncode == 0 and os.path.isfile(dest) and os.path.getsize(dest) > 0.9 * SIZE:
        print(f"    ✓ {human(os.path.getsize(dest))}，用时 {time.time() - t0:.0f}s")
        return True
    print(f"    ✗ 失败（rc={p.returncode}）")
    return False


def try_hf(endpoint, tag, dest):
    import subprocess
    url = f"{endpoint}/{REPO}/resolve/main/{FILENAME}"
    print(f"[{tag}] {url}")
    t0 = time.time()
    p = subprocess.run(["curl", "-L", "-C", "-", "--retry", "2", "-o", dest, url])
    if p.returncode == 0 and os.path.isfile(dest) and os.path.getsize(dest) > 0.9 * SIZE:
        print(f"    ✓ {human(os.path.getsize(dest))}，用时 {time.time() - t0:.0f}s")
        return True
    print(f"    ✗ 失败/不完整")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="third_party/weights")
    ap.add_argument("--force", action="store_true", help="重下（默认断点续传）")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    dst = os.path.join(args.dest, FILENAME)

    if os.path.isfile(dst) and not args.force:
        sz = os.path.getsize(dst)
        if sz >= 0.999 * SIZE:
            print(f"权重已存在且完整：{dst}（{human(sz)}）")
            return
        print(f"已存在但可能不完整（{human(sz)} / {human(SIZE)}），尝试续传")

    ok = try_modelscope(dst, args.force)
    if not ok:
        ok = try_hf(HF_MIRROR, "2/3 hf-mirror", dst)
    if not ok:
        ok = try_hf(HF_OFFICIAL, "3/3 huggingface", dst)

    if not ok:
        print("\n✗ 三个源都失败了。可以手动下载后放到该路径：")
        print(f"   {os.path.abspath(dst)}")
        sys.exit(1)

    sz = os.path.getsize(dst)
    print(f"\n完成：{os.path.abspath(dst)}  {human(sz)}")
    if sz < 0.999 * SIZE:
        print(f"⚠ 大小与官方（{human(SIZE)}）不符，可能不完整")
    else:
        print("大小校验通过")


if __name__ == "__main__":
    main()
