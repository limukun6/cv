"""
00b_fetch_vggt_weights.py — 下载 VGGT / VGGT-Ω 权重，多源自动回退。

为什么需要这个脚本
----------------
1. **huggingface.co 在国内经常连不上**（本机实测直接 `HTTP 000`）
2. `hf-mirror.com` 能连，但速度极不稳定（实测 79 KB/s ~ 300 KB/s）
3. **ModelScope（阿里魔搭）镜像同一个仓库，实测 5–15 MB/s**（快 60 倍）

所以顺序是：ModelScope → hf-mirror → huggingface（带 token）→ huggingface（匿名）。

关于 VGGT-Ω 的授权
-------------------
VGGT-Ω 在 HuggingFace 上是 `gated: manual`（需要申请授权），但 **ModelScope
上有公开镜像**，实测 12 MB/s 可直连。如果用了镜像，报告里应当注明
"权重来自 ModelScope 镜像"，并引用官方仓库与论文。

用法:
    # VGGT-1B（默认）
    python scripts/00b_fetch_vggt_weights.py

    # VGGT-Ω 的 benchmark 版本
    python scripts/00b_fetch_vggt_weights.py --preset omega-416

    # 任意仓库/文件
    python scripts/00b_fetch_vggt_weights.py --repo facebook/VGGT-Omega \
        --file vggt_omega_1b_512.pt --dest third_party/weights/vggt_omega

    # 看看镜像上有哪些文件
    python scripts/00b_fetch_vggt_weights.py --list --repo facebook/VGGT-Omega
"""
import argparse
import json
import os
import subprocess
import sys
import time

MS_API = "https://www.modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={fn}"
MS_FILES = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master"
HF_MIRROR = "https://hf-mirror.com"
HF_OFFICIAL = "https://huggingface.co"

PRESETS = {
    "vggt1b":    ("facebook/VGGT-1B",     "model.pt",                        "third_party/weights",             5026874952),
    "omega-416": ("facebook/VGGT-Omega",  "vggt_omega_1b_416_reproduce.pt",  "third_party/weights/vggt_omega",  None),
    "omega-512": ("facebook/VGGT-Omega",  "vggt_omega_1b_512.pt",            "third_party/weights/vggt_omega",  None),
    "omega-256": ("facebook/VGGT-Omega",  "vggt_omega_1b_256_text.pt",       "third_party/weights/vggt_omega",  None),
}


def human(n):
    return f"{n / 1024 ** 3:.2f} GB"


def ms_list(repo):
    """列出 ModelScope 镜像上的文件及大小。"""
    import urllib.request
    try:
        with urllib.request.urlopen(MS_FILES.format(repo=repo), timeout=30) as r:
            d = json.load(r)
    except Exception as e:
        print(f"查询失败: {type(e).__name__}: {e}")
        return
    files = d.get("Data", {}).get("Files") or d.get("Data") or []
    print(f"ModelScope 上 {repo} 的文件：")
    for f in files:
        sz = f.get("Size")
        print(f"  {f.get('Path',''):<42} {human(sz) if sz else '?'}")


def try_modelscope(repo, fn, dest, force=False):
    import subprocess
    url = MS_API.format(repo=repo, fn=fn)
    print(f"[1/3] ModelScope: {url}")
    cmd = ["curl", "-L", "--retry", "3", "--retry-delay", "3", "-o", dest, url]
    if not force:
        # curl 的断点续传必须是 `-C -`（一个短横），单独写 -C 会吃掉后面的参数
        cmd[1:1] = ["-C", "-"]
    t0 = time.time()
    p = subprocess.run(cmd)
    if p.returncode == 0 and os.path.isfile(dest) and os.path.getsize(dest) > 100 * 1024 ** 2:
        dt = time.time() - t0
        print(f"    ✓ {human(os.path.getsize(dest))}，用时 {dt:.0f}s "
              f"({os.path.getsize(dest)/1024**2/max(dt,1e-9):.1f} MB/s)")
        return True
    print(f"    ✗ 失败（rc={p.returncode}）")
    return False


def try_hf(endpoint, tag, repo, fn, dest, token=None):
    import subprocess
    url = f"{endpoint}/{repo}/resolve/main/{fn}"
    print(f"[{tag}] {url}")
    cmd = ["curl", "-L", "-C", "-", "--retry", "2", "-o", dest]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    cmd.append(url)
    t0 = time.time()
    p = subprocess.run(cmd)
    if p.returncode == 0 and os.path.isfile(dest) and os.path.getsize(dest) > 100 * 1024 ** 2:
        print(f"    ✓ {human(os.path.getsize(dest))}，用时 {time.time() - t0:.0f}s")
        return True
    print("    ✗ 失败/不完整")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="vggt1b")
    ap.add_argument("--repo", default=None, help="覆盖 preset 的仓库")
    ap.add_argument("--file", default=None, help="覆盖 preset 的文件名")
    ap.add_argument("--dest", default=None, help="覆盖 preset 的目标目录")
    ap.add_argument("--force", action="store_true", help="重下（默认断点续传）")
    ap.add_argument("--list", action="store_true", help="只列出镜像上的文件")
    args = ap.parse_args()

    repo, fn, dest_dir, expect = PRESETS[args.preset]
    repo = args.repo or repo
    fn = args.file or fn
    dest_dir = args.dest or dest_dir
    if args.dest:
        dest_dir = args.dest

    if args.list:
        ms_list(repo)
        return

    os.makedirs(dest_dir, exist_ok=True)
    dst = os.path.join(dest_dir, fn)

    if os.path.isfile(dst) and not args.force:
        sz = os.path.getsize(dst)
        if (expect and sz >= 0.999 * expect) or (not expect and sz > 100 * 1024 ** 2):
            print(f"权重已存在：{dst}（{human(sz)}）")
            return
        print(f"已存在但可能不完整（{human(sz)}），尝试续传")

    token = os.environ.get("HF_TOKEN")
    ok = try_modelscope(repo, fn, dst, args.force)
    if not ok:
        ok = try_hf(HF_MIRROR, "2/3 hf-mirror", repo, fn, dst, token)
    if not ok:
        ok = try_hf(HF_OFFICIAL, "3/3 huggingface", repo, fn, dst, token)

    if not ok:
        print("\n✗ 三个源都失败了。可以手动下载后放到该路径：")
        print(f"   {os.path.abspath(dst)}")
        if not token:
            print("   提示：gated 模型需要 HF token，设置环境变量 HF_TOKEN=hf_xxx")
        sys.exit(1)

    sz = os.path.getsize(dst)
    print(f"\n完成：{os.path.abspath(dst)}  {human(sz)}")
    if expect and sz < 0.999 * expect:
        print(f"⚠ 大小与官方（{human(expect)}）不符，可能不完整")
    else:
        print("大小校验通过")


if __name__ == "__main__":
    main()
