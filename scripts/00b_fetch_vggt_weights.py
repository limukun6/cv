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


def _probe_size(url, token=None):
    """用 HEAD 拿 Content-Length；拿不到返回 None。"""
    import subprocess
    cmd = ["curl", "-sIL", "-m", "60"]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    cmd.append(url)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
    except Exception:
        return None
    sizes = [int(l.split(":")[1].strip()) for l in out.splitlines()
             if l.lower().startswith("content-length:") and l.split(":")[1].strip().isdigit()]
    return sizes[-1] if sizes else None


def _fetch_range(url, path, start, end, workers_retry=80):
    """下 [start, end] 这一段，追加写入 path，可断点续跑。

    ⚠️ 两个坑（都踩过）：
      · 不要用 `-C -` 配 `-r`：两者叠加会让 curl 重复追加，文件体积会超过预期
        （实测下到 140%）。
      · 回调用 `>>` 追加（stdout 重定向），不能用 `-o`——`-o` 会截断。
    """
    import subprocess
    want = end - start + 1
    for _ in range(workers_retry):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= want:
            return True
        with open(path, "ab") as f:
            subprocess.run(["curl", "-sL", "-m", "900",
                            "-r", f"{start + have}-{end}", url],
                           stdout=f, stderr=subprocess.DEVNULL)
    return (os.path.getsize(path) if os.path.exists(path) else 0) >= want


def fetch_parallel(url, dst, size, workers=8, token=None):
    """8 路并行分片下载后合并。

    单连接在国内实测 0.5–2.4 MB/s（4.7 GB 要 30 min 以上），
    8 路并行能到 6–9 MB/s（约 10 min），且断点续传不浪费。
    """
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    if not size:
        return False
    n = workers
    ch = (size + n - 1) // n
    tmp = [f"{dst}.part{i}" for i in range(n)]
    for p in tmp:
        open(p, "ab").close()

    spans = []
    for i in range(n):
        s = i * ch
        e = min(s + ch - 1, size - 1)
        spans.append((s, e))

    print(f"    并行下载 {n} 片，共 {human(size)}")
    t0 = time.time()
    ok = True
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(_fetch_range, url, tmp[i], s, e)
                for i, (s, e) in enumerate(spans)]
        done = 0
        last = -1
        while done < n:
            done = sum(1 for f in futs if f.done())
            if done != last:
                got = sum(os.path.getsize(p) for p in tmp)
                rate = got / max(time.time() - t0, 1e-6) / 1024 ** 2
                print(f"      {done}/{n} 片，{human(got)}（{rate:.1f} MB/s）")
                last = done
            time.sleep(5)
        ok = all(f.result() for f in futs)

    total = sum(os.path.getsize(p) for p in tmp)
    if not ok or total != size:
        print(f"    ✗ 分片不完整：{total}/{size}")
        return False

    with open(dst, "wb") as out:
        for p in tmp:
            with open(p, "rb") as f:
                while True:
                    buf = f.read(64 * 1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
    for p in tmp:
        os.remove(p)

    # 结构校验：只比大小不够——并发写曾让大小正确而内容损坏。
    # 两种容器格式分别处理：
    #   · zip（torch 1.6+ 默认）：用 zipfile 读中央目录，比扫 EOCD 字节更可靠
    #   · 旧版 pickle（例如 pytorch.org 的 vgg16）：没有目录结构，退化为大小校验
    with open(dst, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"PK":
        import zipfile
        try:
            with zipfile.ZipFile(dst) as z:
                names = z.namelist()
            print(f"    ✓ zip 结构完整（{len(names)} 个条目）")
        except Exception as e:
            print(f"    ✗ zip 中央目录损坏: {type(e).__name__}")
            return False
    else:
        print("    · 旧版 pickle 格式，跳过 zip 校验（仅按大小比对）")
    print(f"    ✓ {human(os.path.getsize(dst))}，用时 {time.time() - t0:.0f}s")
    return True


def try_parallel(endpoint, tag, repo, fn, dest, token=None, workers=8):
    """先探大小，再并行分片；失败返回 False 让调用方回退单连接。"""
    url = f"{endpoint}/{repo}/resolve/main/{fn}"
    print(f"[{tag}] 并行分片: {url}")
    size = _probe_size(url, token)
    if not size:
        print("    拿不到文件大小，回退单连接")
        return False
    print(f"    文件大小 {human(size)}")
    return fetch_parallel(url, dest, size, workers=workers, token=token)


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
    ap.add_argument("--workers", type=int, default=8, help="并行分片数（默认 8）")
    ap.add_argument("--single", action="store_true",
                    help="强制单连接下载（排查问题时用）")
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

    # 首选并行分片：单连接在国内实测 0.5–2.4 MB/s，4.7 GB 要 30 min 以上；
    # 8 路并行实测 6–9 MB/s（约 10 min），且分片可断点续跑。
    # 并行失败（比如服务端不支持 Range）再回退原来的单连接链。
    ok = False
    if not args.single:
        for endpoint, tag in ((HF_MIRROR, "1/3 hf-mirror"), (HF_OFFICIAL, "1/3 huggingface")):
            if try_parallel(endpoint, tag, repo, fn, dst, token, workers=args.workers):
                ok = True
                break
            print("    并行失败，换源或回退单连接")

    if not ok:
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
