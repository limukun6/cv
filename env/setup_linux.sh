#!/usr/bin/env bash
# =============================================================
# setup_linux.sh — 在租来的 Linux + NVIDIA 卡上，一条命令装完 Part 1 需要的环境
#
# 用法（在仓库根目录）：
#     bash env/setup_linux.sh
#
# 它会做：
#   1. 检查 GPU / 驱动 / nvcc
#   2. 建 venv（不污染系统 python）
#   3. 装 torch（CUDA 版本按显卡自动选；4090=sm_89 用 cu128 即可）
#   4. 编译 3DGS 的三个 CUDA 扩展
#   5. 装 VGGT 依赖
#   6. 装 colmap（conda 或 apt）
#   7. 跑一遍冒烟测试，确认能真的在 GPU 上算出东西
#
# 为什么要在租卡之前写好：租用时间最贵的浪费方式，是在机器上调试自己的脚本。
# =============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "================ 仓库根目录: $ROOT ================"

# ---------- 1. 硬件与环境检查 ----------
echo
echo "[1/7] 检查 GPU / 驱动"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "✗ 找不到 nvidia-smi。这不像一台有 NVIDIA 驱动的机器。"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "显存: ${VRAM_MB} MiB"

# 计算显卡的计算能力，用来决定编译架构
CAP=$(python3 - <<'PY' 2>/dev/null || echo ""
import subprocess, re
out = subprocess.run(["nvidia-smi","--query-gpu=compute_cap","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print(out.splitlines()[0] if out else "")
PY
)
if [ -z "$CAP" ]; then
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
fi
echo "计算能力: ${CAP:-未知}"

# ---------- 2. venv ----------
echo
echo "[2/7] 建 Python 虚拟环境"
PYBIN=${PYBIN:-python3}
$PYBIN -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip wheel setuptools

# ---------- 3. torch ----------
echo
echo "[3/7] 装 PyTorch"
# 4090/3090 等都是 sm_89/sm_86，cu128 轮子覆盖；cu128 也覆盖到 sm_120，
# 所以将来换 Blackwell 卡也能用同一套命令。
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128 || \
pip install -q torch torchvision

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("arch list:", torch.cuda.get_arch_list())
assert torch.cuda.is_available(), "CUDA 不可用"
x = torch.randn(1024, 1024, device="cuda")
y = (x @ x).sum().item()
print("KERNEL TEST: PASS", y)
PY

# ---------- 4. 3DGS ----------
echo
echo "[4/7] 编译 3DGS 的 CUDA 扩展"
if [ ! -d third_party/gaussian-splatting ]; then
    git clone --recursive --depth 1 \
        https://github.com/graphdeco-inria/gaussian-splatting.git \
        third_party/gaussian-splatting
fi

# 架构列表：Linux 上 torch 的 cpp_extension 会用 TORCH_CUDA_ARCH_LIST，不设的话
# 它会枚举机器上所有可见卡，通常也没问题；显式指定更快也更容易复现。
if [ -n "${CAP:-}" ]; then
    export TORCH_CUDA_ARCH_LIST="$CAP"
    echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
fi

pip install -q ninja plyfile tqdm opencv-python-headless scipy matplotlib
for ext in diff-gaussian-rasterization simple-knn fused-ssim; do
    echo "  building $ext ..."
    pip install -q --no-build-isolation "third_party/gaussian-splatting/submodules/$ext"
done

python - <<'PY'
import torch, fused_ssim
a = torch.rand(1, 3, 128, 128, device="cuda")
print("fused_ssim:", float(fused_ssim.fused_ssim(a, a)))
import diff_gaussian_rasterization, simple_knn
print("3DGS extensions OK")
PY

# ---------- 5. VGGT ----------
echo
echo "[5/7] 装 VGGT"
if [ ! -d third_party/vggt ]; then
    git clone --depth 1 https://github.com/facebookresearch/vggt.git third_party/vggt
fi
pip install -q einops safetensors huggingface_hub trimesh pycolmap

# ---------- 6. colmap ----------
echo
echo "[6/7] 装 COLMAP"
if command -v colmap >/dev/null 2>&1; then
    echo "  已有: $(command -v colmap)"
else
    echo "  系统里没有 colmap，尝试用 conda 装（不动系统 python）"
    if command -v conda >/dev/null 2>&1; then
        conda create -y -n colmap -c conda-forge colmap >/dev/null 2>&1 || true
        COLMAP_BIN="$(conda run -n colmap which colmap 2>/dev/null || true)"
        if [ -n "$COLMAP_BIN" ]; then
            echo "  colmap: $COLMAP_BIN"
            echo "  ⚠ 请把它写进环境变量： export COLMAP_BIN=$COLMAP_BIN"
        fi
    else
        echo "  ✗ 没有 conda。请手动装 colmap：apt-get install -y colmap 或编译源码"
    fi
fi

# ---------- 7. 冒烟测试 ----------
echo
echo "[7/7] 冒烟测试"
python - <<'PY'
import os, shutil, torch
print("python ok")
print("colmap:", os.environ.get("COLMAP_BIN") or shutil.which("colmap") or "未找到（COLMAP 阶段会失败）")
print("VGGT 代码:", os.path.isdir("third_party/vggt"))
print("3DGS 代码:", os.path.isdir("third_party/gaussian-splatting"))
PY

cat <<'EOF'

================ 安装完成 ================
接下来：
  1. 上传数据包（见 docs/P1_PLAN_GPU.md 第五节）：
         tar xzf p1_bundle.tar.gz          # 解到仓库根，得到 data/
  2. 下载 VGGT 权重：
         python scripts/00b_fetch_vggt_weights.py
  3. 跑整个 Part 1 实验矩阵：
         python scripts/35_run_p1_matrix.py --stage all

注意：
  · 如果 COLMAP 不是系统装的，记得 export COLMAP_BIN=$(which colmap)
  · 每阶段跑完，把 results/results.csv 和新增的 docs/figures/*.png 备份出去。
EOF
