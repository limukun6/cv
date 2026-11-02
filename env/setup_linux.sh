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

# 无卡模式（AutoDL 的 0.1 元/小时模式）也允许跑这个脚本：
# 装包、编译 CUDA 扩展**都不需要 GPU 在线**（nvcc 只要 toolkit 在就能编译）。
# 所以等卡期间可以先把环境准备好，有卡了一键切过去。
SKIP_GPU=${SKIP_GPU:-0}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    if [ "$SKIP_GPU" = "1" ]; then
        echo "  ⚠ 无 GPU（无卡模式）。跳过 GPU 检查，按默认架构 8.9 编译。"
    else
        echo "✗ 找不到 nvidia-smi。"
        echo "  如果这是 AutoDL 的「无卡模式」，用： SKIP_GPU=1 bash env/setup_linux.sh"
        echo "  如果这是普通模式，说明镜像不对（没有驱动）。"
        exit 1
    fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "显存: ${VRAM_MB} MiB"
fi

# 计算显卡的计算能力，用来决定编译架构
CAP=""
if command -v nvidia-smi >/dev/null 2>&1; then
    CAP=$(python3 - <<'PY' 2>/dev/null || echo ""
import subprocess
out = subprocess.run(["nvidia-smi","--query-gpu=compute_cap","--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
print(out.splitlines()[0] if out else "")
PY
)
    if [ -z "$CAP" ]; then
        CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
    fi
fi
# 无卡模式下拿不到算力，默认按 8.9（4090/4090D/L20 都是 8.9）编译。
# 万一后来换到别的架构，重跑一次脚本加 ARCH=xx 覆盖即可。
CAP=${CAP:-${ARCH:-8.9}}
echo "计算能力: ${CAP}（无卡模式下按默认值 8.9）"

# ★ nvcc 是硬性要求：3DGS 的三个扩展必须在构建时编译。
#   租来的镜像如果只装了 CUDA **runtime**（没有 toolkit），就会在编译阶段才失败，
#   白等几十分钟。所以在这里提前拦。
if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc: $(nvcc --version | grep -o 'release [0-9.]*' | head -1)"
else
    echo "✗ 找不到 nvcc —— 这个镜像只有 CUDA runtime，没有编译器。"
    echo "  3DGS 的三个 CUDA 扩展无法编译。两个选择："
    echo "    1) 换一个带 CUDA toolkit（devel）的镜像（推荐，见 docs/IMAGE_CHOICE.md）"
    echo "    2) 手动装 toolkit： apt-get install -y cuda-toolkit-12-8"
    if [ "${ALLOW_NO_NVCC:-0}" != "1" ]; then
        exit 1
    fi
    echo "  （ALLOW_NO_NVCC=1，继续，但编译阶段可能失败）"
fi

# ---------- 2. venv ----------
echo
echo "[2/7] 建 Python 虚拟环境"
PYBIN=${PYBIN:-python3}
$PYBIN --version

# 租来的镜像经常缺 python3-venv（Ubuntu 把 ensurepip 拆出去了）。
# 这里做三级回退，避免第一步就卡住。
if ! $PYBIN -m venv .venv 2>/dev/null; then
    echo "  venv 不可用，尝试安装 python3-venv"
    if command -v apt-get >/dev/null 2>&1; then
        (sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv) >/dev/null 2>&1 || true
    fi
fi
if [ ! -d .venv ]; then
    $PYBIN -m venv .venv 2>/dev/null || true
fi
if [ ! -d .venv ]; then
    echo "  仍失败，尝试 virtualenv"
    $PYBIN -m pip install -q --user virtualenv 2>/dev/null || true
    $PYBIN -m virtualenv .venv 2>/dev/null || true
fi
if [ ! -d .venv ]; then
    echo "✗ 无法创建虚拟环境。请手动装 python3-venv，或改用 conda："
    echo "    conda create -n p1 python=3.11 -y && conda activate p1"
    echo "  之后把 PYBIN 指向该环境再跑本脚本：PYBIN=\$(which python) bash env/setup_linux.sh"
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip wheel setuptools

# ---------- 3. torch ----------
echo
echo "[3/7] PyTorch"

# ★ 核心原则：**torch 的 CUDA 版本要和镜像里的 nvcc 对齐**
#   3DGS 的三个扩展要用本机 nvcc 编译，而它们链接的是 torch 自带的 CUDA 运行时。
#   两者主版本必须一致，次版本不一致会告警（通常能用，但不干净）。
#   所以：
#     · 镜像自带 torch 且 nvcc 主版本一致 → 直接复用（省 2.5 GB 下载）
#     · 没 torch / 版本太老 → 按 nvcc 的次版本装对应的 cuXXX 轮子
NVCC_REL=$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | awk '{print $2}')
NVCC_MAJOR=$(echo "${NVCC_REL:-12.4}" | awk -F. '{print $1}')
CU_TAG=$(echo "${NVCC_REL:-12.4}" | awk -F. '{print $1$2}')     # 12.4 -> 124
echo "  镜像 nvcc: ${NVCC_REL:-未知}  → CUDA 轮子标签 cu${CU_TAG}"

SYS_TORCH=$($PYBIN -c "import torch;print(torch.__version__)" 2>/dev/null || echo "none")
SYS_CUDA=$($PYBIN -c "import torch;print(torch.version.cuda or '')" 2>/dev/null || echo "")
echo "  镜像自带 torch: $SYS_TORCH (cuda ${SYS_CUDA:-无})"

REUSE_TORCH=${REUSE_TORCH:-auto}
if [ "$REUSE_TORCH" = "auto" ]; then
    REUSE_TORCH=0
    if [ "$SYS_TORCH" != "none" ]; then
        TMINOR=$(echo "$SYS_TORCH" | sed 's/[^0-9.].*//' | awk -F. '{print $1"."$2}')
        TMAJ=$(echo "${SYS_CUDA}" | awk -F. '{print $1}')
        # torch >= 2.4，且 CUDA 主版本和 nvcc 一致，就复用
        if [ "$(printf '%s\n2.4\n' "$TMINOR" | sort -V | head -1)" = "2.4" ] && \
           [ "${TMAJ:-x}" = "${NVCC_MAJOR}" ]; then
            REUSE_TORCH=1
        fi
    fi
    echo "  → 自动判定 REUSE_TORCH=$REUSE_TORCH"
fi

if [ "$REUSE_TORCH" = "1" ]; then
    echo "  复用镜像自带的 torch ${SYS_TORCH}+cu${SYS_CUDA}（省 2.5 GB 下载，且与 nvcc ${NVCC_REL} 对齐）"
    # venv 必须带 --system-site-packages 才看得见镜像里的 torch
    if [ ! -f .venv/.reuse_torch ]; then
        rm -rf .venv
        $PYBIN -m venv --system-site-packages .venv
        # shellcheck disable=SC1091
        source .venv/bin/activate
        touch .venv/.reuse_torch
    fi
    python - <<'PY'
import torch
print("  torch", torch.__version__, "cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("  arch list:", torch.cuda.get_arch_list())
    x = torch.randn(512, 512, device="cuda"); (x @ x).sum().item()
    print("  KERNEL TEST: PASS")
else:
    print("  ⚠ 无 GPU（无卡模式）→ 跳过 kernel test，装包和编译仍可进行")
PY
else
    if [ -f .venv/.reuse_torch ]; then
        rm -rf .venv
        $PYBIN -m venv .venv
        # shellcheck disable=SC1091
        source .venv/bin/activate
    fi
    echo "  安装 torch（cu${CU_TAG}，与镜像 nvcc 对齐）"
    pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/cu${CU_TAG}" || \
    pip install -q torch torchvision
    python - <<'PY'
import torch
print("  torch", torch.__version__, "cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("  arch list:", torch.cuda.get_arch_list())
    x = torch.randn(512, 512, device="cuda"); (x @ x).sum().item()
    print("  KERNEL TEST: PASS")
else:
    print("  ⚠ 无 GPU（无卡模式）→ 跳过 kernel test，装包和编译仍可进行")
PY
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
if torch.cuda.is_available():
    print("arch list:", torch.cuda.get_arch_list())
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    print("KERNEL TEST: PASS", y)
else:
    print("⚠ 无 GPU（无卡模式）→ 跳过 kernel test")
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
