# Windows + RTX 50 系（sm_120）环境搭建实录

> 这份文档记录的是**我们这台机器上真实跑通的配置**，不是抄来的通用教程。
> 换机器时请按这里的顺序走，可以避开我们踩过的 5 个坑。

## 机器

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 **Laptop**，8 GB VRAM，**compute capability 12.0 (sm_120)**, Blackwell |
| 驱动 | 591.74 |
| CPU / RAM | Intel Core Ultra 9 285H / 32 GB |
| OS | Windows |
| VS | Visual Studio 2022 Community，MSVC 14.44 |
| conda | `D:\conda\Anaconda`（注意：PATH 里可能还残留旧路径 `D:\cs31n\Anaconda`） |
| Python | 系统 3.14（**不能用**），conda 里建 3.11 |

## 最终可用的版本组合

```
Python        3.11.18     (conda env "gs")
torch         2.8.0+cu129
torchvision   0.23.0+cu129
nvcc          12.9.86     (conda env "cuda129", 仅提供编译器)
MSVC          14.44
CUDA arch     sm_120  (TORCH_CUDA_ARCH_LIST=12.0)
```

## 五个坑（按踩的顺序）

### 坑 1：`torch.cuda.is_available()` 会骗人

原来的 `pytorch_learning` 环境里是 `torch 2.13.0+cu126`，`is_available()` 返回 **True**，
但真正做一次矩阵乘就崩：

```
CUDA error: no kernel image is available for execution on the device
```

原因：cu126 的 wheel 只编了 `sm_50 … sm_90`，**没有 sm_120**。判断驱动可用 ≠ 有该架构的内核。

**教训**：环境装完必须做一次真运算（矩阵乘），不能只看 `is_available()`。

**检查命令**
```python
import torch
print(torch.cuda.get_arch_list())          # 必须包含 'sm_120'
torch.randn(2048, 2048, device='cuda') @ torch.randn(2048, 2048, device='cuda')
```

### 坑 2：torch 版本与 Python 版本的组合被砍

`torch==2.13.0` 在 Python 3.11 上**没有 cu129 轮子**（只有 2.8.0 / 2.9.0）。
且 cu129 索引里 `torchvision` 只有 `0.23.0`，对应 torch 2.8（不是 2.9）。

**结论**：`torch 2.8.0 + torchvision 0.23.0 + cu129 + py3.11`。

**查可用版本的命令**
```powershell
curl.exe -s https://download.pytorch.org/whl/cu129/torch/ | Select-String "cp311-cp311-win_amd64"
curl.exe -s https://download.pytorch.org/whl/cu129/torchvision/ | Select-String "cp311-cp311-win_amd64"
```

### 坑 3：`--index-url` 会把 PyPI 整个替换掉

直接
```powershell
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```
会报 `Could not find a version that satisfies the requirement filelock` ——
因为 pytorch 索引里没有 torch 的依赖（filelock / sympy / networkx / jinja2 / fsspec）。

**正确做法：两步**
```powershell
# 1) 先用国内镜像装依赖
pip install -i https://mirrors.aliyun.com/pypi/simple `
    filelock typing_extensions sympy networkx jinja2 fsspec numpy pillow setuptools
# 2) 再用 --no-deps 装 torch（不碰依赖）
pip install --no-deps --index-url https://download.pytorch.org/whl/cu129 `
    torch==2.8.0 torchvision==0.23.0
```

> 镜像速度实测：阿里云最快，清华次之，官方 pypi.org 最慢（12 s 响应）。

### 坑 4：nvcc 主版本必须等于 torch 的 CUDA 版本

用 conda-forge 装 `cuda-nvcc` 默认给了 **13.4**，而 torch 是 cu129。
主版本不一致编出来的扩展会崩。必须锁 12.9：

```powershell
conda install -y -n cuda129 --override-channels `
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge `
  "cuda-nvcc=12.9" "cuda-cudart-dev=12.9" "cuda-crt=12.9"
```

装完布局：
```
D:\conda\Anaconda\envs\cuda129\Library\bin\nvcc.exe
D:\conda\Anaconda\envs\cuda129\Library\include\cuda_runtime.h
D:\conda\Anaconda\envs\cuda129\Library\lib\cudart.lib
```
→ 因此 **`CUDA_HOME` 要指到 `...\cuda129\Library`**（不是 env 根目录）。

> 为什么单独建 `cuda129` 环境而不是装进 `gs`：conda-forge 的 CUDA 包和 `gs` 里的
> python/torch 依赖容易打架。编译器放独立环境，只通过 `CUDA_HOME` 引用，最干净。

### 坑 5：两个 Windows 专有错误

**(a) `cl.exe` 找不到** → 必须在 VS 的开发者环境里编译
（`scripts/build_cuda_ext.bat` 已经封装好，它会先 call `vcvars64.bat`）。

**(b) `LINK : fatal error LNK1181: 无法打开输入文件 "cudart.lib"`**
torch 在 Windows 上只去 `CUDA_HOME\lib\x64` 找库，但 conda 把 `cudart.lib` 放在
`CUDA_HOME\lib`。一行修复：

```powershell
Copy-Item "D:\conda\Anaconda\envs\cuda129\Library\lib\*.lib" `
          "D:\conda\Anaconda\envs\cuda129\Library\lib\x64\" -Force
```

**另外**：`.bat` 文件里**不要写中文**。UTF-8 中文注释在 cmd 的 GBK 代码页下会把
后续命令解析坏，症状是环境变量莫名不生效。

## 从零复现（完整命令）

```powershell
$conda = "D:\conda\Anaconda\condabin\conda.bat"
$gs    = "D:\conda\Anaconda\envs\gs\python.exe"

# 1) 运行环境
& $conda create -y -n gs python=3.11

# 2) torch 依赖（国内镜像）
& $gs -m pip install -i https://mirrors.aliyun.com/pypi/simple `
    filelock typing_extensions sympy networkx jinja2 fsspec numpy pillow setuptools `
    opencv-python-headless scipy plyfile tqdm ninja

# 3) torch（--no-deps）
& $gs -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu129 `
    torch==2.8.0 torchvision==0.23.0

# 4) 验证 sm_120
& $gs -c "import torch; print(torch.cuda.get_arch_list()); torch.randn(64,64,device='cuda')@torch.randn(64,64,device='cuda')"

# 5) 编译器环境（独立 env）
& $conda create -y -n cuda129 --override-channels `
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge `
  cuda-nvcc cuda-cudart-dev
& $conda install -y -n cuda129 --override-channels `
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge `
  "cuda-nvcc=12.9" "cuda-cudart-dev=12.9" "cuda-crt=12.9"
Copy-Item "D:\conda\Anaconda\envs\cuda129\Library\lib\*.lib" `
          "D:\conda\Anaconda\envs\cuda129\Library\lib\x64\" -Force

# 6) 第三方代码
git clone --recursive --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting

# 7) 编译三个扩展
cmd /c "scripts\build_cuda_ext.bat third_party\gaussian-splatting\submodules\diff-gaussian-rasterization"
cmd /c "scripts\build_cuda_ext.bat third_party\gaussian-splatting\submodules\simple-knn"
cmd /c "scripts\build_cuda_ext.bat third_party\gaussian-splatting\submodules\fused-ssim"

# 8) 冒烟测试
& $gs -c "import torch,fused_ssim; a=torch.rand(1,3,128,128,device='cuda'); print(fused_ssim.fused_ssim(a,a))"
```

## 耗时记录（实测）

| 步骤 | 耗时 |
|---|---|
| 建 gs 环境 | ~2 min |
| 装 torch + 依赖（3.5 GB 下载） | ~12 min |
| 建 cuda129 + 装 12.9 | ~10 min |
| 编译 diff-gaussian-rasterization | ~3 min |
| 编译 simple-knn / fused-ssim | 各 ~1 min |
| **合计** | **约 30 分钟**（踩坑排查另计约 2 小时） |

## 仍未解决的问题

- **8 GB 显存**：3DGS 训练必须降分辨率（`-r 2` 或 `-r 4`）；VGGT 能吃多少帧还没实测，
  这是 Part 1 的最大未知数。
- **conda 的 PATH**：`D:\cs31n\Anaconda` 已改名成 `D:\conda\Anaconda`，但 PATH 没更新，
  所以命令行里直接敲 `conda` 会失败。用绝对路径 `D:\conda\Anaconda\condabin\conda.bat`，
  或者跑一次 `conda init powershell` 修好。
