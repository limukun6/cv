# 镜像怎么选（租卡时对着这张表选）

> 一句话：**Ubuntu 22.04 + CUDA 12.x + 带 nvcc 的 PyTorch 镜像**。
> 判断标准只有一条硬性要求：**镜像里必须有 `nvcc`**。

---

## 1. 硬性要求

| 项目 | 要求 | 为什么 |
|---|---|---|
| **CUDA toolkit（含 nvcc）** | **必须有，版本 ≥ 12.1** | 3DGS 的三个 CUDA 扩展必须在构建时编译。只有 runtime 的镜像会在编译阶段失败 |
| 显卡 | ≥ 24 GB（4090 / 3090 / A5000 等） | VGGT 需要；3DGS 在 24 GB 上可以跑原生分辨率 |
| 系统 | Ubuntu 20.04 / 22.04 | 22.04 最稳。18.04 的 glibc 太老，新版 wheel 容易不兼容 |
| Python | 3.10 / 3.11 | 我们本机验证过 3.11 |
| 磁盘 | **≥ 50 GB** | 数据 0.85 GB + VGGT 权重 4.7 GB + 12 次实验的 checkpoint（每个约 100–400 MB）+ 渲染图 |
| 内存 | ≥ 16 GB | COLMAP 阶段用 |

> **不要用 Windows 镜像。** 我们本机那套 `build_cuda_ext.bat` 是给 Windows 救急的，
> Linux 上编译简单得多（`pip install ./submodule` 就行）。

---

## 2. 推荐的镜像写法

各平台命名不一样，但找**同时满足这三条**的那个就行：

```
✓ 名字里带 PyTorch（省得自己装）
✓ 名字里带 CUDA 12.x（12.1 / 12.4 / 12.8 都可以）
✓ 名字里带 devel 或 "toolkit"（= 有 nvcc）
```

具体形态大致是：

- `PyTorch 2.x.x / Python 3.11 / CUDA 12.4 / Ubuntu 22.04`
- `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`（如果你用 Docker）
- `nvidia/cuda:12.8-devel-ubuntu22.04` + 自己装 torch

**镜像里预装的 torch 版本不用管**——我们的 `setup_linux.sh` 会建独立 venv，
按 cu128 重新装一份，不依赖镜像里的环境。

---

## 3. 明确避开的镜像

| 避开 | 原因 |
|---|---|
| CUDA **11.x** | 4090（sm_89）需要 ≥11.8，且我们的 torch cu128 轮子要更新 |
| 只有 runtime 没有 devel | **没有 nvcc，编译必失败**（最常见踩坑） |
| Ubuntu 18.04 | glibc 老，新版 wheel 容易报 `GLIBC_2.28 not found` |
| 纯净 base 镜像（无 torch 无 conda） | 能用，但要多花 10 分钟装，不划算 |
| Windows 镜像 | 编译路径完全不同，我们没为它做适配 |

---

## 4. ⭐ 开机后先跑这 5 条命令（1 分钟，能省几小时）

**在装任何东西之前**，确认镜像合格：

```bash
nvidia-smi                      # 1. 有卡吗？显存多少？
nvcc --version                  # 2. ★ 有编译器吗？（最关键）
python3 --version               # 3. Python 3.10/3.11？
df -h /                         # 4. 磁盘 ≥ 50 GB？
free -g                         # 5. 内存 ≥ 16 GB？
```

判读：

| 输出 | 结论 |
|---|---|
| `nvcc --version` 报 command not found | 🔴 **换镜像**。别试图在 runtime 镜像里补装 toolkit，浪费时间 |
| 显存 < 24 GB | 🟡 能跑 3DGS，但 VGGT 帧数会受限，对比会失真 |
| 磁盘 < 30 GB | 🔴 换机器，装不下权重和 checkpoint |
| 其他都 OK | ✅ 可以开始：`git clone` + `bash env/setup_linux.sh` |

---

## 5. 如果平台只提供系统镜像（自己装环境）

按这个顺序装，就是 `setup_linux.sh` 在做的事：

```bash
# 1. CUDA toolkit（必须，含 nvcc）
#    Ubuntu 22.04 上最简单：
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-8

# 2. 其余交给脚本
git clone https://github.com/limukun6/cv.git && cd cv
bash env/setup_linux.sh
```

---

## 6. 一个省钱的细节

**先按小时租、验证通了再续。** 理由：

- 我们第一个 30 分钟做的事是"验证镜像 + 装环境 + 冒烟测试"，这些**不需要一直占着卡**
- 如果镜像选错（没有 nvcc），你会损失的不是钱，是**在线排查的时间**

**推荐流程**：租最小单位（1 小时）→ 跑那 5 条命令 → 通了就续到 12 小时。

---

## 7. 我们的脚本对新镜像的鲁棒性（已加固）

| 可能的镜像差异 | 脚本怎么处理 |
|---|---|
| 缺 `python3-venv` | 自动 `apt-get install python3-venv`，再退到 `virtualenv` |
| 没有 `nvcc` | **开跑前就报错并给方案**，不会等到编译阶段才失败 |
| 预装了别的 torch | 独立 venv，不污染、也不被污染 |
| 没有 `conda` | COLMAP 优先用系统的；没有则尝试 conda；都没有会明确提示 `COLMAP_BIN` |
| 不同显卡架构 | 自动读算力写 `TORCH_CUDA_ARCH_LIST` |
