# 3D Gaussian Splatting · 指定部分精读指南

> **给谁看**：`notes/READING_P1.md` 里"🔴 精读 1：3DGS"的负责人（Part 1 初始化方式对比）。
> **读哪部分**：清单指定的 §3 全部 + §4。⚠️ **但清单的节号和论文的节号对不上**，见 §0.2，别一打开 PDF 就懵。
> **需要多久**：3–4 小时。本文档第一部分 30 min，第二部分 2 h，第四部分（坑）40 min，自测 20 min。
> **论文**：[arXiv:2308.04079](https://arxiv.org/abs/2308.04079) · SIGGRAPH 2023 / ACM TOG 42(4) · [项目页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) · [官方代码](https://github.com/graphdeco-inria/gaussian-splatting)（本仓库已 clone 到 `third_party/gaussian-splatting/`）
> **信息来源**：论文正文 **§1–§6 + §7.1 / §7.2（含 Table 1、Table 2、Table 3）+ §7.3 的两组消融 + §7.4 逐字读到**（arXiv HTML 与 ar5iv 双源交叉，Table 数据逐格抄录）；**没读到的部分在 §0.3 逐条列出**。**官方代码直接读了本仓库里的实现**（`arguments/__init__.py`、`scene/gaussian_model.py`、`train.py`、`utils/camera_utils.py`、`cuda_rasterizer/*`），所以文里"论文说 A、代码是 B"的地方都是真对照过的，不是猜的。

**一句话定位**：这份文档不教你"3DGS 是什么"，而是教你"**3DGS 对初始化有多敏感、敏感在哪几个旋钮上**"——因为那正是 Part 1 的实验变量。读完你应该能指着代码说出：换一套初始化点云/位姿，会从哪一行开始影响结果。

### 清单里那 5 个"读完必须能回答"，答案在哪

| 阅读清单的问题 | 本文档位置 | 一句话答案 |
|---|---|---|
| Q1 一个高斯有几个参数 | **2.1.2** | 59 个：位置 3 + 缩放 3 + 旋转 4 + 不透明度 1 + 球谐 48 |
| Q2 投影后为什么还是椭圆？为什么是前提 | **2.1.4**（前置 **1.3**） | 用均值处的雅可比做局部仿射近似；只有椭圆才能 O(1) 做完剔除、覆盖、混合 |
| Q3 clone/split 各自条件？为什么不统一 | **2.3.2** | 都是"梯度大"，再按"高斯大小"分：缺体积→clone（膨胀），形状不对→split（体积守恒） |
| Q4 opacity reset 解决什么？不做会怎样 | **2.3.3** | 清掉相机附近的浮空雾；不做的话 floaters 会自我繁殖、密度控制失控 |
| Q5 损失是什么？λ 多少？为什么要 DSSIM | **2.2.3** | $\mathcal L=(1-\lambda)\mathcal L_1+\lambda\mathcal L_{\rm D\text{-}SSIM}$，λ=0.2；L1 对结构不敏感、平坦区梯度弱，DSSIM 补这一半 |

---

## 目录

- **0. 开工前 5 分钟**：版本核对 + 节号陷阱（⚠️ 清单的 §3/§4 ≠ 论文的 §3/§4）
- **第一部分 · 打地基**（不含论文内容）：成像公式 / 高斯 / 为什么投影后仍是高斯 / 球谐 / SfM 点云
- **第二部分 · 逐节精读**
  - 2.1 §4 可微 3D 高斯（= 清单 §3.1）
  - 2.2 §5.1 优化（= 清单 §3.2 前半）
  - 2.3 §5.2 自适应密度控制（= 清单 §3.2 后半）
  - 2.4 §6 光栅化（= 清单 §4 前半）
  - 2.5 §7.1 实现细节（= 清单 §4 后半）
  - 2.6 §7.3 消融：Initialization from SfM
  - 2.7 Table 1/2：P1 要引的数字
  - 2.8 §7.4 Limitations：论文自己承认的四个弱点
- **第三部分 · 压缩伪代码**（约 40 行讲完整篇）
- **第四部分 · 论文没明说、但决定成败的 9 件事**（论文 vs 代码对照）
- **第五部分 · 自测 13 问**（答案折叠）
- **第六部分 · 速查卡 + 三分钟口头版 + 和 Part 1 的接口**

---

## 0. 开工前 5 分钟：版本核对 + 节号陷阱

### 0.1 版本核对（已替你做）

| 项 | 值 |
|---|---|
| arXiv 号 | 2308.04079 |
| 版本 | **只有 v1**（2023-08-08 提交），arXiv 上没有 v2，所以不存在"v1 说法 vs 最新版说法"的差异问题 |
| 正式发表 | SIGGRAPH 2023 → ACM TOG 42(4), July 2023, DOI 10.1145/3592433。**arXiv v1 就是 TOG 版排版**（页面顶部有 `Journal: TOG685 Volume: 42(4)`），所以节号与正式版一致 |
| 许可 | CC BY 4.0（可以截图进你的报告，注明出处即可） |
| 官方代码 | `graphdeco-inria/gaussian-splatting`。⚠️ 注意：**你现在 clone 到的版本比论文新**，多了 exposure 补偿、depth regularization、`--optimizer_type sparse_adam`、`antialiasing` 开关等论文里没有的东西。读代码时以 `arguments/__init__.py` 的默认值为准 |

### 0.2 ⚠️ 节号陷阱：清单里的 §3 / §4 不是论文的 §3 / §4

这是本文档要提醒你的第一件事，能省你 20 分钟翻不到页的焦虑。清单写的是"§3.1 三维高斯+投影、§3.2 优化+自适应密度控制、§4 实现细节"——**那是大多数中文博客自己编的章号**。论文的真实结构是：

| 清单里的说法 | 论文真实章节 | 论文里的小节 |
|---|---|---|
| §3.1 三维高斯 + 投影（$\Sigma'=JW\Sigma W^TJ^T$） | **§4 Differentiable 3D Gaussian Splatting** | 无小节，Eq.(4)(5)(6) 都在这 |
| §3.2 优化 + 自适应密度控制 | **§5 Optimization with Adaptive Density Control** | 5.1 Optimization、5.2 Adaptive Control of Gaussians |
| §4 实现细节 / 光栅化 | **§6 Fast Differentiable Rasterizer** + **§7.1 Implementation** | 6 无小节；7.1 有 "Optimization Details" |

对照过来的完整骨架（读 PDF 时按这个找）：

```
1 Introduction
2 Related Work          ← 清单说跳过，同意（真要读就只看 2.3 节 point-based 那段，它交代了 Eq.(3) 从哪来）
3 Overview              ← 1 页，当摘要读
4 Differentiable 3D Gaussian Splatting      ★ 清单 §3.1
5 Optimization with Adaptive Density Control
  5.1 Optimization                          ★ 清单 §3.2 上半
  5.2 Adaptive Control of Gaussians         ★ 清单 §3.2 下半（全篇最实用）
6 Fast Differentiable Rasterizer for Gaussians   ★ 清单 §4 上半
7 Implementation, results and evaluation
  7.1 Implementation                        ★ 清单 §4 下半
  7.2 Results and Evaluation                ★ Table 1/2 在这
  7.3 Ablations                             ★ "Initialization from SfM" 在这（Part 1 的直接依据）
  7.4 Limitations
8 Discussion and Conclusions
附录 A 梯度推导 / B 优化与稠密化算法 / C 光栅化细节 / D 逐场景指标
```

> **读的时候留个心眼**：§3 Overview 只有一页，但它把"输入是什么、三个组件分别在哪节"说清楚了。**如果你的时间和精力只够读一页，读 §3**——它是全篇唯一同时包含输入、表示、优化、渲染四个环节的地方。

### 0.3 我抓到了什么、没抓到什么

| 章节 | 状态 | 本文档对应 |
|---|---|---|
| 摘要、§1、§3、§4、§5、§6 | ✅ 逐字读到（HTML 双源交叉核对） | 第二部分 2.1–2.4 |
| §7.1、§7.2（Table 1/2）、§7.4 | ✅ 逐字读到（Table 1/2 逐格抄录） | 2.5、2.7、2.8 |
| §7.3 Ablations | ⚠️ **部分**：只读到 "Initialization from SfM" 和 "Densification" 两段；另三组消融（unlimited depth complexity / anisotropic covariance / spherical harmonics）的正文**没抓到** | 2.6（含缺口标注） |
| 附录 A（梯度推导）、附录 B（伪代码式算法表） | ❌ 未抓全文。**附录 B 是一张"优化与稠密化"的完整伪代码表，和本文第三部分几乎等价**，值得你翻 PDF 对一遍 | — |
| 补充视频、项目页交互 demo | ❌ 未看（也不需要） | — |

> 📌 **附录 B 特别提一句**：论文正文把 clone/split/prune 的条件散在 §5.2 的散文里，附录 B 把它们整理成了一张算法表（Algorithm 1）。**如果你要复现或改这套机制，看附录 B 比看正文快**。我没抓到它的原文，所以本文的伪代码是按正文 + 代码重建的，不是抄附录 B。

### 0.4 三小时阅读节奏（建议）

| 时段 | 干什么 | 产出 |
|---|---|---|
| 0:00–0:30 | 本文第一部分 + 论文 §3 | 能说出 3DGS 的输入是什么、三个组件各解决什么 |
| 0:30–1:20 | 论文 §4，对照本文 2.1 | 能手推 Eq.(5)(6)，能答清单 Q1、Q2 |
| 1:20–2:10 | 论文 §5，对照本文 2.2、2.3 | 能答清单 Q3、Q4、Q5，能画 clone/split 判断树 |
| 2:10–2:40 | 论文 §6 + §7.1，对照本文 2.4、2.5 | 能讲清"为什么反向要重走一遍" |
| 2:40–3:10 | §7.3 + Table 1/2 + §7.4，对照本文 2.6、2.7、2.8、第四部分 | 拿到 P1 报告要引的数字 + 9 个坑 + 论文自认的四个弱点 |
| 3:10–3:30 | 第五部分自测 | 答不上就回去补对应小节 |

---

## 第一部分 · 打地基（不含论文内容）

> 判断标准：读完这五小节，第二部分的三段式里**不该再出现需要解释的术语**。这里一个论文结论都不讲，只补前置。

### 1.1 先看清一件事：NeRF 和 3DGS 用的是同一个成像公式

这是理解全文的钥匙，也是 3DGS 敢说"我用光栅化，但我和 NeRF 等价"的底气。

**New view synthesis（NVS）**：给一组拍了同一静态场景的照片 + 每张照片的相机位姿，合成出没拍过的视角。所有方法都在做同一件事：**定义一个从"空间中的点 + 方向"到"颜色 + 密度"的函数，然后沿光线积分**。

体渲染（volumetric rendering）沿一条光线、在 $N$ 个采样点上的颜色是：

$$
C=\sum_{i=1}^{N}T_i\,(1-\exp(-\sigma_i\delta_i))\,\mathbf{c}_i,\qquad T_i=\exp\Big(-\sum_{j=1}^{i-1}\sigma_j\delta_j\Big)
$$

- $\sigma_i$：第 $i$ 个采样点的**体密度**（有多"实"）
- $\delta_i$：相邻采样点的**间隔**（这一步光线走了多长）
- $T_i$：**透射率**（transmittance），从相机走到第 $i$ 点还没被挡住的比例
- $\mathbf{c}_i$：该点朝这个方向发出的颜色

把 $\alpha_i = 1-\exp(-\sigma_i\delta_i)$ 代进去，就变成经典的 **alpha 混合**：

$$
C=\sum_{i=1}^{N}T_i\alpha_i\mathbf{c}_i,\qquad T_i=\prod_{j=1}^{i-1}(1-\alpha_j)
$$

**关键在于**：点云 alpha 混合的写法 $C=\sum_{i\in\mathcal N}c_i\alpha_i\prod_{j=1}^{i-1}(1-\alpha_j)$ 和上面这个式子**长得一模一样**。也就是说，只要你把每个点看成一个有"不透明度 $\alpha$"的小图元，并且**按深度从前往后**混合，你的点渲染器在数学上就是一个体渲染器。

> 🗣 大白话
> 这就是 3DGS 全部合法性的来源：它没有放弃 NeRF 的成像模型，只是把"沿光线撒点、每点问一遍 MLP"换成了"场景里本来就有几百万个点，按深度排序后让 GPU 直接混"。前者是**采样**（慢、有噪声），后者是**光栅化**（快、硬件擅长）。**公式一样，执行方式完全不同**——一句话概括整篇论文。

### 1.2 高斯是什么：从"一个模糊的球"到"一个有朝向的椭球"

一个 3D 各向同性（球形）高斯由**中心** $\boldsymbol\mu\in\mathbb R^3$ 和**一个标量宽度** $\sigma$ 决定，密度随距离衰减：$G(\mathbf{x})\propto \exp(-\|\mathbf{x}-\boldsymbol\mu\|^2/2\sigma^2)$。

各向异性高斯把这个"宽度"换成一个 **$3\times3$ 协方差矩阵** $\Sigma$：

$$
G(\mathbf{x})=\exp\!\Big(-\tfrac12(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\Big)
$$

协方差矩阵的含义只有一句话：**它的特征向量 = 椭球的三个轴方向，特征值 = 三个轴的方差（半轴长的平方）**。等值面 $\{\mathbf{x}:(x-\mu)^\top\Sigma^{-1}(x-\mu)=k^2\}$ 是一个椭球面。

所以"一个高斯"在几何上就是**一个可以任意拉伸、任意旋转的三维椭球 + 一个中心**。3DGS 的整个场景 = 几百万个这样的椭球叠在一起。

> 🗣 大白话
> 为什么不用球？因为**真实表面是二维薄片**。一面墙、一片叶子，用球去填要叠很多层（层与层之间还互相遮挡、浪费）；用一个压扁的椭球（一个轴很小、另两个轴很大）一层就贴上了。这就是论文反复强调 **anisotropic（各向异性）** 的原因——它直接决定"用多少个高斯才能表示一个场景"。

### 1.3 为什么"投影后还是高斯"是整件事的地基

先把结论记住，2.1.4 会用论文原话再讲一遍：

- **线性变换**（旋转、缩放、剪切、平移）作用在高斯上，结果**仍然是高斯**，新均值 = 变换后的均值，新协方差 = $A\Sigma A^\top$。
- **透视投影不是线性变换**（除以了深度 $z$），所以严格来说，3D 高斯投影到图像上**不是**精确的高斯。
- 3DGS 的做法：在均值点附近对投影做**一阶泰勒展开**，用这个**局部仿射近似**的雅可比矩阵 $J$ 代替真实投影。于是投影又变回线性变换，又可以用 $A\Sigma A^\top$ 的公式。

**为什么这一步是"整个方法成立的前提"，而不只是一个数学技巧**：

1. 只有"投影后是高斯"，你才能把三维椭球投影成一个**有解析表达式的二维椭圆**（2×2 协方差），才能用它做剔除、算覆盖了哪些 tile、解析地算出每个像素的不透明度——**这些都是 O(1) 的闭式计算，不用采样**。
2. 如果投影后不是高斯，你就只能对每个像素沿光线采样（回到 NeRF 的老路），或者在屏幕上近似成圆形（那就丢掉了各向异性，也就是丢掉了"一个椭球贴一片薄墙"的紧凑性）。
3. 顺带解释了为什么**远景大平面会糊**：投影的仿射近似在离均值越远的地方误差越大，而一个几百像素宽的椭圆，其边缘离均值很远——这正好是 §7.4 Limitations 里会点出的问题之一。

> 🗣 大白话
> 你可以把它想成"贴图": 3DGS 不是往屏幕上撒点，而是往屏幕上**贴一堆有方向的小椭圆片**。只有当这片能被精确地算成一个椭圆时，"贴"才能是一次 GPU 调用而不是几百次采样。

### 1.4 球谐（SH）：为什么颜色要 48 个数

同一个表面，从左边看和从右边看，亮度不一样（镜面反射、菲涅尔）。如果每个高斯只存一个 RGB，那它就是个纯漫反射体，高光会丢。

**球谐函数**是球面上的一组正交基函数，作用和傅里叶基在圆上做的一样：**用一组系数逼近球面上的任意函数**。3DGS 用 3 阶（degree 3）球谐，每个颜色通道需要 $(3+1)^2=16$ 个系数，RGB 三通道共 **48 个**。

所以"一个高斯的颜色"不是一个 RGB，而是**这条视线方向 $\mathbf d$ 的函数**：

$$
\mathbf c(\mathbf d)=\sum_{\ell=0}^{3}\sum_{m=-\ell}^{\ell} \mathbf k_{\ell m}\,Y_{\ell m}(\mathbf d)
$$

- degree 0 的 1 个系数 = 基础色（漫反射项，代码里叫 `f_dc`）
- degree 1–3 的 15 个 = 方向相关的修正（高光、视角依赖，代码里叫 `f_rest`）

**代码里就按这个切**：`features_dc` 是 `3×1`，`features_rest` 是 `3×15`，加起来 48。

> 🗣 大白话
> 48 个系数不是为了炫技，是为了让"闪光的金属、湿润的地面"能被表示。代价也很实在：**48/59 ≈ 81% 的参数都花在颜色上了**，位置+形状只有 10 个。这个比例在 §4 第四部分会直接关系到显存。

### 1.5 SfM 点云：你 Part 1 的唯一变量

论文的输入是"**一组图 + SfM 标定好的相机 + SfM 顺带产出的稀疏点云**"。这三个东西里：

- 相机（内参 + 外参）→ 决定"每个高斯被投到屏幕的哪儿"
- 稀疏点云 → 决定"一开始有几个高斯、在哪儿、多大"

**Part 1 的全部实验设计，就是把这个输入换掉**：COLMAP 给一套，VGGT 给一套，VGGT-Ω 给一套，后面 3DGS 代码一行不改。所以你必须知道 3DGS 到底**用**了这套输入的哪些性质：

| 输入的性质 | 3DGS 用它做什么 | 换初始化时会变吗 |
|---|---|---|
| 点的**位置** | 初始高斯的均值 $\mu$ | ✅ 会（VGGT 点云更密、更"贴着表面"、可能有系统性偏移） |
| 点的**数量** | 初始高斯个数 | ✅ 会（COLMAP 几万 vs VGGT 点图上百万） |
| 点的**局部密度** | 初始尺度（到最近 3 个点的距离） | ✅ 会（稀疏处初始高斯特别大） |
| 点的**颜色** | 初始 SH 的 DC 项 | ✅ 会（COLMAP 有点颜色，VGGT 的点图没有颜色——**这是一个必须你自己决定的接口问题**） |
| 相机**内参/外参** | 每一步投影 | ✅ 会（这是 Part 1 的另一半变量） |
| 场景**尺度** | `spatial_lr_scale`、`cameras_extent`（见 4.4） | ✅ 会（单目 VGGT 的尺度是任意的） |

> 🗣 大白话
> 这张表就是 Part 1 的"变量清单"。**你的报告里只要有一条没能对上（比如 VGGT 的点云没颜色，你却忘了做颜色初始化），初始条件就不再是"只差一个变量"了。**

---

## 第二部分 · 逐节精读

> 每节按 📄 原文 → 🗣 大白话 → 🧩 伪代码 三段走。伪代码用论文符号、不可运行，只用来对齐思路。

### 2.1 §4 可微 3D 高斯（= 清单 §3.1）

#### 2.1.1 为什么是"不带法向的椭球"，而不是"带法向的圆盘"

#### 📄 论文原文

> "Our representation has similarities to previous methods that use 2D points and assume each point is a small planar circle with a normal. **Given the extreme sparsity of SfM points it is very hard to estimate normals. Similarly, optimizing very noisy normals from such an estimation would be very challenging.** Instead, we model the geometry as a set of 3D Gaussians that do not require normals."

#### 🗣 大白话

前一派做法（Surface Splatting、Pulsar 那一支）把每个点当成一个**带朝向的小圆片**：圆片必须"贴着表面"躺，所以需要法向。而 SfM 给的稀疏点云**没有法向**，硬估出来的法向非常噪。

于是 3DGS 把"朝向"这件事**下放给协方差矩阵**去自己学：椭球的三个轴可以任意倾斜，倾斜成什么角度是优化出来的——**朝向变成了参数，而不是先验**。这换来两件事：① 不需要法向估计，直接吃 SfM 点云；② 椭球的"压扁方向"天然就是表面法向，等于**顺便学出了几何**（论文 Fig.3 把那 60% 缩小的椭球画出来就是这个意思）。

**不这么设计的代价**：你必须先跑一遍 MVS 拿到稠密带法向的点云（这正是它引用的那些方法要做的事），于是"输入只有 SfM 点"这个卖点就没了——而"只要 SfM 点"恰好是你 Part 1 最在意的性质：**VGGT 给的正是没有法向的点图**。

#### 🧩 伪代码

```
function 初始化一个高斯(点 p):
    μ ← p                            # 均值就是点的位置
    Σ ← s²·I                         # 各向同性：一个球，还没有朝向
    α ← 0.1                          # 不透明度初值（论文没写，代码是 0.1）
    c ← SH(点云颜色)                  # 只有 DC 项非零，方向项初始为 0
    return (μ, Σ, α, c)
# 注意：没有 "法向 n" 这个字段。椭球的倾斜角在训练中被学出来。
```

#### 2.1.2 一个高斯到底有几个参数（清单 Q1）

**答案：59 个浮点数**（不含 batch/内部缓存）。

| 参数 | 维度 | 论文符号 | 代码字段 | 说明 |
|---|---|---|---|---|
| 位置（均值） | 3 | $\boldsymbol\mu$ | `_xyz` | 直接就是世界坐标 |
| 缩放 | 3 | $S$（对角） | `_scaling` | 存的是 $\log s$，取用时 `exp` |
| 旋转 | 4 | $q$（单位四元数） | `_rotation` | 存 4 个，取用时归一化 |
| 不透明度 | 1 | $\alpha$ | `_opacity` | 存 logit，取用时 `sigmoid` |
| 球谐颜色 | 48 | $\mathbf k_{\ell m}$ | `_features_dc`(3) + `_features_rest`(45) | 3 通道 × 16 系数（degree 3） |
| **合计** | **59** | | | |

> 📌 一个容易记错的坑：**旋转只有 4 个参数，不是 9 个**（不存矩阵），**缩放只有 3 个，不是 6 个**（不存协方差矩阵）。论文 Eq.(6) 的 $\Sigma=RSS^\top R^\top$ 是**计算出来的**，不是存下来的。所以"3DGS 每个高斯存 59 个数"是对的，"存 3×3 协方差"是错的。

#### 2.1.3 Eq.(4)：高斯的定义

#### 📄 论文原文

> "Our Gaussians are defined by a full 3D covariance matrix $\Sigma$ defined in world space centered at point (mean) $\mu$:  $G(x)=e^{-\frac{1}{2}(x)^{T}\Sigma^{-1}(x)}$. This Gaussian is multiplied by $\alpha$ in our blending process."

#### 🗣 大白话

注意这句 "This Gaussian is multiplied by $\alpha$"——**高斯函数本身不含幅度**，幅度由外部的 $\alpha$ 给。这个设计很妙：形状（协方差）和"有多实"（不透明度）解耦，可以分别用不同的学习率、不同的激活函数优化。

还有一个隐含的点：$G$ **没有归一化系数** $1/\sqrt{(2\pi)^3|\Sigma|}$。也就是说"大椭球和小椭球，中心处的 $G$ 值都是 1"。这看起来是错的（体积大的应该更"淡"），但因为它后面要乘 $\alpha$、并且 $\alpha$ 是可学的，所以模型可以自己补偿。**代价是：优化初期大高斯会过度覆盖**——这也是后面 opacity reset 要处理的问题之一。

#### 🧩 伪代码

```
# 屏幕上一个像素的最终颜色（论文 Eq.3 的写法）
function 混合像素(按深度排序的高斯列表 G_sorted):
    C ← 0
    T ← 1                            # 透射率
    for g in G_sorted:               # 从近到远
        α_g ← g.α × G_2D(像素中心; g.Σ')     # 二维高斯在该像素的取值
        C ← C + T · α_g · g.color(视线方向)
        T ← T · (1 - α_g)
        if T < 1e-4: break           # α 饱和即停
    return C
```

#### 2.1.4 Eq.(5)：投影——清单 Q2 的正主

#### 📄 论文原文

> "Given a viewing transformation $W$ the covariance matrix $\Sigma'$ in camera coordinates is given as follows: $\Sigma'=JW~\Sigma~W^{T}J^{T}$ where $J$ is the **Jacobian of the affine approximation of the projective transformation**."
>
> "Zwicker et al. also show that **if we skip the third row and column of $\Sigma'$, we obtain a 2×2 variance matrix with the same structure and properties as if we would start from planar points with normals**."

#### 🗣 大白话

**"为什么投影后还是椭圆？"** 分三步：

1. 高斯族在**仿射变换下封闭**：$x\mapsto Ax+b$ 把高斯映成高斯，新协方差 = $A\Sigma A^\top$。所以 $W$（世界→相机，刚体变换）之后还是高斯。
2. **透视投影不是仿射的**（$x/z$ 是非线性的），严格来说投影后不是高斯。所以论文用 $J$ 做**一阶泰勒近似**：在均值附近把投影函数线性化。$J$ 就是投影函数在均值处的雅可比。于是 $W$ 之后再用 $J$ 做一次线性变换，仍然保持高斯性。
3. 最后只保留屏幕上的 **2×2** 部分：论文的写法是先得到 3×3 的 $\Sigma'$、再**扔掉第三行第三列**（因为屏幕是二维的，深度方向的延展对成像没有贡献，留着还会和深度耦合出麻烦）；代码里等价于把 $J$ 写成 2×3（第三行全 0）。这一步来自 EWA splatting，得到的就是"每个高斯在屏幕上的脚印（footprint）"。

**为什么这一步是整个方法成立的前提**（这是清单 Q2 的后半问）：

- 有了 2×2 的解析椭圆，你才能 **O(1) 算出一个高斯覆盖了哪些 tile**（用 99% 置信区间当包围盒）、**O(1) 算出每个像素的 $\alpha$**、**O(1) 算梯度**。
- 这三件事一旦是解析的，渲染就退化成"排序 + 混合"，GPU 每秒能做几十亿次。**如果投影后不是高斯，你就只能回到光线采样。**

**一个必须知道的近似代价**：一阶泰勒在**离均值越远越不准**。一个大椭圆（几百像素）的边缘，误差不可忽略——所以远景的大平面、极端近的物体容易出问题。论文 §7.4 的 limitations 里点了这个。

> **⚠️ 符号陷阱（第四部分会再提）**：论文写 $\Sigma'=JW\Sigma W^\top J^\top$（列向量约定），代码 `forward.cu` 里写的是 `T = W*J; cov = transpose(T) * transpose(Vrk) * T`（行向量约定），数学上是同一件事的两种转置写法。对着符号读代码时，**别以为代码写错了**。

#### 🧩 伪代码

```
function 投影一个高斯(g, 相机(K, W)):
    # 1. 3D 协方差（论文 Eq.6）
    Σ_3D ← R(q_g) · diag(s_g)² · R(q_g)ᵀ

    # 2. 到相机空间（刚体，仍是高斯）
    Σ_cam ← W · Σ_3D · Wᵀ

    # 3. 透视投影的局部仿射近似 J（在 μ 处一阶展开）
    J ← ∂(π(x)) / ∂x |_{x=μ_cam}      # 2×3，含 fx/z²、fy/z² 这些项
    Σ_2D ← J · Σ_cam · Jᵀ

    # 4. 只要屏幕上的 2×2 脚印
    Σ_2D ← Σ_2D[0:2, 0:2]

    # 5. 代码里还额外加了一个"低通"（论文 §4 没写！见 4.2）
    Σ_2D[0,0] += 0.3 ; Σ_2D[1,1] += 0.3

    return (μ_2D ← π(μ_cam), Σ_2D)
```

#### 2.1.5 Eq.(6)：为什么优化 $s$ 和 $q$，而不是直接优化 $\Sigma$

#### 📄 论文原文

> "An obvious approach would be to directly optimize the covariance matrix $\Sigma$ ... However, **covariance matrices have physical meaning only when they are positive semi-definite. For our optimization of all our parameters, we use gradient descent that cannot be easily constrained to produce such valid matrices, and update steps and gradients can very easily create invalid covariance matrices.**"

> "To allow independent optimization of both factors, we store them separately: a 3D vector $s$ for scaling and a quaternion $q$ to represent rotation. These can be trivially converted to their respective matrices and combined, **making sure to normalize $q$ to obtain a valid unit quaternion**."

#### 🗣 大白话

$\Sigma$ 必须是**半正定**的（否则不是"方差"，椭球会变成双曲面，几何上无意义）。而梯度下降是一步一步"往梯度反方向挪"，它**根本不知道**你在要求半正定——挪一步就可能挪出界。

解法：**换一组参数，让"合法"变成结构上自动成立**。

- $\Sigma = RSS^\top R^\top$，其中 $S$ 是对角缩放、$R$ 是旋转。
- 只要 $R$ 是真正的旋转矩阵、$S$ 的对角元是正的，$\Sigma$ 就**必然**半正定——不需要任何约束、不需要投影、不需要特征值分解。
- 所以参数化对象变成 $s$（3 个，任意实数，取 `exp` 变正）和 $q$（4 个，归一化后变旋转）。

> 这是一个反复出现的招式，叫"**用参数化把约束吸收掉**"。你在 Part 3（生成先验）里还会遇到同一招：与其惩罚"生成图要几何一致"，不如把一致性做成结构的一部分（BRPO 的置信掩码就是这个思路）。

**为什么不直接优化 6 个独立的协方差元素**（对称矩阵有 6 个自由元）？因为 6 个元素构成的矩阵**不一定**是半正定的（行列式可能为负）。$RSS^\top R^\top$ 的 7 个自由度（3 缩放 + 4 四元数，四元数归一化后实际 3 自由度 → 共 6 自由度）恰好铺满所有合法协方差，**没有冗余也没有遗漏**。

#### 🧩 伪代码

```
function Σ_from_params(s, q):
    q ← q / ‖q‖                      # 归一化 → 合法旋转
    R ← quat2mat(q)                  # 3×3 旋转矩阵
    S ← diag(exp(s))                 # 保证三个轴长为正（代码存 log s）
    return R · S · Sᵀ · Rᵀ           # 必然半正定，白送

# 反例：如果直接优化对称矩阵 Σ 的 6 个元素
#   Σ ← Σ - lr · ∂L/∂Σ   →  可能 det(Σ) < 0  →  Σ⁻¹ 无意义 → 优化直接崩
```

#### 2.1.6 初始化：这一节是 Part 1 的入口，读三遍

#### 📄 论文原文

> "We estimate the initial covariance matrix as an **isotropic Gaussian with axes equal to the mean of the distance to the closest three points**."

#### 🗣 大白话

一个 SfM 点变成高斯时，**初始大小**是这么定的：找到离它最近的 3 个点，算平均距离，用这个距离当半径，做成一个球（各向同性）。

**为什么用"到最近 3 个点的距离"**？因为这是你手上唯一能免费拿到的**局部密度**估计。点密的地方（表面细节多）→ 初始高斯小；点稀的地方（大片墙面、天空）→ 初始高斯大。

**这对 Part 1 意味着什么**（这是本文档最想让你记住的一条）：

> 初始化点云的**空间分布**直接决定初始高斯的**大小分布**。而高斯的大小决定了：
> ① 它一开始能覆盖多少像素（大高斯 → 早期梯度信号强、但也更容易"糊"）；
> ② 它会不会立刻触发 `split`（超过 `percent_dense × extent` 的高斯走 split 而不是 clone）；
> ③ 训练结束时的高斯总数（也就决定了显存和训练时间）。
>
> 所以 **COLMAP 点云 vs VGGT 点云，即使位姿完全一样，3DGS 的训练轨迹也会不同**——不是因为它俩"位姿差多少",而是因为**点云的密度分布不同**。你的报告必须把这两件事分开讲，否则 P1 的结论会站不住。

> 📌 一个必须写进报告的接口问题：**VGGT 的 dense pointmap 没有颜色**。而 3DGS 的初始化代码会用点云颜色去填 SH 的 DC 项（`features[:, :3, 0] = RGB2SH(color)`）。如果你给 VGGT 点云一个"全 0 颜色"或"某个常数颜色"，初始 SH 就和 COLMAP 组不在同一起点上了。**推荐的公平做法**：从对应的像素直接取 RGB 作为点颜色（VGGT 的点图本来就是逐像素的，取颜色是 O(1) 的事）。这条在清单里没写，但它会实打实影响前 1k iter 的收敛曲线。

#### 🧩 伪代码

```
function 从点云初始化高斯(P: 点云, colors):
    for p_i in P:
        d_i ← mean(‖p_i - p_j‖ for p_j in knn(P, p_i, k=3))   # 局部密度
        g_i ← { μ: p_i,
                s: [log d_i]*3,        # 各向同性球，半径 = 平均近邻距离
                q: [1,0,0,0],          # 单位四元数（还没转起来）
                α: 0.1,
                SH: [RGB2SH(colors_i), 0, ..., 0] }
    return {g_i}
# ⚠️ 代码细节：distCUDA2 返回的是"到 3 近邻的【平方】距离的均值"，
#    然后取 sqrt。所以严格说是 RMS 距离，不是算术平均距离。见 4.2。
```

---

### 2.2 §5.1 优化（= 清单 §3.2 前半）

#### 2.2.1 为什么优化必须"能创造、能销毁"

#### 📄 论文原文

> "**Inevitably, geometry may be incorrectly placed due to the ambiguities of 3D to 2D projection. Our optimization thus needs to be able to create geometry and also destroy or move geometry if it has been incorrectly positioned.**"
>
> "The quality of the parameters of the covariances of the 3D Gaussians is critical for the compactness of the representation since **large homogeneous areas can be captured with a small number of large anisotropic Gaussians**."

#### 🗣 大白话

3D→2D 的投影**天生是多对一的**：无数个三维点投到同一个像素。所以任何"只看图、不看真值"的优化都可能把几何放错地方（浮空、穿墙、表面糊成一坨）。你没法靠损失函数阻止它——**唯一办法是让优化过程自己有增删改的能力**。

这就引出 §5.2 的整套机制。注意这里的因果关系：**不是"3DGS 设计了 clone/split 所以效果好"，而是"投影不可逆 → 必须能增删 → 才有 clone/split"**。动机先于细节。

第二句还给了个关于"紧凑性"的判据：**大片同质区域应该被少量大高斯覆盖**。这句话是你判断"训练是否健康"的实用标准——如果你训出来的场景里，一面白墙上有几千个小高斯，说明协方差没学好。

#### 🧩 伪代码

```
function 优化一步(G, 相机k, 图I_k):
    Î_k ← 可微光栅化(G, 相机k)          # §6
    L ← 0.8·L1(Î_k, I_k) + 0.2·(1 - SSIM(Î_k, I_k))    # §2.2.3
    ∇ ← 反向传播(G.μ, G.s, G.q, G.α, G.SH)              # 附录 A 是手推的
    G ← G - lr · Adam(∇)
    return G
```

#### 2.2.2 激活函数：sigmoid 给 $\alpha$，exp 给 scale

#### 📄 论文原文

> "We use a **sigmoid activation function for $\alpha$ to constrain it in the [0−1) range and obtain smooth gradients**, and an **exponential activation function for the scale of the covariance** for similar reasons."

#### 🗣 大白话

和 2.1.5 是同一招：**用激活函数把物理约束吸收进参数化**。

- $\alpha$ 必须 $\in[0,1)$。存 logit（任意实数），取用时 sigmoid → 永远合法，且梯度处处光滑（相比"钳位/clamp"在边界处梯度为 0 的死区）。
- 尺度必须 $>0$。存 $\log s$（任意实数），取用时 `exp` → 永远为正，且**学习率在小尺度上等效于"相对步长"**（`exp` 的导数就是自己，尺度小的地方步子也自动变小）。这个性质和第四部分 4.4 的"场景尺度归一"是一套的。

> 📌 一个能省你半天 debug 的事实：**你调 `--scaling_lr` 时，调的其实是 $\log s$ 的学习率**，不是"米"的学习率。所以它的合适取值和场景绝对尺度无关——这也是为什么同一套超参能跑 Mip-NeRF 360（米级）和 synthetic（单位球级）。

#### 🧩 伪代码

```
取用时:  α = sigmoid(_opacity)      ;  s = exp(_scaling)      ;  q = normalize(_rotation)
存盘时:  _opacity = logit(α)        ;  _scaling = log(s)      ;  _rotation = q
# 优化的是左边的下划线变量（无约束），右边才是物理量
```

#### 2.2.3 损失函数（清单 Q5）：$\mathcal L=(1-\lambda)\mathcal L_1+\lambda\mathcal L_{\text{D-SSIM}}$，$\lambda=0.2$

#### 📄 论文原文

> "The loss function is $\mathcal L_1$ combined with a D-SSIM term: $\mathcal L=(1-\lambda)\mathcal L_{1}+\lambda\mathcal L_{\mathrm{D\text{-}SSIM}}$. **We use $\lambda~=~0.2$ in all our tests.**"
>
> "We use a standard exponential decay scheduling technique similar to Plenoxels, **but for positions only**."

#### 🗣 大白话

**为什么不能只用 L1？**

L1 是逐像素的绝对差，它有两个毛病：
1. **对结构不敏感**：一整块区域整体偏移 1 个像素，L1 可能和"逐像素各错一点"差不多；但人眼立刻就看出"歪了/糊了"。
2. **在平坦区域梯度信号弱**：白墙上上下挪一点，L1 变化很小，于是优化器不知道该怎么调。

**D-SSIM（1 − SSIM）** 是在**局部窗口**上比较"均值、方差、协方差"三个统计量，天生对**结构、对比度、边缘**敏感。它补的正是 L1 缺的那一半。

所以这个组合的分工是：**L1 守住光度/PSNR，D-SSIM 守住结构/主观质量**。这也解释了为什么 3DGS 的 PSNR 和 SSIM **同时**都很好——如果只用 L1，PSNR 会更高一点但看起来糊；只用 D-SSIM，指标会难看。

**为什么 $\lambda=0.2$（结构项只占 20%）？** 论文没给消融（§7.3 的五组消融里没有 $\lambda$），所以他只说了 "in all our tests"。**我的解读（不是原文）**：D-SSIM 是在局部窗口上算的，权重太高会让优化器过度优化"局部统计量的匹配"，把高频纹理压平（over-smoothing），甚至在不同窗口的交界处引入块状伪影。0.2 是"够用但不喧宾夺主"的经验值。**你复现时不要动它**，动了就和论文数字不可比了。

**"位置才做 lr 指数衰减"** 这句话很关键，别漏：位置学习率用指数衰减（论文没给具体数值，代码里是从 $1.6\times10^{-4}$ 降到 $1.6\times10^{-6}$，即 $\times0.01$），**而颜色、不透明度、缩放的 lr 是常数**。动机很直白：早期高斯要"跑位"去找结构，后期要"稳住"只调外观。如果位置 lr 一直不降，后期会一直抖、永远收敛不干净。

#### 🧩 伪代码

```
function 渲染+损失(G, 相机k, GT图I_k):
    Î,k ← 光栅化(G, 相机k)                 # 同时返回每个高斯的屏幕空间位置（用于 §5.2）
    L1   ← mean|Î - I_k|
    DSSIM← 1 - SSIM(Î, I_k)               # 局部窗口统计量比较
    L    ← (1-0.2)·L1 + 0.2·DSSIM
    return L, Î, 屏幕空间位置
```

---

### 2.3 §5.2 自适应密度控制（= 清单 §3.2 后半）

> 这一节是全篇**最实用**的一节，也是 Part 2/Part 3 里"浮空伪影从哪来"的答案来源。慢读。

#### 2.3.1 判据：为什么用"屏幕空间位置梯度"

#### 📄 论文原文

> "It focuses on regions with missing geometric features ("under-reconstruction"), but also in regions where Gaussians cover large areas in the scene (which often correspond to "over-reconstruction"). **We observe that both have *large* view-space positional gradients.** Intuitively, this is likely because they correspond to regions that are not yet well reconstructed, and the optimization tries to move the Gaussians to correct this."
>
> "we densify Gaussians with an **average magnitude of view-space position gradients above a threshold $\tau_{\mathrm{pos}}$, which we set to 0.0002** in our tests."

#### 🗣 大白话

**要"补高斯"，得先知道哪里缺。** 但你没有真值几何，怎么知道哪里缺？

论文的观察非常漂亮：**不管缺（该有东西没有）还是多（一个大气斯糊了一片），优化器都会使劲把附近的高斯往某个方向推**。所以"位置梯度大"= "这里还没重建好"。

- **欠重建**（细节没覆盖）：这个区域该有几何，但没有高斯在那里，于是附近边缘的高斯被往那边拽 → 梯度大。
- **过重建**（一个大气斯跨了多个表面）：一个椭球盖住了两个不同颜色的表面，无论怎么调都不可能同时对上两边，于是它被两边来回拉 → 梯度大。

一个判据同时抓住两种病，这是这篇论文里最"省事"的设计。

**为什么是"屏幕空间的"、而且是"平均"的**（这两点论文都是一句话带过，但决定成败）：

- **屏幕空间**：因为损失是在图像上算的，梯度最有意义的度量是"这个高斯在屏幕上该移动多少像素"。世界空间的梯度会随距离变化（远处 1 米很小，近处 1 米很大），不能跨深度比较。
- **平均**：代码里是 `xyz_gradient_accum / denom`，其中 `denom` 是**这个高斯被看见的次数**（只在 `visibility_filter` 为真时累加）。也就是说它是"**在被看到的那些帧上，平均每次的梯度**"。⚠️ 这个细节论文完全没写，见 4.5。

> 💡 **对 Part 1 的直接提示**：$\tau_{\mathrm{pos}}=0.0002$ 是个**绝对阈值**。如果你的初始化让前期的屏幕空间梯度整体偏大/偏小（比如点云太密 → 高斯太小 → 要移动更多像素才能覆盖），触发 densify 的时刻就会整体前移/后移，于是**收敛曲线的形状会变**。这就是为什么 2.7 里我会说"收敛曲线要画出来，而且要知道它有多少是超参决定的"。

#### 🧩 伪代码

```
function 该不该稠密化(g, 阈值τ):
    avg_grad ← g.累积梯度 / g.被看见次数       # 屏幕空间位置梯度，按可见性平均
    return avg_grad > τ                        # τ = 0.0002

# 每个 iter 都对所有"可见"的高斯累加：
#   累积梯度 += ‖∂L/∂p_screen‖   ;   被看见次数 += 1
# 每 100 iter，用上面的平均梯度判一次
```

#### 2.3.2 `clone` vs `split`：清单 Q3

#### 📄 论文原文

> "For small Gaussians that are in under-reconstructed regions, we need to cover the new geometry that must be created. **For this, it is preferable to clone the Gaussians, by simply creating a copy of the same size, and moving it in the direction of the positional gradient.**"
>
> "On the other hand, large Gaussians in regions with high variance need to be split into smaller Gaussians. **We replace such Gaussians by two new ones, and divide their scale by a factor of $\phi~=~1.6$** which we determined experimentally. **We also initialize their position by using the original 3D Gaussian as a PDF for sampling.**"
>
> "**In the first case we detect and treat the need for increasing both the total volume of the system and the number of Gaussians, while in the second case we conserve total volume but increase the number of Gaussians.**"

#### 🗣 大白话

两种病，两种手术。判据是**这个高斯本身大不大**（阈值：`percent_dense × extent = 0.01 × 场景半径`，代码里叫 `percent_dense`）：

| | 触发条件 | 症状 | 手术 | 对"总体积"的影响 |
|---|---|---|---|---|
| **clone**（克隆） | 梯度大 **且** 高斯很小 | 细节区域缺覆盖，需要**新体积** | 复制一份（同样大小，挪到梯度方向） | **膨胀**（体积增加） |
| **split**（分裂） | 梯度大 **且** 高斯很大 | 一个椭球跨了多个表面，需要**拆细** | 换成 2 个，尺度除以 1.6，位置从原高斯里采样 | **守恒**（体积大致不变，个数 ×2） |

**"为什么不统一用一种？"** 这就是清单 Q3 的答案：

- 如果全用 clone：一个"过重建"的大高斯会被复制成两个**一样大**的——问题一点没解决，反而多了一份"错的东西"，高斯数暴涨。
- 如果全用 split：在"欠重建"的空白区，把一个**本来就小**的高斯切成两个更小的，两个小高斯加起来还填不满那个洞 → 永远补不上。
- 所以判据必须是"**缺体积** vs **体积够了但形状不对**"，这正好对应"小 vs 大"。

**$\phi=1.6$ 的来历**：论文原话是 "which we determined experimentally"，即**实验调出来的，没有理论依据**。代码里是 `scale / (0.8 * N)`、$N=2$，正好 1.6。$\phi$ 太小（比如 1.1）→ 拆了跟没拆一样，要拆很多次；太大（比如 4）→ 一次拆太碎，出现"空洞感"。**复现时别动它**。

**"用原高斯当 PDF 采样位置"** 是一句信息量很大的话：新位置 $\sim\mathcal N(\mu, \Sigma)$，也就是**按这个椭球的形状撒**——沿长轴撒得远、沿短轴撒得近。这样拆出来的两个子椭球正好沿着原来的延展方向铺开，**天然贴合原椭球描述的那个平面**。（如果改成在球内均匀撒，就会把"薄片"拆成"两团"，细节立刻糊掉。）

#### 🧩 伪代码

```
function 稠密化(G, 相机, extent, τ=0.0002, percent_dense=0.01):
    for g in G where avg_grad(g) > τ:
        if max(g.s) <= percent_dense * extent:      # 小 → 缺体积
            g' ← copy(g)                            # 论文说"挪向梯度方向"
            G.add(g')
        else:                                        # 大 → 形状不对
            for _ in 1..2:
                ε ~ N(0, g.Σ)                        # 用原高斯当 PDF 采样
                G.add( μ: g.μ + ε,
                       s: g.s / 1.6,                 # 每个轴都缩
                       q: g.q, α: g.α, SH: g.SH )
            G.remove(g)
    return G
# ⚠️ 代码里 clone 那一支【没有】"挪向梯度方向"这一步，见 4.1
```

#### 2.3.3 opacity reset：清单 Q4

#### 📄 论文原文

> "**Similar to other volumetric representations, our optimization can get stuck with floaters close to the input cameras; in our case this may result in an unjustified increase in the Gaussian density. An effective way to moderate the increase in the number of Gaussians is to set the $\alpha$ value close to zero every $N=3000$ iterations.** The optimization then increases the $\alpha$ for the Gaussians where this is needed while allowing our culling approach to remove Gaussians with $\alpha$ less than $\epsilon_\alpha$ as described above."

#### 🗣 大白话

**问题**：训练早期，相机附近会积一堆"雾"状的高斯（floaters）。它们存在的唯一理由是"能糊住某个视角的某个像素"，从几何上讲纯属垃圾。但它们是**有效的**——能降低损失，所以优化器不会主动删它们。更糟的是，它们位置梯度也大，于是被判为"重建不足"，还会被 clone/split 继续**繁殖**。

**解法**：每 3000 iter，把所有高斯的 $\alpha$ **压到接近 0**。然后：

- 真正"有用"的高斯（有多个视角在支撑它）会**很快把 $\alpha$ 学回来**；
- 纯糊的 floaters（只为一个视角糊住一小块）**学不回来**，$\alpha$ 一直很低，于是被 $\alpha<\epsilon_\alpha$ 的剪枝规则删掉。

**不做会怎样（清单 Q4 的后半问）**：高斯数会单调暴涨（因为 floaters 自己还在 clone），显存和训练时间失控，而且**渲染出来的近景有一层半透明的雾**。注意这不是"稍微差一点",而是**结构性失控**——密度控制机制被浮空高斯劫持了。

> ⚠️ **一个数值上的魔鬼细节**（论文没写，见 4.3）：代码里 reset 是把 $\alpha \leftarrow \min(\alpha, 0.01)$，而剪枝阈值 $\epsilon_\alpha = 0.005$。**0.01 > 0.005**，所以 reset 之后**没有任何高斯会立刻因为 $\alpha$ 太小被删掉**。真正的清理发生在 reset 之后的若干次迭代里（那些撑不住的会继续往 0 掉）。也就是说："reset → 立即 prune" 这个理解是**错的**，reset 之后要等几百个 iter 才见效。如果你在 reset 那一刻统计高斯数，会看到"没变化",别以为机制没生效。

#### 🧩 伪代码

```
function opacity_reset(G):
    for g in G:
        g.α ← min(g.α, 0.01)        # 注意：不是设成 0，也不是设成 0.01
    # 不删除任何高斯。删除留给下一步的 prune（α < 0.005）
    # 于是"有效清理"发生在 reset 之后的几百个 iter 里

# 主循环里的节奏（代码）：
#   it % 3000 == 0  →  opacity_reset
#   it % 100  == 0  →  判梯度、clone/split、prune
```

#### 2.3.4 剪枝：论文一句话，代码四个条件

#### 📄 论文原文

> "Gaussians may shrink or grow and considerably overlap with others, but **we periodically remove Gaussians that are very large in worldspace and those that have a big footprint in viewspace**."

#### 🗣 大白话

论文只说"删太大的"，代码里有**四个**删除条件（见 4.6 的对照表）：

1. $\alpha < 0.005$ —— 透明的（主要清理对象）
2. 屏幕空间半径 > 20 px —— 一个高斯盖了 20 像素以上，通常是"糊墙"的
3. 世界空间尺度 > `0.1 × extent` —— 相对场景尺度太大的（会带来严重遮挡错误）
4. 前 3000 iter **不启用** 2、3 两条（`size_threshold = 20 if iteration > opacity_reset_interval else None`）

**为什么世界空间也有条件**：一个大高斯哪怕在屏幕上看起来还行，只要它的世界尺度大得离谱（比如 0.1×场景半径），它就会挡住后面真实的内容，而且它的梯度会把周围全搞乱。这是"防患于未然"。

#### 2.3.5 稀疏视角下，这套机制为什么会失控（连到 Part 2/Part 3）

这一段不是论文内容，是我把 §5.2 的机制推到稀疏设定下的推论——但它是你 P2/P3 的理论基础，值得先想清楚：

- clone/split 的判据是**屏幕空间位置梯度**。而在稀疏视角下，一大片区域**在所有训练视角里都不可见**（没拍到），那里**没有任何光度梯度**。
- 于是：判据 $\approx 0$ → **不 clone 也不 split** → 那些区域的高斯只能靠初始化"猜"。
- 同时，**可见区域**的 floaters 依然会繁殖，reset 之后它们学到的是"在这几个视角上糊住就行"（过拟合），而不是"在三维里对"。
- 结果就是你会在稀疏场景里看到：**高斯总数疯长 + 空白区域是雾 + 已拍区域过拟合**。这三件事同一个病因。

> 🗣 一句话版：**3DGS 的密度控制是"完全由观测驱动"的，它没有先验。观测不到的地方，它不会长东西，也不会知道自己长错了。** —— 这句话可以直接进你 P2 的 Intro。

---

### 2.4 §6 光栅化（= 清单 §4 前半）

> 这一节回答"为什么能实时"。读的时候时刻问自己：**这个设计省掉了什么**。

#### 2.4.1 两个目标：预排序 + 不限梯度个数

#### 📄 论文原文

> "Our goals are to have fast overall rendering and fast sorting to allow approximate $\alpha$-blending – including for anisotropic splats – and to **avoid hard limits on the number of splats that can receive gradients** that exist in previous work."
>
> "we design a **tile-based rasterizer** for Gaussian splats ... to **pre-sort primitives for an entire image at a time, avoiding the expense of sorting per pixel** that hindered previous $\alpha$-blending solutions"
>
> "Our fast rasterizer allows efficient backpropagation over an arbitrary number of blended Gaussians with low additional memory consumption, **requiring only a constant overhead per pixel**."

#### 🗣 大白话

**为什么"每像素排序"是灾难**：α 混合要求从前往后按深度混合。如果每个像素各自维护一个有序列表，那么一个覆盖 10 万个像素的大高斯就要被插入 10 万次列表；而且不同的像素看到的遮挡关系不同，你没法复用。老方法（Kopanas 那一支）就卡在这。

**3DGS 的取舍**：**整张图只排一次序**。代价是：如果两个高斯在某个像素处"深度顺序和全局排序不一致"（深度交错的情形），混合就不是严格正确的。但论文算过一笔账——**当每个 splat 的尺寸接近一个像素时，这种交错几乎不存在**，近似误差可以忽略。这就是原文那句 "these approximations become negligible as splats approach the size of individual pixels"。

**"不限梯度个数"为什么难**：如果你最多只混合前 K 个高斯，那么 K 之外的高斯拿不到梯度，它们就永远学不好（而且 K 要按场景手调）。3DGS 想去掉这个 K，于是反向传播时必须**知道每个像素到底混了哪些**——这才是 2.4.4 那个漂亮技巧要解决的问题。

#### 🧩 伪代码

```
# 对比两种排序策略（这是本节的核心 trade-off）
策略A（老方法）: for 每个像素: 对该像素覆盖的所有 splat 排序   → O(像素 × splat)，爆炸
策略B（3DGS）:   for 整个屏幕: 对所有 splat 按 (深度, tile_id) 排一次  → 一次基数排序
# 策略B 的代价：深度交替处混合不严格正确 → 但 splat≈1 像素时误差可忽略
```

#### 2.4.2 三步：剔除 → 实例化 + 造 key → 单次基数排序

#### 📄 论文原文

> "Our method starts by **splitting the screen into 16×16 tiles**, and then proceeds to cull 3D Gaussians against the view frustum and each tile. Specifically, **we only keep Gaussians with a 99% confidence interval intersecting the view frustum**. Additionally, we use a **guard band** to trivially reject Gaussians at extreme positions (i.e., those with means close to the near plane and far outside the view frustum), since computing their projected 2D covariance would be unstable."
>
> "We then **instantiate each Gaussian according to the number of tiles they overlap and assign each instance a key that combines view space depth and tile ID**. We then sort Gaussians based on these keys using a **single fast GPU Radix sort**."

#### 🗣 大白话

三步，每步都在为下一步减少工作量：

1. **剔除（cull）**：整个高斯只要不在视锥里就扔掉；再对每个 tile 算出它覆盖了哪几个 tile。判据用 **99% 置信区间**（即椭球的 $\pm3\sigma$ 范围，取整成包围盒）——这是一个"宁可多算一点也不能漏"的保守选择，因为**漏掉一个高斯会产生可见的洞，多算一个只是浪费一点算力**。
2. **实例化（instantiate）**：一个高斯覆盖了 $k$ 个 tile，就在列表里出现 $k$ 次（$k$ 个"实例"）。每个实例带一个 key = `(深度 << 32) | tile_id`。
3. **排序**：对**所有实例**做一次 GPU 基数排序（radix sort，按位排，$O(n)$）。排完的效果是：**同一个 tile 的所有实例在数组里是连续的一段，并且段内按深度有序**。

**guard band 是什么**：屏幕外围一圈"看不见但存在"的缓冲区。均值落在视锥外极远处的高斯，投影时会出现除零/协方差爆炸（$J$ 里有 $1/z^2$），代码里干脆把它们**先钳到 guard band 边界再算**（`t.x = clamp(t.x, ±limx)`），算出来再丢掉。这是纯工程稳健性措施，论文只用一句话带过，但**没有它，训练会偶尔出现 NaN**。

#### 🧩 伪代码

```
function 排序并分桶(G, 相机):
    实例列表 ← ∅
    for g in G:
        if g 的 99% 置信椭球 ∩ 视锥 = ∅: continue
        tiles ← g 的屏幕包围盒覆盖到的 tile 集合        # 16×16 一块
        实例列表 += [ (key=(g.深度, t.id), g) for t in tiles ]
    radix_sort(实例列表, by=key)                      # 一次，全局
    for t in 所有 tile:
        t.起点, t.终点 ← 该 tile 在排序后数组里的连续区间
    return 实例列表, 每个 tile 的区间
```

#### 2.4.3 前向：每 tile 一个线程块，front-to-back，饱和即停

#### 📄 论文原文

> "For rasterization, **we launch one thread block for each tile**. Each block first **collaboratively loads packets of Gaussians into shared memory** and then, for a given pixel, **accumulates color and $\alpha$ values by traversing the lists front-to-back**, thus maximizing the gain in parallelism both for data loading/sharing and processing. **When we reach a target saturation of $\alpha$ in a pixel, the corresponding thread stops.** At regular intervals, threads in a tile are queried and the processing of the entire tile terminates when all pixels have saturated (i.e., $\alpha$ goes to 1)."

#### 🗣 大白话

为什么是 tile 而不是 pixel？因为 **GPU 的 shared memory 是按 block 共享的**：一个 tile 内所有像素看到的高斯列表高度重叠，把高斯数据**一次性搬进 shared memory**，之后该 tile 内所有像素都从这儿读——访存次数从"像素数 × 高斯数"降到"tile 数 × 高斯数"。这是 CUDA 编程的标准套路（也解释了为什么 tile 必须是 16×16 这种"一个 block 能装下"的尺寸）。

**"饱和即停"** 是性能的关键：前面的高斯已经把 $\alpha$ 累到接近 1（完全不透明）时，后面再多的高斯也贡献不了颜色，直接停。这个优化对户外场景特别有效（前景遮挡很快饱和）。

**为什么要"定期查询整个 tile 是否都饱和"**：单个像素停了不等于整个 tile 停了。要定期问一遍"这个 tile 里还有没有活着的像素"，全没了就整个 block 退出，把 SM 让给别的 tile。

#### 🧩 伪代码

```
function 光栅化一个tile(t, 实例区间):
    把 实例区间 里的高斯数据 一次性搬进 shared_memory
    for 像素 p in t (并行):
        C ← 0 ; T ← 1
        for g in 实例区间 (从前往后):
            α ← g.α · exp(-½ dᵀ Σ'⁻¹ d)      # d = 像素中心相对 g 中心的偏移
            C ← C + T·α·g.color(视线)
            T ← T·(1-α)
            if T < 阈值: break               # 该像素饱和
        Ẽ[p] ← C
    # 定期检查：tile 内所有像素都饱和了吗？都饱和 → 整个 block 退出
```

#### 2.4.4 反向：为什么"重走一遍"比"存列表"更聪明

**这一小段是全篇工程上最漂亮的地方，也最容易被跳过。别跳。**

#### 📄 论文原文

> "One solution would be to store arbitrarily long lists of blended points per-pixel in global memory. **To avoid the implied dynamic memory management overhead, we instead choose to traverse the per-tile lists again; we can reuse the sorted array of Gaussians and tile ranges from the forward pass. To facilitate gradient computation, we now traverse them back-to-front.**"
>
> "**each point stores the final accumulated opacity $\alpha$ in the forward process; we divide this by each point's $\alpha$ in our back-to-front traversal to obtain the required coefficients for gradient computation.**"

#### 🗣 大白话

反向传播需要什么？需要**前向时每一步的透射率 $T_i$**（因为 $\partial C/\partial \alpha_i$ 里有 $\prod_{j<i}(1-\alpha_j)$ 这一项）。

**朴素方案**：前向时把每个像素"混了哪些高斯、各自 $\alpha$ 多少"全存下来。问题是——不知道有多少个（取决于饱和在哪），要么开一个很大的上界（浪费内存），要么动态分配（GPU 上很慢）。

**3DGS 的方案**：
1. 前向时**只存一个数**：每个（高斯, 像素）对在混合结束后的**最终累计不透明度**。
2. 反向时**从后往前走**（back-to-front），同时维护一个"当前累计值"。
3. 关键等式：如果从后往前走，那么在处理到点 $i$ 时，你手上已有的累计量除以 $i$ 自己贡献的 $\alpha_i$，就还原出了 $i$ 之前的透射率。

数学上就是：记最终累计为 $A_{\text{final}}=\sum$ 那一串，那么回溯时每一步都可以**用除法从后一项恢复前一项**。

> 🗣 一个比喻
> 你有一串连乘 $T_1,T_1(1-\alpha_2),T_1(1-\alpha_2)(1-\alpha_3),\dots$。如果你只记了**最后一项**和每个 $\alpha_i$，你不需要把中间每一项都抄下来——**一路除回去**就行。
>
> 这个技巧的收益是：**每个像素的额外内存是常数**（一个 float），而不管它混了 1 个还是 1000 个高斯。这就是原文 "requiring only a constant overhead per pixel" 的确切含义。

**而且它顺带解决了"不限梯度个数"**：因为反向不依赖任何预存的列表长度，所以混合了多少个高斯都可以正确反传。

#### 🧩 伪代码

```
# 前向（同时刻）
for 像素 p, 从前往后:
    T ← T · (1-α_i)
记录: 该 (p, g_i) 的最终累计不透明度 A_final[p, i]     # 只存这一个数

# 反向（back-to-front 重走同一份排序数组，不用重新排序）
A ← 1                                                  # 反向累计
for 像素 p, 从后往前:
    T_before_i ← A_final[p,i] / α_i                    # 除法还原 i 之前的透射率
    ∂L/∂α_i    ← f(A, T_before_i, C_i)                 # 用还原出的系数算梯度
    A ← T_before_i                                     # 继续往前
# 内存: 每像素 O(1)。与混合了多少个高斯无关。
```

#### 2.4.5 "近似"到底有多近似

论文自己承认 $\alpha$ 混合是近似的（因为只做了全局排序，没有逐像素排序）。你要记住的判据是：

> **splat 越小，近似越准。** 当每个 splat 接近单像素尺寸时，误差可以忽略。

反过来说：**如果你把高斯训得特别大（或渲染分辨率特别低），近似误差会显现**——表现为深度交替处的颜色错误、闪烁。这也解释了为什么 3DGS 在"高分辨率 + 高斯自然较小"的设定下效果最好，以及为什么后来那么多工作（Mip-Splatting、Taming 3DGS 等）要专门处理"小/大高斯"问题。

---

### 2.5 §7.1 实现细节（= 清单 §4 后半）

> ⚠️ **先纠正一个流传很广的误解**：**§7.1 里没有学习率，也没有 100/3000/0.2/0.0002 这些数字。** 它只有两段——一段讲分辨率 warm-up，一段讲 SH 分带。你清单里要的那 5 个答案所需的数字全在 §5.1、§5.2 和 Table 1 里。这一段我逐字核过原文，别再去 §7.1 里找超参了。

#### 2.5.1 训练前 500 步：分辨率 warm-up

#### 📄 论文原文

> "For stability, we **"warm-up" the computation in lower resolution**. Specifically, we start the optimization using **4 times smaller image resolution** and we **upsample twice after 250 and 500 iterations**."

#### 🗣 大白话

**为什么需要**：初始化时每个高斯是个球，半径等于"到最近 3 个点的距离"——这个尺寸在点云稀疏处**往往比它最终该有的大得多**。如果一上来就在全分辨率下训练，一个大高斯会盖住几百个像素，每个像素都给它一个梯度，早期更新非常"暴力"，容易抖、容易炸，也容易在开头就长出一堆错的大高斯。

**降到 1/4 分辨率（像素数变成 1/16）后**：每个高斯覆盖的像素少了、每次更新的"有效样本"少了、梯度更平滑，形状能先粗调到位；到第 250、500 步再逐级提回原分辨率。

**还有一个顺带的好处**：前 500 步只花 1/16 的渲染成本。论文报的 41 分钟训完 Mip-NeRF 360，这部分省的时间不算多，但省的是**最难调的那段**。

**注意 500 这个数字和 `densify_from_iter = 500` 是同一个数**。这不是巧合：§5.2 开头就写 "**After optimization warm-up (see Sec. 7.1), we densify every 100 iterations**"——论文明确把"warm-up 结束"和"开始 densify"绑在一起。所以 500 就是那个交界点：**在高斯还没"粗调到位"之前，不该判"哪里缺"。**

#### 🧩 伪代码

```
for it in 1 .. 30000:
    if it < 250:        分辨率 ← 原分辨率 / 4
    elif it < 500:      分辨率 ← 原分辨率 / 2
    else:               分辨率 ← 原分辨率
    渲染(分辨率) ; 算 L1+SSIM ; 反传
# 只在 >= 500 时才允许 clone/split（densify_from_iter = 500）
```

> ⚠️ **官方发布的代码没有实现这一步**（见 4.9）。`train.py` 里没有任何上采样逻辑，图像从第一个 iter 就是最终分辨率。这意味着：①论文的 41m33s 你不能直接对标（你的前 500 步更贵）；②**论文 Table 1 里的 "Ours-7K" 和你在代码里跑 7K 不是一回事**——它的 7K 是"warm-up 后还跑了 6500 步"，你的是"从头全分辨率 7K"。这个坑会让你复现数字对不上却找不到原因。

#### 2.5.2 SH 分带训练：先只学漫反射色

#### 📄 论文原文

> "SH coefficient optimization is **sensitive to the lack of angular information**. For typical "NeRF-like" captures where a central object is observed by photos taken in the entire hemisphere around it, the optimization works well. However, **if the capture has angular regions missing (e.g., when capturing the corner of a scene, or performing an "inside-out" capture) completely incorrect values for the zero-order component of the SH (i.e., the base or diffuse color) can be produced by the optimization. To overcome this problem we start by optimizing only the zero-order component, and then introduce one band of the SH after every 1000 iterations until all 4 bands of SH are represented.**"

#### 🗣 大白话

SH 的零阶项（DC）就是**基础色/漫反射色**。如果 16 个系数一开始全部放开优化，而某些**观察方向压根没有数据**（拍墙角：你看不到墙角的背面；inside-out 采集：相机围在外面往中心看），那些高阶基函数在无数据方向上的取值**没有任何约束**，优化器就会拿它去凑局部残差——而且会把零阶项一起带歪，表现是"整个物体的颜色变得莫名其妙"。

**解法**：分带放开。前 1000 步只优化 DC（等价于一个纯漫反射模型，最稳），之后每 1000 步加一阶，第 3000 步之后 4 个带全开（degree 0/1/2/3 = 1+3+5+7 = 16 个系数）。

> 💡 **这条对你的 Part 3 是意外收获**。生成的新视角本质上是"**伪造的角向观测**"。如果这些伪视角正好填补了原本缺失的角向区域，SH 就会被它们带偏 → **颜色漂移**。这是"生成视角污染重建"的一个**具体、可指认的机制**（比笼统说"引入错误梯度"强得多），可以直接写进 P3 的 failure analysis：**去看生成轮次之间的 SH 系数变化量**，就是一个现成的诊断指标。

> 📌 这是论文与代码**对得上**的少数几处之一：`train.py` 里 `if iteration % 1000 == 0: gaussians.oneupSHdegree()`，和原文完全一致。

#### 🧩 伪代码

```
for it in 1 .. 30000:
    可用阶数 ← min(3, it // 1000)          # 0,1,2,3
    只对 ℓ ≤ 可用阶数 的 SH 系数计算梯度    # 其余系数梯度置零（不是"不存"，是"不学"）
# 效果：前 1000 步 = 纯漫反射；3000 步后 = 完整视角相关着色
```

#### 2.5.3 学习率：论文一个字都没给

#### 📄 论文原文（§7.4 Limitations 里唯一提到 lr 的地方）

> "While we used the same hyperparameters for our full evaluation, **early experiments show that reducing the position learning rate can be necessary to converge in very large scenes (e.g., urban datasets)**."

#### 🗣 大白话

论文从未列出学习率、迭代数、densify 参数的具体数值（除了 §5.1 的 λ=0.2、§5.2 的 τ=0.0002 和 100/3000 这两个频率）。**所有超参数字都只存在于代码里**（`arguments/__init__.py`），我在 2.2.3 和第四部分引用它们时都标了来源。

**但上面这句 limitations 里的话对你有直接价值**：论文自己说，**超大场景（点名 urban 数据集）可能需要降低位置学习率**。而你的必做数据里 **Waymo 就是城市街景**。所以：

- 如果 Waymo 上出现"训了很久还是糊/浮空多/收敛慢"，第一个该试的不是加迭代，而是**降位置 lr**（或降 `densify_grad_threshold` 让它晚点开始 densify）。
- 报告里如果做了这件事，可以引这句话作为依据，而不是"我们瞎调的"。

---

### 2.6 §7.3 消融：Initialization from SfM（Part 1 的直接依据）

> 这一节是**全篇唯一直接研究"初始化影响"的实验**。你的 P1 就是在同一个问题上做更细的版本（SfM vs 前馈模型，而不是 SfM vs 随机），所以这一段要逐字读。

#### 📄 论文原文

> "We also assess the importance of initializing the 3D Gaussians from the SfM point cloud. **For this ablation, we uniformly sample a cube with a size equal to three times the extent of the input camera's bounding box.** We observe that our method performs relatively well, **avoiding complete failure even without the SfM points**. Instead, **it degrades mainly in the background, see Fig. 7**. Also **in areas not well covered from training views, the random initialization method appears to have more floaters that cannot be removed by optimization**. On the other hand, **the synthetic NeRF dataset does not have this behavior because it has no background and is well constrained by the input cameras**."

> Figure 7 caption: "Initialization with SfM points helps. Above: initialization with a random point cloud. Below: initialization using SfM points."

#### 🗣 大白话

**他们做了什么**：把 SfM 点云扔掉，在"相机包围盒的 3 倍大立方体"里**均匀撒随机点**当初始化。这是一个**极端的初始化破坏**（相当于"完全不知道场景在哪"）。

**结果三句话**：

1. **没有彻底崩**。3DGS 的密度控制确实有能力"自己长出几何"——随机点也能训出一个看着像样的场景。
2. **坏在背景**。前景（被很多视角覆盖的物体）几乎不受影响；背景（大片、远、视角少）明显变差。**为什么**：背景的约束最弱，能不能"猜到"它，全靠初始化给的先验位置。
3. **覆盖不到的地方会留一堆删不掉的浮空物**（"floaters that cannot be removed by optimization"）。注意这个措辞——**删不掉**。机制是：几乎没有视角在看它们 → 光度损失给不出信号 → 既没有梯度把它们移走，也没有梯度把它们的 $\alpha$ 压到剪枝阈值（0.005）以下。opacity reset 最多把它压到 0.01，之后**没人管它，它就永远停在 0.01**：视觉上淡成雾，但**不会被删掉**，一直占着显存。在"部分覆盖"的区域（P2 稀疏视角下的典型情况）更糟——偶尔一个视角看到它，$\alpha$ 又被抬回去。

**而 synthetic（NeRF-synthetic）完全不受影响**——因为它是"一个物体 + 纯白背景"，没有背景要猜，而且每个方向都被半球相机覆盖死了。

**Table 3 里的数字**（同一张表，PSNR，5K / 30K 两个训练长度）：

| 配置 | Truck-5K | Garden-5K | Bicycle-5K | Truck-30K | Garden-30K | Bicycle-30K | Avg-5K | Avg-30K |
|---|---|---|---|---|---|---|---|---|
| **Random Init** | 16.75 | 20.90 | 19.86 | 18.02 | 22.19 | 21.05 | **19.17** | **20.42** |
| **Full（SfM 初始化）** | 22.71 | 25.82 | 23.18 | 24.81 | 27.70 | 25.65 | **23.90** | **26.05** |
| 差值 | −5.96 | −4.92 | −3.32 | −6.79 | −5.51 | −4.60 | **−4.73** | **−5.63** |

（来源：论文 Table 3，逐行逐格抄的。这张表还包含 Limited-BW / No-Split / No-SH / No-Clone / Isotropic 五行，见下面的 ⚠️。）

#### 🗣 这条消融给你 Part 1 的三条结论

**① 初始化影响的量级是"几 dB"，不是"零点几 dB"。** 把初始化从 SfM 换成"完全随机"，平均掉 4.7 dB（5K）到 5.6 dB（30K）。**注意这个数字的性质**：它是"极端破坏"下的**上界**，不能拿来预测"COLMAP 换成 VGGT 会掉多少"——后者的点云质量远好于随机点。**但它给了你一个坐标系**：如果 COLMAP→VGGT 只掉 0.2 dB，说明前馈几何已经"够用"；如果掉 3 dB，说明初始化依然是瓶颈。这正是清单里那个贯穿问题（"初始化差 1 度，PSNR 差多少"）的量化参照。

**② 初始化影响的是"背景"和"覆盖不足的区域"，不是前景。** 你的三个必做场景 **全部是户外/室内真实场景，背景占比很大**（Waymo 街景、DL3DV 户外、Re10k 室内）。所以**你的场景是初始化差异被放大的那一类**。反过来，如果你只在物体级数据上做实验，会得出"初始化不重要"的错误结论——这就是论文 synthetic 那段的教训。

**③ 它改的是"浮空物能不能被优化掉"，不只是"起点好不好"。** "floaters that cannot be removed by optimization" 这句话解释了你在 P2/P3 会看到的一个现象：**某些伪影无论怎么训都存在**。它们不是"没训够"，而是"没人在看它们"。这对 P3 里"生成视角补梯度"的动机是一个直接支撑：**生成视角的作用本质上是给这些"没人在看"的区域派一个观察者**。

> ⚠️ **一处我没读到的内容（请自己核对原文）**：§7.3 里除了 "Initialization from SfM" 和 "Densification" 这两段，还有三组消融（unlimited depth complexity of splats with gradients / anisotropic covariance / spherical harmonics）。这三段的原文我没抓到，所以 **Table 3 里 `Limited-BW`、`No-SH`、`Isotropic` 三行的精确定义我不能替你确认**。完整 Table 3 我照抄在下面，你自己对着原文那三段读：
>
> | 配置 | T-5K | G-5K | B-5K | T-30K | G-30K | B-30K | Avg-5K | Avg-30K |
> |---|---|---|---|---|---|---|---|---|
> | Limited-BW | 14.66 | 22.07 | 20.77 | 13.84 | 22.88 | 20.87 | 19.16 | 19.19 |
> | Random Init | 16.75 | 20.90 | 19.86 | 18.02 | 22.19 | 21.05 | 19.17 | 20.42 |
> | No-Split | 18.31 | 23.98 | 22.21 | 20.59 | 26.11 | 25.02 | 21.50 | 23.90 |
> | No-SH | 22.36 | 25.22 | 22.88 | 24.39 | 26.59 | 25.08 | 23.48 | 25.35 |
> | No-Clone | 22.29 | 25.61 | 22.15 | 24.82 | 27.47 | 25.46 | 23.35 | 25.91 |
> | Isotropic | 22.40 | 25.49 | 22.81 | 23.89 | 27.00 | 24.81 | 23.56 | 25.23 |
> | **Full** | 22.71 | 25.82 | 23.18 | 24.81 | 27.70 | 25.65 | **23.90** | **26.05** |
>
> 顺便说两个不用读原文就能看出来的结论：
> ① **`No-Split`（21.50 / 23.90）比 `No-Clone`（23.35 / 25.91）惨得多**：30K 上 split 掉 **2.15 dB**，clone 只掉 **0.14 dB**。也就是说 **split 是刚需，clone 是增益**——这正好对上 §7.3 "Densification" 那段的自述："splitting big Gaussians is important to allow good reconstruction of the background ... while cloning the small Gaussians ... allows for a better and **faster convergence** especially when thin structures appear"。
> ② **clone 的价值在"早"不在"终"**：5K 时 clone 掉 0.55 dB、30K 时只掉 0.14 dB——它是**加速项**，不是最终精度的来源。这个区分对你 P1 的"收敛速度"实验有直接指导意义：**如果某条初始化路线在 5K 落后但 30K 追平，很可能是 densify 的动力学差异，而不是几何质量差异。**

#### 🧩 伪代码（如果他们做了你的实验，会怎么写）

```
function 消融_初始化(scene, 训练器):
    点云_SfM  ← COLMAP(图像)                       # 真实条件
    点云_随机 ← 在 3×相机包围盒 内均匀采样 |点云_SfM| 个点
    for 点云 in [点云_随机, 点云_SfM]:
        高斯 ← 初始化(点云)                          # 位姿完全相同！
        PSNR ← 训练并评测(高斯, 相机, 迭代=5K/30K)
    # 关键：只改点云，位姿一模一样 → 这是"只改一个变量"的干净对照
    return PSNR表
```

---

### 2.7 Table 1/2：P1 要引的数字

> ⚠️ **结构先说清，别找错表**：论文的 **Table 1 是一张合并表**，三个数据集块（Mip-NeRF360 / Tanks&Temples / Deep Blending）并排，每块内是 `SSIM↑ | PSNR↑ | LPIPS↓ | Train | FPS | Mem`。**Table 2 是 synthetic（Blender）的 PSNR**，不是 T&T/DB。清单里说的 "Table 1/2 速度和质量对比" 指的就是这两张。

#### 2.7.1 Table 1（真实场景）

> 📄 表标题原文：**"Table 1. Quantitative evaluation of our method compared to previous work, computed over three datasets. Results marked with dagger † have been directly adopted from the original paper, all others were obtained in our own experiments."**

| 方法 | Mip-NeRF360<br>SSIM↑ PSNR↑ LPIPS↓ | Train | FPS | Mem | Tanks&Temples<br>SSIM↑ PSNR↑ LPIPS↓ | Train | FPS | Mem | Deep Blending<br>SSIM↑ PSNR↑ LPIPS↓ | Train | FPS | Mem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Plenoxels | 0.626 / 23.08 / 0.463 | 25m49s | 6.79 | 2.1GB | 0.719 / 21.08 / 0.379 | 25m5s | 13.0 | 2.3GB | 0.795 / 23.06 / 0.510 | 27m49s | 11.2 | 2.7GB |
| INGP-Base | 0.671 / 25.30 / 0.371 | 5m37s | 11.7 | 13MB | 0.723 / 21.72 / 0.330 | 5m26s | 17.1 | 13MB | 0.797 / 23.62 / 0.423 | 6m31s | 3.26 | 13MB |
| INGP-Big | 0.699 / 25.59 / 0.331 | 7m30s | 9.43 | 48MB | 0.745 / 21.92 / 0.305 | 6m59s | 14.4 | 48MB | 0.817 / 24.96 / 0.390 | 8m | 2.79 | 48MB |
| M-NeRF360 | 0.792† / 27.69† / 0.237† | 48h | 0.06 | 8.6MB | 0.759 / 22.22 / 0.257 | 48h | 0.14 | 8.6MB | 0.901 / 29.40 / 0.245 | 48h | 0.09 | 8.6MB |
| **Ours-7K** | 0.770 / 25.60 / 0.279 | 6m25s | 160 | 523MB | 0.767 / 21.20 / 0.280 | 6m55s | 197 | 270MB | 0.875 / 27.78 / 0.317 | 4m35s | 172 | 386MB |
| **Ours-30K** | **0.815 / 27.21 / 0.214** | 41m33s | 134 | 734MB | **0.841 / 23.14 / 0.183** | 26m54s | 154 | 411MB | **0.903 / 29.41 / 0.243** | 36m2s | 137 | 676MB |

#### 2.7.2 Table 2（synthetic Blender，只有 PSNR）

> 📄 表标题原文：**"Table 2. PSNR scores for Synthetic NeRF, we start with 100K randomly initialized points. Competing metrics extracted from respective papers."**

| 方法 | Mic | Chair | Ship | Materials | Lego | Drums | Ficus | Hotdog | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| Plenoxels | 33.26 | 33.98 | 29.62 | 29.14 | 34.10 | 25.35 | 31.83 | 36.81 | 31.76 |
| INGP-Base | 36.22 | 35.00 | 31.10 | 29.78 | 36.39 | 26.02 | 33.51 | 37.40 | 33.18 |
| Mip-NeRF | 36.51 | 35.14 | 30.41 | 30.71 | 35.70 | 25.48 | 33.29 | 37.48 | 33.09 |
| Point-NeRF | 35.95 | 35.40 | 30.97 | 29.61 | 35.04 | 26.06 | 36.13 | 37.30 | 33.30 |
| **Ours-30K** | 35.36 | 35.83 | 30.80 | 30.00 | 35.78 | 26.15 | 34.87 | 37.72 | **33.32** |

> 📌 **Table 2 的表标题里藏着 Part 1 最关键的一句话**："**we start with 100K randomly initialized points**"。也就是说，**在 synthetic 上，3DGS 用 10 万个随机点初始化就拿到了 SOTA**。配合 §7.2 那句 "Our approach quickly and automatically prunes them to about 6–10K meaningful Gaussians. The final size of the trained model after 30K iterations reaches about 200–500K Gaussians per scene"，你就得到了"初始化敏感性"的**反面证据**。两条一起看，才是完整的结论（见 2.6 的结论②）。

#### 🗣 怎么读这两张表（P1 报告里最容易写错的三处）

**① 别只报 PSNR。** Mip-NeRF360 数据集上，3DGS 的 PSNR（27.21）**低于** Mip-NeRF360（27.69），但 SSIM（0.815 vs 0.792）和 LPIPS（0.214 vs 0.237）**都更好**。所以论文的措辞是 "on par and sometimes slightly better"。**如果你的 P1 表只列 PSNR，就会得出"3DGS 不如 NeRF"的结论——这是错的，而且会被看出来。**

**② Mem 那一列可以直接换算成高斯数（这是你的显存预算公式）。**
一个高斯 59 个 float32 = **236 字节**。所以：

- Mip-NeRF360 的 734 MB ≈ **3.1M** 高斯
- T&T 的 411 MB ≈ **1.7M** 高斯
- Deep Blending 的 676 MB ≈ **2.9M** 高斯

和 §1 那句 "1-5 million Gaussians for all scenes tested" 完全对上。**公式：显存 ≈ 高斯数 × 236 B × 约 4（参数 + 梯度 + Adam 一阶 + 二阶）**。也就是说 1M 高斯 ≈ 0.9 GB 只用于参数状态。**再加 §7.4 说的"rasterizer 额外 30–500 MB"。**

> ⚠️ **但这个公式只能当下界，不能直接外推**：论文 §7.4 说他们训练大场景时**峰值显存超过 20 GB**。按 236 B×4 算，3M 高斯才 2.8 GB——差的那十几 GB 在**光栅化的实例化/排序缓冲、图像批次、框架开销**上。所以 **8 GB 卡上到底能扛多少高斯，只能实测**（用 `nvidia-smi` 盯峰值，或直接看 `train.log` 里的高斯数曲线）。**别按公式推一个数写进报告。** 本仓库 README 的 FAQ 给了正确的降显存顺序（调 `--densify_grad_threshold` / `--densification_interval` / `--densify_until_iter`，或 `--test_iterations -1`），照那个做。

**③ 跨数据集比 PSNR 没意义。** 论文自己说 "Note that image resolutions also vary over datasets"、"we kept the native input resolution for all renders"。**PSNR 只能在同一个数据集的同一套 test split 内比较**。所以你的表 1 必须按数据集分块（就像论文这张表一样），不能横着比。

> 另外两个引用时要说清的细节：
> - **M-NeRF360 在 Mip-NeRF360 那一块带 †**，即数字来自原论文而不是自己跑；T&T / DB 两块是他们自己跑的。**引用时如果混用会被质疑口径。**
> - **FPS 是在 A6000 上测的**，且是 1080p 级原生分辨率下的实时渲染速度。你的机器（消费级显卡）数字会不同，**报告里要标 GPU 型号，别直接抄 134 FPS**。

---

### 2.8 §7.4 Limitations：论文自己承认的四个弱点

> 这一节是**清单没要求、但强烈建议读**的——因为它把你 P2/P3 要遇到的失败模式提前点名了。四条，逐条给原文。

#### ① 没观测到的地方必有伪影 + 拉长/斑块状高斯

> 📄 "In regions where the scene is not well observed we have artifacts; in such regions, other methods also struggle. Even though the anisotropic Gaussians have many advantages as described above, **our method can create elongated artifacts or "splotchy" Gaussians**."

🗣 这正是 2.3.5 推论的**作者自述版**。注意 "elongated"（拉长）——形状自由度是把双刃剑：它让一个椭球能贴住一片薄墙，也让优化器能把一个高斯"拉"成一根穿过空洞的细长条去凑像素。你在 P2 的 failure case 里会大量看到这种"细长条"。

#### ② popping（跳变）伪影：来自 guard band 和"粗糙的可见性排序"

> 📄 "We also occasionally have **popping artifacts** when our optimization creates large Gaussians; this tends to happen in regions with view-dependent appearance. One reason for these popping artifacts is **the trivial rejection of Gaussians via a guard band in the rasterizer**... Another factor is **our simple visibility algorithm, which can lead to Gaussians suddenly switching depth/blending order**. This could be addressed by antialiasing, which we leave as future work. Also, **we currently do not apply any regularization to our optimization**; doing so would help with both the unseen region and popping artifacts."

🗣 这段话等于作者承认了 2.4.1 那个"整图排一次序"的近似**会露馅**：两个高斯深度接近时，换一个视角就可能"谁在前谁在后"翻转，于是那块像素突然变色 = 跳变（popping）。

**"we currently do not apply any regularization"** 这句对 Part 3 极其重要：**3DGS 本身没有任何正则项**，所以它对"错误的监督信号"是**完全无免疫**的。你喂进去一张幻觉图，它就会照做——因为没有东西在拉它回来。这句话可以直接支撑你 P3 的动机段。

#### ③ 超大场景可能要降位置学习率

> 📄 "early experiments show that **reducing the position learning rate can be necessary to converge in very large scenes (e.g., urban datasets)**."

🗣 见 2.5.3。**Waymo 就是 urban**。

#### ④ 显存比 NeRF 高得多（这条直接威胁你的机器）

> 📄 "During training of large scenes, **peak GPU memory consumption can exceed 20 GB in our unoptimized prototype**... Rendering the trained scene requires sufficient GPU memory to store the full model (several hundred megabytes for large-scale scenes) and **an additional 30–500 MB for the rasterizer**, depending on scene size and image resolution."

🗣 **把这句话和 2.7 的 236 B/高斯 合起来**，你就能算出自己的预算。而且注意两个数字的口径不同：

- **训练峰值 > 20 GB**：包含参数 + 梯度 + Adam 状态 + rasterizer 的实例化/排序缓冲，而且是"unoptimized prototype"（他们自己说的，可以优化）
- **渲染**：模型几百 MB + rasterizer 30–500 MB

**对你的实际意义**：8 GB 卡上跑 Mip-NeRF 360 那种规模的场景，**必须先压高斯数**。压的手段按"伤害从小到大"排：①`--data_device cpu`（把图像放内存，省一点）；②`--test_iterations -1`（避开测试时的显存尖峰）；③降 `densify_until_iter`（比如 15000 → 10000），代价是细节少一点；④降 `densify_grad_threshold`（更难触发 clone/split），代价是空白区补不上。

> ⚠️ **注意 ③④ 的副作用**：它们改的是**密度控制的行为**，而密度控制正是 2.3.5 说的"稀疏视角失控"的那套机制。所以**一旦你在 P1 改了这些参数，P2 就必须用同一套**，否则两部分的数字不可比。这条写进 `configs/`，别靠记忆。

---

## 第三部分 · 压缩伪代码（约 40 行讲完整篇）

> 检验你是否真读懂的唯一标准：能不能把这 40 行口述给别人。**如果有一行你讲不出来，回去补对应小节。**

```
# ============ 3DGS 全景（论文 §4 + §5 + §6） ============
输入: 图像 {I_k}, 相机 {(K_k, W_k)}, SfM 稀疏点云 P (含颜色)

# ---------- 初始化（§4，2.1.6）----------
G ← ∅
for p_i in P:
    d_i ← mean(‖p_i − p_j‖, j ∈ 3-NN(p_i))       # 局部密度
    G += { μ: p_i, s: [log d_i]³, q: (1,0,0,0),
           α: 0.1, SH: [RGB2SH(color_i), 0×45] }

# ---------- 训练主循环（§5）----------
for it in 1 .. 30000:
    if it % 1000 == 0: 升一级 SH 阶数 (0→3)        # §7.1 原文 + 代码一致
    取一张【随机】训练视角 k

    # ---- 前向：投影 + 光栅化（§4 + §6）----
    for g in G 可见:
        Σ_g  ← R(q)·diag(exp(s))²·R(q)ᵀ            # Eq.6，天然半正定
        Σ'_g ← J·W·Σ_g·Wᵀ·Jᵀ ; 留左上 2×2 (+0.3 低通)  # Eq.5，仿射近似
        屏幕包围盒 → 覆盖的 tile 列表
    按 (深度, tile_id) 对【所有实例】做一次基数排序   # §6
    每 tile 一个 block: 前向累积 C = Σ T·α·c，α 饱和即停
    记录每个 (像素,高斯) 的最终累计不透明度         # 反向要用

    # ---- 损失（§5.1）----
    L ← 0.8·L1(Î,I_k) + 0.2·(1−SSIM(Î,I_k))        # Eq.7, λ=0.2
    反向（back-to-front 重走排序数组，用除法还原 T）
    每个高斯的屏幕空间位置梯度 → 累加（只算可见的）

    # ---- 密度控制（§5.2，只在 500 < it < 15000 开）----
    if it % 100 == 0 and 500 < it < 15000:
        avg_grad ← 累加梯度 / 被看见次数
        for g where avg_grad > 0.0002:              # τ_pos
            if max(g.s) ≤ 0.01·extent:  G += copy(g)         # clone
            else:                       G += 2 × (s/1.6 的子高斯) ; G −= g   # split
        删 α < 0.005
        if it > 3000: 再删 屏幕半径>20px 或 世界尺度>0.1·extent
    if it % 3000 == 0:  ∀g: α ← min(α, 0.01)         # opacity reset

# ---------- 推理（渲染一个视角）----------
给定位姿: 重复上面"前向"那 5 行，不做反向、不改 α  ← 这就是 >30fps 的来源
```

**口述版（30 秒）**：
> 场景是一堆各向异性高斯，每个 59 个参数（位置 3 + 缩放 3 + 旋转 4 + 不透明度 1 + 球谐 48）。渲染时把每个高斯用局部仿射近似投影成屏幕上的椭圆（$\Sigma'=JW\Sigma W^\top J^\top$），按 tile 分桶、全局排一次深度序、每个 tile 一个 CUDA block 做前向 α 混合、饱和即停。训练就是拿随机一个视角渲染出来和真图比 L1+SSIM，反传；每 100 步根据"屏幕空间位置梯度大不大、高斯大不大"决定 clone 还是 split，每 3000 步把不透明度压到 0.01 清掉浮空雾。

---

## 第四部分 · 论文没明说、但决定成败的 9 件事

> 这一部分是精读最大的增值点。每一条都是"论文里读不到、但复现时一定会撞上"的。**证据全部来自本仓库里 clone 的官方代码，你可以自己去对行号。**

### 4.0 先给总表

| # | 论文说 | 代码是 | 对你的影响 |
|---|---|---|---|
| 1 | clone 时"moving it in the direction of the positional gradient" | **位置完全不变**，直接复制 | 别在报告里写"新高斯会先偏移"；抄实现时要自己补 |
| 2 | "We use Stochastic Gradient Descent techniques" | **Adam**，`eps=1e-15`，每参数组独立 lr | 步长几乎恒等于 lr（不因梯度小而衰减）；"收敛速度"是动量的速度，不是梯度的速度 |
| 3 | reset 后"allow culling to remove α<ε" | reset 到 **0.01**，而 ε = **0.005** → 当下**删不掉任何东西** | 别在 reset 那一步统计高斯数变化，会以为没生效 |
| 4 | 初始尺度 = "mean of the distance to the closest three points" | `sqrt(mean(d²))` = **RMS** 距离 | 初始高斯比论文描述**偏大**（偏态分布下约 1.2–1.4×） |
| 5 | 稠密化"every 100 iterations" | 只开 **500–15000** iter；且梯度是"按可见次数的平均" | 收敛曲线在 15k 处形状由实现决定，不由数据决定 |
| 6 | 剪枝"periodically remove" | 四条件 + 屏幕裁剪**只在 3000 iter 后开启** | 早期大高斯不受惩罚 |
| 7 | 论文 §4 没提投影后的低通 | 代码对所有高斯加 `+0.3` 到 2×2 协方差对角 | 小于像素的高斯不会消失/走样；去掉会掉 PSNR |
| 8 | 论文完全没提"白背景额外 reset" | `if it == densify_from_iter and white_background: reset_opacity()` | **synthetic 数据集多一次 reset**，与真实场景不可直接比 |
| 9 | §7.1 说前 500 iter 用 **1/4 分辨率 warm-up** | **代码里没有这一步**，全程全分辨率 | 论文的 7K/41min 数字与你复现的 7K 不是同一件事 |

### 4.1 clone 不动位置：论文比实现"理想"

**证据**：`scene/gaussian_model.py` 的 `densify_and_clone` 里，`new_xyz = self._xyz[selected_pts_mask]`——**原封不动复制**。论文 §5.2 那句 "by simply creating a copy of the same size, and moving it in the direction of the positional gradient" 在代码里**没有对应实现**。

**为什么会这样（我的推断，非原话）**：把新高斯沿梯度方向挪一点，等价于给它一个"出生速度"，而位置本来就会在下一个优化步自己朝梯度方向走。所以这一步是"可省的加速项",省掉不影响正确性，只影响前几十步的速度。

**你要注意什么**：
- 写报告时如果引用原文，**别顺手说"3DGS 让新高斯沿梯度方向偏移"**——审稿人（助教）如果对过代码，会认为你没读实现。
- 但反过来说，**如果你想让 3DGS 收敛更快**，这是一个免费的实验变量（"clone 时加 0.5×梯度方向的偏移"）。这可能是一个很小的加分项。

### 4.2 论文说 SGD，代码是 Adam（且 eps = 1e-15）

**证据**：`gaussian_model.py` 的 `training_setup`：`torch.optim.Adam(l, lr=0.0, eps=1e-15)`，每个参数组（xyz / f_dc / f_rest / opacity / scaling / rotation）各给一个 lr。

**为什么这件事重要**：
1. **Adam 有动量**。所以你看到的"收敛速度"不是"梯度下降的速度"，是"Adam 动量的速度"。你没法用 lr × 步数 去估"高斯挪了多远"。
2. **`eps=1e-15` 是个非常小的值**。默认 Adam 的 eps 是 1e-8，作用是在梯度极小时防止除零、同时**让二阶矩估计衰减**。1e-15 几乎等于关掉了这个保护——意味着**即使是极小的梯度也会产生接近 ±lr 的步长**。对 $\alpha$ 和 scale 这些"存的是 logit/log"的参数，这一步会放大噪声。
3. **densify/prune 重建参数 tensor 时，Adam 状态怎么处理？** 这是复现时最容易搞错的一点，代码里两种做法不同：
   - `cat_tensors_to_optimizer`（clone/split）：**老高斯的 exp_avg / exp_avg_sq 保留，新高斯补 0**——所以 densify **不会**打断动量。
   - `replace_tensor_to_optimizer`（opacity reset 用）：`exp_avg` / `exp_avg_sq` **全部置 0**——所以**每次 opacity reset 都会把不透明度的动量清零**。
   也就是说：**每 3000 iter，α 的优化会"重启"一次**。这解释了为什么 reset 之后 α 的学习曲线会有一个明显的"重新爬坡"段。

**对 Part 1 的意义**：如果你改了初始化点数（比如 VGGT 点云降采样到 5 万），densify 的**时机和次数**会变，而 Adam 的动量是按高斯逐个维护的——**点数不同，动量的统计性质就不同**。所以"只改初始化"这个说法在实现层面其实**改了两个东西**：初始几何 + 优化动力学。报告里承认这一点，比假装"只改了一个变量"更专业。

### 4.3 opacity reset 是 $\min(\alpha,0.01)$，而剪枝阈值是 0.005

**证据**：`reset_opacity`：`inverse_opacity_activation(torch.min(self.get_opacity, 0.01))`；`train.py` 里 `densify_and_prune(opt.densify_grad_threshold, 0.005, ...)`。

**两个反直觉的后果**：
1. **reset 不是把 $\alpha$ 设成 0**，而是"把大于 0.01 的压到 0.01，小于等于 0.01 的保持原样"。所以已经快死的高斯**不会因为 reset 而复活**，只是不再被"抬高"。
2. **0.01 > 0.005** → reset 之后**立即**执行的那次 prune 里，**没有任何高斯会被 $\alpha$ 条件删掉**。清理只能发生在后续迭代中，让那些"学不回来"的高斯自己掉到 0.005 以下。

**你要注意什么**：如果你在做"reset 频率"这个消融（比如 3000 vs 5000 vs 10000），**别用"reset 后高斯数下降多少"当指标**——那个数几乎恒为 0。应该看的是"reset 后的 200–500 iter 内高斯数下降多少"。

### 4.4 初始尺度是 RMS 距离，且位置学习率被场景尺度乘了一道

**第一条证据**：`create_from_pcd` 里 `dist2 = distCUDA2(...)`，`scales = torch.log(torch.sqrt(dist2))`。注意 `distCUDA2`（simple-knn 的 CUDA 核）返回的是**到 3 个最近邻的平方距离的均值**，所以 `sqrt` 之后是 **RMS 距离**，不是算术平均距离。对偏态分布（近处点密、远处点稀），RMS 比算术平均**大**。

**第二条证据**（这条更关键）：`training_setup` 里 `'lr': position_lr_init * self.spatial_lr_scale`，而 `spatial_lr_scale` 来自 `scene.dataset_readers.getNerfppNorm`，值是 $1.1\times$（相机中心到场景中心的**最大距离**）。

也就是说：**位置学习率是相对场景尺度归一化的**。设计意图很好（同一套超参能跑米级和单位球级场景），但它意味着：

> **如果你喂进去的点云尺度和相机尺度不一致，位置学习率就不是你以为的那个值了。**

这正是本仓库 `README.md §3.2` 那条"场景契约"存在的原因（`frames[].w2c` 与 `points3d.ply` 必须同尺度）。VGGT 输出的是"以首帧为参考的 OpenCV 系、尺度任意"——你把它转成 canonical 时**必须把点云和位姿放在同一个尺度里**，否则：
- `cameras_extent` 算出来是错的 → `percent_dense × extent` 错 → clone/split 判据错
- 世界空间剪枝阈值 `0.1 × extent` 错 → 删错东西
- 位置 lr 错 → 收敛速度不可比

**这条是你 P1 里最值得写进报告的"隐藏变量"。**

### 4.5 梯度判据是"按可见次数的平均值"，而训练是"每 iter 随机一帧"

**证据**：`add_densification_stats`：`xyz_gradient_accum[visibility_filter] += ‖grad‖`，`denom[visibility_filter] += 1`；`densify_and_prune` 里 `grads = xyz_gradient_accum / denom`。而 `train.py` 每个 iter **只取一张随机视角**（`randint`）。

**后果**：
- 一个高斯如果只在 1 帧里可见，`denom=1`，它的一次梯度就被当成"平均梯度"→ **极易越过阈值**被 clone/split。这在**稀疏视角**下尤其危险（因为很多高斯只被 1–2 帧看到）。
- 一个在 100 帧里都可见的高斯，要"平均梯度"大，需要持续被推——这是健康的判据。

**对你 P2 的直接意义**：稀疏视角下 `denom` 普遍偏小 → densify 触发**更激进** → 高斯数暴涨。这和 2.3.5 的推论是同一个机制的两个侧面。**所以"高斯数曲线"在 P2 里应该是一条急剧上升的曲线，而且上升速度随稀疏度增加**——这是一条可以写进报告的、可验证的预测。

### 4.6 屏幕空间剪枝要等 3000 iter 才开

**证据**：`train.py` 的 `size_threshold = 20 if iteration > opt.opacity_reset_interval else None`。

论文只说 "periodically remove"。"periodically" 的具体含义是：**前 3000 iter 不删大高斯，之后才删**。这与 opacity reset 同一时刻——因为"删大高斯"和"压不透明度"是同一套"清理浮空物"的策略，一起启用才合理。

### 4.7 那个 0.3 的低通滤波，论文 §4 完全没提

**证据**：`forward.cu` 里 `constexpr float h_var = 0.3f; cov.x += h_var; cov.z += h_var;`（对所有高斯无条件执行，不以 `antialiasing` 开关为转移）。

**它在干什么**：给屏幕上的 2×2 协方差对角线加 0.3（单位是像素²）。效果是**保证每个高斯至少覆盖约 1 个像素**。如果不加：
- 一个远小于 1 像素的高斯，采样点可能整个落在它的"支撑"之外 → **它在屏幕上消失**（它的贡献是 0，梯度也是 0）→ 永远学不好。
- 极端情况下会出现**高频闪烁**（camera 微微一动，高斯忽隐忽现）。

这是 EWA splatting 的标准做法，但论文 §4 一个字都没提。**你在读 §4 的时候如果觉得"投影公式到这就完了"，那就漏了一个必做的工程补丁。**

### 4.8 白背景场景多一次 opacity reset（synthetic 专用）

**证据**：`train.py`：
```
if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
    gaussians.reset_opacity()
```

也就是说，**当数据集是白背景（NeRF-synthetic 就是）时，在 `densify_from_iter = 500` 这一步会额外做一次 opacity reset**。论文完全没提。

**为什么会这么写（推断）**：synthetic 数据集是**纯白背景 + 物体**，物体外的大片白区最容易长出浮空高斯（因为白背景很容易被"糊"出来）。所以在刚开始 densify 时就先压一次，**趁高斯还少，把这些"一出生就想糊背景"的干掉**。真实数据集（有复杂背景）没有这个必要。

**对你的意义**：如果你的额外数据里有 synthetic（比如你想加 NeRF-synthetic 做对照），**必须知道这个分支**，否则你和论文的数字对不上，而且找不到原因。

### 4.9 论文 §7.1 的分辨率 warm-up，代码里完全没有

**证据**：论文 §7.1 明写 "we start the optimization using 4 times smaller image resolution and we upsample twice after 250 and 500 iterations"。而在本仓库 `third_party/gaussian-splatting/` 里 `grep` 不到任何 `upsample` / 分辨率 warm-up 逻辑：`train.py` 从第 1 个 iter 就用 `scene.getTrainCameras()` 里的最终分辨率图像，`arguments/__init__.py` 里也没有对应参数。

**为什么这件事比它看起来重要**：

1. **前 500 iter 的"训练成本"不同**。论文前 500 步只花约 1/16 的渲染成本，你的前 500 步是全成本。所以**你不能用论文的 41m33s 来估你的训练时间**。
2. **"7K"这个标签不可比**。论文的 Ours-7K 是"warm-up 之后又训了 6500 步"，代码的 7K 是"从头全分辨率 7K"。清单里那个"3DGS 收敛速度"实验如果要和论文数字对话，**必须说明这一点**。
3. **它影响早期 densify 的行为**。densify 从 500 iter 开始，而论文正好在 500 iter 才回到全分辨率——"先看清轮廓、再决定哪里缺"是一个连贯的设计。代码里 500 iter 时已经全分辨率跑了一段，判据看到的图和论文不是一回事。
4. **这也是很多"复现不掉"现象的来源之一**。如果你在 7K checkpoint 上和论文的 25.60 dB 对比，差个零点几 dB 很正常，**别去怀疑数据或位姿**。

**你可以怎么利用它**：这本身是一个免费的、可写进报告的实验变量——"在代码里补上论文说的分辨率 warm-up，看 5K/7K 的 PSNR 和最终 PSNR 变多少"。如果正收益明显，这就是一条很实在的小贡献（而且能证明你真的逐字读了 §7.1）。

---

## 第五部分 · 自测 13 问

> 答案折叠。**考机制，不考名词**——能答上第 2、4、6、9、13 题，说明你真的读懂了。

<details><summary>1. 一个高斯有几个参数？为什么旋转存 4 个、缩放存 3 个，而不是存 9 个的旋转矩阵和 6 个的协方差？</summary>

59 个：位置 3、缩放 3（存 log）、旋转 4（单位四元数）、不透明度 1（存 logit）、球谐 48（3 通道 × 16 系数）。

不存 3×3 矩阵的原因：① 参数量少（4 vs 9，3 vs 6）；② **合法性是结构自带的**——单位四元数归一化后必然是旋转，`exp(s)` 必然是正尺度，于是 $RSS^\top R^\top$ 必然半正定，不需要任何约束或投影；③ 由 2×2 的 $RS$ 组合出协方差只是一次小矩阵乘法，开销可忽略。
</details>

<details><summary>2. 透视投影是非线性的，为什么 3D 高斯投影到屏幕后还能被当成椭圆？这个近似在哪里会失效？</summary>

因为它没有用真实的透视投影，而是**在均值处做一阶泰勒展开**，用雅可比 $J$ 代替投影函数（"局部仿射近似"）。仿射变换把高斯映成高斯，所以投影后仍是高斯，新协方差 = $J W\Sigma W^\top J^\top$，再砍掉第三行第三列得到 2×2 的屏幕脚印。

失效的地方：**离均值越远，一阶近似误差越大**。所以①超大椭圆（很近的物体、或低分辨率下被放大）边缘不准；②跨越大深度范围的投影（同样的高斯，一部分离相机近一部分远）最糟。这也解释了为什么"远景大平面"和"极近物体"是 3DGS 的典型弱点。
</details>

<details><summary>3. 为什么不直接优化 6 个独立的协方差元素？</summary>

因为梯度下降无法保证结果仍是**半正定**矩阵。一步更新就可能让某个特征值变负，此时 $\Sigma^{-1}$ 无意义、椭球变成双曲面，优化直接崩（表现为 NaN 或几何爆炸）。

用 $s,q$ 参数化之后，"半正定"从**约束**变成了**结构性质**：只要 $q$ 单位、$s$ 为正（`exp` 保证），$RSS^\top R^\top$ 必然合法。这是"用参数化吸收约束"的标准套路。
</details>

<details><summary>4. `clone` 和 `split` 的触发条件分别是什么？为什么判据里要包含"高斯本身大不大"？</summary>

两者都要求**平均屏幕空间位置梯度 > 0.0002**（"这里还没重建好"），区别在**高斯自身的尺度**：

- `max(s) ≤ 0.01 × 场景半径` → **clone**：复制一份。语义是"这里缺体积，需要更多高斯来填"。
- 否则 → **split**：换成 2 个，尺度各除以 1.6，位置从原高斯的分布里采样。语义是"体积够了，但一个椭球跨了多个表面，要拆细"。

判据里必须含"大小"，因为梯度大只说明"没重建好"，**没说明是缺还是多**。统一用一种会坏：全 clone → 过重建区域的高斯数爆炸、错误被复制；全 split → 欠重建区域的稀疏高斯被切得更碎，洞永远补不上。

论文 Table 3 的消融从侧面支持这个分工：**拆掉 split 掉 2.15 dB（30K），拆掉 clone 只掉 0.14 dB**。也就是说 split 负责"最终精度"（背景大平面的重建），clone 主要负责"早期收敛速度"（细结构）。
</details>

<details><summary>5. `opacity reset` 解决什么问题？不做会怎样？它和剪枝是什么关系？</summary>

解决**相机附近的浮空高斯（floaters）**：它们只为了在某几个视角上"糊住一块像素"而生，几何上是垃圾，但能降低损失，所以优化器不但不删它们，还会因为它们梯度大而继续 clone 繁殖。

reset 的做法是每 3000 iter 把所有 $\alpha$ 压到接近 0：真正有多个视角支撑的高斯会迅速把 $\alpha$ 学回来；纯糊的学不回来，$\alpha$ 持续走低，最终被 $\alpha<0.005$ 的剪枝删掉。

不做的话：高斯数单调暴涨（浮空高斯自己繁殖）、显存与训练时间失控、近景出现半透明雾状层——**密度控制机制被浮空高斯劫持**，是结构性失控而非"稍差一点"。

和剪枝的关系：reset **不直接删任何高斯**，它只是"把分数清零，让下一轮考试重考"。删人是剪枝干的。
</details>

<details><summary>6. 为什么损失里需要 D-SSIM？只用 L1 会怎样？为什么 λ 只有 0.2？</summary>

L1 是逐像素绝对差，对**结构**不敏感（整块偏移和逐像素各错一点，L1 可能差不多），且在**平坦区域**梯度信号弱（白墙上下挪一点，L1 几乎不变）。D-SSIM 在局部窗口上比较均值/方差/协方差，天生对结构、对比度、边缘敏感，补的正是这一半。

所以分工是：L1 守 PSNR，D-SSIM 守主观质量/SSIM。只用 L1 会更"糊"（虽然 PSNR 未必低）；只用 D-SSIM 指标会难看。

λ=0.2 是论文给的经验值，**§7.3 里没有 λ 的消融**，所以别声称它是最优的。我的理解（非原文）：D-SSIM 权重过大会过度优化局部统计量、压平高频纹理。复现时不要改动，动了就和论文不可比。
</details>

<details><summary>7. 光栅化为什么是"整图排一次序"，而不是"每像素排序"？这个选择的代价是什么？</summary>

代价对比：每像素排序意味着一个大高斯要被插入它覆盖的每个像素的列表（几十万次），且无法复用；整图排序只需一次 GPU 基数排序，排完每个 tile 拿一段连续区间即可。

代价是**正确性打了折扣**：全局排序只保证"大致按深度"，当两个高斯在某个像素处的真实深度顺序与全局顺序不一致（深度交错），α 混合就不是严格正确的。论文的辩护是：**当 splat 尺寸接近单像素时，这种交错几乎不存在，误差可忽略**。反过来说，高斯训得特别大（或分辨率很低）时，误差会显现为深度交替处的颜色错误。
</details>

<details><summary>8. 反向传播为什么"重走一遍"比"存下每个像素的混合列表"更好？它凭什么只要常数内存？</summary>

朴素方案要给每个像素存"混了哪些高斯、各自 $\alpha$ 多少"，长度不定 → 要么开大上界（浪费），要么动态分配（GPU 上很慢）。

3DGS 的方案：前向时每个（像素,高斯）只存**一个数**（混合结束后的最终累计不透明度），反向时**从后往前走**，用"当前累计量 ÷ 该点自己的 $\alpha$"**除法还原**出该点之前的透射率。因为透射率本来就是连乘，从最后一项一路除回去即可。

收益：**每像素额外内存是 O(1)**，与混合了多少个高斯无关。而且它顺带实现了"不限制接收梯度的高斯个数"。
</details>

<details><summary>9. 投影公式在论文里是 JWΣWᵀJᵀ，代码里写的是 TᵀΣT。哪个对？</summary>

两个都对，是**同一个式子的两种转置约定**。论文用列向量约定（$p_{cam}=Wp_{world}$，协方差变 $A\Sigma A^\top$），代码用行向量约定（`cov = transpose(T)*transpose(Vrk)*T`，其中 `T = W*J`）。对同一个物理变换，取 2×2 子块后得到的屏幕椭圆是同一个。

**读代码时的实用结论**：别纠结转置，只要确认①$W$ 是 world→camera 的旋转部分，②$J$ 里有 $f_x/z$ 和 $f_x x/z^2$ 这类项，就说明对上了。
</details>

<details><summary>10. 如果我把 COLMAP 的位姿和点云换成 VGGT 的，但忘了统一尺度（点云在米、位姿在任意尺度），会发生什么？</summary>

三处会一起坏：

1. `getNerfppNorm` 算出的 `cameras_extent` 只从**相机中心**算，与点云尺度不一致 → `percent_dense × extent` 这个 clone/split 的判据错位 → 要么全 clone（高斯数爆炸），要么全 split（细节永远补不上）。
2. 世界空间剪枝阈值 `0.1 × extent` 错 → 把正常高斯当"过大"删掉，或该删的没删。
3. 位置学习率被乘了 `spatial_lr_scale`（∝ extent）→ 有效步长差好几个数量级，收敛曲线完全不可比。

**这也是本仓库 README §3.2 场景契约那条"frames[].w2c 与 points3d.ply 必须同尺度"存在的原因。**
</details>

<details><summary>11. 论文说"clip/split 每 100 iter"，那为什么收敛曲线在 15000 iter 处会有拐点？</summary>

因为代码里 densify 只开在 **500–15000** iter（`densify_from_iter=500`、`densify_until_iter=15000`）。15000 之后高斯数量**冻结**，只剩纯优化。

所以 PSNR-vs-iteration 曲线的"形状"由三件事叠加决定：①密度控制在 15k 前不断加容量；②15k 后容量固定、只精修参数；③位置学习率一直在指数衰减。**如果你拿 7k / 15k / 30k 三个 checkpoint 比三种初始化，7k 和 15k 的差异有很大一部分是"densify 还在进行"，不是"初始化更优"。** 这是 P1 收敛曲线实验里必须说明的混淆因素。
</details>

<details><summary>12. 稀疏视角下，为什么高斯数会暴涨，而且空白区域是雾？</summary>

两个机制同时发生：

① **判据失真**：densify 用的是"按可见次数的平均屏幕梯度"。稀疏视角下多数高斯只被 1–2 帧看到，`denom` 很小 → 一次梯度就被当平均值 → 轻易越过 0.0002 → 过度 clone/split。

② **没有梯度的区域**：大片区域在所有训练视角里都不可见 → 光度梯度为 0 → 判据为 0 → **永远不触发补高斯**。这些地方的几何完全由初始化和正则"猜"，初始猜成什么样就是什么样（通常是雾/浮空）。

也就是说：**3DGS 的密度控制完全由观测驱动，它没有"这里应该有个平面"的先验**。观测不到的地方，它既不会长大东西，也不知道自己长错了。——这就是 P2/P3 要引的那句话。
</details>

<details><summary>13. 论文 Table 3 说"把 SfM 初始化换成随机点，平均掉 5.6 dB"；可 Table 2 又说"在 synthetic 上我们用 10 万个随机点就拿到了 SOTA"。这两句矛盾吗？哪一句该写进你的报告？</summary>

不矛盾，两句都对——**它们描述的是两种不同的场景结构**，而这个区别恰好是你 Part 1 里最该写清楚的一段话。

- **Truck / Garden / Bicycle（Table 3）**：无界户外场景。有大片背景、有天空、相机只沿着一条路径扫过去 → **大量表面只被少量视角看到，甚至完全看不到**。这时初始化在做什么？它是在"猜那些没被约束的区域"。随机点 = 完全乱猜 → 背景崩、浮空物删不掉 → 掉 5.6 dB。
- **Synthetic（Table 2）**：一个孤立物体 + 纯白背景，相机绕半球密集拍摄 → **每个表面都有大量、多方向的观测**。这时初始化几乎不重要，因为梯度会把随机点"拉"到该去的地方。100K 随机点被自动剪到 6–10K，最终 200–500K，拿到 33.32 dB 的 SOTA。

**所以正确的表述是**："3DGS 对初始化敏感，但**敏感程度取决于观测约束的强弱**——无界、背景大、覆盖不全的真实场景最敏感；物体级、密视角、纯背景下几乎不敏感。"

**该写进报告的是这个条件化的版本**，而不是"3DGS 对初始化很敏感"这种无条件断言。理由：你的三个必做场景全是真实场景（户外/室内、背景占比大），所以**你的实验落在"敏感"那一侧**——但如果你把结论写成无条件的，助教会问"synthetic 上呢"。
</details>

---

## 第六部分 · 速查卡 + 三分钟口头版

### 6.1 一页速查卡

```
【一句话】
场景 = 几百万个【各向异性 3D 高斯】；渲染 = 把它们【投影成屏幕椭圆】再【按 tile 排序 + α 混合】；
训练 = 用【随机视角的 L1+D-SSIM】反传，同时【按屏幕空间梯度】clone/split 改密度。

【一个高斯 = 59 个数】
位置 3 + 缩放 3(log) + 旋转 4(单位四元数) + 不透明度 1(logit) + 球谐 48(3ch × 16)
→ 不存协方差矩阵！Σ = R S Sᵀ Rᵀ 是算出来的，所以天然半正定。

【两个核心公式】
Σ   = R S Sᵀ Rᵀ                       (Eq.6，合法性来自参数化)
Σ'  = J W Σ Wᵀ Jᵀ                     (Eq.5，J = 透视投影在 μ 处的雅可比)
L   = 0.8·L1 + 0.2·(1−SSIM)           (Eq.7，λ=0.2)

【训练超参（λ、τ_pos 来自论文；其余数字只存在于代码里）】
iterations 30k | densify 每 100 步，仅 500–15000 | τ_pos = 0.0002 | split φ = 1.6
opacity reset 每 3000 → α ← min(α, 0.01) | prune α < 0.005 | 屏幕半径 > 20px（3000 步后才开）
位置 lr 1.6e-4 → 1.6e-6（指数衰减，其余常数）| Adam, eps=1e-15
图像宽度 > 1600 px 自动降采样（1600 本身不降！）

【输入输出】
输入：图像 + 相机(K, W) + SfM 稀疏点云（含颜色）  →  输出：一个 .ply（59×N 个 float）
初始化：μ = 点位置；s = 到 3 近邻的 RMS 距离（各向同性）；α = 0.1；SH = 点颜色 → DC

【论文 Table 1 关键数字（A6000 上测）】
Mip-NeRF360 / T&T / DeepBlending 三个块，Ours-30K 分别是：
  PSNR 27.21 / 23.14 / 29.41，SSIM 0.815 / 0.841 / 0.903，LPIPS 0.214 / 0.183 / 0.243
  训练 41m33s / 26m54s / 36m2s，FPS 134 / 154 / 137，Mem 734MB / 411MB / 676MB
对照：M-NeRF360 48h、0.06 FPS、8.6MB；INGP-Base 5m37s、11.7 FPS、13MB
显存换算：≈ 236 B / 高斯 → 734MB ≈ 3.1M 高斯（训练峰值 >20GB，见 §7.4）

【初始化有多重要（Table 3，SfM vs 随机点）】
Avg-30K：Full 26.05 vs Random Init 20.42 → 差 5.6 dB；Avg-5K 23.90 vs 19.17 → 差 4.7 dB
但 synthetic 上 10 万随机点就能到 SOTA（Table 2）→ 敏感的是"无界 + 背景大 + 覆盖不全"的真实场景

【为什么快】
整图排一次序（不是每像素）+ tile 内共享内存 + α 饱和即停 + 反向"重走一遍"用除法还原 T（每像素 O(1) 内存）
```

### 6.2 容易记错的坑（三个月后你只会回来看这一段）

| # | 坑 | 正确理解 |
|---|---|---|
| 1 | "3DGS 每个高斯存协方差矩阵" | ❌ 存 $s$(3) 和 $q$(4)，$\Sigma$ 是**算出来的** |
| 2 | "投影是精确的透视投影" | ❌ 是**局部仿射近似**（雅可比 $J$），离均值越远越不准 |
| 3 | "clone 时新高斯会沿梯度方向偏移" | ❌ 论文这么写，**代码没做** |
| 4 | "opacity reset 把 α 设成 0" | ❌ 是 $\min(\alpha, 0.01)$；而且 **0.01 > 剪枝阈值 0.005**，reset 当下删不掉任何东西 |
| 5 | "densify 一直开到 30k" | ❌ 只到 **15000**；之后高斯数冻结 |
| 6 | "论文说 SGD" | ❌ 代码是 **Adam (eps=1e-15)**；且 reset 会清零 α 的动量 |
| 7 | "1600 px 上限对所有图生效" | ⚠️ 条件是 `width > 1600`，**宽度正好 1600 的不降**（本项目 meta.yaml 里 Waymo 是 1600×1067，正好卡在这条线上：**不会**被降采样，显存和耗时都要按全分辨率算） |
| 8 | "论文 §4 的投影公式就是全部" | ❌ 还有一步没写的 `+0.3` 像素低通（否则小于 1 像素的高斯会消失） |
| 9 | "稀疏视角下高斯数会变少（信息少了）" | ❌ 恰恰相反：**暴涨**，因为可见次数少 → 平均梯度虚高 → 过度 densify |
| 10 | 符号：$W$ 是 world→camera（不是 camera→world）；$J$ 是 2×3 的雅可比（代码里写成 3×3、最后一行全 0） | 论文 $W$ 是 viewing transformation，作用在列向量上；代码里还有一层转置约定。论文说"skip 第三行第三列"，是因为它把 $\Sigma'$ 当 3×3 写 |
| 11 | "论文的 7K 就是训 7000 步" | ⚠️ 论文的 7K 前面还有一段 1/4→1/2→全分辨率的 warm-up（§7.1），**代码里没实现** |

### 6.3 三分钟口头版（讲给组员/助教）

> 3DGS 的出发点是：NeRF 那个体渲染公式没错，错的是**执行方式**——沿光线撒点、每点问一遍 MLP，太慢。同样的公式，如果场景里本来就有几百万个"点"，那我只要把它们**按深度排好序**，让 GPU 做 alpha 混合就完事了。
>
> 难点在"点"不能是真正的点，否则屏幕上就是一堆洞。所以每个点被换成一个**各向异性椭球**（3D 高斯）：可旋转、可拉伸，一个压扁的椭球正好能贴住一片薄墙。一个椭球 59 个参数（位置 3、缩放 3、旋转 4、不透明度 1、球谐颜色 48）。
>
> 关键是"椭球投影到屏幕还是椭圆"——透视投影是非线性的，所以用**均值处的雅可比做一阶近似**，于是 $\Sigma'=JW\Sigma W^\top J^\top$，投影后仍是椭圆。这一步是整个方法能实时的前提：只有椭圆才能 O(1) 算覆盖了哪些 tile、O(1) 算像素不透明度。
>
> 训练就是把一堆高斯扔进去，随机取视角渲染、和真图比 L1+SSIM、反传。但光靠优化参数不够——投影是多对一的，几何一定会放错位置。所以每 100 步要判断"哪里缺、哪里多"：**屏幕空间位置梯度大 = 没重建好**；高斯小就 clone（缺体积），高斯大就 split（形状不对）。再每 3000 步把所有不透明度压到 0.01，让那些"只为了糊住某个视角"的浮空高斯饿死。
>
> 对我们 Part 1 来说最关键的一点：**这套机制完全是观测驱动的**。初始点云决定了一开始有几个高斯、各多大（到 3 近邻距离），而这个初始分布会一路影响 densify 的时机、最终高斯数、以及收敛曲线的形状。所以换初始化不只是"换一套位姿"，是**换了整个优化轨迹的起点**——报告里必须把"几何质量"和"点云密度分布"分开讲。

### 6.4 和 Part 1 的接口：读完这篇你应该能填的表

这张表是你读完 3DGS 之后的**直接产物**，可以原样搬进报告的 Method 一节：

| 接口点 | COLMAP 给什么 | VGGT 给什么 | 必须自己做的决定 |
|---|---|---|---|
| 位姿 | `qvec/tvec`（w2c） | 以首帧为参考的 OpenCV 系 | 转成 canonical（见 README §3.2） |
| 内参 | 估出来的 `PINHOLE` | 逐帧预测焦距 | 报告里给两者差多少 %，并做一次"用 GT 内参"的对照 |
| 点云数量 | 几万 | 逐像素点图 → 上百万 | **降采样 + 置信过滤到同量级**（否则初始点数变混淆变量） |
| 点云颜色 | 有 | **没有** | 从像素取 RGB 填 SH 的 DC 项（否则初始条件不等价） |
| 点云尺度 | 与相机同尺度 | 任意尺度 | **必须和位姿统一尺度**（否则 `extent` / 位置 lr 全错，见 4.4） |
| 场景 extent | `getNerfppNorm` 从相机算 | 同 | 两者都算出来对一下，差太多说明尺度有问题 |

---

## 附：接下来的阅读顺序

1. **本文第四部分 → 第五部分**（如果时间紧，先做这两节，它们比正文更能帮你写报告）
2. 回论文读 **§7.3 的 "Initialization from SfM" 那一段**（本文 2.6）：那是唯一直接讨论"初始化影响"的实验，也是你 P1 的对照依据
3. 按 `notes/READING_P1.md` 的清单去读 **`scene/dataset_readers.py` 的 `readColmapSceneInfo`**——那是 `21_to_3dgs.py` 要照抄的格式契约
4. 然后才是 VGGT（清单里的精读 2）

> ⚠️ 最后提醒一句：读 3DGS 的时候**不要顺手去读 Related Work**。清单说得对，那 3 页是给审稿人看的。你真正需要从 Related Work 里拿的只有一句话——"点云的 alpha 混合和 NeRF 的体渲染是同一个成像模型"（§2.3，我在 1.1 已经替你讲了）。
