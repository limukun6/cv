# Part 1 阅读清单

> **给谁看**：负责 Part 1（初始化方式对比）的同学。
> **总时间**：精读 3 篇（约 8 小时）+ 略读 3 篇（约 1.5 小时）。分两天读完最舒服。
> **核心原则**：不是"读完"，是**读到能回答问题**。每篇下面都列了"读完必须能回答什么"，答不上就回去补那一段。

---

## 一张表看完

| 级别 | 论文 | 链接 | 读哪几节 | 时间 |
|---|---|---|---|---|
| 🔴 精读 | **3DGS** | [arXiv:2308.04079](https://arxiv.org/abs/2308.04079) | §3 全部 + §4 | 3–4 h |
| 🔴 精读 | **VGGT** | [arXiv:2503.11651](https://arxiv.org/abs/2503.11651) | §1 + §3 + Table 1 | 2–3 h |
| 🔴 精读 | **VGGT-Ω** | [arXiv:2605.15195](https://arxiv.org/abs/2605.15195) | 摘要 + §1 + 三处改动 | 2 h |
| 🟡 略读 | COLMAP | [官网文档](https://colmap.github.io/) | Tutorial 两页 | 1 h |
| 🟢 可选 | DA3 | [arXiv:2511.10647](https://arxiv.org/abs/2511.10647) | 摘要 + 总览图 | 20 min |
| 🟢 可选 | Pi3 | [arXiv:2507.13347](https://arxiv.org/abs/2507.13347) | 摘要 | 15 min |
| ⚪ 暂不读 | LingBot-Map | [arXiv:2604.14141](https://arxiv.org/abs/2604.14141) | 摘要 | 15 min |
| ⚪ 暂不读 | HorizonStream | [arXiv:2605.23889](https://arxiv.org/abs/2605.23889) | 摘要 | 15 min |

---

## 🔴 精读 1：3DGS

**《3D Gaussian Splatting for Real-Time Radiance Field Rendering》** — SIGGRAPH 2023
[arXiv:2308.04079](https://arxiv.org/abs/2308.04079) · [项目页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

> 📘 **配套精读文档 → [`READING_3DGS.md`](READING_3DGS.md)**：指定部分的逐节精读（📄原文/🗣大白话/🧩伪代码）、**论文 vs 官方代码对照的 9 个坑**、Table 1/2/3 的逐格数字、§7.3 初始化消融、13 道自测题。
> ⚠️ 注意：本文下面的 §3.1 / §3.2 / §4 是博客式编号，**对应论文的 §4 / §5 / §6 + §7.1**，对照表见配套文档 §0.2。

### 读哪几节

| 节 | 内容 | 为什么 |
|---|---|---|
| §3.1 | 三维高斯 + 投影 | **整篇的数学核心**。重点看 $\Sigma' = JW\Sigma W^TJ^T$ |
| §3.2 | 优化 + 自适应密度控制 | clone/split/opacity reset 全在这 |
| §4 | 实现细节 | 光栅化怎么做的，为什么快 |
| Table 1/2 | 速度和质量对比 | P1 报告要引用的数字 |

**跳过**：Related Work（全部）、附录、补充视频。

### 读完必须能回答

1. 一个高斯有几个参数？各是什么、各几维？（答案：位置 3 + 缩放 3 + 旋转 4 + 透明度 1 + 球谐 48 = 59）
2. 三维椭球投影到屏幕后**为什么还是椭圆**？这一步为什么是整个方法成立的前提？
3. `clone` 和 `split` 分别在什么条件下触发？为什么不统一用一种？
4. `opacity reset` 解决什么问题？不做会怎样？
5. 损失函数是什么？$\lambda$ 取多少？为什么需要 DSSIM 而不只用 L1？

### 读的时候特别注意

- 它**不是**神经网络。整个场景就是一堆显式参数，优化它们就等同于"训练"。
- 渲染是**光栅化**（splatting），不是 NeRF 那种光线步进。这是速度的来源。
- 它对**初始化**极度敏感——这正好是 Part 1 的研究对象。

---

## 🔴 精读 2：VGGT

**《VGGT: Visual Geometry Grounded Transformer》** — CVPR 2025
[arXiv:2503.11651](https://arxiv.org/abs/2503.11651) · [代码](https://github.com/facebookresearch/vggt) · [项目页](https://vgg-t.github.io/)

### 读哪几节

| 节 | 内容 | 为什么 |
|---|---|---|
| §1 引言 | 它想解决什么问题 | 一句话：**把多视图几何变成一个回归问题** |
| §3.1 | 网络结构（交替注意力） | 关键：帧内注意力 + 全局注意力交替 |
| §3.2 | 预测头（相机/深度/点图） | 输出了哪些东西，各什么形状 |
| Table 1 | 相机参数估计精度 | 和 COLMAP 的直接对比，**P1 必引** |

**跳过**：3D 点跟踪、下游任务（非刚体跟踪、前馈新视角合成）——那些和 Part 1 无关。

### 读完必须能回答

1. 输入是什么、输出有哪四项？
2. 它**没有**做哪些传统步骤？（特征提取、匹配、RANSAC、三角化、束调整——全都没有）
3. "帧内注意力 + 全局注意力交替"是为了什么？全用全局注意力不行吗？
4. 它的相机坐标系是相对谁的？（答：以第一帧为参考）
5. **显存随什么增长？** 为什么？（答：随帧数大致平方增长——这直接决定了我们在 8 GB 上能跑几帧）

### 读的时候特别注意

第 5 个问题是我们这台机器的**瓶颈所在**。读到相关处，想一下：200 帧塞进去要多少显存？这就是为什么我们可能要"三边统一用 20 帧".

---

## 🔴 精读 3：VGGT-Ω

**《VGGT-Ω》** — CVPR 2026 Oral
[arXiv:2605.15195](https://arxiv.org/abs/2605.15195) · [项目页](http://vggt-omega.github.io/)

### 读哪几节

只读**摘要 + §1 + 方法里三处改动的段落**。这篇的细节不用全啃，重点是搞清"它相对 VGGT 改了什么、为什么能放大"。

### 三个必抓的点

| 改动 | 作用 |
|---|---|
| **单头多任务 + 去掉高分辨率卷积层** | 简化架构，训练更快 |
| **registers + register attention** | 用少量"寄存器 token"汇总场景信息，**跨帧信息只通过寄存器交换**，替代部分全局注意力 |
| **15× 监督数据 + 自监督协议** | 规模放大 |

关键数字（摘要原文）：
- 训练时显存只需 VGGT 的 **30%**
- 监督数据多 **15 倍**
- Sintel 上相机估计精度提升 **77%**

### 读完必须能回答

1. 相对 VGGT，它改了哪三处？
2. 为什么显存能降到 30%？（提示：register attention 把跨帧注意力从"全连接"变成"经过寄存器中转"）
3. **⚠️ 最重要的一题**：那 77% 的提升，是"数据多了"带来的，还是"结构更适合放大"带来的？**这两件事怎么从实验上分离？**

> 第 3 题就是 Part 1 报告里最有价值的一段讨论。原文的 Key Analysis 明确要求你讨论 "what further scaling brings"。如果你只写"VGGT-Ω 比 VGGT 好"，这一段就白丢了。

---

## 🟡 略读：COLMAP

**不要读论文**（Building Rome in a Day 太老，而且讲的是大规模场景，和你的需求不匹配）。

**读官网文档更有用**：[https://colmap.github.io/tutorial.html](https://colmap.github.io/tutorial.html)

只看这两部分：

1. **Structure-from-Motion → Incremental**：四个步骤（特征提取 → 匹配 → 几何验证 → 增量重建）
2. **什么情况下会失败**：纹理重复、反光、纯色、视角间隔太大

### 读完必须能回答

1. 增量式 SfM 的四个步骤分别是什么？
2. 什么是"注册"（registration）？为什么会"注册失败"？
3. 束调整（Bundle Adjustment）在优化什么？为什么它是瓶颈？

> 这三个问题的答案，就是 Part 2 里"为什么稀疏视角下 COLMAP 会崩"的理论基础。

---

## 🟢 可选：DA3 / Pi3

**只读摘要 + 一张总览图**，目标是一句话说出"和 VGGT 的差别在哪"：

- **DA3**（Depth Anything 3）：[arXiv:2511.10647](https://arxiv.org/abs/2511.10647) —— 主打"从任意视角恢复视觉空间"，通用性强。⚠️ 注意：i3dgs 就是用它来做位姿 bootstrap 的。
- **Pi3**（Permutation-equivariant）：[arXiv:2507.13347](https://arxiv.org/abs/2507.13347) —— 主打**置换等变**，即**输入的图片顺序不影响输出**。这是个很漂亮的归纳偏置。

**这两个是 P1 的加分项**（原文列为 "Advanced Exploration"）。时间不够就不做，但摘要值得看——报告 Related Work 里能引。

---

## ⚪ 暂不读（但要知道它们是什么）

| 论文 | 一句话 | 什么时候才需要读 |
|---|---|---|
| LingBot-Map [2604.14141] | 流式重建（一次吃一段，滚动更新） | 想做"离线 vs 在线"这条对比轴时 |
| HorizonStream [2605.23889] | 长时程注意力的流式重建 | 同上 |
| Scaffold-GS | 锚点式结构化高斯 | 想换 3DGS 底座时 |
| Wavelet-GS | 小波分解的高斯表示 | 同上 |

> 这几篇**现在完全不用读**。它们的价值是"如果你想让 P1 从复现升级成 benchmark"，那才需要。

---

## 不必读（省下的时间干别的）

| 论文 | 为什么不用读 |
|---|---|
| Waymo / DL3DV / Re10k 数据集论文 | Related Work 引一句"我们用 XXX 数据集"就够 |
| Mip-NeRF 360 / Tanks & Temples | 同上 |
| NeRF 原论文 | 除非你不懂体积渲染。不懂就读 §4（体渲染方程那一节），够用 |
| Building Rome in a Day | 读 COLMAP 文档更实用 |

**很多同学在这几篇上浪费两天，纯属自杀。**

---

## 仓库：只读这几个文件（比读论文更实用）

### 3DGS（`third_party/gaussian-splatting/`）

| 文件 | 看什么 | 为什么 |
|---|---|---|
| `arguments/__init__.py` | 所有默认超参 | 我们要改分辨率和迭代次数就在这 |
| `scene/dataset_readers.py` | `readColmapSceneInfo` 函数 | **★ 最重要**：搞清它期望的目录结构和二进制格式，`21_to_3dgs.py` 就照它写 |
| `scene/gaussian_model.py` | `create_from_pcd` / `densify_and_clone` / `densify_and_split` | 对应论文 §3.2 的三个函数 |
| `train.py` | 主循环 | 看 `scene.loaded_iter`、`opt.iterations` 怎么用 |
| `render.py` | 怎么渲染测试视角 | 写 `40_eval_render.py` 时照抄 |

### VGGT（将来 clone 到 `third_party/vggt/`）

| 文件 | 看什么 |
|---|---|
| `README.md` | 推理示例那一段 |
| `vggt/models/vggt.py` | `forward()` 的签名和返回值 |
| 输出字段 | `pose_enc` / `depth` / `point_map` 各是什么形状、什么坐标约定 |

> **VGGT 的位姿输出是"以第一帧为参考的 OpenCV 系"** —— 记住这句，`11_run_vggt.py` 要把它转成我们的 canonical 格式。

---

## 带着这三个问题读（贯穿性）

这三题没有标准答案，但读的时候一直想着，读完你的报告就有主线了：

1. **同一个场景，COLMAP 和 VGGT 给出的位姿差多少？差在旋转还是平移？**
   → 提示：COLMAP 靠几何，误差通常是"局部漂移"；VGGT 靠统计，误差可能是"系统性偏置"。这两种误差对 3DGS 的影响**完全不同**。

2. **初始化质量差 1 度，最终 PSNR 差多少？**
   → 这决定了 P1 的结论强度。如果差 1 度只掉 0.2 dB，那"用 VGGT 初始化"的实用价值就很高。

3. **"scaling" 到底是数据量还是结构？**
   → VGGT-Ω 同时放大了两者，怎么分离？可以考虑：VGGT-Ω vs VGGT 在**同一帧数**下的对比，以及**同一模型**在不同帧数下的对比。

---

## 读完自测（能答上就过关）

<details><summary>1. 3DGS 里一个高斯有多少个参数？为什么不直接用球体？</summary>

59 个：位置 3、缩放 3、旋转 4（四元数）、透明度 1、球谐系数 48（degree 3，每通道 16）。不用球体是因为真实表面是二维薄片，球形高斯需要叠很多层才能填满，浪费；压扁的椭球贴一层就够。
</details>

<details><summary>2. 为什么 3DGS 能实时而 NeRF 不能？</summary>

NeRF 每条光线要采样上百个点、每点过一遍 MLP，1080p 一帧 = 2 亿次网络调用。3DGS 是显式表示，渲染只需把高斯投影到屏幕然后按深度 alpha 混合，是 GPU 光栅化硬件擅长的操作。
</details>

<details><summary>3. VGGT 为什么不需要特征匹配和束调整？它的代价是什么？</summary>

因为它把整个多视图几何学成了一个端到端回归问题，跨视图对应关系隐含在注意力里。代价是：没有显式几何保证、精度不可解释、显存随帧数平方增长、对训练分布外场景泛化有限。
</details>

<details><summary>4. VGGT-Ω 的 register attention 具体解决了什么？</summary>

原来跨帧所有 token 两两做注意力（O(N²)），显存爆炸。改成：每帧先把信息汇总到少量"寄存器 token"，跨帧只在这些寄存器之间交换。复杂度大降，显存只要 30%，于是能用 15 倍数据训更大的模型。
</details>

<details><summary>5. 为什么 COLMAP 在稀疏视角下会崩，而 VGGT 不会（至少不会崩得那么突然）？</summary>

COLMAP 靠"匹配到足够的特征点对"来注册新视角，视角一旦稀疏、重叠一低，匹配就断，增量式注册直接卡死——是**硬失败**。VGGT 没有"注册"这个概念，它总是给出一个答案，只是稀疏时答案**更不靠谱**——是**软退化**。这个区别是 Part 2 的核心叙事。
</details>

---

## 阅读顺序建议

```
Day 1 上午   3DGS §3.1–3.2（数学核心，慢读）
Day 1 下午   3DGS §4 + Table，然后去读 dataset_readers.py
Day 1 晚上   VGGT §1 + §3（快读，抓结构）
Day 2 上午   VGGT Table 1 + 仓库 README
Day 2 下午   VGGT-Ω（重点想第 3 个贯穿问题）
Day 2 晚上   COLMAP 文档两页
空余时间     DA3 / Pi3 摘要
```
