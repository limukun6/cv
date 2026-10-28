# Project 4 · Generative Sparse-View 3D Reconstruction

> AIAA 3201 Introduction to Computer Vision · Term Project · 2 人组
> 组员：A（几何线）· B（生成线）

**仓库状态**：`W0 · 搭建地基`
**上次验收**：无

---

## 目录

- [一、这个项目在干什么](#一这个项目在干什么)
- [二、两人分工与时间表](#二两人分工与时间表)
- [三、场景契约（组内法律）](#三场景契约组内法律)
  - [3.1 数据根目录](#31-数据根目录)
  - [3.2 位姿格式 canonical](#32-位姿格式-canonical)
  - [3.3 Splits](#33-splits)
  - [3.4 转 3DGS](#34-转-3dgs)
  - [3.5 评测口径](#35-评测口径)
  - [3.6 运行产物与 results.csv](#36-运行产物与-resultscsv)
- [四、代码规范与 Owner](#四代码规范与-owner)
- [五、快速开始](#五快速开始)
- [六、第 3 天验收](#六第-3-天验收)
- [七、已知问题 / 风险](#七已知问题--风险)

---

## 一、这个项目在干什么

用一条统一的管线，跑完三个递进的实验：

| Part | 输入条件 | 要回答的问题 |
|---|---|---|
| **P1** | 稠密视频 + 有位姿 | COLMAP / VGGT / VGGT-Ω 三种初始化，谁又快又准？ |
| **P2** | 抽稀 1/10、1/30 + **无位姿** | 视角一稀疏，位姿和几何在哪崩？ |
| **P3** | 承接 P2 | 生成模型补出的视角，能不能**真的**提升重建？ |

必做场景：`waymo-405841`（户外 1/10）、`dl3dv-2`（户外 1/30）、`re10k-1`（室内 1/30）。

---

## 二、两人分工与时间表

| 阶段 | A（几何线） | B（生成线） |
|---|---|---|
| **W0（3 天）** | `00` 抽帧 · `01` splits · `10` COLMAP · `11` VGGT · `20` 校验 | `21` → 3DGS 转换 · `30` 训练 · `40` 渲染评测 · `41` ATE 评测 |
| **W1** | **P1 三条路线全包** | **P2 风险排查**（见下） |
| **W2** | P1 收尾 → 支援 P2 失败分析 | P2 正式实验（full / sparse10 / sparse30） |
| **W3** | P3 闭环主程序（插值→渲染→生成→重训） | P3 一致性掩码 + 消融矩阵 |
| **W4** | P3 第二轮 + 额外数据 | P3 图表 + 视频生成加速 |
| **W5–6** | 报告主体 | README + demo 视频 |

> **⚠️ W1 头 3 天，B 的任务不是跑实验，是「探雷」**：确认 i3dgs（备选 S3PO-GS / InstantSplat）**有没有代码、能不能装、要什么输入格式**。跑不起来立刻上报换船，别拖到 W3。

**三条组队铁律**

1. **共享代码只能有一个人写**（评测器、数据转换、训练模板）。两人各写一份评测器是灾难开端。
2. **结果必须共享**：所有实验数字进 `results/results.csv`，每跑完立刻追加一行。不记的记录等于没跑。
3. **风险最高的事最先做**。P2 能不能跑起来，这个不确定性要在 W1 消灭。

---

## 三、场景契约（组内法律）

> **本节是组内法律。** 任何脚本、任何实验、任何表格都要遵守。
> 修改本节需两人同意，并在文末「变更记录」写一行。
> 契约**管**：目录结构、文件格式、字段名、坐标系、评测口径。
> 契约**不管**：用多少帧、test 留几帧、超参——那些写进 `configs/`，不写在这里。

### 3.1 数据根目录

```
data/<scene>/
├── images/                   # 抽出来的帧，唯一真值来源
│   ├── 00000.png
│   └── ...
├── splits/                   # 见 3.3
│   ├── _index.csv
│   ├── test.txt
│   ├── full.txt
│   ├── sparse10.txt
│   └── sparse30.txt
├── raw/                      # 原始下载物（视频/tfrecord），永不修改
├── colmap/                   # Plan A 原始产物（COLMAP 直出，不动它）
├── canonical/                # ★ 所有方法输出必须落到这里（见 3.2）
│   ├── colmap.json
│   ├── vggt.json
│   ├── vggt_omega.json
│   ├── i3dgs.json
│   └── gt.json               # 有真值位姿的场景
└── meta.yaml
```

**场景 id 固定**：`waymo-405841` · `dl3dv-2` · `re10k-1`。额外场景用 `optional-<name>`（如 `optional-mipnerf360-bicycle`）。

`meta.yaml` 模板：

```yaml
scene: waymo-405841
source_dataset: waymo
scene_type: outdoor          # indoor | outdoor | object
raw_format: tfrecord         # tfrecord | mp4 | torch | frames
frame_count: 199
image: { width: 1600, height: 1067, extension: png }
has_gt_pose: true
gt_pose_source: waymo_official
scale_is_metric: true
notes: ""
```

### 3.2 位姿格式 `canonical`

**坐标系约定（所有方法必须先转成这个再存）**

| 项目 | 规定 |
|---|---|
| 轴向 | **OpenCV 相机系**：x 右、y 下、**z 向前**（指向场景） |
| 变换方向 | 存 **w2c**（world-to-camera）：`p_cam = w2c @ p_world` |
| 矩阵 | 4×4，**行主序**，最后一行 `[0,0,0,1]` |
| 单位 | 不强制。但同一 scene 内，`frames[].w2c` 与 `points3d.ply` **必须同尺度** |
| 世界原点 | 不强制。推荐第一帧相机中心为原点（便于目视调试） |

> **为什么必须钉死**：COLMAP 存 `qvec/tvec`（world→cam 四元数）、VGGT 输出以首帧为参考的 OpenCV 系、3DGS 的 `transforms.json` 存 **c2w 且 y 向上看向 -z** 的 Blender 系。三者互不相同，**不统一第一步就全错**。

**JSON 格式**

```jsonc
{
  "schema": "scene-poses/v1",          // 固定字符串，用于校验
  "scene": "waymo-405841",
  "source": "vggt",                    // 见下「合法取值」
  "created_at": "2026-03-01T12:00:00Z",

  "coordinate_frame": "w2c_opencv",    // 固定字符串
  "units": "unknown",                  // "unknown" | "meter"

  "image": { "dir": "images", "width": 1600, "height": 1067, "ext": "png" },

  "intrinsics": {                      // 默认内参
    "model": "PINHOLE",                // PINHOLE | SIMPLE_PINHOLE
    "fx": 1200.5, "fy": 1200.5, "cx": 800.0, "cy": 533.5
  },

  "frames": [
    {
      "name": "00000.png",             // 必须与 images/ 下文件名完全一致
      "w2c": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "intrinsics": null               // null=用默认值；VGGT 这类逐帧预测焦距的填覆盖值
    }
  ],

  "points": "points3d.ply",            // 相对本 json 目录；可为 null
  "points_count": 48213,

  "meta": {
    "wall_clock_sec": 412.3,
    "wall_clock_breakdown": { "extract": 12.0, "match": 260.0, "solve": 140.3 },
    "model_load_sec": 8.5,             // 模型加载耗时，不计入 wall_clock
    "gpu": "RTX 4090",
    "notes": ""
  }
}
```

**`source` 合法取值**（不许自创）：
`colmap` · `vggt` · `vggt_omega` · `da3` · `pi3` · `i3dgs` · `hi_slam2` · `s3po_gs` · `instant_splat` · `long_splat` · `gt` · `interpolated`

**`points3d.ply` 要求**

- `binary_little_endian` PLY，属性 `x y z`（可选 `red green blue`）
- 顶点数 **20k – 200k**（超了降采样，不足保留但不许人为补点）
- 必须与 `w2c` 同世界系、同尺度
- 命名固定 `points3d.ply`，与 json 同目录

**落地前必须通过校验**（`scripts/20_validate_canonical.py`）

1. `schema` 与 `coordinate_frame` 字段正确
2. `frames` 数量 == 该实验所用 split 行数
3. 每个 `w2c` 是 4×4、末行 `[0,0,0,1]`、左上 3×3 是**旋转矩阵**（`R@Rᵀ≈I`、`det≈+1`，误差 `<1e-4`）
4. 每个 `name` 在 `images/` 下真实存在
5. `fx, fy > 0`
6. 相机中心 `-Rᵀt` 的分布不应挤在点云质心（快速体检姿态错误）
7. `points_count` 与实际 PLY 顶点数一致

> 第 3 条能抓出 90% 的四元数顺序错误和矩阵转置错误。**不要跳过。**

### 3.3 Splits

纯文本，一行一个文件名（含扩展名），UTF-8 无 BOM，顺序同 `images/`。

```
00000.png
00008.png
```

**必须存在四个**

| 文件 | 含义 | 约束 |
|---|---|---|
| `test.txt` | **测试视角** | ★ 三个稀疏档**共用这一个**，任何训练集不得包含它 |
| `full.txt` | 全视角训练集 | = `images/` 全部 − `test` |
| `sparse10.txt` | 1/10 抽帧训练集 | ⊆ `full`，且 ≠ `test` |
| `sparse30.txt` | 1/30 抽帧训练集 | ⊆ `full`，且 ≠ `test` |

**生成规则**（`scripts/01_make_splits.py`）

```
pool     = images/ 下所有帧，按文件名排序
test     = pool[::test_every]          # test_every 来自 config，默认 8
full     = pool \ test
sparse_k = pool[::k] \ test            # k = 10 或 30
```

**三条硬规则**

1. `test.txt` 一旦生成就**冻结**。改 test = 作废之前所有数字。
2. 稀疏集先在 **pool** 上按 k 抽，再减 test——保留「原序列 1/k」语义。
3. `sparse30` 减完 test 若 **< 5 帧**，记进 `_index.csv` 的 `warning` 列，并在报告声明实际帧数。

`splits/_index.csv`（可追溯性）

```csv
split_name,frames,test_every,source_pool,created_at,warning
test,25,8,all,2026-03-01,
full,174,-,all,2026-03-01,
sparse10,18,-,pool[::10],2026-03-01,
```

> ⚠️ 若 BRPO / S3PO-GS 仓库有官方 split 脚本（原文说 "follow BRPO"），**优先用他们的数字**，再填进这张表。契约只管格式，不管留几帧。

### 3.4 转 3DGS

**唯一实现：`scripts/21_to_3dgs.py`**（B 写，A 验）。只允许一个版本。

**产物目录**

```
outputs/scenes/<scene>__<split>/
├── images/                # ★ 只放该 split 的【训练帧】（硬链接，不复制）
├── sparse/0/
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.txt
└── test_images/           # 测试帧（硬链接），仅供 40_eval_render.py 用
```

**为什么用 COLMAP 格式而不是 `transforms.json`**

> 3DGS 的 Blender 加载器只用 `camera_angle_x` 推焦距，并**强制 `fx = fy`、主点在图像中心**。若某方法给出非中心主点（`cx ≠ W/2`）或 `fx ≠ fy`，用 `transforms.json` 会**静默丢失**这部分内参，表现为整体轻微发虚，且极难排查。
> COLMAP 的 `PINHOLE fx fy cx cy` 能完整表达任意内参。**所以 canonical → COLMAP 是主路径。**

`cameras.txt`（一行一个相机）

```
1 PINHOLE 1600 1067 1200.5 1200.5 800.0 533.5
```

`images.txt`（**每图两行**，第二行是空 2D 观测，可留空但**必须在**）

```
1 0.9999 0.0001 0.0002 0.10 -0.02 0.50 1 00000.png

2 0.9998 0.0001 0.0003 -0.02 0.11 0.49 1 00008.png

```

> `images.txt` 的四元数是 **world-to-camera**（与我们的 `w2c` 一致），但**顺序是 `QW QX QY QZ`（实部在前）**，从旋转矩阵转换时极易搞错。另外注意：COLMAP 的 `tvec` 是 `t = -R @ C`（C 是相机中心），不是直接的相机位置。

`points3D.txt`（一行一个点，无 track 时可留空）

```
1 12.3 -4.5 8.9 128 128 128 0.0
```

**次选（仅可视化用）**：`transforms.json`

```
camera_angle_x = 2 * atan(W / (2*fx))          # 弧度
c2w_cv      = inv(w2c_cv)
c2w_blender = c2w_cv @ diag(1, -1, -1, 1)      # 翻转 y、z 轴方向
```

必须在其 header 写：
`"_warning": "assumes fx==fy and principal point at image center; use sparse/0 for accurate intrinsics"`

### 3.5 评测口径

**唯一实现：`scripts/40_eval_render.py` + `scripts/41_eval_ate.py`。**全组只有一份。

| 项目 | 规定 |
|---|---|
| 在哪些帧上算 | **只在本 split 的 `test.txt` 上**（稀疏场景训练视角能虚高 5–10 dB） |
| 分辨率 | 渲染到 `meta.yaml` 原始尺寸，**评估前不 resize** |
| PSNR | 逐图算后取**算术平均**（不是按像素合并） |
| SSIM | 标准实现，`gaussian_weights=True` 固定 |
| LPIPS | `lpips` 包，**backbone 锁定 `alex`**，输入范围 `[-1,1]` |
| 背景色 | 3DGS 默认白底 `bg_color=1,1,1`，全组统一 |
| 额外必记 | `train_psnr`（在训练帧里抽 10 张算），用来报 overfit gap |

**位姿指标**

- 转 TUM 格式 → `evo_ape ... -a`（**Sim(3)，含尺度**）→ 报 `ATE RMSE`
- 单目方法尺度不可观测，**禁止**用无 `-a` 的对齐；报告里写明对齐方式

**计时三段分开记，不许合并**

| 字段 | 从 | 到 |
|---|---|---|
| `t_init` | 读入帧 | 位姿 + 点云写盘完成 |
| `t_train` | 启动 3DGS 训练 | 最后一个 iteration |
| `t_eval` | 启动评测 | 所有 test 帧渲染完成 |

同时必须记 GPU 型号、分辨率、输入帧数。

### 3.6 运行产物与 results.csv

每次运行产出：

```
outputs/runs/<run_id>/
├── config.json          # 完整超参 + git commit hash + 时间戳
├── metrics.json
├── test_renders/
└── train.log
```

`run_id` 命名：`<scene>__<source>__<split>__<method>__<yyyyMMdd-HHMM>`
例：`waymo-405841__vggt__sparse10__3dgs__20260301-1430`

`results/results.csv` 列定义：

| 列 | 含义 |
|---|---|
| `run_id` | 上面的命名 |
| `scene` / `part` / `source` / `split` | 定位实验 |
| `train_frames` / `test_frames` | 实际帧数 |
| `psnr` / `ssim` / `lpips` | test 指标 |
| `train_psnr` | overfit 参考 |
| `ate_rmse` | 仅 P2/P3 |
| `t_init_s` / `t_train_s` / `t_eval_s` | 三段耗时（秒） |
| `gpu` / `git_commit` | 环境可追溯 |
| `notes` | 一句话说明 |

**每跑完一个实验立刻追加一行。不记的记录等于没跑。**

---

## 四、代码规范与 Owner

```
repo/
├── README.md                   # 本文件（含契约）
├── .gitignore
├── scripts/                    # 编号即执行顺序
│   ├── 00_extract_frames.py    # A
│   ├── 01_make_splits.py       # A
│   ├── 10_run_colmap.sh        # A
│   ├── 11_run_vggt.py          # A
│   ├── 20_validate_canonical.py# A
│   ├── 21_to_3dgs.py           # B（A 验）
│   ├── 30_train_3dgs.sh        # B
│   ├── 40_eval_render.py       # B
│   ├── 41_eval_ate.py          # B
│   └── 50_closed_loop.py       # P3 共同
├── configs/
│   ├── scene/*.yaml
│   └── exp/*.yaml
├── data/                       # gitignore
├── outputs/                    # gitignore
└── results/results.csv         # ★ 入库
```

| 文件 | Owner | 另一方 |
|---|---|---|
| `00 / 01 / 10 / 11 / 20` | **A** | 可读、可提 issue，不改 |
| `21 / 30 / 40 / 41` | **B** | 可读、可提 issue，不改 |
| `50` / `configs` / `README` | 共同 | 改前打招呼 |
| 本节（契约） | 共同 | 改需两人同意 |

**Git 规则**

- `main` 开分支保护，只接受 PR
- 分支：`feat/a-<what>` / `feat/b-<what>`
- 提交信息：`feat(p2): i3dgs canonical converter`（type 用 `feat`/`fix`/`data`/`exp`/`docs`）
- `results/results.csv` **必须入库**——它是全组唯一的数字真相
- `data/`、`outputs/`、模型权重一律不进仓库

---

## 五、快速开始

### 1. 环境

```bash
# 每个 Part 独立环境，避免依赖打架
conda env create -f env/p1-colmap-vggt.yml   # A
conda env create -f env/p2-3dgs.yml          # B（3DGS + 评测）
conda env create -f env/p3-diffusion.yml     # P3 需要时再建
```

> 别把所有依赖塞进一个环境：3DGS 的 `diff-gaussian-rasterization`、VGGT 的权重库、扩散模型的 xformers 对 torch 版本要求经常互斥。

### 2. 数据

```bash
python scripts/00_extract_frames.py --scene waymo-405841 --raw data/_raw/waymo-405841
python scripts/01_make_splits.py   --scene waymo-405841 --test-every 8
```

### 3. 跑通一条链路

```bash
python scripts/10_run_colmap.sh   --scene waymo-405841
python scripts/20_validate_canonical.py --scene waymo-405841 --source colmap
python scripts/21_to_3dgs.py      --scene waymo-405841 --source colmap --split full
python scripts/30_train_3dgs.sh   --scene waymo-405841 --source colmap --split full
python scripts/40_eval_render.py  --run waymo-405841__colmap__full__3dgs__<ts>
```

预期：`outputs/runs/<run_id>/metrics.json` 里有非零 PSNR/SSIM/LPIPS。

### 4. 记录

把 `outputs/runs/<run_id>/metrics.json` 的内容追加一行到 `results/results.csv`，然后 `git commit`。

---

## 六、第 3 天验收

以下 **6 条必须全绿**。做不到就别进 P1/P2，先修。

- [ ] **1.** 两人各自完整跑一遍 `00→01→10→20→21→30→40`，同一 scene / 同一 split，PSNR **相差 < 0.01 dB**
- [ ] **2.** `40_eval_render.py` 在 3DGS 官方 demo（`tandt/truck`）上跑出 **25–26 dB**（说明评测器是对的）
- [ ] **3.** `20_validate_canonical.py` 对 `canonical/` 下所有 json 通过，特别是旋转矩阵正交性
- [ ] **4.** 挑一帧，把 `canonical` 里的位姿渲染出来与原图**叠加目视对齐**
- [ ] **5.** 新机器 clone 后按 README 能跑通
- [ ] **6.** `results.csv` 至少有 **2 行真实数据**

> 第 1 条最重要：它一次性证明「数据一致 + 位姿转换一致 + 评测器一致 + 环境一致」。**这四条里任意一条不一致，第 1 条就过不了。**

---

## 七、已知问题 / 风险

> 每次踩坑后加一行。这是仓库里最有价值的段落之一。

| 日期 | 问题 | 状态 |
|---|---|---|
| | VGGT 在 200 帧上 OOM（显存随帧数平方） | 待验证 |
| | i3dgs / GeoQuery / VGGT-Ω 代码是否公开 | **待核实（W1 必须完成）** |

---

## 变更记录

| 日期 | 改了什么 | 谁 | 两人同意？ |
|---|---|---|---|
| 2026-03-01 | 初版 | | |

---

## 引用

3DGS · COLMAP · VGGT · VGGT-Ω · i3dgs · HI-SLAM2 · S3PO-GS · BRPO · GeoQuery · GSFixer · GenFusion · Difix3D+
