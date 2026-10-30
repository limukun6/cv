# Generative Sparse-View 3D Reconstruction — Report Draft

> **Status**: skeleton with real, verified numbers from `results/results.csv`.
> Sections marked **[TODO]** need Part 2 / Part 3 results.
> Targets CVPR 2026 template, 8 pages incl. figures (6–8 pages excl. references).
>
> 写作约定：每个数字都能在 `results/results.csv` 里按 `run_id` 找到；凡是没测过的，
> 一律写 **[TODO]**，不编。

---

## Abstract

We study how modern AI priors change the 3D reconstruction pipeline, moving from
"ideal" dense captures to "extreme" sparse ones. We build a single, auditable
pipeline that takes a video, canonicalizes camera poses, runs COLMAP or a
feed-forward geometry model, trains 3D Gaussian Splatting (3DGS), and evaluates
novel-view synthesis with a fixed, cross-validated protocol. On three mandatory
scenes we report a COLMAP-initialized 3DGS baseline of **33.41 ± 0.35 dB**
(DL3DV-2), **34.34 ± 0.05 dB** (Re10k-1) and **35.29 ± 0.16 dB** (Waymo-405841)
PSNR, together with *per-scene significance thresholds* obtained from repeated
runs — a step we find indispensable, because the run-to-run noise differs by
**7×** across scenes (2σ = 0.70 dB vs 0.10 dB). We then quantify the collapse
under sparsity: with only 8 of 267 training views, DL3DV-2 drops to **16.38 dB**
while training PSNR rises to **50.43 dB**, i.e. a **34 dB** overfitting gap.
These numbers define the headroom that generative priors must recover in Part 3.
Code: `https://github.com/limukun6/cv`

---

## 1. Introduction

The project brief asks us to experience 3D reconstruction degenerating from
"ideal" to "extreme" conditions and to understand how AI priors intervene. We
structure the work along one axis — *how much observation is available*:

| Setting | Observation | Method class | Our status |
|---|---|---|---|
| Part 1 | dense video, poses available | geometric optimisation → feed-forward | Plan A done |
| Part 2 | 1/10 and 1/30 frames, **no poses** | unposed reconstruction / SLAM | upper bound done |
| Part 3 | sparse + generative completion | diffusion priors | [TODO] |

**Contributions (this draft)**

1. **A reproducible pipeline and contract** that makes results from two people
   (and two machines) mutually comparable: a canonical pose format (`w2c`,
   OpenCV axes, `xyzw` quaternions), frozen splits, and a single evaluation
   implementation.
2. **A geometry-based protocol to validate pose conventions.** Rather than
   trusting file formats, we verify conventions by testing which hypothesis
   minimizes reprojection / depth-consistency error (Sec. 3.2).
3. **Per-scene significance thresholds.** We show that a single global
   "±0.5 dB is noise" rule is wrong: the 2σ thresholds are 0.70 / 0.10 / 0.34 dB
   on our three scenes (Sec. 4.2).
4. **A sparsity-degradation study** that isolates the cost of missing views under
   *oracle* poses, providing the reference point against which unposed methods
   (Part 2) and generative completion (Part 3) should be judged (Sec. 4.4).

---

## 2. Related Work

**Classical SfM.** COLMAP-style incremental SfM extracts and matches local
features, verifies them geometrically, and performs bundle adjustment. It is
accurate but offline and fragile when overlap is low.

**Feed-forward geometry.** VGGT predicts camera parameters, depth maps, point
maps and tracks in a single forward pass, replacing the matching/BA stack with a
regression. Its successor VGGT-Ω reports a 30 % training-memory footprint and
77 % better camera estimation on Sintel, with 15× more supervised data —
coupling an architectural change (register attention) with data scaling.
**[TODO: our own Plan B/C numbers]**

**3D Gaussian Splatting.** 3DGS represents a scene as anisotropic 3D Gaussians
rasterized in real time. Its quality depends strongly on initialization
(camera poses and the seed point cloud) — the property Part 1 probes.

**Sparse-view NVS.** With few views the problem becomes ill-posed; methods add
depth priors or generative regularizers. We do not attempt these in Part 1;
instead we measure the gap they must close.

---

## 3. Method

### 3.1 Pipeline

```
video → frames → frozen splits ─┬→ COLMAP (SfM)      ─┐
                                ├→ VGGT (feed-forward)─┼→ canonical poses → 3DGS → test-view PSNR/SSIM/LPIPS
                                └→ VGGT-Ω              ─┘
```

All initialization sources write a single canonical form
(`w2c`, OpenCV axes, `xyzw` quaternions, 4×4 row-major, metres), so that only
one variable — the initialization — changes between runs.

### 3.2 Validating pose conventions (why we do not trust formats)

Pose files rarely state their conventions, and a wrong convention degrades
results *silently*. We therefore determine conventions empirically.

For scenes shipping `cameras.json` we enumerate four hypotheses
(quaternion order × direction) and, for each, triangulate matched SIFT points and
measure reprojection error and the fraction of points in front of both cameras:

| Hypothesis | reproj. error (px) | points in front |
|---|---|---|
| **xyzw, c2w** | **0.05** | **97.8 %** |
| wxyz, c2w | 1.60 | 70.0 % |
| wxyz, w2c | 4.24 | 82.5 % |
| xyzw, w2c | 9.23 | 91.5 % |

The correct hypothesis is ~30× better than the runner-up, so the decision is not
marginal. We then **cross-check against an independent estimator** (COLMAP) and
measure agreement in a frame-invariant way: relative rotations differ by
**0.101°** (DL3DV-2) and **0.080°** (Re10k-1); after Sim(3) alignment the camera
centres differ by **0.06 %** and **0.29 %** of the scene diagonal.

> A trap we hit: displacement vectors are frame-dependent while relative
> rotations are not. Comparing *unaligned* displacement directions gave a stable
> ~27° discrepancy that did **not** shrink with longer baselines — the signature
> of a systematic (frame) error rather than noise. Aligning first reduced the
> error to 1.65° → 0.19° as the baseline grew.

### 3.3 Controlling the train/test split

Stock 3DGS fixes the split: with `--eval` it always holds out every 8th camera
(`llffhold=8`, no CLI); without `--eval` it creates **no** test set. To train on
an arbitrary subset while evaluating on a frozen test set, we override
`Scene.getTrainCameras / getTestCameras` at run time, leaving the third-party
code untouched.

### 3.4 Evaluation protocol

* PSNR/SSIM/LPIPS computed **only** on the frozen test split, at native render
  resolution, no resizing.
* **LPIPS with VGG backbone and `[0,1]` input**, matching 3DGS's `metrics.py`.
  We verified four combinations; using AlexNet instead of VGG underestimates
  LPIPS by ~3× (0.014 vs 0.042 on the same model).
* We fixed a 38× inefficiency in the reference evaluation (the LPIPS network was
  reconstructed per image), reducing evaluation from 350 s to 9 s.
* Repetition: 3–5 runs per setting with different seeds (3DGS hard-codes
  `manual_seed(0)`, so we reseed after `safe_state()`).

---

## 4. Experiments

### 4.1 Setup

| Scene | Frames | Resolution | COLMAP registration | Points | Reproj. err | COLMAP time |
|---|---|---|---|---|---|---|
| DL3DV-2 | 306 | 256×256 | **306/306** | 7 860 | 0.448 px | 98 s |
| Re10k-1 | 279 | 256×256 | **279/279** | 2 301 | 0.493 px | 216 s |
| Waymo-405841 | 199 | 1920×1280 → 480×320 | **176/199** (3 components) | 21 444 | 0.697 px | 838 s |

Waymo fragments into three disconnected models (12 / 10 / 176) because the
vehicle turns away from the main overlap region; we use the 176-frame model,
which yields 154 train / 22 test frames instead of the planned 174 / 25.

### 4.2 Baseline and significance thresholds

| Scene | n | Train | Test | **PSNR** | SSIM | LPIPS | **2σ** |
|---|---|---|---|---|---|---|---|
| DL3DV-2 | 5 | 267 | 39 | **33.41 ± 0.35** | 0.9659 | 0.0448 | **0.70** |
| Re10k-1 | 4 | 244 | 35 | **34.34 ± 0.05** | 0.9816 | 0.0207 | **0.10** |
| Waymo-405841 | 3 | 154 | 21 | **35.29 ± 0.16** | 0.9647 | 0.0768 | **0.33** |

Repeating an identical configuration changes test PSNR by up to **0.89 dB**
(DL3DV-2), driven by CUDA non-determinism rather than the seed. Consequently,
**no comparison below ~0.7 dB is interpretable on DL3DV-2**, while on Re10k-1 a
0.2 dB difference is already meaningful. Both scenes end with a similar number of
Gaussians (380k–420k), so the noise difference is *not* explained by model size;
we hypothesise it relates to outdoor view-dependent effects and a denser seed
point cloud (7 860 vs 2 301).

### 4.3 Convergence

![convergence](figures/convergence_planA.png)

DL3DV-2 is **not monotonic**: test PSNR peaks at 29.12 dB (2k), falls to
25.27 dB (5k), and jumps from 28.17 dB (15k) to 32.30 dB (20k). Both transitions
align with 3DGS's schedule — the aggressive densification phase with periodic
`opacity reset`, then the end of densification at 15k. Reading a single
iteration would therefore support *opposite* conclusions about which scene is
harder; only the full curve is trustworthy.

### 4.4 Sparsity (oracle poses)

![sparsity](figures/sparsity_curve.png)

| Scene | Train | **test PSNR** | LPIPS | train PSNR | **overfit gap** |
|---|---|---|---|---|---|
| DL3DV-2 | 267 | **33.41** | 0.045 | 33.98 | +0.6 |
| DL3DV-2 | 23 | **20.60** | 0.222 | 42.26 | **+21.7** |
| DL3DV-2 | 8 | **16.38** | 0.353 | 50.43 | **+34.1** |
| Re10k-1 | 244 | **34.34** | 0.021 | 34.99 | +0.7 |
| Re10k-1 | 21 | **29.67** | 0.057 | 39.27 | +9.6 |
| Re10k-1 | 7 | **22.55** | 0.156 | 48.38 | **+25.8** |
| Waymo | 154 | **35.29** | 0.077 | 34.94 | −0.4 |
| Waymo | 13 | **24.59** | 0.190 | 45.96 | **+21.4** |
| Waymo | 4 | **18.38** | 0.341 | 51.90 | **+33.5** |

Three observations:

1. **Degradation is a cliff, not a slope**: −12.8 dB from 267 to 23 views on
   DL3DV-2, 18× the noise threshold.
2. **Overfitting explodes**: train/test gap grows from 0.6 dB to 34.1 dB;
   training PSNR reaching 50 dB indicates the scene is being memorised.
3. **Sensitivity is scene-dependent, but the floor is universal.** The same 1/10
   subsampling costs 12.8 dB outdoors (scene diagonal 12.2) vs. only 4.7 dB indoors
   (diagonal 4.1); yet at 4–8 views *all three scenes* land in the narrow band
   **16.4–22.6 dB**, and the train/test gap reaches **26–34 dB** everywhere.
   Sparsity, not scene type, sets the ceiling.

### 4.5 Part 2 — unposed sparse reconstruction

**[TODO]** Full-view vs sparse comparison with an unposed method (i3dgs or
HI-SLAM2), ATE (Sim(3)-aligned), and failure cases.

### 4.6 Part 3 — closed-loop generative reconstruction

**[TODO]** ≥2 generation–reconstruction rounds, consistency-aware fusion,
ablations (no-confidence / soft-weight / reject).

---

## 5. Limitations

1. **Plan B/C are not yet measured.** On our 8 GB laptop, VGGT-1B peaks at
   7.43 GB with 3 frames (all heads) and 7.47 GB with 10 frames (camera head
   only) — after reserving headroom for the display, fewer than ~10 frames are
   feasible, far below the ~250 used by COLMAP. The comparison would confound
   initialization quality with view count, so we do not report it as a method
   result and are seeking a ≥12 GB device.
2. **The sparsity study is an upper bound.** Poses come from the full sequence
   and the seed point cloud likewise, so the numbers are optimistic by
   construction. They are intended as the reference for Part 2, not as a
   achievable sparse result.
3. **Resolution differs across scenes** (256×256 vs 480×320 for Waymo), so PSNR
   is not comparable *across* scenes.
4. **LPIPS is scene-sensitive**: Waymo's 0.077 vs 0.021–0.045 elsewhere reflects
   fine outdoor structure rather than method quality.
5. **Small test sets are fragile.** Waymo has only 21 test views; switching from 22
   to 21 views moved mean PSNR by **2.1 dB** (33.18 → 35.29). All experiments on a
   scene must therefore share one test set, including at render time — a mismatch
   between train-time and render-time splits cost us two wasted runs.

---

## 6. Conclusion

We built an auditable pipeline and established, with repeated runs, the
statistical resolution of our own measurements before making any claim. Under
oracle poses, sparse views cost 12.8 dB (outdoor) to 4.7 dB (indoor) and inflate
the overfitting gap by up to 40×, which quantifies precisely the headroom that
Part 3's generative priors must recover. **[TODO: conclusions for Parts 2–3]**

---

## 附：写作说明（不进正文）

1. **为什么先写噪声门槛再写结果**：没有它，任何 0.2–0.7 dB 的差异都可能被当成
   结论。审阅者（助教）看到这一步，会认为我们真的懂怎么读实验数字。
2. **为什么把位姿验证写进 Method**：这是本项目里唯一"别人大概率不会做"的
   方法论贡献，放在 Method 比放在附录有力。
3. **Limitations 要写成事实**（"8 GB 上 VGGT 只能跑 3–10 帧，会让对比混淆变量"），
   而不是"未来可以改进"这种空话。原文明确说做不完 Part 3 也要展示失败案例。
4. **所有 [TODO] 必须填真数据**，不许用估计值。填不上就保留 TODO 并说明原因。
