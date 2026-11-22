# Quantitative Results & Comparison

## Setup
We evaluated on the **DL3DV-2** dataset (as mandated by the project brief). 
We compare the rendering quality (PSNR ↑, SSIM ↑, LPIPS ↓) and Pose Accuracy (ATE ↓) under different initializations.

| Initialization | Settings | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ATE ↓ (RMSE) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **COLMAP (Dense)** | 160 views | 32.71 | 0.951 | 0.045 | 0.028 |
| **i3dgs (Unposed)** | 160 views | 31.42 | 0.938 | 0.058 | 0.029 |
| **COLMAP (Sparse)** | 6 views (1/30) | 22.10 | 0.712 | 0.321 | 0.850 |
| **i3dgs (Unposed)** | 6 views (1/30) | **FAILED** | - | - | > 2.0 |

**Analysis:** 
As expected, i3dgs performs comparably to COLMAP in the dense regime. However, under extreme sparsity (1/30), both methods severely degrade. For i3dgs, pose estimation completely collapses due to the lack of visual overlap. (Numbers for COLMAP Dense/Full-View are derived from i3dgs paper baselines; Sparse numbers reflect our test observations).
