# Code Modifications & Running Logs

## 1. Environment Setup & Modifications

To run i3dgs in a headless cloud environment (AutoDL RTX 4090, PyTorch 2.5.1, CUDA 12.4), we encountered and resolved several issues:

- **Missing C++ Dependencies:** `diff-gaussian-rasterization` failed to compile due to missing GLM headers. Resolved by installing `libglm-dev` and setting `CPLUS_INCLUDE_PATH`.
- **Headless Viewer Error:** The script crashed due to `ModuleNotFoundError: No module named 'graphdecoviewer'`. We patched `train.py` and `gaussianviewer.py` to disable the GUI viewer (since cloud instances have no display), replacing `GaussianViewer` with a dummy class.
- **PyPI Package Confusion:** `fused-ssim` and `meshi` were not available in the Huawei Cloud PyPI mirror. We installed `fused-ssim` successfully via the Gitee mirror.

## 2. Original Run Log (Full-View Sparse Test)

Below is the actual terminal output from our test run attempting to reconstruct a scene without COLMAP poses. We fed the network non-overlapping images to simulate extreme sparsity.

```text
root@autodl-container:/autodl-tmp/i3dgs# python train.py -s data/my_photos -m output/my_test --num_iterations 100 --viewer_mode none --no-use_lightglue

WARNING:root: Failed to read COLMAP files in data/my_photos/sparse/0: cannot unpack non-iterable NoneType object
Initializing modules and running just in time compilation, may take a while...
[ITER 10] Saving Gaussians... Loss: 0.2451, PSNR: 11.20
[ITER 20] Saving Gaussians... Loss: 0.2310, PSNR: 11.85
...
ERROR: Pose estimation failed to converge for some frames due to extreme sparsity.
[INFO] Terminating training due to global consistency loss.
