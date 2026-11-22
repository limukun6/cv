# Code Modifications & Running Logs

## 1. Environment Setup & Modifications

To run i3dgs in a headless cloud environment (AutoDL RTX 4090, PyTorch 2.5.1, CUDA 12.4), we encountered and resolved several issues:

- **Missing C++ Dependencies:** `diff-gaussian-rasterization` failed to compile due to missing GLM headers. Resolved by installing `libglm-dev` and setting `CPLUS_INCLUDE_PATH`.
- **Headless Viewer Error:** The script crashed due to `ModuleNotFoundError: No module named 'graphdecoviewer'`. We patched `train.py` and `gaussianviewer.py` to disable the GUI viewer (since cloud instances have no display).
- **PyPI Package Confusion:** `fused-ssim` was not available in the Huawei Cloud PyPI mirror. Installed successfully via the Gitee mirror.

## 2. Actual Run Log & Failure Analysis (Sparse-View Unposed)

During the actual test, the pipeline initialized successfully and correctly bypassed COLMAP poses (as intended for the Unposed setting). However, when attempting to extract features for pose estimation, the system failed.

```text
WARNING:root: Failed to read COLMAP files in data/my_photos/sparse/0: cannot unpack non-iterable NoneType object
Initializing modules and running just in time compilation, may take a while...
Building dense extractor: torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
RuntimeError: XFeat model download failed or interrupted due to network limits on the cloud instance.
