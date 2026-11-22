# Part 2: Unposed Sparse Reconstruction

## Overview
This folder contains the implementation and experimental results for **Part 2** of the AIAA 3201 Term Project. 

The goal is to reconstruct 3D scenes from extremely sparse views (1/10 or 1/30) **without any known camera poses**. We evaluate the performance of [i3dgs](https://github.com/graphdeco-inria/i3dgs) (Immediate 3DGS) and analyze its physical limitations under extreme sparsity.

## Environment Setup
Experiments were run on an NVIDIA RTX 4090 (AutoDL cloud instance) with PyTorch 2.5.1 and CUDA 12.4.

```bash
# Clone the official i3dgs repository
git clone --recursive https://github.com/graphdeco-inria/i3dgs.git
cd i3dgs

# Install dependencies
pip install -r requirements.txt --no-build-isolation
pip install cupy-cuda12x
