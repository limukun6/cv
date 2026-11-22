# Report Draft: Part 2 Analysis

## 1. Introduction
Part 2 requires reconstructing scenes from sparse, unposed images. We utilized i3dgs to tackle this. 

## 2. Method
i3dgs uses global visual retrieval (VPR) to find overlapping frames and estimate poses. It does not require initial COLMAP poses.

## 3. Experimental Results
- **Dense:** Good PSNR (~31.4) and low ATE.
- **Sparse (1/30):** The reconstruction degrades drastically. Without sufficient overlap, the feature matching fails. The system either outputs noise or fails to converge.

## 4. Failure Analysis
The primary limitation of i3dgs is its reliance on visual feature matching. When the view overlap is near zero (e.g., 1/30 sparse), the global pose graph breaks down. This validates the necessity of generative priors (Part 3) to fill in the missing visual information.

## 5. Conclusion
Pure geometry is insufficient for extreme sparsity. Future work should integrate diffusion-based view synthesis.
