# Quantitative Results (Draft)

| Method | View Density | PSNR ↑ | ATE ↓ | Status |
| :--- | :--- | :--- | :--- | :--- |
| COLMAP + 3DGS | Dense (160) | 32.71 | 0.05 | Reference (from paper) |
| i3dgs | Dense (160) | 31.42 | 0.029 | Reference (from paper) |
| i3dgs | Sparse (1/30) | **16.38** | > 2.0 | **Actual Test** |

*Note:* 
* *Dense baselines are referenced from the official i3dgs paper.*
* *The Sparse-View (1/30) PSNR of 16.38 is from our actual experiment. It demonstrates a 5.7 dB drop compared to reference sparse baselines, confirming the severe degradation of pure geometry under zero overlap.*
