# Code Modifications & Running Logs

## 1. Modifications to i3dgs
To run i3dgs in a headless cloud environment (AutoDL RTX 4090), we made the following adjustments:
- Disabled the interactive viewer in `train.py` by passing `--viewer_mode none`.
- Reduced iteration count to `1000` for rapid iteration during debugging.

## 2. Running Log (Sample)
```text
# Command executed:
python train.py -s /root/autodl-tmp/tandt_db/truck -m output/full_truck --num_iterations 100 --viewer_mode none

# Output (Screenshot attached in results/sparse_failure.png):
[ITER 10] Loss: 0.0421, PSNR: 18.2, Time: 0.05s
[ITER 20] Loss: 0.0398, PSNR: 19.1, Time: 0.05s
...
ERROR: Pose estimation failed to converge for some frames due to extreme sparsity.
