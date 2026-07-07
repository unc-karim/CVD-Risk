# Weights Directory

Place the trained model weights in this folder. The backend loads them at startup.

Expected files:
- `weights/hypertension.pt` (HTN classifier)
- `weights/cimt_reg.pth` (CIMT regression)
- `weights/vessel.pth` (vessel segmentation)
- `weights/fusion_cvd_notskewed.pth` (fusion meta-classifier)

Notes:
- If you have legacy weights under `pth/`, you can either move them here or set `WEIGHTS_DIR=./pth`.
- The fusion checkpoint must include `standardization` parameters (`fusion_mean`, `fusion_std`).
- Large files are intentionally gitignored.
