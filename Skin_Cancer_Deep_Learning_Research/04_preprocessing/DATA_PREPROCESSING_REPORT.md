# Data preprocessing report

- Dataset: HAM10000 (local data root: `/home/user/Skin-cancer-detection-using-Deep-Learning/DATA/HAM10000`)
- Dataset audit status: **BLOCKED**
- Image size configured: 160 × 160 RGB.
- Normalization: ImageNet channel mean/std. Validation/test never contribute to normalization estimates.
- Augmentation: random horizontal flip p=0.5, random vertical flip p=0.5, random rotation within +/-15 degrees; training split only, after group-aware splitting.
- Split: StratifiedGroupKFold; outer 5-fold test holdout, inner 4-fold validation from the remaining data; seed 42; approximate 60/20/20.
- Original data: read-only; no image files are copied or transformed on disk.

## Execution status

Preprocessing was **not executed on clinical data** because no validated dataset was available. No split manifest, resized images, or augmentation outputs were generated.

- Blocker: Dataset root does not exist: /home/user/Skin-cancer-detection-using-Deep-Learning/DATA/HAM10000
