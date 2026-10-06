# Data leakage check

**Status: NOT EXECUTABLE — no local image/metadata records were found.**

No split was created and no model was trained. Consequently, patient/lesion overlap and exact-image duplicate overlap across train/validation/test are **NOT AVAILABLE**, not passed.

The reproducible pipeline is configured to form groups from patient ID (when supplied), lesion ID, and exact-image SHA-256. It uses a stratified grouped holdout, validates group/hash disjointness, writes the split manifest before training, and applies augmentation only through the training data loader.

The held-out test split is reserved for one final evaluation; validation macro-F1 is the only checkpoint-selection metric.

Blocker(s):
- Dataset root does not exist: /home/user/Skin-cancer-detection-using-Deep-Learning/DATA/HAM10000
