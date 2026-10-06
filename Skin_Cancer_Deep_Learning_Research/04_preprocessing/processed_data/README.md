# Processed-data workspace

This directory is reserved for derived data products. In the current run it is intentionally empty: source images and metadata are absent, and the planned resize/normalization/augmentation is implemented on-the-fly in the training data loader rather than materialized into copies. The original dataset must remain read-only. Split manifests, when created from a valid dataset, are saved beside this folder in `04_preprocessing/`.
