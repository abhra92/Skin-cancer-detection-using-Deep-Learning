# Skin Cancer Detection Using Deep Learning — reproducible research workspace

This repository contains the supplied research materials and a reproducible experiment pipeline in `Skin_Cancer_Deep_Learning_Research/`.

## Current execution status

The full workspace and research-paper PDFs were inspected. **No HAM10000 image files or metadata CSV are present**; the only supplied CSV is a 45-row bibliography. The existing 25-cell PyTorch notebook has zero executed code cells and zero outputs. This run audited the dataset location and correctly stopped before model training. Therefore:

- No model was trained; no checkpoint, confusion matrix, class distribution, training curve, or model metric exists.
- Experimental values in the result tables are blank/`NOT AVAILABLE`, not zero.
- The supplied papers' numbers are treated as literature results and are not presented as this project's results.
- The audit/preflight and five software tests were executed. The tests use temporary synthetic fixtures only to verify code behavior; their values are excluded from research results.

See `Skin_Cancer_Deep_Learning_Research/10_reports/FINAL_RESEARCH_EXPERIMENT_REPORT.md` and `01_project_audit/PROJECT_AUDIT.md` for full findings.

## Directory map

```text
Skin_Cancer_Deep_Learning_Research/
├── 01_project_audit/       workspace inventory, machine-readable audit
├── 02_literature/          supplied-paper inventory and source-based review
├── 03_dataset/             data-quality report and manifest
├── 04_preprocessing/       configuration, split/leakage outputs, processed_data/
├── 05_experiments/         experiment matrix, run log, histories/configs
├── 06_models/              checkpoints from successful real runs
├── 07_results/             CSV/XLSX tables, reports, confusion matrices
├── 08_figures/             data/status and, after training, performance figures
├── 09_statistics/          statistical-analysis record
├── 10_reports/             results, discussion, tables, figures, final report
└── 11_reproducibility/     code, dependencies, tests, run instructions
```

The original papers and notebook remain in their supplied locations; no original dataset files were modified.

## Run the pipeline after obtaining HAM10000

The data directory is git-ignored. Place a properly sourced/licensed copy under:

```text
DATA/HAM10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/   # JPEGs, if distributed in split folders
└── HAM10000_images_part_2/   # JPEGs, if distributed in split folders
```

Then install dependencies and run the project audit, tests, and experiment pipeline using the instructions in `Skin_Cancer_Deep_Learning_Research/11_reproducibility/README.md`.

Quick start after installing the dependencies:

```bash
python Skin_Cancer_Deep_Learning_Research/11_reproducibility/audit_workspace.py
PYTHONPATH=Skin_Cancer_Deep_Learning_Research/11_reproducibility \
  python -m unittest discover \
  -s Skin_Cancer_Deep_Learning_Research/11_reproducibility/tests -v
python Skin_Cancer_Deep_Learning_Research/11_reproducibility/run_pipeline.py \
  --data-root DATA/HAM10000
```

The last command exits with code `2` when the data audit is blocked. That is an intentional preflight failure status—not a successful training run. It will not download data, ask for Kaggle credentials, or fabricate samples.

## Planned experiment matrix

The pipeline pre-registers four controlled experiments on one grouped split:

1. Compact CNN from scratch, unweighted cross-entropy.
2. Same CNN, inverse-frequency weighted cross-entropy.
3. ImageNet-pretrained MobileNetV3-Small, unweighted cross-entropy.
4. Same transfer model, inverse-frequency weighted cross-entropy.

All use identical train-only augmentation, approximately 60/20/20 train/validation/test splits grouped by patient (when available), lesion, and exact-image SHA-256, and validation macro-F1 for checkpoint selection. The test split is reserved for final evaluation.

## Safety and research integrity

This is educational research code and is **not** a clinical diagnostic system. HAM10000-only test results cannot establish clinical validity, prospective performance, or generalization to other institutions, imaging devices, or skin tones. No paper's results should be copied into the project's experiment-result table.
