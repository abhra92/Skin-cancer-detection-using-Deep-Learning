# Final Research Experiment Report

## 1. Research topic
Skin Cancer Detection Using Deep Learning.

## 2. Research problem and objective
The supplied materials identify multi-class skin-lesion recognition, severe class imbalance, and the need for reliable evaluation as relevant challenges. The focused executable study compares a small CNN with a compact ImageNet-pretrained transfer model and measures inverse-frequency class weighting while controlling split, preprocessing, and augmentation. This is a benchmark research study, not a claim that a trained system improves diagnosis.

## 3. Dataset and observed statistics
- Intended dataset: HAM10000, seven diagnoses (`akiec, bcc, bkl, df, mel, nv, vasc`).
- Requested local data root: `/home/user/Skin-cancer-detection-using-Deep-Learning/DATA/HAM10000`.
- Metadata CSV: `NOT AVAILABLE`.
- Image files found: `0`.
- Metadata rows: `NOT AVAILABLE`.
- Class distribution, missing-value summaries, duplicate rates, corrupted-image counts, and metadata-image consistency: NOT AVAILABLE from valid research data because the source data are absent or failed audit.

## 4. Preprocessing and augmentation
The code audits metadata/image IDs, missing values, labels, duplicate rows, exact image hashes, corruption, and lesion-label consistency. It is configured for square resizing to 160×160 RGB and ImageNet channel normalization (no test-derived statistics). Group-aware splitting precedes augmentation. Horizontal/vertical flips and ±15° rotation are applied on-the-fly to training images only; validation/test are deterministic. The raw dataset is never modified. No training-time transformations were executed because no experiment completed.

## 5. Experimental design and leakage controls
Four experiments are preregistered in `05_experiments/experiment_matrix.csv`: a small CNN and MobileNetV3-Small transfer model, each with unweighted versus inverse-frequency weighted cross-entropy. The split is approximately 60/20/20 with StratifiedGroupKFold, grouping by patient ID when supplied, lesion ID, and exact-image SHA-256. Validation macro-F1 selects checkpoints; the test set is reserved for one final evaluation. Assertions prevent groups and exact-image hashes crossing splits.

## 6. Models and training configuration
- CNN baseline: compact four-block CNN trained from scratch.
- Transfer-learning model: torchvision MobileNetV3-Small with ImageNet weights; weight download/load failure does not silently switch to random initialization.
- Configuration for this invocation: seed=42, image size=160, batch size=16, maximum epochs=10, early-stopping patience=3, workers=0; AdamW, learning rate 1e-3 for CNN and 3e-4 for MobileNetV3-Small, weight decay 1e-4, and ReduceLROnPlateau on validation macro-F1.
- Runtime at report generation: 2 CPU cores; RAM 3939.9 MiB; GPU: NOT DETECTED (nvidia-smi unavailable); PyTorch: NOT INSTALLED; torchvision: NOT INSTALLED; configured device: CPU (if training dependencies are installed).
- Training device recorded for completed run(s): NOT AVAILABLE — no experiment completed.

## 7. Metrics and observed results
Planned metrics include accuracy; macro and weighted precision, recall, and F1; per-class sensitivity/specificity; one-vs-rest ROC-AUC and PR-AUC when class representation permits; confusion matrices; parameter count; training duration; and forward-pass latency.
**No experiment completed** (4 planned row(s)). All experimental metrics are NOT AVAILABLE; blocked/skip statuses and blank metric fields are retained in `07_results/master_results.csv` and `.xlsx`.
No test predictions, classification reports, confusion matrices, training histories, or model checkpoints were produced.

## 8. Model comparison and best model
No model comparison or best-model selection is possible without completed experiments. The highest literature-reported result is not substituted for this study's empirical performance.

## 9. Class-wise analysis and statistical analysis
Class-wise precision/recall/F1, sensitivity, specificity, and confusion matrices are NOT AVAILABLE. No hypothesis test, p-value, confidence interval, or statistical-significance statement is reported because no real-data experiment completed.
Repeated-seed comparisons are required before making inferential or statistical-significance claims.

## 10. Execution log and validation
The workspace/PDF inventory and dataset audit were executed. The pipeline correctly stopped before training because the audited data and/or required dependencies were not ready. All supplied PDFs were text-extracted for inspection and SHA-256 deduplicated; the existing notebook was confirmed unexecuted. Software tests exercise split/leakage and metric functions using ephemeral toy fixtures only; those fixtures are not research data and their scores are not saved in result tables.

## 11. Limitations
- Primary limitation: the supplied workspace has no valid HAM10000 image/metadata table; see `03_dataset/dataset_report.md` for the exact blockers.
- No model was trained or checkpoint saved. Runtime at report generation: 2 CPU cores; RAM 3939.9 MiB; GPU: NOT DETECTED (nvidia-smi unavailable); PyTorch: NOT INSTALLED; torchvision: NOT INSTALLED; configured device: CPU (if training dependencies are installed).
- PyTorch/torchvision availability is recorded above; no results were fabricated to work around missing data or dependencies.
- Dataset provenance/version, patient grouping, image quality, and external validation remain unverified until valid data are supplied.
- Any HAM10000-only results would not establish real-world generalization or clinical utility.

## 12. Reproducibility and next steps
Install the dependencies in `11_reproducibility/requirements.txt` (use a platform-matched PyTorch/torchvision wheel if needed), place the original HAM10000 metadata/images under `DATA/HAM10000/` (or pass `--data-root`), and resolve audit blockers. Exact commands from the repository root:

```bash
python -m pip install -r Skin_Cancer_Deep_Learning_Research/11_reproducibility/requirements.txt
python Skin_Cancer_Deep_Learning_Research/11_reproducibility/run_pipeline.py --data-root DATA/HAM10000
```

With valid real data and compatible dependencies, the pipeline writes grouped splits/leakage checks, per-run histories/checkpoints, held-out-test metrics/reports/confusion matrices, master CSV/XLSX tables, and figures.
The audit is currently blocked; with the present missing/invalid dataset, the second command is expected to exit 2 with `BLOCKED_MISSING_OR_INVALID_DATA` and will not report training metrics.
See `11_reproducibility/README.md` for data layout and optional flags.

## 13. Future work
Obtain a properly licensed, checksum-recorded HAM10000 copy; install compatible PyTorch/torchvision; resolve all audit blockers; rerun the four-run comparison; then repeat with multiple seeds. Predefine clinical class mappings; add an external evaluation only with compatible labels and patient-level deduplication; quantify uncertainty/calibration; and review minority-class errors with dermatology expertise.
