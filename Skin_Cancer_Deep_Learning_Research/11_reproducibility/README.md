# Reproducibility and execution guide

## What is reproducible here

- A read-only HAM10000 metadata/image audit, including missing/malformed labels, image matching, exact duplicate-content hashes, image decoding, and lesion-label consistency.
- A deterministic stratified group split, approximately 60/20/20, that keeps patient IDs together when supplied, otherwise lesion IDs, and exact-image duplicates together.
- Two model families (compact CNN from scratch and ImageNet-pretrained MobileNetV3-Small), each with an unweighted and an inverse-frequency weighted cross-entropy run.
- Train-only on-the-fly augmentation, validation-macro-F1 checkpoint selection, one held-out test evaluation, per-class metrics, confusion matrices, histories, checkpoints, and CSV/XLSX exports.
- Software tests for split/leakage and metric behavior. The tests create disposable synthetic fixtures and do **not** write their scores to research results.

The model-training path has **not** been executed in this workspace: HAM10000 files are absent, and the initial runtime did not contain PyTorch/torchvision. The available runtime has two visible CPU cores, about 4 GiB RAM, and no detected GPU. A CPU-only PyTorch wheel fetch was attempted but failed at the external wheel index TLS step. The current output tables retain blank metrics and blocked statuses.

## Environment setup

Python 3.11 is the inspected interpreter. Create a virtual environment outside version-controlled data/results if desired:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Skin_Cancer_Deep_Learning_Research/11_reproducibility/requirements.txt
```

PyTorch wheels vary by operating system and CPU/CUDA runtime. If the combined requirements install cannot resolve a compatible PyTorch wheel, install the scientific/report packages first, then install a matching PyTorch + torchvision build from the official PyTorch installation selector. For a CPU-only example:

```bash
python -m pip install numpy pandas scikit-learn Pillow matplotlib seaborn openpyxl pypdf tqdm
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For CUDA, use the official PyTorch command appropriate to the machine's supported CUDA runtime. The current sandbox has no GPU; CUDA performance is not claimed.

## Data placement and provenance

The default root is `DATA/HAM10000/`, which is ignored by Git. It must contain the original `HAM10000_metadata.csv` plus the JPG/JPEG image files, including when those images are nested in the dataset's image-part subfolders. Alternatively, pass `--data-root /path/to/HAM10000`.

Before training, record the dataset source, download/access date, license/terms, and any supplied checksum in your research notebook. The pipeline does not download data or require a Kaggle token. It will not treat a bibliography CSV as labels. Keep the original data read-only. Do not mix in ISIC/PH2 files without a separately specified, audited label mapping and external-test protocol.

## Validate and run

From the repository root:

```bash
python Skin_Cancer_Deep_Learning_Research/11_reproducibility/audit_workspace.py
PYTHONPATH=Skin_Cancer_Deep_Learning_Research/11_reproducibility \
  python -m unittest discover \
  -s Skin_Cancer_Deep_Learning_Research/11_reproducibility/tests -v
python Skin_Cancer_Deep_Learning_Research/11_reproducibility/run_pipeline.py \
  --data-root DATA/HAM10000
```

Optional controls (defaults shown):

```bash
python Skin_Cancer_Deep_Learning_Research/11_reproducibility/run_pipeline.py \
  --data-root DATA/HAM10000 --seed 42 --image-size 160 \
  --batch-size 16 --epochs 10 --patience 3 --num-workers 0
```

- The pipeline first audits source files and data. If metadata or images are missing/invalid, it writes a blocker report, four blocked experiment rows with null metrics, and exits with code `2`. This is expected until real data pass the audit.
- If data pass audit but PyTorch/torchvision are unavailable, runs are marked blocked due to dependencies. No test split is used for hyperparameter selection.
- A MobileNet pretrained-weight download/load failure is visible and does not silently switch to random initialization.
- An actual execution error prints a traceback, logs the run as failed, and is not marked completed.

## Experiment design

| ID | Model | Initialization | Loss intervention | Paired comparison |
|---|---|---|---|---|
| EXP-001 | SmallCNN | Random/from scratch | Unweighted cross-entropy | Reference for EXP-002 |
| EXP-002 | SmallCNN | Random/from scratch | Inverse-frequency weighted cross-entropy | Compare loss intervention with EXP-001 |
| EXP-003 | MobileNetV3-Small | ImageNet-pretrained | Unweighted cross-entropy | Transfer reference for EXP-004 |
| EXP-004 | MobileNetV3-Small | ImageNet-pretrained | Inverse-frequency weighted cross-entropy | Compare loss intervention with EXP-003 |

Fixed controls: seed 42; square 160×160 RGB resize; ImageNet mean/std; AdamW (1e-3 for SmallCNN, 3e-4 for MobileNetV3-Small), weight decay 1e-4; ReduceLROnPlateau on validation macro-F1; max 10 epochs; early-stop patience 3; batch size 16. Loss weights are calculated using the **training split only**. The small CNN and MobileNet are intended as practical baselines, not as an exhaustive architecture search.

The split uses `StratifiedGroupKFold` for an approximately 20% test group holdout, then a second grouped split of the remaining records to create approximately 60% train and 20% validation. Exact proportions may vary because whole groups are kept together. The manifest and pairwise group/hash overlap checks are saved before training. If the available data cannot support the specified group-stratified split (e.g. too few independent lesion groups in a class), training is blocked instead of falling back to a leaking image-level split.

## Metrics and averaging

- Accuracy and balanced accuracy.
- Macro precision/recall/F1: unweighted mean across the seven fixed HAM10000 classes.
- Weighted precision/recall/F1: support-weighted mean across classes.
- Per-class precision, recall/sensitivity, F1, specificity, and support.
- Multiclass ROC-AUC and PR-AUC as one-vs-rest, macro/weighted where class support permits; undefined classes are reported as unavailable/partial with an explanation.
- Raw and row-normalized confusion matrices.
- Training time, model parameter counts, and per-image forward-pass time.

Checkpoint selection uses validation macro-F1. Test metrics are computed from the selected checkpoint only; they are not used to choose an architecture or tune a threshold.

## Output files

- `01_project_audit/PROJECT_AUDIT.md`, `workspace_inventory.csv`, `audit_environment.json`
- `02_literature/LITERATURE_SUMMARY.md`, `source_paper_inventory.csv`
- `03_dataset/dataset_report.csv`, `dataset_report.md`, `dataset_manifest.json`
- `04_preprocessing/preprocessing_config.json`, `DATA_PREPROCESSING_REPORT.md`, `DATA_LEAKAGE_CHECK.md`, `splits.csv` (only after valid data), `processed_data/`
- `05_experiments/experiment_matrix.csv`, `experiment_log.csv`, `configs/`, `training_histories/`
- `06_models/` checkpoint files only for completed, real-data runs
- `07_results/master_results.csv`, `master_results.xlsx`, `model_comparison.csv`, plus per-class reports/predictions/confusion matrices
- `08_figures/` data-availability/status figures in the current blocked run; class/model/training figures only when observed data support them
- `09_statistics/STATISTICAL_ANALYSIS.md`
- `10_reports/RESULTS.md`, `DISCUSSION.md`, `TABLES.md`, `FIGURES.md`, `FINAL_RESEARCH_EXPERIMENT_REPORT.md`

No confusion-matrix placeholder, class-frequency estimate, model-performance plot, or fake checkpoint is emitted when the necessary real labels/predictions are absent.

## Repeated runs and statistical claims

The default matrix is one fixed seed and is descriptive. One run does not justify a p-value, confidence interval, or statistical-significance claim. For inferential comparisons, prespecify several seeds/runs and use paired comparisons on matched grouped splits; retain the independent test-set policy. Do not use the test set for model selection.

## Limitations and intended use

HAM10000 benchmark results are not prospective clinical evidence. The dataset has limited demographic/site/device scope and task-specific label characteristics. External ISIC testing requires an independent, harmonized set and exact duplicate/patient leakage checks. This software is for education/research only and must not guide diagnosis or treatment.
