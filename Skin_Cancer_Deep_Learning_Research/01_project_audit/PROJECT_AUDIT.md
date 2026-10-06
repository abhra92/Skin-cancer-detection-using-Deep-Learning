# Project audit

- Audit time (UTC): `2026-10-06T09:36:04+00:00`
- Repository: `/home/user/Skin-cancer-detection-using-Deep-Learning`
- Branch: `arena/5a5efa09-skin-cancer-detection-using-de`
- Commit at audit: `d0240649e4355f3462230bb6fdfb27f2f08483c8`
- Source inventory file count: **76** (generated outputs and `.git` excluded)

## Workspace inventory

| Category | Count |
|---|---:|
| notebook | 1 |
| other | 2 |
| research_methodology_or_draft_pdf | 4 |
| research_paper_pdf | 35 |
| tabular_file | 1 |
| text_or_configuration | 26 |
| word_document | 7 |

The full file-by-file inventory is `workspace_inventory.csv`. Source PDFs are catalogued with SHA-256 deduplication and page-one text extracts in `../02_literature/source_paper_inventory.csv`.

## Research papers and documents

- PDF paths: **39**; unique PDF contents: **34**.
- Research-paper PDF paths: **35**; unique paper contents: **30**.
- PDF text extraction errors: **0**.
- Word documents: **7**; paragraph counts: `{"Extra/Experiment 1 - Introduction to Research and Identification of Research Problem.docx": 135, "Extra/Experiment_2_Literature_Review_Skin_Cancer_DL.docx": 269, "Extra/Research_Paper_Formatting_Guidelines.docx": 17, "Research questions.docx": 0, "Research_Objectives_Questions_Hypothesis.docx": 672, "Review and reference.docx": 108, "Skin_Cancer_Research_Questions_IEEE.docx": 241}`.
- Existing paper text extracts and JSON/knowledge-graph files are identified in `audit_environment.json` and `workspace_inventory.csv`.
- The existing `Extra/skin_cancer_papers_clean.csv` is bibliography metadata (45 rows in the supplied file), not image labels or HAM10000 metadata.

## Dataset inventory

- Image files (JPG/JPEG/PNG/WebP/BMP): **0**.
- CSV files: `Extra/skin_cancer_papers_clean.csv`.
- HAM10000 metadata CSV candidate(s): `NONE`.
- HAM10000 image folders, ISIC image data, PH2 image data, and metadata labels are not present in the inspected workspace. The directories with those names contain research papers/reviews, not data.
- As no image/metadata rows exist, class frequencies, missing-value rates, duplicates, corrupted images, and metadata-image consistency are **NOT AVAILABLE** and are not estimated.

## Existing code and results

- Existing Python scripts before this generated pipeline: **0**.
- Notebook(s): `[{"path": "HAM10000_Deep_Learning_PyTorch.ipynb", "cell_count": 25, "code_cells": 21, "executed_code_cells": 0, "stored_outputs": 0}]`.
- The only notebook is a 25-cell PyTorch EfficientNet-B0/HAM10000 workflow. It has zero executed code cells and zero stored outputs; it is not evidence of executed training.
- It specifies lesion-group splitting, train-only augmentation, weighted cross entropy, ImageNet transfer weights and test evaluation, but has no local data, metric exports, or model artifacts.
- Existing model checkpoints: **0**; existing plot/image files: **0**.
- No existing experiment metrics or confusion matrices were supplied. The literature bibliography CSV is not counted as results.

## Runtime and compute

- Python: `3.11.2` at `/home/user/.venv/bin/python`.
- OS/platform: `Linux-6.1.158+-x86_64-with-glibc2.36`.
- CPU cores visible: `2`.
- RAM total/available: `3939.9 / 3564.1 MiB`.
- GPU: `NOT DETECTED (nvidia-smi is unavailable)`; `nvidia-smi` available: `False`.
- PyTorch installed: `False`; CUDA available: `NOT CHECKED`.
- Free disk: `18.46 GiB` of `20.34 GiB`.
- Analysis virtualenv packages: `{"pypdf": "6.19.0", "numpy": "2.4.6", "pandas": "3.0.6", "scikit-learn": "1.9.1", "matplotlib": "3.11.2", "pillow": "12.3.0", "seaborn": "0.13.2", "openpyxl": "3.1.5", "torch": "NOT INSTALLED", "torchvision": "NOT INSTALLED"}`.
- In the initial system Python, the common ML stack (PyTorch, torchvision, TensorFlow, scikit-learn, pandas, NumPy, Pillow, Matplotlib, seaborn and openpyxl) was absent. An isolated `/home/user/.venv` was used for PDF/data/report tooling. A CPU-only PyTorch wheel fetch was attempted but the PyTorch wheel index terminated TLS before providing a package; PyTorch/torchvision remain unavailable.

## Execution conclusion

The source audit, PDF text extraction, document inspection, environment checks, and data-availability checks were executable and were performed. Supervised model training/evaluation cannot validly run in this workspace because HAM10000 images and metadata are absent; there is no GPU, and PyTorch/torchvision are not installed. No research metrics are fabricated or substituted with synthetic results.

### Key artifacts

- `workspace_inventory.csv` — file-level inventory.
- `audit_environment.json` — machine-readable file/data/environment audit.
- `../02_literature/source_paper_inventory.csv` — PDF content hashes, duplicate paths, page counts, PDF metadata and extracts.
- `../03_dataset/dataset_report.csv` and `dataset_report.md` — actual dataset check (currently blocked).
