"""Run the audited HAM10000 experiment matrix.

Usage from the repository root:
    python Skin_Cancer_Deep_Learning_Research/11_reproducibility/run_pipeline.py \
        --data-root DATA/HAM10000

The default action audits local files and attempts training only if the actual
metadata and image files pass data-integrity checks. It never downloads data or
substitutes synthetic samples.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
from typing import Any

import numpy as np
import pandas as pd

from pipeline_core import (
    CLASS_NAMES,
    DEFAULT_DATA_ROOT,
    EXPERIMENTS,
    LOG_COLUMNS,
    RESULT_COLUMNS,
    SEED,
    audit_dataset,
    ensure_output_structure,
    git_version,
    make_grouped_split,
    split_quality_table,
    utc_now,
    write_experiment_matrix,
    write_json,
    write_master_outputs,
    write_preprocessing_config,
    write_execution_status_figure,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = OUTPUT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Directory containing HAM10000_metadata.csv and image files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Research output directory")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0, help="Default 0 is conservative for low RAM/CPU environments")
    parser.add_argument("--audit-only", action="store_true", help="Audit/report inputs but do not train even when data is present")
    return parser.parse_args()


def _hash_dataset(frame: pd.DataFrame, metadata_path: str | None) -> str:
    digest = hashlib.sha256()
    if metadata_path and Path(metadata_path).is_file():
        with Path(metadata_path).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    if "image_sha256" in frame:
        for image_hash in sorted(frame["image_sha256"].dropna().astype(str)):
            digest.update(image_hash.encode("ascii"))
    digest.update(f"n={len(frame)}".encode("ascii"))
    return digest.hexdigest()


def _save_splits(splits: pd.DataFrame, data_root: Path, output_dir: Path, seed: int) -> pd.DataFrame:
    manifest = splits.copy()
    root = data_root.resolve()
    manifest["image_path"] = manifest["image_path"].map(
        lambda path: Path(path).resolve().relative_to(root).as_posix()
    )
    desired_columns = [
        column for column in [
            "image_id", "dx", "lesion_id", "patient_id", "split",
            "leakage_group", "image_sha256", "image_path",
        ] if column in manifest.columns
    ]
    manifest[desired_columns].to_csv(output_dir / "04_preprocessing" / "splits.csv", index=False)
    counts = split_quality_table(splits)
    counts.to_csv(output_dir / "04_preprocessing" / "split_class_counts.csv", index=False)

    groups_by_split = {
        split: set(splits.loc[splits["split"] == split, "leakage_group"].astype(str))
        for split in ("train", "validation", "test")
    }
    hashes_by_split = {
        split: set(splits.loc[splits["split"] == split, "image_sha256"].dropna().astype(str))
        for split in ("train", "validation", "test")
    }
    group_overlap = {
        f"{first}_vs_{second}": sorted(groups_by_split[first] & groups_by_split[second])
        for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]
    }
    hash_overlap = {
        f"{first}_vs_{second}": sorted(hashes_by_split[first] & hashes_by_split[second])
        for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]
    }
    passed = all(not overlap for overlap in [*group_overlap.values(), *hash_overlap.values()])
    status = "PASS" if passed else "FAIL"
    leakage_lines = [
        "# Data leakage check",
        "",
        f"- Dataset split file: `04_preprocessing/splits.csv`",
        f"- Grouping: patient ID when present, lesion ID when present, and exact image SHA-256; components are kept within one split.",
        f"- Group disjointness: **{status}**",
        f"- Exact-byte-image disjointness: **{status}**",
        "- Split assignment happened before augmentation; only the train dataset uses random image transforms.",
        "- Validation macro-F1 selects the checkpoint. The held-out test set is not used for tuning.",
        "",
        "## Split sizes",
        "",
        "| Split | Images |",
        "|---|---:|",
    ]
    for split in ("train", "validation", "test"):
        leakage_lines.append(f"| {split} | {int((splits['split'] == split).sum())} |")
    leakage_lines.extend(["", "## Group overlap by split pair", "", "```json", json.dumps(group_overlap, indent=2), "```", "", "## Exact-image hash overlap by split pair", "", "```json", json.dumps(hash_overlap, indent=2), "```", ""])
    (output_dir / "04_preprocessing" / "DATA_LEAKAGE_CHECK.md").write_text("\n".join(leakage_lines), encoding="utf-8")
    return counts


def _write_no_data_leakage_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Data leakage check",
        "",
        "**Status: NOT EXECUTABLE — no local image/metadata records were found.**",
        "",
        "No split was created and no model was trained. Consequently, patient/lesion overlap and exact-image duplicate overlap across train/validation/test are **NOT AVAILABLE**, not passed.",
        "",
        "The reproducible pipeline is configured to form groups from patient ID (when supplied), lesion ID, and exact-image SHA-256. It uses a stratified grouped holdout, validates group/hash disjointness, writes the split manifest before training, and applies augmentation only through the training data loader.",
        "",
        "The held-out test split is reserved for one final evaluation; validation macro-F1 is the only checkpoint-selection metric.",
        "",
        "Blocker(s):",
    ]
    lines.extend(f"- {reason}" for reason in summary.get("blocking_reasons", ["Dataset metadata and images are missing."]))
    lines.append("")
    (output_dir / "04_preprocessing" / "DATA_LEAKAGE_CHECK.md").write_text("\n".join(lines), encoding="utf-8")


def _experiment_log_row(config: dict[str, Any], status: str, dataset_version: str, seed: int, image_size: int, batch_size: int, epochs: int, notes: str, result: dict[str, Any] | None = None) -> dict[str, Any]:
    result = result or {}
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "timestamp_utc": timestamp,
        "experiment_id": config["experiment_id"],
        "status": status,
        "dataset_version": dataset_version,
        "code_version": git_version(REPO_ROOT),
        "random_seed": seed,
        "model": config["model"],
        "pretrained": config["pretrained"],
        "imbalance_strategy": config["imbalance_strategy"],
        "split_policy": "StratifiedGroupKFold; approximately 60/20/20; patient/lesion/exact-image grouped",
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_run": result.get("epochs_run"),
        "best_epoch": result.get("best_epoch"),
        "training_seconds": result.get("training_seconds"),
        "test_inference_seconds": result.get("test_inference_seconds"),
        "metrics_json": json.dumps(result.get("metrics", {}), sort_keys=True),
        "output_paths_json": json.dumps(result.get("output_paths", {}), sort_keys=True),
        "notes": notes,
    }


def _blocked_result(config: dict[str, Any], status: str, note: str, seed: int, image_size: int, batch_size: int, epochs: int, dataset_version: str) -> dict[str, Any]:
    result = {column: None for column in RESULT_COLUMNS}
    result.update({
        "experiment_id": config["experiment_id"],
        "status": status,
        "model": config["model"],
        "imbalance_strategy": config["imbalance_strategy"],
        "dataset_version": dataset_version,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "notes": note,
    })
    return result


def _write_preprocessing_report(output_dir: Path, summary: dict[str, Any], split_counts: pd.DataFrame | None, config: dict[str, Any]) -> None:
    lines = [
        "# Data preprocessing report",
        "",
        f"- Dataset: HAM10000 (local data root: `{summary.get('data_root')}`)",
        f"- Dataset audit status: **{'READY' if summary.get('valid_for_training') else 'BLOCKED'}**",
        f"- Image size configured: {config['image_size']} × {config['image_size']} RGB.",
        f"- Normalization: ImageNet channel mean/std. Validation/test never contribute to normalization estimates.",
        f"- Augmentation: {', '.join(config['augmentation']['operations'])}; training split only, after group-aware splitting.",
        f"- Split: {config['split']['method']}; seed {config['split']['seed']}; approximate 60/20/20.",
        f"- Original data: read-only; no image files are copied or transformed on disk.",
        "",
        "## Execution status",
        "",
    ]
    if split_counts is None:
        lines.append("Preprocessing was **not executed on clinical data** because no validated dataset was available. No split manifest, resized images, or augmentation outputs were generated.")
        lines.append("")
        lines.extend(f"- Blocker: {reason}" for reason in summary.get("blocking_reasons", []))
    else:
        lines.extend([
            "Metadata validation and group-aware split assignment were executed. Resizing, normalization, and stochastic augmentation are implemented on-the-fly in the model input pipeline; the raw data remain unchanged.",
            "",
            "| Split | Diagnosis | Image count |",
            "|---|---|---:|",
        ])
        for row in split_counts.itertuples(index=False):
            lines.append(f"| {row.split} | {row.dx} | {row.image_count} |")
    lines.append("")
    (output_dir / "04_preprocessing" / "DATA_PREPROCESSING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_missing_artifact_notes(output_dir: Path) -> None:
    notes = {
        output_dir / "06_models" / "README.md": "# Model artifacts\n\nNo model checkpoints were produced in the current run because the HAM10000 images and metadata are absent. The training code saves PyTorch state-dict checkpoints here after a real run.\n",
        output_dir / "07_results" / "classification_reports" / "README.md": "# Classification reports\n\nNo test-set classification report is available: no model was trained or evaluated on HAM10000. Reports are generated per experiment only after training and held-out test evaluation.\n",
        output_dir / "07_results" / "confusion_matrices" / "README.md": "# Confusion matrices\n\nNo confusion matrix was generated because there are no test labels or predictions. A blank/fabricated matrix is intentionally not supplied. Each real completed experiment will create raw and row-normalized matrices.\n",
        output_dir / "05_experiments" / "training_histories" / "README.md": "# Training histories\n\nNo training histories exist for the current run because training was blocked before the model stage. The folder will contain one CSV per completed training run.\n",
    }
    for path, text in notes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(text, encoding="utf-8")


def _create_metric_figures(output_dir: Path, result_rows: list[dict[str, Any]]) -> None:
    completed = [row for row in result_rows if row.get("status") == "COMPLETED"]
    if not completed:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [f"{row['experiment_id']}\n{row['model']}\n{row['imbalance_strategy'].replace('_', ' ')}" for row in completed]
    for metric, filename, title in [
        ("accuracy", "model_comparison_accuracy", "Held-out test accuracy"),
        ("balanced_accuracy", "model_comparison_balanced_accuracy", "Held-out test balanced accuracy"),
        ("macro_f1", "model_comparison_macro_f1", "Held-out test macro-F1"),
        ("macro_precision", "model_comparison_macro_precision", "Held-out test macro precision"),
        ("macro_recall", "model_comparison_macro_recall", "Held-out test macro recall"),
    ]:
        values = [row.get(metric) for row in completed]
        if not any(value is not None and np.isfinite(float(value)) for value in values):
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.7), 5.0))
        bars = ax.bar(names, [np.nan if value is None else value for value in values], color="#4c72b0")
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title, weight="bold")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        target = output_dir / "08_figures" / f"{filename}.png"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    per_class_paths = [
        output_dir / "07_results" / "classification_reports" / f"{row['experiment_id']}_per_class_metrics.csv"
        for row in completed
    ]
    available = [path for path in per_class_paths if path.exists()]
    if available:
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        for row, path in zip(completed, per_class_paths):
            if not path.exists():
                continue
            per_class = pd.read_csv(path)
            ax.plot(per_class["class"], per_class["f1"], marker="o", label=row["experiment_id"])
        ax.set_ylabel("Per-class F1")
        ax.set_xlabel("HAM10000 diagnosis")
        ax.set_ylim(0, 1)
        ax.set_title("Held-out test per-class F1", weight="bold")
        ax.legend()
        ax.grid(alpha=0.22)
        fig.tight_layout()
        target = output_dir / "08_figures" / "per_class_f1.png"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    # Combined training curves are strictly sourced from saved actual histories.
    histories = sorted((output_dir / "05_experiments" / "training_histories").glob("EXP-*_training_history.csv"))
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for path in histories:
            history = pd.read_csv(path)
            if history.empty:
                continue
            label = path.name.split("_")[0]
            axes[0].plot(history["epoch"], history["train_loss"], alpha=0.65, linestyle="--", label=f"{label} train")
            axes[0].plot(history["epoch"], history["validation_loss"], marker="o", label=f"{label} validation")
            axes[1].plot(history["epoch"], history["train_macro_f1"], alpha=0.65, linestyle="--", label=f"{label} train")
            axes[1].plot(history["epoch"], history["validation_macro_f1"], marker="o", label=f"{label} validation")
        axes[0].set_title("Training and validation loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross-entropy loss")
        axes[1].set_title("Training and validation macro-F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        fig.tight_layout()
        target = output_dir / "08_figures" / "training_curves.png"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _runtime_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "cpu_cores": os.cpu_count(),
        "ram_total_mib": None,
        "gpu": "NOT DETECTED (nvidia-smi unavailable)",
        "torch": "NOT INSTALLED",
        "torchvision": "NOT INSTALLED",
        "device": "CPU (if training dependencies are installed)",
    }
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                facts["ram_total_mib"] = round(int(line.split()[1]) / 1024, 1)
                break
    except OSError as exc:
        facts["ram_read_error"] = f"{type(exc).__name__}: {exc}"
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            probe = subprocess.run([nvidia_smi, "-L"], capture_output=True, text=True, timeout=10)
            facts["gpu"] = (probe.stdout or probe.stderr).strip() or "nvidia-smi reported no GPU"
        except (OSError, subprocess.TimeoutExpired) as exc:
            facts["gpu"] = f"PROBE ERROR: {type(exc).__name__}: {exc}"
    if importlib.util.find_spec("torchvision"):
        try:
            torchvision_spec = importlib.util.find_spec("torchvision")
            facts["torchvision"] = str(getattr(torchvision_spec, "origin", "INSTALLED"))
        except (ImportError, ValueError) as exc:
            facts["torchvision"] = f"DISCOVERY ERROR: {type(exc).__name__}: {exc}"
    if importlib.util.find_spec("torch"):
        try:
            import torch
            facts["torch"] = torch.__version__
            if torch.cuda.is_available():
                facts["device"] = "CUDA"
                facts["gpu"] = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
            else:
                facts["device"] = "CPU"
        except Exception as exc:
            facts["torch"] = f"IMPORT ERROR: {type(exc).__name__}: {exc}"
            facts["device"] = "NOT AVAILABLE"
    return facts


def _write_current_run_reports(output_dir: Path, summary: dict[str, Any], result_rows: list[dict[str, Any]], matrix: pd.DataFrame, args: argparse.Namespace) -> None:
    completed = [row for row in result_rows if row.get("status") == "COMPLETED"]
    blocked = [row for row in result_rows if str(row.get("status", "")).startswith("BLOCKED")]
    runtime = _runtime_facts()
    ram_text = f"{runtime['ram_total_mib']} MiB" if runtime.get("ram_total_mib") is not None else "NOT AVAILABLE"
    runtime_text = (
        f"{runtime.get('cpu_cores') or 'NOT AVAILABLE'} CPU cores; RAM {ram_text}; "
        f"GPU: {runtime.get('gpu')}; PyTorch: {runtime.get('torch')}; torchvision: {runtime.get('torchvision')}; "
        f"configured device: {runtime.get('device')}"
    )
    table_rows = []
    for row in result_rows:
        table_rows.append(
            f"| {row.get('experiment_id')} | {row.get('model')} | {row.get('status')} | "
            f"{row.get('accuracy') if row.get('accuracy') is not None else 'NOT AVAILABLE'} | "
            f"{row.get('macro_f1') if row.get('macro_f1') is not None else 'NOT AVAILABLE'} | "
            f"{row.get('training_seconds') if row.get('training_seconds') is not None else 'NOT AVAILABLE'} |"
        )
    if not table_rows:
        table_rows = ["| — | — | No experiment rows | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE |"]

    if completed:
        best = max(completed, key=lambda row: float(row.get("macro_f1") or -1))
        best_text = f"Best macro-F1 among completed experiments: **{best['experiment_id']} ({best['model']})**, macro-F1={best['macro_f1']:.4f}. This is a benchmark result, not clinical validation."
        result_interpretation = "Only completed rows in `master_results.csv` contain observed model metrics. The test set was evaluated after validation-based selection."
    else:
        best_text = "**No best model can be selected:** no completed experiment has empirical validation/test metrics."
        result_interpretation = "No research accuracy, precision, recall, F1, AUC, confusion matrix, training time, or best-model claim is available. Empty metric fields are intentionally left blank."

    results_md = [
        "# Results",
        "",
        f"- Dataset audit: **{'passed' if summary.get('valid_for_training') else 'blocked'}**.",
        f"- Experiments completed: **{len(completed)}** of **{len(result_rows)}** planned.",
        "",
        "| Experiment | Model | Status | Accuracy | Macro-F1 | Training seconds |",
        "|---|---|---|---:|---:|---:|",
        *table_rows,
        "",
        result_interpretation,
        "",
        best_text,
        "",
        "Model metrics in this report are populated only from executed runs saved in `07_results/master_results.csv`; literature values are not mixed into these tables.",
        "",
    ]
    (output_dir / "10_reports" / "RESULTS.md").write_text("\n".join(results_md), encoding="utf-8")

    discussion = [
        "# Discussion",
        "",
        "## Interpretation",
        "",
    ]
    if completed:
        discussion.append("The current results are restricted to the supplied HAM10000 sample and the fixed split/configuration. Macro-F1 and class-wise recall are considered alongside accuracy because class frequencies may be imbalanced. The highest macro-F1 is not automatically treated as clinically superior; minority-class sensitivity and computational cost are reported together.")
    else:
        discussion.append("There are no empirical results to interpret. The local workspace contains research papers and an unexecuted workflow notebook, but no HAM10000 image files or diagnosis metadata. Model training and clinical metrics were therefore blocked. Reporting a literature accuracy as a project result, or creating synthetic model scores, would be misleading and was not done.")
    discussion.extend([
        "",
        "## Statistical analysis",
        "",
        "No hypothesis test or confidence interval is reported without repeated real-data runs. A single split/model run does not support a claim of statistically significant architecture or loss-function differences. The primary comparison is paired by fixed split and seed; repeat-seed analysis is needed for inferential claims.",
        "",
        "## Limitations",
        "",
    ])
    if completed:
        discussion.extend([
            f"- This run completed {len(completed)} experiment(s) on the available HAM10000 data; results are descriptive for the audited split only.",
            "- The default matrix uses one seed/split; between-seed variability and inferential uncertainty are not estimated.",
            "- HAM10000-only evaluation is a benchmark, not prospective or clinical validation; external testing requires an independently sourced, label-compatible set and a prespecified mapping.",
            "- Group leakage controls depend on patient/lesion identifiers actually supplied; absent patient IDs cannot be inferred.",
        ])
    else:
        discussion.extend([
            "- No image dataset or label metadata was present, so dataset statistics and all supervised evaluation measures are unavailable.",
            f"- No training could be executed or checkpoint saved. Runtime at report generation: {runtime_text}.",
            "- Future runs must first audit real source images, maintain patient/lesion/content-level disjointness, and report the resulting split counts.",
            "- HAM10000-only evaluation would remain a benchmark study, not prospective or clinical validation; external testing requires an independently sourced, label-compatible set and a prespecified mapping.",
        ])
    discussion.append("")
    (output_dir / "10_reports" / "DISCUSSION.md").write_text("\n".join(discussion), encoding="utf-8")

    table_md = [
        "# Tables",
        "",
        "## Table 1. Experiment status and observed metrics",
        "",
        "| Experiment | Model | Status | Accuracy | Macro-F1 | Training seconds |",
        "|---|---|---|---:|---:|---:|",
        *table_rows,
        "",
        "`NOT AVAILABLE` means the value was not experimentally determined; it does not mean zero.",
        "",
        "## Table 2. Pre-registered experiment matrix",
        "",
        "| ID | Model | Imbalance strategy | Execution status |",
        "|---|---|---|---|",
    ]
    for row in matrix.to_dict(orient="records"):
        table_md.append(f"| {row['experiment_id']} | {row['model']} | {row['imbalance_strategy']} | {row['execution_status']} |")
    table_md.append("")
    (output_dir / "10_reports" / "TABLES.md").write_text("\n".join(table_md), encoding="utf-8")

    figures = sorted(path.name for path in (output_dir / "08_figures").glob("*.png"))
    figure_lines = [
        "# Figures",
        "",
        "Figures are generated from observed local data/metrics only. Status/availability plots are not model-performance figures.",
        "",
    ]
    if figures:
        figure_lines.extend(f"- `{name}`" for name in figures)
    else:
        figure_lines.append("No figures have been generated.")
    if not summary.get("valid_for_training"):
        figure_lines.extend([
            "",
            "- `dataset_asset_availability.png` shows only whether source assets were found; it is explicitly not a class-distribution plot.",
            "- Class distribution, training/validation curves, confusion matrices, model comparisons, per-class F1, precision, and recall plots are **NOT AVAILABLE** because there are no real labels/predictions.",
        ])
    figure_lines.append("")
    (output_dir / "10_reports" / "FIGURES.md").write_text("\n".join(figure_lines), encoding="utf-8")

    def display(value: Any) -> str:
        if value is None or pd.isna(value):
            return "NOT AVAILABLE"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{float(value):.4f}"
        return str(value)

    result_metric_table = [
        "| Experiment | Model | Accuracy | Macro-F1 | Weighted-F1 | Macro ROC-AUC | Macro PR-AUC | Training seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in completed:
        result_metric_table.append(
            f"| {row.get('experiment_id')} | {row.get('model')} | {display(row.get('accuracy'))} | "
            f"{display(row.get('macro_f1'))} | {display(row.get('weighted_f1'))} | "
            f"{display(row.get('macro_roc_auc_ovr'))} | {display(row.get('macro_pr_auc_ovr'))} | "
            f"{display(row.get('training_seconds'))} |"
        )
    if completed:
        best = max(completed, key=lambda row: float(row.get("macro_f1") or -1))
        observed_results = [
            f"**{len(completed)} of {len(result_rows)} planned experiment(s) completed.** Metrics below are from completed held-out-test evaluations; incomplete rows remain explicitly blocked/skipped in `master_results.csv`.",
            "",
            *result_metric_table,
            "",
            "All available class-wise metrics, per-experiment classification reports, confusion matrices, histories, and checkpoints are linked in `master_results.csv`. Metrics are empirical results, not literature values.",
        ]
        best_text = f"Highest macro-F1 among completed experiments: **{best['experiment_id']} ({best['model']})**, macro-F1={display(best.get('macro_f1'))}. This is a benchmark result, not clinical validation. No uncompleted experiment is treated as a comparator."
        classwise_text = "Per-class classification reports and confusion matrices are saved for each completed run. No hypothesis tests or confidence intervals are reported: the matrix uses one seed/split and does not estimate between-seed variability."
        completed_devices = sorted({
            note.split("device=", 1)[1].split(";", 1)[0]
            for row in completed
            if (note := str(row.get("notes", ""))).find("device=") >= 0
        })
        training_device = ", ".join(completed_devices) if completed_devices else "NOT AVAILABLE in completed-run notes"
        preprocessing_execution = "The audited split was used for training; resizing/normalization and on-the-fly flips/rotation were applied by the training loader to train samples only. Validation and test transformations were deterministic. Raw source files were read only."
    else:
        observed_results = [
            f"**No experiment completed** ({len(result_rows)} planned row(s)). All experimental metrics are NOT AVAILABLE; blocked/skip statuses and blank metric fields are retained in `07_results/master_results.csv` and `.xlsx`.",
            "No test predictions, classification reports, confusion matrices, training histories, or model checkpoints were produced.",
        ]
        best_text = "No model comparison or best-model selection is possible without completed experiments. The highest literature-reported result is not substituted for this study's empirical performance."
        classwise_text = "Class-wise precision/recall/F1, sensitivity, specificity, and confusion matrices are NOT AVAILABLE. No hypothesis test, p-value, confidence interval, or statistical-significance statement is reported because no real-data experiment completed."
        training_device = "NOT AVAILABLE — no experiment completed"
        preprocessing_execution = "The code audits metadata/image IDs, missing values, labels, duplicate rows, exact image hashes, corruption, and lesion-label consistency. It is configured for square resizing to 160×160 RGB and ImageNet channel normalization (no test-derived statistics). Group-aware splitting precedes augmentation. Horizontal/vertical flips and ±15° rotation are applied on-the-fly to training images only; validation/test are deterministic. The raw dataset is never modified. No training-time transformations were executed because no experiment completed."

    class_counts = summary.get("diagnosis_label_counts") or {}
    if summary.get("valid_for_training"):
        class_distribution_text = "Observed diagnosis counts: " + ", ".join(
            f"{label}={int(class_counts.get(label, 0))}" for label in CLASS_NAMES
        ) + ". Integrity and split details are in `03_dataset/dataset_report.csv` and `04_preprocessing/DATA_LEAKAGE_CHECK.md`."
    else:
        class_distribution_text = "Class distribution, missing-value summaries, duplicate rates, corrupted-image counts, and metadata-image consistency: NOT AVAILABLE from valid research data because the source data are absent or failed audit."

    if not summary.get("valid_for_training"):
        limitation_lines = [
            "- Primary limitation: the supplied workspace has no valid HAM10000 image/metadata table; see `03_dataset/dataset_report.md` for the exact blockers.",
            f"- No model was trained or checkpoint saved. Runtime at report generation: {runtime_text}.",
            "- PyTorch/torchvision availability is recorded above; no results were fabricated to work around missing data or dependencies.",
            "- Dataset provenance/version, patient grouping, image quality, and external validation remain unverified until valid data are supplied.",
        ]
        future_text = "Obtain a properly licensed, checksum-recorded HAM10000 copy; install compatible PyTorch/torchvision; resolve all audit blockers; rerun the four-run comparison; then repeat with multiple seeds. Predefine clinical class mappings; add an external evaluation only with compatible labels and patient-level deduplication; quantify uncertainty/calibration; and review minority-class errors with dermatology expertise."
        expected_run_text = "The audit is currently blocked; with the present missing/invalid dataset, the second command is expected to exit 2 with `BLOCKED_MISSING_OR_INVALID_DATA` and will not report training metrics."
    elif not completed:
        limitation_lines = [
            f"- Dataset audit passed, but no experiment completed; inspect row-specific statuses in `07_results/master_results.csv`. Runtime: {runtime_text}.",
            "- Results remain unavailable until a non-audit-only run completes with compatible training dependencies.",
            "- HAM10000-only evaluation is a benchmark, not prospective or clinical validation; external evaluation and multiple seeds remain future work.",
        ]
        future_text = "Resolve any dependency/runtime blockers shown in the experiment log, complete the planned runs, repeat with multiple seeds, and add an independent compatible external evaluation with patient-level deduplication. Quantify uncertainty/calibration and review minority-class errors with dermatology expertise."
        expected_run_text = "The dataset audit passed, but no experiment completed in this invocation; see the experiment statuses/log for whether audit-only mode or another blocker prevented training."
    else:
        limitation_lines = [
            f"- This run completed {len(completed)} experiment(s) on the audited HAM10000 split; results are descriptive for this dataset/configuration only.",
            "- The default matrix uses one seed/split; between-seed variability and inferential uncertainty are not estimated.",
            "- HAM10000-only evaluation is a benchmark, not prospective or clinical validation; external testing requires an independent, label-compatible set and a prespecified mapping.",
            "- Leakage control depends on patient/lesion identifiers actually supplied; absent identifiers cannot be inferred.",
        ]
        future_text = "Repeat the comparison with multiple seeds; predefine clinical class mappings; add a truly external evaluation only with compatible labels and patient-level deduplication; quantify uncertainty/calibration; and review minority-class errors with dermatology expertise. No clinical-use claim should be made from benchmark images alone."
        expected_run_text = f"This invocation completed {len(completed)} experiment(s); any remaining blocked/skipped rows are identified in the result table."

    if args.audit_only:
        execution_text = "The workspace/PDF inventory and dataset audit were executed. Audit-only mode was explicitly requested, so no training was attempted."
    elif completed:
        execution_text = f"The workspace/PDF inventory and dataset audit were executed; the pipeline completed {len(completed)} training/evaluation run(s) and recorded their artifacts."
    else:
        execution_text = "The workspace/PDF inventory and dataset audit were executed. The pipeline correctly stopped before training because the audited data and/or required dependencies were not ready."

    final_report = [
        "# Final Research Experiment Report",
        "",
        "## 1. Research topic",
        "Skin Cancer Detection Using Deep Learning.",
        "",
        "## 2. Research problem and objective",
        "The supplied materials identify multi-class skin-lesion recognition, severe class imbalance, and the need for reliable evaluation as relevant challenges. The focused executable study compares a small CNN with a compact ImageNet-pretrained transfer model and measures inverse-frequency class weighting while controlling split, preprocessing, and augmentation. This is a benchmark research study, not a claim that a trained system improves diagnosis.",
        "",
        "## 3. Dataset and observed statistics",
        f"- Intended dataset: HAM10000, seven diagnoses (`{', '.join(CLASS_NAMES)}`).",
        f"- Requested local data root: `{summary.get('data_root')}`.",
        f"- Metadata CSV: `{summary.get('metadata_path') or 'NOT AVAILABLE'}`.",
        f"- Image files found: `{summary.get('image_files_found', 0)}`.",
        f"- Metadata rows: `{summary.get('metadata_rows') if summary.get('metadata_rows') is not None else 'NOT AVAILABLE'}`.",
        f"- {class_distribution_text}",
        "",
        "## 4. Preprocessing and augmentation",
        preprocessing_execution,
        "",
        "## 5. Experimental design and leakage controls",
        "Four experiments are preregistered in `05_experiments/experiment_matrix.csv`: a small CNN and MobileNetV3-Small transfer model, each with unweighted versus inverse-frequency weighted cross-entropy. The split is approximately 60/20/20 with StratifiedGroupKFold, grouping by patient ID when supplied, lesion ID, and exact-image SHA-256. Validation macro-F1 selects checkpoints; the test set is reserved for one final evaluation. Assertions prevent groups and exact-image hashes crossing splits.",
        "",
        "## 6. Models and training configuration",
        "- CNN baseline: compact four-block CNN trained from scratch.",
        "- Transfer-learning model: torchvision MobileNetV3-Small with ImageNet weights; weight download/load failure does not silently switch to random initialization.",
        f"- Configuration for this invocation: seed={args.seed}, image size={args.image_size}, batch size={args.batch_size}, maximum epochs={args.epochs}, early-stopping patience={args.patience}, workers={args.num_workers}; AdamW, learning rate 1e-3 for CNN and 3e-4 for MobileNetV3-Small, weight decay 1e-4, and ReduceLROnPlateau on validation macro-F1.",
        f"- Runtime at report generation: {runtime_text}.",
        f"- Training device recorded for completed run(s): {training_device}.",
        "",
        "## 7. Metrics and observed results",
        "Planned metrics include accuracy; macro and weighted precision, recall, and F1; per-class sensitivity/specificity; one-vs-rest ROC-AUC and PR-AUC when class representation permits; confusion matrices; parameter count; training duration; and forward-pass latency.",
        *observed_results,
        "",
        "## 8. Model comparison and best model",
        best_text,
        "",
        "## 9. Class-wise analysis and statistical analysis",
        classwise_text,
        "Repeated-seed comparisons are required before making inferential or statistical-significance claims.",
        "",
        "## 10. Execution log and validation",
        execution_text + " All supplied PDFs were text-extracted for inspection and SHA-256 deduplicated; the existing notebook was confirmed unexecuted. Software tests exercise split/leakage and metric functions using ephemeral toy fixtures only; those fixtures are not research data and their scores are not saved in result tables.",
        "",
        "## 11. Limitations",
        *limitation_lines,
        "- Any HAM10000-only results would not establish real-world generalization or clinical utility.",
        "",
        "## 12. Reproducibility and next steps",
        "Install the dependencies in `11_reproducibility/requirements.txt` (use a platform-matched PyTorch/torchvision wheel if needed), place the original HAM10000 metadata/images under `DATA/HAM10000/` (or pass `--data-root`), and resolve audit blockers. Exact commands from the repository root:",
        "",
        "```bash",
        "python -m pip install -r Skin_Cancer_Deep_Learning_Research/11_reproducibility/requirements.txt",
        "python Skin_Cancer_Deep_Learning_Research/11_reproducibility/run_pipeline.py --data-root DATA/HAM10000",
        "```",
        "",
        "With valid real data and compatible dependencies, the pipeline writes grouped splits/leakage checks, per-run histories/checkpoints, held-out-test metrics/reports/confusion matrices, master CSV/XLSX tables, and figures.",
        expected_run_text,
        "See `11_reproducibility/README.md` for data layout and optional flags.",
        "",
        "## 13. Future work",
        future_text,
        "",
    ]
    (output_dir / "10_reports" / "FINAL_RESEARCH_EXPERIMENT_REPORT.md").write_text("\n".join(final_report), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    ensure_output_structure(output_dir)
    preprocessing_config = write_preprocessing_config(output_dir / "04_preprocessing" / "preprocessing_config.json", data_root)
    write_json(output_dir / "11_reproducibility" / "configs" / "default_experiment_config.json", {
        "dataset_root": str(data_root),
        "seed": args.seed,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "num_workers": args.num_workers,
        "experiment_matrix": EXPERIMENTS,
    })

    frame, summary = audit_dataset(data_root, output_dir)
    dataset_version = _hash_dataset(frame, summary.get("metadata_path")) if frame is not None else "NOT AVAILABLE — no audited dataset"
    write_json(output_dir / "03_dataset" / "dataset_manifest.json", {
        **summary,
        "dataset_version_sha256": dataset_version,
    })

    blockers = list(summary.get("blocking_reasons", []))
    if not summary.get("valid_for_training"):
        _write_no_data_leakage_report(output_dir, summary)
    split_counts: pd.DataFrame | None = None
    splits: pd.DataFrame | None = None
    if frame is not None:
        try:
            splits = make_grouped_split(frame, seed=args.seed)
            split_counts = _save_splits(splits, data_root, output_dir, args.seed)
        except Exception as exc:
            blockers.append(f"Grouped split creation failed: {type(exc).__name__}: {exc}")
            summary["valid_for_training"] = False
            summary["blocking_reasons"] = blockers
            splits = None
            split_counts = None
            _write_no_data_leakage_report(output_dir, summary)
            traceback.print_exc()

    # Update the preprocessing record to distinguish config generation from data processing.
    preprocessing_config["split"]["seed"] = args.seed
    preprocessing_config["image_size"] = args.image_size
    write_json(output_dir / "04_preprocessing" / "preprocessing_config.json", preprocessing_config)
    _write_preprocessing_report(output_dir, summary, split_counts, preprocessing_config)

    dataset_status = "READY" if summary.get("valid_for_training") and splits is not None else "BLOCKED"
    matrix = write_experiment_matrix(
        output_dir / "05_experiments" / "experiment_matrix.csv",
        dataset_status,
        blockers,
    )
    for config in EXPERIMENTS:
        write_json(output_dir / "05_experiments" / "configs" / f"{config['experiment_id']}.json", {
            **config,
            "seed": args.seed,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "num_workers": args.num_workers,
            "split": preprocessing_config["split"],
            "augmentation": preprocessing_config["augmentation"],
        })

    result_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    if args.audit_only:
        status = "BLOCKED_AUDIT_ONLY" if not summary.get("valid_for_training") else "SKIPPED_AUDIT_ONLY"
        note = "Audit-only mode requested; no training was attempted."
        for config in EXPERIMENTS:
            result_rows.append(_blocked_result(config, status, note, args.seed, args.image_size, args.batch_size, args.epochs, dataset_version))
            log_rows.append(_experiment_log_row(config, status, dataset_version, args.seed, args.image_size, args.batch_size, args.epochs, note))
    elif not summary.get("valid_for_training") or splits is None:
        note = "Training was not started because audited metadata and images are missing or failed validation."
        if not importlib.util.find_spec("torch") or not importlib.util.find_spec("torchvision"):
            note += " PyTorch and/or torchvision is also missing; install compatible packages before a future run."
        for config in EXPERIMENTS:
            result_rows.append(_blocked_result(config, "BLOCKED_MISSING_OR_INVALID_DATA", note + " Blockers: " + "; ".join(blockers), args.seed, args.image_size, args.batch_size, args.epochs, dataset_version))
            log_rows.append(_experiment_log_row(config, "BLOCKED_MISSING_OR_INVALID_DATA", dataset_version, args.seed, args.image_size, args.batch_size, args.epochs, note + " Blockers: " + "; ".join(blockers)))
    elif not importlib.util.find_spec("torch") or not importlib.util.find_spec("torchvision"):
        note = "Dataset checks passed, but compatible PyTorch/torchvision are not installed. No training was attempted."
        for config in EXPERIMENTS:
            result_rows.append(_blocked_result(config, "BLOCKED_MISSING_DEPENDENCY", note, args.seed, args.image_size, args.batch_size, args.epochs, dataset_version))
            log_rows.append(_experiment_log_row(config, "BLOCKED_MISSING_DEPENDENCY", dataset_version, args.seed, args.image_size, args.batch_size, args.epochs, note))
    else:
        from train_models import train_experiment

        for config in EXPERIMENTS:
            try:
                result_row, partial_log = train_experiment(
                    config=config,
                    splits=splits,
                    output_dir=output_dir,
                    dataset_version=dataset_version,
                    seed=args.seed,
                    image_size=args.image_size,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    patience=args.patience,
                    num_workers=args.num_workers,
                )
                result_rows.append(result_row)
                log = _experiment_log_row(
                    config, "COMPLETED", dataset_version, args.seed, args.image_size, args.batch_size, args.epochs,
                    str(partial_log.get("notes", "")),
                    {**result_row, "metrics": {key: result_row.get(key) for key in result_row if key in {
                        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1", "macro_specificity", "macro_roc_auc_ovr", "weighted_roc_auc_ovr", "macro_pr_auc_ovr",
                    }}, "output_paths": {key: result_row.get(key) for key in ["checkpoint_path", "history_path", "classification_report_path", "confusion_matrix_path"]}},
                )
                log.update({key: value for key, value in partial_log.items() if key in {"epochs_run", "best_epoch", "training_seconds", "test_inference_seconds"}})
                log_rows.append(log)
            except Exception as exc:
                traceback.print_exc()
                if "requires compatible CPU/GPU builds" in str(exc):
                    status = "BLOCKED_MISSING_DEPENDENCY"
                elif "pretrained MobileNetV3-Small weights could not be loaded" in str(exc):
                    status = "FAILED_PRETRAINED_WEIGHTS_UNAVAILABLE"
                else:
                    status = "FAILED"
                note = f"{type(exc).__name__}: {exc}; full traceback was emitted to the run console."
                result_rows.append(_blocked_result(config, status, note, args.seed, args.image_size, args.batch_size, args.epochs, dataset_version))
                log_rows.append(_experiment_log_row(config, status, dataset_version, args.seed, args.image_size, args.batch_size, args.epochs, note))

    # Record every run as a structured artifact, even when it is honestly blocked.
    log_frame = pd.DataFrame(log_rows, columns=LOG_COLUMNS)
    for index, row in log_frame.iterrows():
        config = next((item for item in EXPERIMENTS if item["experiment_id"] == row["experiment_id"]), None)
        if config:
            matrix.loc[matrix["experiment_id"] == config["experiment_id"], "execution_status"] = row["status"]
    matrix.to_csv(output_dir / "05_experiments" / "experiment_matrix.csv", index=False)
    write_master_outputs(output_dir, result_rows, log_rows, matrix, summary)
    _write_missing_artifact_notes(output_dir)
    write_execution_status_figure(output_dir, log_rows)
    _create_metric_figures(output_dir, result_rows)
    _write_current_run_reports(output_dir, summary, result_rows, matrix, args)

    statistical_text = [
        "# Statistical analysis",
        "",
        "## Current run",
        "",
    ]
    completed = [row for row in result_rows if row.get("status") == "COMPLETED"]
    if completed:
        statistical_text.extend([
            f"Completed experiments: {len(completed)}.",
            "Each configuration currently uses a single prespecified random seed/split. Descriptive metrics are reported, but repeated-seed uncertainty and inferential tests are not estimated.",
            "",
            "No p-value or confidence interval is emitted for one run. Future paired comparisons should repeat the same grouped split protocol across several seeds and use a prespecified paired analysis; independent test leakage must be avoided.",
        ])
    else:
        statistical_text.extend([
            "No model was trained on study data; there are no empirical metric observations to summarize or compare.",
            "Means, standard deviations, confidence intervals, t-tests, ANOVA, p-values, effect sizes, and statistical decisions are therefore **NOT AVAILABLE / NOT JUSTIFIED**.",
            "Synthetic software fixtures are excluded from all research analyses.",
        ])
    statistical_text.append("")
    (output_dir / "09_statistics" / "STATISTICAL_ANALYSIS.md").write_text("\n".join(statistical_text), encoding="utf-8")

    _write_current_run_reports(output_dir, summary, result_rows, matrix, args)
    print(f"Dataset status: {'READY' if summary.get('valid_for_training') else 'BLOCKED'}")
    print(f"Dataset blockers: {summary.get('blocking_reasons', [])}")
    print(f"Completed experiments: {len(completed)} / {len(EXPERIMENTS)}")
    print(f"Master results: {output_dir / '07_results' / 'master_results.csv'}")
    if not completed:
        return 2 if not args.audit_only else 0
    if any(row.get("status") != "COMPLETED" for row in result_rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
