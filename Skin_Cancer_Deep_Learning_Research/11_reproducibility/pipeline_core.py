"""Shared utilities for the Skin Cancer Detection research pipeline.

The module deliberately separates workspace/data auditing from model training.
No experiment is reported as completed unless it trained and was evaluated on
real, audited source images.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent
REPO_ROOT = OUTPUT_DIR.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "DATA" / "HAM10000"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42

EXPERIMENTS: list[dict[str, Any]] = [
    {
        "experiment_id": "EXP-001",
        "model": "SmallCNN",
        "pretrained": False,
        "imbalance_strategy": "unweighted_cross_entropy",
        "description": "From-scratch CNN reference; same train-only augmentation as paired run.",
    },
    {
        "experiment_id": "EXP-002",
        "model": "SmallCNN",
        "pretrained": False,
        "imbalance_strategy": "inverse_frequency_class_weighted_cross_entropy",
        "description": "Controlled imbalance experiment paired with EXP-001.",
    },
    {
        "experiment_id": "EXP-003",
        "model": "MobileNetV3-Small",
        "pretrained": True,
        "imbalance_strategy": "unweighted_cross_entropy",
        "description": "ImageNet transfer-learning reference; same split and augmentation.",
    },
    {
        "experiment_id": "EXP-004",
        "model": "MobileNetV3-Small",
        "pretrained": True,
        "imbalance_strategy": "inverse_frequency_class_weighted_cross_entropy",
        "description": "Controlled imbalance experiment paired with EXP-003.",
    },
]

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "macro_specificity",
    "macro_roc_auc_ovr",
    "weighted_roc_auc_ovr",
    "macro_pr_auc_ovr",
]

RESULT_COLUMNS = [
    "experiment_id",
    "status",
    "model",
    "imbalance_strategy",
    "dataset_version",
    "seed",
    "image_size",
    "batch_size",
    "epochs_requested",
    "epochs_run",
    "best_epoch",
    "best_validation_macro_f1",
    "training_seconds",
    "test_inference_seconds",
    "test_inference_ms_per_image",
    "parameters_total",
    "parameters_trainable",
    *METRIC_COLUMNS,
    "test_images",
    "checkpoint_path",
    "history_path",
    "classification_report_path",
    "confusion_matrix_path",
    "notes",
]

LOG_COLUMNS = [
    "timestamp_utc",
    "experiment_id",
    "status",
    "dataset_version",
    "code_version",
    "random_seed",
    "model",
    "pretrained",
    "imbalance_strategy",
    "split_policy",
    "image_size",
    "batch_size",
    "epochs_requested",
    "epochs_run",
    "best_epoch",
    "training_seconds",
    "test_inference_seconds",
    "metrics_json",
    "output_paths_json",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_version(repo_root: Path = REPO_ROOT) -> str:
    """Return commit identity and whether uncommitted source edits are present."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "NOT AVAILABLE"
    tree_state = "dirty working tree" if status.stdout.strip() else "clean working tree"
    return f"{completed.stdout.strip()} ({tree_state})"


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_csv_row(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    frame = pd.DataFrame([{key: row.get(key) for key in columns}], columns=columns)
    frame.to_csv(path, mode="a" if exists else "w", header=not exists, index=False)


def _check_row(key: str, value: Any, status: str, details: str = "") -> dict[str, str]:
    return {
        "record_type": "quality_check",
        "key": key,
        "value": "NOT AVAILABLE" if value is None else str(value),
        "status": status,
        "details": details,
    }


def _label_row(label: str, count: int, details: str) -> dict[str, str]:
    return {
        "record_type": "class_distribution",
        "key": label,
        "value": str(count),
        "status": "OBSERVED",
        "details": details,
    }


def _metadata_candidates(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    csv_files = sorted(data_root.rglob("*.csv"))
    exact = [p for p in csv_files if p.name.lower() == "ham10000_metadata.csv"]
    if exact:
        return exact
    candidates: list[Path] = []
    required = {"image_id", "dx"}
    for path in csv_files:
        try:
            columns = {str(c).strip().lower() for c in pd.read_csv(path, nrows=0).columns}
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        if required.issubset(columns):
            candidates.append(path)
    return candidates


def audit_dataset(data_root: Path, output_dir: Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Inspect metadata, image integrity, duplicate records, and image linkage.

    Raw files are opened read-only. The audit never rewrites or moves source data.
    A dataframe is returned only when a structurally usable, fully matched dataset
    is available. All discovered quality issues remain visible in the report.
    """
    data_root = Path(data_root).expanduser().resolve()
    report_dir = output_dir / "03_dataset"
    figures_dir = output_dir / "08_figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    frame: pd.DataFrame | None = None
    images = sorted(
        p for p in data_root.rglob("*")
        if data_root.exists() and p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    metadata_files = _metadata_candidates(data_root)
    rows.append(_check_row("data_root", data_root, "PRESENT" if data_root.exists() else "MISSING"))
    rows.append(_check_row("metadata_csv_count", len(metadata_files), "OBSERVED" if metadata_files else "MISSING"))
    rows.append(_check_row("image_file_count", len(images), "OBSERVED" if images else "MISSING"))

    summary: dict[str, Any] = {
        "data_root": str(data_root),
        "data_root_exists": data_root.exists(),
        "metadata_paths": [str(p) for p in metadata_files],
        "metadata_path": str(metadata_files[0]) if len(metadata_files) == 1 else None,
        "metadata_files_found": len(metadata_files),
        "image_files_found": len(images),
        "metadata_rows": None,
        "matched_rows": 0,
        "missing_image_rows": None,
        "unreferenced_image_files": None,
        "duplicate_metadata_rows": None,
        "duplicate_image_ids": None,
        "corrupted_images": None,
        "duplicate_image_content_files": None,
        "diagnosis_label_counts": {},
        "class_names": CLASS_NAMES.copy(),
        "group_column": None,
        "valid_for_training": False,
        "blocking_reasons": [],
    }

    if not data_root.exists():
        rows.append(_check_row("dataset_status", "HAM10000 assets were not found under the requested root", "BLOCKED"))
        summary["blocking_reasons"].append(f"Dataset root does not exist: {data_root}")
    elif not metadata_files:
        rows.append(_check_row("dataset_status", "No HAM10000 metadata CSV with image_id/dx columns was found", "BLOCKED"))
        summary["blocking_reasons"].append("HAM10000 metadata CSV is absent.")
    elif len(metadata_files) > 1:
        rows.append(_check_row("dataset_status", "More than one candidate metadata CSV was found; select one explicitly", "BLOCKED", "\n".join(map(str, metadata_files))))
        summary["blocking_reasons"].append("Multiple candidate metadata files were found; an unambiguous dataset root is required.")
    else:
        metadata_path = metadata_files[0]
        try:
            frame = pd.read_csv(metadata_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            rows.append(_check_row("metadata_read", f"{type(exc).__name__}: {exc}", "BLOCKED"))
            summary["blocking_reasons"].append("Metadata CSV could not be parsed.")
            frame = None

        if frame is not None:
            frame.columns = [str(c).strip().lower() for c in frame.columns]
            required = {"image_id", "dx"}
            missing_columns = sorted(required.difference(frame.columns))
            summary["metadata_rows"] = len(frame)
            summary["group_column"] = "patient_id" if "patient_id" in frame.columns else ("lesion_id" if "lesion_id" in frame.columns else "image_id")
            rows.append(_check_row("metadata_columns", ",".join(frame.columns), "OBSERVED"))
            rows.append(_check_row("required_columns_missing", ",".join(missing_columns) if missing_columns else "none", "BLOCKED" if missing_columns else "PASS"))

            if missing_columns:
                summary["blocking_reasons"].append(f"Required metadata columns are missing: {missing_columns}")
                frame = None
            elif frame.empty:
                rows.append(_check_row("metadata_rows", 0, "BLOCKED"))
                summary["blocking_reasons"].append("Metadata CSV contains no records.")
                frame = None
            else:
                frame["image_id"] = frame["image_id"].astype("string").str.strip()
                frame["dx"] = frame["dx"].astype("string").str.strip().str.lower()
                missing_image_id = int(frame["image_id"].isna().sum() + frame["image_id"].eq("").sum())
                missing_dx = int(frame["dx"].isna().sum() + frame["dx"].eq("").sum())
                duplicate_rows = int(frame.duplicated(keep="first").sum())
                duplicate_ids = int(frame["image_id"].duplicated(keep="first").sum())
                labels = set(frame["dx"].dropna().astype(str))
                unknown_labels = sorted(labels.difference(CLASS_NAMES))
                summary["duplicate_metadata_rows"] = duplicate_rows
                summary["duplicate_image_ids"] = duplicate_ids
                summary["diagnosis_label_counts"] = {
                    str(key): int(value) for key, value in frame["dx"].value_counts(dropna=False).items()
                }
                rows.extend([
                    _check_row("duplicate_metadata_rows", duplicate_rows, "WARNING" if duplicate_rows else "PASS"),
                    _check_row("duplicate_image_ids", duplicate_ids, "BLOCKED" if duplicate_ids else "PASS"),
                    _check_row("missing_image_id_values", missing_image_id, "BLOCKED" if missing_image_id else "PASS"),
                    _check_row("missing_diagnosis_values", missing_dx, "BLOCKED" if missing_dx else "PASS"),
                    _check_row("unknown_diagnosis_labels", ",".join(unknown_labels) if unknown_labels else "none", "BLOCKED" if unknown_labels else "PASS"),
                ])
                if duplicate_ids:
                    summary["blocking_reasons"].append(f"{duplicate_ids} duplicate image_id value(s) were found.")
                if missing_image_id or missing_dx:
                    summary["blocking_reasons"].append("One or more metadata rows have missing image IDs or diagnosis labels.")
                if unknown_labels:
                    summary["blocking_reasons"].append(f"Diagnosis labels outside the HAM10000 seven-class schema were found: {unknown_labels}")

                stem_to_paths: dict[str, list[Path]] = defaultdict(list)
                for path in images:
                    stem_to_paths[path.stem].append(path)
                ambiguous_stems = {stem for stem, paths in stem_to_paths.items() if len(paths) > 1}
                image_lookup = {
                    stem: paths[0]
                    for stem, paths in stem_to_paths.items()
                    if len(paths) == 1
                }
                # When duplicate stems are byte-identical, mapping to one physical copy is
                # deterministic; all copies are still reported and grouped against leakage.
                for stem in ambiguous_stems:
                    candidates = stem_to_paths[stem]
                    try:
                        digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates}
                    except OSError:
                        digests = set()
                    if len(digests) == 1 and candidates:
                        image_lookup[stem] = candidates[0]
                matched_paths = frame["image_id"].map(image_lookup)
                frame["image_path"] = matched_paths
                missing_image_rows = int(frame["image_path"].isna().sum())
                summary["matched_rows"] = int(frame["image_path"].notna().sum())
                summary["missing_image_rows"] = missing_image_rows
                referenced_ids = set(frame["image_id"].dropna().astype(str))
                unreferenced_images = int(sum(1 for image in images if image.stem not in referenced_ids))
                summary["unreferenced_image_files"] = unreferenced_images
                rows.extend([
                    _check_row("ambiguous_image_id_stems", len(ambiguous_stems), "WARNING" if ambiguous_stems else "PASS"),
                    _check_row("metadata_rows_without_image", missing_image_rows, "BLOCKED" if missing_image_rows else "PASS"),
                    _check_row("unreferenced_image_files", unreferenced_images, "WARNING" if unreferenced_images else "PASS"),
                ])
                if missing_image_rows:
                    summary["blocking_reasons"].append(f"{missing_image_rows} metadata row(s) do not have a matching image file.")

                valid_frame = frame[frame["image_path"].notna()].copy()
                corrupt_ids: list[str] = []
                hash_to_ids: dict[str, list[str]] = defaultdict(list)
                sha_values: list[str | None] = []
                for row in valid_frame.itertuples(index=False):
                    image_path = Path(row.image_path)
                    try:
                        with Image.open(image_path) as image:
                            image.verify()
                        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                        sha_values.append(digest)
                        hash_to_ids[digest].append(str(row.image_id))
                    except (OSError, UnidentifiedImageError, ValueError) as exc:
                        corrupt_ids.append(str(row.image_id))
                        sha_values.append(None)
                valid_frame["image_sha256"] = sha_values
                summary["corrupted_images"] = len(corrupt_ids)
                duplicate_content_files = sum(max(0, len(ids) - 1) for ids in hash_to_ids.values())
                summary["duplicate_image_content_files"] = duplicate_content_files
                duplicate_hash_labels: list[str] = []
                if duplicate_content_files:
                    hash_to_labels: dict[str, set[str]] = defaultdict(set)
                    for digest, dx in zip(valid_frame["image_sha256"], valid_frame["dx"]):
                        if digest:
                            hash_to_labels[str(digest)].add(str(dx))
                    duplicate_hash_labels = [digest for digest, labelset in hash_to_labels.items() if len(labelset) > 1]
                rows.extend([
                    _check_row("corrupted_or_unreadable_images", len(corrupt_ids), "BLOCKED" if corrupt_ids else "PASS", ", ".join(corrupt_ids[:25])),
                    _check_row("duplicate_image_content_files", duplicate_content_files, "WARNING" if duplicate_content_files else "PASS"),
                    _check_row("exact_duplicate_images_with_conflicting_labels", len(duplicate_hash_labels), "BLOCKED" if duplicate_hash_labels else "PASS"),
                ])
                if corrupt_ids:
                    summary["blocking_reasons"].append(f"{len(corrupt_ids)} image(s) are unreadable or corrupted.")
                if duplicate_hash_labels:
                    summary["blocking_reasons"].append("Byte-identical image content maps to conflicting diagnosis labels.")

                if "lesion_id" in valid_frame.columns:
                    lesion_labels = valid_frame.dropna(subset=["lesion_id"]).groupby("lesion_id")["dx"].nunique()
                    inconsistent_lesions = int((lesion_labels > 1).sum())
                    rows.append(_check_row("lesion_ids_with_multiple_diagnoses", inconsistent_lesions, "BLOCKED" if inconsistent_lesions else "PASS"))
                    if inconsistent_lesions:
                        summary["blocking_reasons"].append(f"{inconsistent_lesions} lesion_id group(s) have inconsistent diagnoses.")
                else:
                    rows.append(_check_row("lesion_id_column", "not supplied", "WARNING", "Image-level split fallback is possible, but lesion-level leakage cannot be excluded."))

                summary["diagnosis_label_counts"] = {
                    str(key): int(value) for key, value in valid_frame["dx"].value_counts().items()
                }
                missing_expected_classes = [
                    diagnosis for diagnosis in CLASS_NAMES
                    if not bool((valid_frame["dx"] == diagnosis).any())
                ]
                rows.append(_check_row(
                    "expected_ham10000_classes_missing",
                    ",".join(missing_expected_classes) if missing_expected_classes else "none",
                    "BLOCKED" if missing_expected_classes else "PASS",
                ))
                if missing_expected_classes:
                    summary["blocking_reasons"].append(
                        f"Expected HAM10000 diagnosis classes are absent: {missing_expected_classes}"
                    )
                for diagnosis in CLASS_NAMES:
                    count = int((valid_frame["dx"] == diagnosis).sum())
                    rows.append(_label_row(diagnosis, count, "Matched, readable metadata-image rows"))
                extra_classes = sorted(set(valid_frame["dx"].astype(str)).difference(CLASS_NAMES))
                for diagnosis in extra_classes:
                    rows.append(_label_row(diagnosis, int((valid_frame["dx"] == diagnosis).sum()), "Unexpected label; training is blocked"))

                blocking_checks = [
                    missing_columns,
                    missing_image_id,
                    missing_dx,
                    duplicate_ids,
                    unknown_labels,
                    missing_image_rows,
                    corrupt_ids,
                    duplicate_hash_labels,
                    missing_expected_classes,
                ]
                valid = not any(bool(value) for value in blocking_checks) and len(valid_frame) > 0
                summary["valid_for_training"] = bool(valid)
                if valid:
                    # Keep only validated records; do not alter raw source files.
                    frame = valid_frame.reset_index(drop=True)
                else:
                    frame = None

    if not summary["valid_for_training"] and not summary["blocking_reasons"]:
        summary["blocking_reasons"].append("Dataset audit did not yield a usable metadata-image table.")

    if summary["valid_for_training"]:
        rows.append(_check_row("dataset_status", "metadata and image files passed blocking checks", "PASS"))
    else:
        rows.append(_check_row("dataset_status", "not suitable for training until blockers are resolved", "BLOCKED", "; ".join(summary["blocking_reasons"])))

    report_csv = report_dir / "dataset_report.csv"
    pd.DataFrame(rows, columns=["record_type", "key", "value", "status", "details"]).to_csv(report_csv, index=False)
    summary["dataset_report_csv"] = str(report_csv)
    summary["audit_timestamp_utc"] = utc_now()

    md_lines = [
        "# Dataset audit report",
        "",
        f"- Audit time (UTC): `{summary['audit_timestamp_utc']}`",
        f"- Requested dataset root: `{data_root}`",
        f"- HAM10000 metadata CSV: `{summary['metadata_path'] or 'NOT AVAILABLE'}`",
        f"- Metadata rows: `{summary['metadata_rows'] if summary['metadata_rows'] is not None else 'NOT AVAILABLE'}`",
        f"- Candidate image files: `{summary['image_files_found']}`",
        f"- Metadata rows linked to a readable image: `{summary['matched_rows']}`",
        f"- Status: **{'READY FOR PREPROCESSING' if summary['valid_for_training'] else 'BLOCKED — NO EXPERIMENTAL TRAINING' }**",
        "",
        "## Checks",
        "",
        "| Check | Value | Status | Details |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        details = row["details"].replace("|", "\\|").replace("\n", "; ")
        md_lines.append(f"| {row['key']} | {row['value']} | {row['status']} | {details} |")
    md_lines.extend(["", "## Interpretation", ""])
    if summary["valid_for_training"]:
        md_lines.append("The discovered metadata and images passed blocking integrity checks. See `dataset_report.csv` for class counts, warnings, and image-level checks. Original files were read only.")
    else:
        md_lines.append("No class frequencies or dataset-level statistics are inferred when the metadata/images are absent. The reason(s) training is blocked are:")
        md_lines.extend([f"- {reason}" for reason in summary["blocking_reasons"]])
    md_lines.extend([
        "",
        "A bibliography CSV is not treated as dataset metadata. Image classes, lesion counts, corruption rates, and duplicate rates are reported only when calculated from the actual image/metadata files.",
        "",
    ])
    (report_dir / "dataset_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    # The chart is strictly an availability status figure when there is no dataset;
    # it is not a substitute for a missing class-distribution plot.
    if not summary["valid_for_training"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        present = [int(bool(metadata_files)), int(bool(images))]
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        bars = ax.bar(["Metadata CSV", "Image files"], present, color=["#4776b4", "#ef9f45"], width=0.58)
        ax.set_ylim(0, 1.3)
        ax.set_yticks([0, 1], labels=["Not found", "Found"])
        ax.set_title("Dataset asset availability in inspected workspace", pad=12, weight="bold")
        ax.grid(axis="y", alpha=0.22)
        for bar, found, count in zip(bars, present, [len(metadata_files), len(images)]):
            label = f"Found ({count})" if found else "Not found"
            ax.text(bar.get_x() + bar.get_width() / 2, 0.04 if not found else 1.04, label, ha="center", va="bottom", fontsize=10)
        fig.text(0.5, 0.01, "Availability only — no class distribution or model metric is implied.", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(figures_dir / "dataset_asset_availability.png", dpi=220, bbox_inches="tight")
        fig.savefig(figures_dir / "dataset_asset_availability.pdf", bbox_inches="tight")
        plt.close(fig)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = pd.Series(summary["diagnosis_label_counts"]).reindex(CLASS_NAMES, fill_value=0)
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        bars = ax.bar(counts.index, counts.values, color="#4776b4")
        ax.set_title("HAM10000 class distribution (audited metadata)", weight="bold")
        ax.set_xlabel("Diagnosis label")
        ax.set_ylabel("Matched image records")
        ax.grid(axis="y", alpha=0.22)
        ax.bar_label(bars, padding=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "class_distribution.png", dpi=300, bbox_inches="tight")
        fig.savefig(figures_dir / "class_distribution.pdf", bbox_inches="tight")
        plt.close(fig)

    return frame, summary


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first != root_second:
            # Lexical root selection makes the resulting grouping deterministic.
            low, high = sorted((root_first, root_second))
            self.parent[high] = low


def build_leakage_groups(frame: pd.DataFrame) -> np.ndarray:
    """Join records sharing patient, lesion, or exact image content."""
    dsu = _DisjointSet()
    keys_by_row: list[list[str]] = []
    group_columns = [column for column in ("patient_id", "lesion_id") if column in frame.columns]
    for row in frame.itertuples(index=False):
        record = row._asdict()
        keys: list[str] = []
        for column in group_columns:
            value = record.get(column)
            if value is not None and not pd.isna(value) and str(value).strip():
                keys.append(f"{column}:{str(value).strip()}")
        image_hash = record.get("image_sha256")
        if image_hash is not None and not pd.isna(image_hash) and str(image_hash).strip():
            keys.append(f"image_sha256:{str(image_hash).strip()}")
        if not keys:
            image_id = record.get("image_id")
            keys.append(f"image_id:{image_id}")
        for key in keys:
            dsu.find(key)
        for key in keys[1:]:
            dsu.union(keys[0], key)
        keys_by_row.append(keys)
    components = []
    for keys in keys_by_row:
        root = dsu.find(keys[0])
        components.append(hashlib.sha1(root.encode("utf-8")).hexdigest()[:16])
    return np.asarray(components, dtype=object)


def make_grouped_split(frame: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Create deterministic, approximately 60/20/20 stratified group splits."""
    if frame.empty:
        raise ValueError("Cannot split an empty dataset.")
    required = {"dx", "image_id", "image_path", "image_sha256"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Cannot split dataset; required validated columns are missing: {missing}")
    labels = frame["dx"].astype(str).to_numpy()
    groups = build_leakage_groups(frame)
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError("At least two diagnosis classes are required for classification.")
    groups_per_class = {
        class_name: int(pd.Series(groups[labels == class_name]).nunique())
        for class_name in classes
    }
    insufficient = {name: count for name, count in groups_per_class.items() if count < 5}
    if insufficient:
        raise ValueError(f"Fewer than five independent leakage groups for class(es): {insufficient}; 60/20/20 grouped split is not defensible.")

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(outer.split(np.zeros(len(frame)), labels, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_rel, val_rel = next(
        inner.split(np.zeros(len(trainval_idx)), labels[trainval_idx], groups[trainval_idx])
    )
    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel]
    split_values = np.full(len(frame), "", dtype=object)
    split_values[train_idx] = "train"
    split_values[val_idx] = "validation"
    split_values[test_idx] = "test"
    if np.any(split_values == ""):
        raise RuntimeError("Internal split error: one or more rows were not assigned.")

    result = frame.copy().reset_index(drop=True)
    result["split"] = split_values
    result["leakage_group"] = build_leakage_groups(result)
    group_sets = {name: set(result.loc[result["split"] == name, "leakage_group"]) for name in ("train", "validation", "test")}
    if group_sets["train"] & group_sets["validation"] or group_sets["train"] & group_sets["test"] or group_sets["validation"] & group_sets["test"]:
        raise RuntimeError("Data leakage check failed: a patient/lesion/exact-image group crosses splits.")
    hash_sets = {
        name: set(result.loc[result["split"] == name, "image_sha256"].dropna().astype(str))
        for name in ("train", "validation", "test")
    }
    if hash_sets["train"] & hash_sets["validation"] or hash_sets["train"] & hash_sets["test"] or hash_sets["validation"] & hash_sets["test"]:
        raise RuntimeError("Data leakage check failed: identical image bytes cross splits.")

    train_classes = set(result.loc[result["split"] == "train", "dx"].astype(str))
    if train_classes != set(classes):
        raise RuntimeError(f"Training split lacks diagnosis classes: {sorted(set(classes) - train_classes)}")
    return result


def split_quality_table(splits: pd.DataFrame) -> pd.DataFrame:
    counts = (
        splits.groupby(["split", "dx"], observed=False)
        .size()
        .rename("image_count")
        .reset_index()
    )
    return counts


def calculate_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_prob: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Calculate multiclass metrics and one-vs-rest class-level statistics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    labels = np.arange(len(class_names))
    if y_prob.shape != (len(y_true), len(class_names)):
        raise ValueError(f"Probability matrix has shape {y_prob.shape}; expected {(len(y_true), len(class_names))}.")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_rows: list[dict[str, Any]] = []
    specificities: list[float] = []
    for idx, name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        specificity = tn / (tn + fp) if tn + fp else None
        if specificity is not None:
            specificities.append(specificity)
        metrics_for_class = report[name]
        per_class_rows.append({
            "class": name,
            "support": int(metrics_for_class["support"]),
            "precision": float(metrics_for_class["precision"]),
            "recall_sensitivity": float(metrics_for_class["recall"]),
            "f1": float(metrics_for_class["f1-score"]),
            "specificity": specificity,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        })
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_specificity": float(np.mean(specificities)) if specificities else None,
    }
    reasons: dict[str, Any] = {}
    y_true_bin = label_binarize(y_true, classes=labels)
    class_support = y_true_bin.sum(axis=0)
    valid_auc = [idx for idx, count in enumerate(class_support) if 0 < count < len(y_true)]
    if len(valid_auc) == len(class_names):
        metrics["macro_roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, labels=labels, multi_class="ovr", average="macro"))
        metrics["weighted_roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, labels=labels, multi_class="ovr", average="weighted"))
        metrics["macro_pr_auc_ovr"] = float(average_precision_score(y_true_bin, y_prob, average="macro"))
    else:
        auc_values: list[float] = []
        pr_values: list[float] = []
        weights: list[int] = []
        for idx in valid_auc:
            auc_values.append(float(roc_auc_score(y_true_bin[:, idx], y_prob[:, idx])))
            pr_values.append(float(average_precision_score(y_true_bin[:, idx], y_prob[:, idx])))
            weights.append(int(class_support[idx]))
        metrics["macro_roc_auc_ovr"] = float(np.mean(auc_values)) if auc_values else None
        metrics["weighted_roc_auc_ovr"] = float(np.average(auc_values, weights=weights)) if auc_values else None
        metrics["macro_pr_auc_ovr"] = float(np.mean(pr_values)) if pr_values else None
        reasons["auc_unavailable_or_partial"] = (
            "Only classes with both positive and negative test examples were included; missing-class support: "
            + ", ".join(class_names[idx] for idx in range(len(class_names)) if idx not in valid_auc)
        )
    return metrics, pd.DataFrame(per_class_rows), cm, {"classification_report": report, **reasons}


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], png_path: Path, csv_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    pd.DataFrame(cm, index=class_names, columns=class_names).rename_axis("actual").to_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True)
    ax.set_xlabel("Predicted diagnosis")
    ax.set_ylabel("True diagnosis")
    ax.set_title(title, weight="bold")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_normalized_confusion_matrix(cm: np.ndarray, class_names: list[str], png_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    sns.heatmap(normalized, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True)
    ax.set_xlabel("Predicted diagnosis")
    ax.set_ylabel("True diagnosis")
    ax.set_title(title, weight="bold")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_preprocessing_config(path: Path, data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    config = {
        "dataset": "HAM10000",
        "data_root_default": str(data_root),
        "expected_metadata_filename": "HAM10000_metadata.csv",
        "label_column": "dx",
        "image_identifier_column": "image_id",
        "grouping_priority": ["patient_id", "lesion_id", "exact_image_sha256", "image_id"],
        "classes": CLASS_NAMES,
        "image_size": 160,
        "color_mode": "RGB",
        "resize": "PIL bilinear resize to a fixed square; aspect ratio is not preserved",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "source": "ImageNet constants for pretrained MobileNetV3; no statistics are estimated from validation/test",
        },
        "split": {
            "method": "StratifiedGroupKFold; outer 5-fold test holdout, inner 4-fold validation from the remaining data",
            "approximate_proportions": {"train": 0.60, "validation": 0.20, "test": 0.20},
            "seed": SEED,
            "selection_metric": "validation macro-F1",
            "test_policy": "Held-out test set is evaluated once after checkpoint/model selection; it is not used for tuning",
        },
        "augmentation": {
            "applies_to": "training split only, on the fly after split assignment",
            "operations": ["random horizontal flip p=0.5", "random vertical flip p=0.5", "random rotation within +/-15 degrees"],
            "validation_test": "deterministic resize, tensor conversion, ImageNet normalization only",
        },
        "raw_data_policy": "Read-only. No raw images or metadata are rewritten. Split manifest and model inputs are stored separately.",
        "duplicate_policy": "Exact byte duplicates and patient/lesion identifiers are joined into one leakage group; conflicting duplicate labels block training.",
    }
    write_json(path, config)
    return config


def ensure_output_structure(output_dir: Path = OUTPUT_DIR) -> None:
    directories = [
        "01_project_audit",
        "02_literature",
        "03_dataset",
        "04_preprocessing/processed_data",
        "05_experiments/training_histories",
        "06_models",
        "07_results/classification_reports",
        "07_results/confusion_matrices",
        "08_figures",
        "09_statistics",
        "10_reports",
        "11_reproducibility/configs",
        "11_reproducibility/tests",
    ]
    for relative in directories:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def write_experiment_matrix(path: Path, dataset_status: str, blockers: list[str]) -> pd.DataFrame:
    records = []
    for config in EXPERIMENTS:
        records.append({
            "experiment_id": config["experiment_id"],
            "model": config["model"],
            "pretrained": config["pretrained"],
            "imbalance_strategy": config["imbalance_strategy"],
            "image_size": 160,
            "batch_size": 16,
            "epochs_requested": 10,
            "early_stopping": "patience=3, validation macro-F1",
            "seed": SEED,
            "split_policy": "lesion/patient/exact-image grouped stratified 60/20/20",
            "augmentation": "train only: horizontal/vertical flip, +/-15 degree rotation",
            "optimizer": "AdamW",
            "learning_rate": 0.001 if config["model"] == "SmallCNN" else 0.0003,
            "loss": "CrossEntropyLoss" if config["imbalance_strategy"] == "unweighted_cross_entropy" else "inverse-frequency weighted CrossEntropyLoss computed on train split only",
            "dataset_status_at_plan": dataset_status,
            "execution_status": "BLOCKED_MISSING_DATA" if blockers else "PLANNED",
            "notes": config["description"] + (" Blocker(s): " + "; ".join(blockers) if blockers else ""),
        })
    frame = pd.DataFrame(records)
    frame.to_csv(path, index=False)
    return frame


def write_master_outputs(output_dir: Path, result_rows: list[dict[str, Any]], log_rows: list[dict[str, Any]], experiment_matrix: pd.DataFrame, dataset_summary: dict[str, Any]) -> None:
    results_dir = output_dir / "07_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame(result_rows)
    if results_frame.empty:
        results_frame = pd.DataFrame(columns=RESULT_COLUMNS)
    else:
        for column in RESULT_COLUMNS:
            if column not in results_frame.columns:
                results_frame[column] = None
        results_frame = results_frame[RESULT_COLUMNS]
    results_frame.to_csv(results_dir / "master_results.csv", index=False)
    results_frame.to_csv(output_dir / "master_results.csv", index=False)

    log_frame = pd.DataFrame(log_rows, columns=LOG_COLUMNS)
    log_frame.to_csv(output_dir / "05_experiments" / "experiment_log.csv", index=False)
    log_frame.to_csv(output_dir / "experiment_log.csv", index=False)

    # Model-comparison exports contain only actual executed result rows. Blocked
    # attempts remain in master_results.csv and experiment_log.csv with null metrics.
    completed = results_frame[results_frame.get("status", pd.Series(dtype=str)).eq("COMPLETED")] if not results_frame.empty else results_frame
    compare_columns = ["experiment_id", "model", "imbalance_strategy", *METRIC_COLUMNS, "training_seconds", "test_inference_ms_per_image", "status"]
    comparison = completed.reindex(columns=compare_columns)
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    try:
        with pd.ExcelWriter(results_dir / "master_results.xlsx", engine="openpyxl") as writer:
            results_frame.to_excel(writer, sheet_name="master_results", index=False)
            log_frame.to_excel(writer, sheet_name="experiment_log", index=False)
            experiment_matrix.to_excel(writer, sheet_name="experiment_matrix", index=False)
            pd.DataFrame([{
                "dataset_root": dataset_summary.get("data_root"),
                "metadata_path": dataset_summary.get("metadata_path"),
                "metadata_rows": dataset_summary.get("metadata_rows"),
                "image_files_found": dataset_summary.get("image_files_found"),
                "matched_rows": dataset_summary.get("matched_rows"),
                "valid_for_training": dataset_summary.get("valid_for_training"),
                "blocking_reasons": "; ".join(dataset_summary.get("blocking_reasons", [])),
            }]).to_excel(writer, sheet_name="dataset_status", index=False)
        shutil.copy2(results_dir / "master_results.xlsx", output_dir / "master_results.xlsx")
    except ImportError as exc:
        raise RuntimeError("Excel export requires openpyxl; install the reproducibility requirements.") from exc


def write_execution_status_figure(output_dir: Path, log_rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    statuses = Counter(row.get("status", "UNKNOWN") for row in log_rows if row.get("experiment_id", "").startswith("EXP-"))
    if not statuses:
        statuses = Counter({"NO MODEL RUNS": 1})
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = list(statuses)
    values = [statuses[label] for label in labels]
    colors = ["#c44e52" if "BLOCKED" in label or "FAILED" in label else "#4c72b0" for label in labels]
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, padding=4)
    ax.set_ylabel("Experiment count")
    ax.set_title("Experiment execution status (not a model-performance plot)", weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(0, max(values, default=1) + 1.2)
    fig.text(0.5, 0.01, "No clinical metrics are inferred from blocked or software-test runs.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    target = output_dir / "08_figures" / "experiment_execution_status.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=240, bbox_inches="tight")
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
