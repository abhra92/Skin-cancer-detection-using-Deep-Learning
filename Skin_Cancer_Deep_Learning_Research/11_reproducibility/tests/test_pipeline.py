"""Non-research software tests. Fixtures are synthetic and never enter result exports."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pipeline_core import CLASS_NAMES, audit_dataset, calculate_metrics, ensure_output_structure, make_grouped_split
from run_pipeline import _write_current_run_reports


class PipelineTests(unittest.TestCase):
    def _toy_frame(self) -> pd.DataFrame:
        records = []
        for class_index, diagnosis in enumerate(CLASS_NAMES):
            for group_index in range(10):
                image_id = f"toy_{class_index}_{group_index}"
                records.append({
                    "image_id": image_id,
                    "dx": diagnosis,
                    "lesion_id": f"lesion_{class_index}_{group_index}",
                    "image_path": f"/synthetic-not-a-dataset/{image_id}.jpg",
                    "image_sha256": f"hash_{class_index}_{group_index}",
                })
        return pd.DataFrame(records)

    def test_grouped_split_is_complete_disjoint_and_reproducible(self):
        frame = self._toy_frame()
        first = make_grouped_split(frame, seed=123)
        second = make_grouped_split(frame, seed=123)
        self.assertEqual(first["split"].tolist(), second["split"].tolist())
        self.assertEqual(set(first["split"]), {"train", "validation", "test"})
        group_sets = {
            split: set(first.loc[first["split"] == split, "leakage_group"])
            for split in ("train", "validation", "test")
        }
        self.assertFalse(group_sets["train"] & group_sets["validation"])
        self.assertFalse(group_sets["train"] & group_sets["test"])
        self.assertFalse(group_sets["validation"] & group_sets["test"])
        hash_sets = {
            split: set(first.loc[first["split"] == split, "image_sha256"])
            for split in ("train", "validation", "test")
        }
        self.assertFalse(hash_sets["train"] & hash_sets["validation"])
        self.assertFalse(hash_sets["train"] & hash_sets["test"])
        self.assertFalse(hash_sets["validation"] & hash_sets["test"])
        self.assertEqual(len(first), len(frame))

    def test_exact_duplicate_image_content_joins_records_into_one_group(self):
        frame = self._toy_frame()
        frame.loc[1, "image_sha256"] = frame.loc[0, "image_sha256"]
        split = make_grouped_split(frame, seed=7)
        first_group = split.loc[0, "leakage_group"]
        duplicate_group = split.loc[1, "leakage_group"]
        self.assertEqual(first_group, duplicate_group)
        self.assertEqual(split.loc[0, "split"], split.loc[1, "split"])

    def test_temporary_image_fixture_audits_and_splits_without_becoming_research_data(self):
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="skin-cancer-synthetic-audit-") as temp:
            root = Path(temp) / "toy_ham"
            images_dir = root / "images"
            images_dir.mkdir(parents=True)
            records = []
            for class_index, diagnosis in enumerate(CLASS_NAMES):
                for group_index in range(6):
                    image_id = f"toy_{class_index}_{group_index}"
                    # Deterministic, visibly varied test-only image; no clinical content.
                    pixels = [
                        ((class_index * 31 + group_index * 7 + x) % 255,
                         (class_index * 13 + group_index * 19 + y) % 255,
                         (class_index * 47 + group_index * 11 + x + y) % 255)
                        for y in range(12) for x in range(12)
                    ]
                    image = Image.new("RGB", (12, 12))
                    image.putdata(pixels)
                    image.save(images_dir / f"{image_id}.png")
                    records.append({"image_id": image_id, "dx": diagnosis, "lesion_id": f"lesion_{class_index}_{group_index}"})
            pd.DataFrame(records).to_csv(root / "HAM10000_metadata.csv", index=False)
            output = Path(temp) / "reports"
            frame, summary = audit_dataset(root, output)
            self.assertTrue(summary["valid_for_training"])
            self.assertEqual(len(frame), 42)
            self.assertEqual(summary["corrupted_images"], 0)
            self.assertEqual(summary["missing_image_rows"], 0)
            self.assertTrue((output / "08_figures" / "class_distribution.png").exists())
            splits = make_grouped_split(frame, seed=99)
            self.assertEqual(set(splits["split"]), {"train", "validation", "test"})
            self.assertEqual(len(splits), len(records))

    def test_multiclass_metrics_on_known_fixture(self):
        y_true = np.asarray([0, 0, 1, 1, 2, 2])
        y_pred = np.asarray([0, 1, 1, 1, 2, 0])
        probabilities = np.eye(3)[y_pred] * 0.8 + 0.2 / 3
        metrics, per_class, matrix, report = calculate_metrics(
            y_true,
            y_pred,
            probabilities,
            ["class_a", "class_b", "class_c"],
        )
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(len(per_class), 3)
        self.assertIn("classification_report", report)
        self.assertIn("macro_f1", metrics)
        self.assertIn("weighted_f1", metrics)
        self.assertIn("macro_roc_auc_ovr", metrics)
        self.assertTrue(np.isfinite(metrics["macro_f1"]))

    def test_temporary_completed_run_populates_report_without_touching_research_results(self):
        with tempfile.TemporaryDirectory(prefix="skin-cancer-report-generator-test-") as temp:
            output = Path(temp) / "temporary_output"
            ensure_output_structure(output)
            result = {
                "experiment_id": "TEST-ONLY",
                "status": "COMPLETED",
                "model": "TemporaryTestModel",
                "accuracy": 0.75,
                "macro_f1": 0.70,
                "weighted_f1": 0.72,
                "macro_roc_auc_ovr": None,
                "macro_pr_auc_ovr": None,
                "training_seconds": 1.2,
                "notes": "device=CPU test fixture; not a research experiment",
            }
            matrix = pd.DataFrame([{
                "experiment_id": "TEST-ONLY",
                "model": "TemporaryTestModel",
                "imbalance_strategy": "temporary test fixture",
                "execution_status": "COMPLETED",
            }])
            summary = {
                "valid_for_training": True,
                "data_root": "temporary test-only data",
                "metadata_path": "temporary test-only metadata",
                "image_files_found": 7,
                "metadata_rows": 7,
                "diagnosis_label_counts": {label: 1 for label in CLASS_NAMES},
            }
            args = Namespace(
                seed=7,
                image_size=32,
                batch_size=2,
                epochs=1,
                patience=1,
                num_workers=0,
                audit_only=False,
            )
            _write_current_run_reports(output, summary, [result], matrix, args)
            report = (output / "10_reports" / "FINAL_RESEARCH_EXPERIMENT_REPORT.md").read_text()
            self.assertIn("1 of 1 planned experiment(s) completed", report)
            self.assertIn("Highest macro-F1 among completed experiments: **TEST-ONLY", report)
            self.assertIn("CPU test fixture", report)
            self.assertFalse((output / "07_results" / "master_results.csv").exists())

    def test_absent_dataset_is_reported_as_blocked_without_invented_counts(self):
        with tempfile.TemporaryDirectory(prefix="skin-cancer-audit-test-") as temp:
            root = Path(temp) / "absent_ham10000"
            out = Path(temp) / "output"
            frame, summary = audit_dataset(root, out)
            self.assertIsNone(frame)
            self.assertFalse(summary["valid_for_training"])
            self.assertEqual(summary["image_files_found"], 0)
            self.assertIsNone(summary["metadata_rows"])
            report = pd.read_csv(out / "03_dataset" / "dataset_report.csv")
            self.assertIn("BLOCKED", set(report.loc[report["key"] == "dataset_status", "status"]))
            self.assertTrue((out / "08_figures" / "dataset_asset_availability.png").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
