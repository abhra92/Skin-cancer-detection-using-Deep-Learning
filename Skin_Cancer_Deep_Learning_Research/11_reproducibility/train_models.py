"""PyTorch training/evaluation code for the pre-registered experiment matrix.

This module is only invoked after the raw HAM10000 metadata and images pass the
read-only dataset audit. No synthetic data is used by this training code.
"""
from __future__ import annotations

import copy
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from pipeline_core import (
    CLASS_NAMES,
    EXPERIMENTS,
    RESULT_COLUMNS,
    calculate_metrics,
    save_confusion_matrix,
    save_normalized_confusion_matrix,
    write_json,
)


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Model training requires compatible CPU/GPU builds of torch and torchvision. "
            "Install the dependencies listed in 11_reproducibility/requirements.txt; "
            "for CUDA, install the PyTorch wheel matching the target CUDA runtime."
        ) from exc
    return torch, nn, DataLoader, Dataset, models, transforms


class _SmallCNNFactory:
    """Factory; dynamically creates a torch Module after lazy import."""

    @staticmethod
    def build(nn, class_count: int):
        class SmallCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(192),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=0.25),
                    nn.Linear(192, class_count),
                )

            def forward(self, inputs):
                return self.classifier(self.features(inputs))

        return SmallCNN()


def _build_model(config: dict[str, Any], models, nn, class_count: int):
    if config["model"] == "SmallCNN":
        return _SmallCNNFactory.build(nn, class_count)
    if config["model"] == "MobileNetV3-Small":
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if config["pretrained"] else None
            model = models.mobilenet_v3_small(weights=weights)
        except Exception as exc:
            if config["pretrained"]:
                raise RuntimeError(
                    "ImageNet pretrained MobileNetV3-Small weights could not be loaded. "
                    "The transfer-learning experiment was not silently changed to random initialization. "
                    "Provide cached weights or a working download connection, then rerun."
                ) from exc
            raise
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, class_count)
        return model
    raise ValueError(f"Unsupported model configured: {config['model']}")


def _set_seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_transform(transforms, image_size: int, training: bool):
    common = [transforms.Resize((image_size, image_size))]
    if training:
        common.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ])
    common.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return transforms.Compose(common)


def _make_dataset_class(torch, Dataset):
    class HAMLesionDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, transform, class_to_index: dict[str, int]):
            self.frame = frame.reset_index(drop=True)
            self.transform = transform
            self.class_to_index = class_to_index

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            row = self.frame.iloc[index]
            with Image.open(row["image_path"]) as source_image:
                image = source_image.convert("RGB")
            image = self.transform(image)
            label = self.class_to_index[str(row["dx"])]
            return image, label, str(row["image_id"])

    return HAMLesionDataset


def _run_epoch(torch, model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    seen = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels, _image_ids in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            count = int(images.shape[0])
            running_loss += float(loss.detach().item()) * count
            predictions = logits.detach().argmax(dim=1)
            labels_all.extend(labels.detach().cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
            seen += count
    if seen == 0:
        raise RuntimeError("A data loader yielded no image batches.")
    return {
        "loss": running_loss / seen,
        "accuracy": float(accuracy_score(labels_all, predictions_all)),
        "macro_f1": float(f1_score(labels_all, predictions_all, labels=np.arange(len(CLASS_NAMES)), average="macro", zero_division=0)),
    }


def _evaluate_test(torch, model, loader, criterion, device, class_to_index):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    image_ids: list[str] = []
    forward_seconds = 0.0
    with torch.no_grad():
        for images, labels, ids in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - started
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            count = int(images.shape[0])
            total_loss += float(loss.item()) * count
            total += count
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_prob.extend(probabilities.cpu().tolist())
            image_ids.extend(ids)
    if total == 0:
        raise RuntimeError("The held-out test data loader yielded no images.")
    metrics, per_class, cm, report = calculate_metrics(
        np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob), CLASS_NAMES
    )
    return {
        "loss": total_loss / total,
        "metrics": metrics,
        "per_class": per_class,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "image_ids": image_ids,
        "forward_seconds": forward_seconds,
        "image_count": total,
    }


def train_experiment(
    config: dict[str, Any],
    splits: pd.DataFrame,
    output_dir: Path,
    dataset_version: str,
    seed: int = 42,
    image_size: int = 160,
    batch_size: int = 16,
    epochs: int = 10,
    patience: int = 3,
    num_workers: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train one configured experiment and evaluate once on the test split."""
    torch, nn, DataLoader, Dataset, models, transforms = _require_torch()
    _set_seed(torch, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    id_to_class = {index: name for name, index in class_to_index.items()}

    train_frame = splits.loc[splits["split"] == "train"].copy()
    val_frame = splits.loc[splits["split"] == "validation"].copy()
    test_frame = splits.loc[splits["split"] == "test"].copy()
    if min(len(train_frame), len(val_frame), len(test_frame)) == 0:
        raise ValueError("Train, validation, and test splits must all contain records.")
    unknown_labels = sorted(set(splits["dx"].astype(str)).difference(CLASS_NAMES))
    if unknown_labels:
        raise ValueError(f"Unknown diagnosis labels in split manifest: {unknown_labels}")

    DatasetClass = _make_dataset_class(torch, Dataset)
    train_dataset = DatasetClass(train_frame, _make_transform(transforms, image_size, True), class_to_index)
    val_dataset = DatasetClass(val_frame, _make_transform(transforms, image_size, False), class_to_index)
    test_dataset = DatasetClass(test_frame, _make_transform(transforms, image_size, False), class_to_index)
    generator = torch.Generator().manual_seed(seed)
    use_cuda = device.type == "cuda"
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = _build_model(config, models, nn, len(CLASS_NAMES)).to(device)
    parameters_total = sum(parameter.numel() for parameter in model.parameters())
    parameters_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    class_weight_tensor = None
    if config["imbalance_strategy"] == "inverse_frequency_class_weighted_cross_entropy":
        train_labels = np.asarray([class_to_index[str(label)] for label in train_frame["dx"]])
        counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
        if np.any(counts == 0):
            absent = [CLASS_NAMES[index] for index, count in enumerate(counts) if count == 0]
            raise ValueError(f"Cannot compute training-only class weights; no train examples for: {absent}")
        weights = len(train_labels) / (len(CLASS_NAMES) * counts.astype(float))
        class_weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    learning_rate = 1e-3 if config["model"] == "SmallCNN" else 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    models_dir = output_dir / "06_models"
    histories_dir = output_dir / "05_experiments" / "training_histories"
    report_dir = output_dir / "07_results" / "classification_reports"
    confusion_dir = output_dir / "07_results" / "confusion_matrices"
    for directory in (models_dir, histories_dir, report_dir, confusion_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = models_dir / f"{config['experiment_id']}_{config['model'].lower().replace('-', '_')}_best.pth"
    history_path = histories_dir / f"{config['experiment_id']}_training_history.csv"
    history: list[dict[str, Any]] = []
    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    non_improving_epochs = 0
    training_started = time.perf_counter()
    epochs_run = 0
    print(f"[{config['experiment_id']}] model={config['model']} loss={config['imbalance_strategy']} device={device}", flush=True)

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = _run_epoch(torch, model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(torch, model, val_loader, criterion, device)
        scheduler.step(val_metrics["macro_f1"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "experiment_id": config["experiment_id"],
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        epochs_run = epoch
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"[{config['experiment_id']}] epoch {epoch}/{epochs} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} lr={current_lr:.2e} "
            f"seconds={epoch_seconds:.1f}",
            flush=True,
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({
                "model_state_dict": best_state,
                "class_names": CLASS_NAMES,
                "experiment_id": config["experiment_id"],
                "best_validation_macro_f1": best_val_f1,
                "best_epoch": best_epoch,
                "seed": seed,
                "image_size": image_size,
                "dataset_version": dataset_version,
            }, checkpoint_path)
            non_improving_epochs = 0
        else:
            non_improving_epochs += 1
        if non_improving_epochs >= patience:
            print(f"[{config['experiment_id']}] early stopping after {epoch} epochs", flush=True)
            break

    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("Training ended without producing a validation-selected checkpoint.")
    model.load_state_dict(best_state)
    test_result = _evaluate_test(torch, model, test_loader, criterion, device, class_to_index)
    metrics = test_result["metrics"]
    confusion_path = confusion_dir / f"{config['experiment_id']}_test_confusion_matrix.png"
    confusion_csv = confusion_dir / f"{config['experiment_id']}_test_confusion_matrix.csv"
    save_confusion_matrix(
        test_result["confusion_matrix"],
        CLASS_NAMES,
        confusion_path,
        confusion_csv,
        f"{config['experiment_id']} — held-out test confusion matrix",
    )
    save_normalized_confusion_matrix(
        test_result["confusion_matrix"],
        CLASS_NAMES,
        confusion_dir / f"{config['experiment_id']}_test_confusion_matrix_normalized.png",
        f"{config['experiment_id']} — row-normalized held-out test confusion matrix",
    )

    report_path = report_dir / f"{config['experiment_id']}_per_class_metrics.csv"
    test_result["per_class"].to_csv(report_path, index=False)
    report_json_path = report_dir / f"{config['experiment_id']}_classification_report.json"
    write_json(report_json_path, {
        "experiment_id": config["experiment_id"],
        "averaging": "macro = unweighted mean over the seven fixed diagnosis labels; weighted = support-weighted mean; AUC/PR-AUC = one-vs-rest where defined",
        "test_metrics": metrics,
        "per_class": test_result["per_class"].to_dict(orient="records"),
        "sklearn_report": test_result["classification_report"],
        "test_loss": test_result["loss"],
    })
    predictions_path = report_dir / f"{config['experiment_id']}_test_predictions.csv"
    predictions = pd.DataFrame({
        "image_id": test_result["image_ids"],
        "true_label": [id_to_class[int(value)] for value in test_result["y_true"]],
        "predicted_label": [id_to_class[int(value)] for value in test_result["y_pred"]],
    })
    probabilities = pd.DataFrame(test_result["y_prob"], columns=[f"probability_{label}" for label in CLASS_NAMES])
    pd.concat([predictions, probabilities], axis=1).to_csv(predictions_path, index=False)

    result_row = {
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED",
        "model": config["model"],
        "imbalance_strategy": config["imbalance_strategy"],
        "dataset_version": dataset_version,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "training_seconds": training_seconds,
        "test_inference_seconds": test_result["forward_seconds"],
        "test_inference_ms_per_image": 1000.0 * test_result["forward_seconds"] / test_result["image_count"],
        "parameters_total": parameters_total,
        "parameters_trainable": parameters_trainable,
        **metrics,
        "test_images": test_result["image_count"],
        "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
        "history_path": str(history_path.relative_to(output_dir)),
        "classification_report_path": str(report_path.relative_to(output_dir)),
        "confusion_matrix_path": str(confusion_path.relative_to(output_dir)),
        "notes": f"device={device}; test metrics were calculated once from the held-out test split after best-checkpoint selection by validation macro-F1.",
    }
    log_row = {
        "status": "COMPLETED",
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "training_seconds": training_seconds,
        "test_inference_seconds": test_result["forward_seconds"],
        "metrics_json": metrics,
        "output_paths_json": {
            "checkpoint": result_row["checkpoint_path"],
            "history": result_row["history_path"],
            "classification_report": result_row["classification_report_path"],
            "confusion_matrix": result_row["confusion_matrix_path"],
            "predictions": str(predictions_path.relative_to(output_dir)),
        },
        "notes": f"device={device}; parameters_total={parameters_total}; parameters_trainable={parameters_trainable}",
    }
    return result_row, log_row
