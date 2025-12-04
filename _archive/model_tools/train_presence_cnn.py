#!/usr/bin/env python3
"""
Train a lightweight CNN to detect human presence directly from CSI windows.

Inputs:
    - data/processed/windows_binary/train.csv
    - data/processed/windows_binary/val.csv
    - data/processed/windows_binary/test.csv

Each CSV must contain a `window_file` column (relative to the windows
directory) and a `presence_label` column (0 = no person, 1 = person).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
ARCHIVE_DIR = ROOT_DIR / "_archive"
for path in (ARCHIVE_DIR, ROOT_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from src.train.dataset import CSIDataset, create_dataloaders  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleCSICNN(nn.Module):
    """
    Shallow 2D CNN tailored for CSI heatmaps (subcarriers × time).
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # (30, 256) -> (30, 128)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # (30, 128) -> (30, 64)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # (30, 64) -> (~15, 32)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.encoder(x)
        logits = self.classifier(logits)
        return logits.squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows-dir",
        default="data/processed/windows",
        help="Directory containing window_*.npy tensors.",
    )
    parser.add_argument(
        "--splits-dir",
        default="data/processed/windows_binary",
        help="Directory containing train/val/test CSV splits.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization coefficient.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience (in epochs).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device identifier.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability in the classifier head.",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the best model to ONNX format.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to store checkpoints and metrics.",
    )
    return parser.parse_args()


def compute_metrics(probas: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = (probas >= 0.5).astype(int)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    roc_auc = float("nan")
    if len(np.unique(labels)) > 1:
        roc_auc = roc_auc_score(labels, probas)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    optimizer=None,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    epoch_loss = 0.0
    all_probas: list[float] = []
    all_labels: list[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.float().to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)

        probas = torch.sigmoid(logits).detach().cpu().numpy()
        all_probas.extend(probas.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss /= len(loader.dataset)
    metrics = compute_metrics(np.array(all_probas), np.array(all_labels))
    return epoch_loss, metrics


def evaluate_test_set(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    probas_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            logits = model(inputs)
            probas = torch.sigmoid(logits).cpu().numpy()
            probas_list.extend(probas.tolist())
            labels_list.extend(labels.cpu().numpy().tolist())
    return compute_metrics(np.array(probas_list), np.array(labels_list))


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    windows_dir = Path(args.windows_dir)
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_train = splits_dir / "train.csv"
    csv_val = splits_dir / "val.csv"
    csv_test = splits_dir / "test.csv"

    for path in (csv_train, csv_val, csv_test):
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")

    train_loader, val_loader = create_dataloaders(
        csv_train=csv_train,
        csv_val=csv_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        file_column="window_file",
        label_column="presence_label",
        root_dir=windows_dir,
        seed=args.seed,
    )
    test_dataset = CSIDataset(
        csv_test,
        root_dir=windows_dir,
        file_column="window_file",
        label_column="presence_label",
        augment=False,
        seed=args.seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SimpleCSICNN(dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    history = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        scheduler.step(val_loss)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "duration_sec": time.time() - start,
        }
        history.append(epoch_summary)
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics['f1']:.3f} val_roc_auc={val_metrics['roc_auc']:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model.")

    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "presence_cnn.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(args),
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )
    print(f"✓ Saved best checkpoint from epoch {best_epoch} to {checkpoint_path}")

    if args.export_onnx:
        onnx_path = output_dir / "presence_cnn.onnx"
        dummy_input = torch.randn(1, 1, 30, 256, device=device)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["csi_window"],
            output_names=["presence_logit"],
            opset_version=12,
        )
        print(f"✓ Exported ONNX model to {onnx_path}")

    test_metrics = evaluate_test_set(model, test_loader, device)
    print(
        "Test metrics -> "
        + ", ".join(f"{k}: {v:.3f}" for k, v in test_metrics.items())
    )

    metrics_payload = {
        "best_epoch": best_epoch,
        "history": history,
        "test_metrics": test_metrics,
    }
    with (output_dir / "presence_cnn_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

