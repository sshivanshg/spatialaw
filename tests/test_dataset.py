import csv
from pathlib import Path

import numpy as np
import torch

from src.train.dataset import CSIDataset, create_dataloaders, DEFAULT_SEED


def _create_sample_windows(tmp_path: Path, num_windows: int = 6) -> Path:
    windows_dir = tmp_path
    rows = []
    for idx in range(num_windows):
        arr = np.random.rand(30, 60).astype(np.float32)
        filename = windows_dir / f"window_{idx:06d}.npy"
        np.save(filename, arr)
        rows.append(
            {
                "window_file": filename.name,
                "label": idx % 2,
            }
        )

    csv_path = windows_dir / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["window_file", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def test_csi_dataset_shapes(tmp_path):
    csv_path = _create_sample_windows(tmp_path)
    dataset = CSIDataset(csv_path, seed=DEFAULT_SEED)

    tensor, label = dataset[0]
    assert tensor.shape == (1, 30, 60)
    assert tensor.dtype == torch.float32
    assert label.dtype == torch.long


def test_csi_dataset_augmentation_preserves_shape(tmp_path):
    csv_path = _create_sample_windows(tmp_path)
    dataset = CSIDataset(
        csv_path,
        augment=True,
        seed=DEFAULT_SEED,
    )

    tensor, _ = dataset[0]
    assert tensor.shape == (1, 30, 60)


def test_create_dataloaders_batch_shapes(tmp_path):
    csv_path = _create_sample_windows(tmp_path, num_windows=8)
    train_loader, val_loader = create_dataloaders(
        csv_path, csv_path, batch_size=4, shuffle=False, seed=DEFAULT_SEED
    )

    batch = next(iter(train_loader))
    inputs, labels = batch
    assert inputs.shape == (4, 1, 30, 60)
    assert labels.shape == (4,)

    val_batch = next(iter(val_loader))
    assert val_batch[0].shape[0] == 4

