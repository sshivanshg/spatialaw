"""
PyTorch Dataset helpers for CSI window tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class AugmentationConfig:
    """Configurable knobs for CSI augmentations."""

    noise_std: float = 0.01
    crop_ratio: float = 0.9
    channel_drop_prob: float = 0.1


class CSIDataset(Dataset):
    """
    Dataset that loads CSI windows stored as .npy files.

    Each item yields a tuple ``(tensor, label)`` where the tensor shape is
    ``(1, n_subcarriers, T)``.
    """

    def __init__(
        self,
        paths_csv: str | Path,
        root_dir: str | Path | None = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        augment: bool = False,
        augment_config: AugmentationConfig | None = None,
        file_column: str = "window_file",
        label_column: str = "label",
    ) -> None:
        self.paths_csv = Path(paths_csv)
        if not self.paths_csv.exists():
            raise FileNotFoundError(self.paths_csv)

        self.df = pd.read_csv(self.paths_csv)
        if file_column not in self.df.columns:
            raise ValueError(f"Column `{file_column}` missing from {paths_csv}")
        if label_column not in self.df.columns:
            raise ValueError(f"Column `{label_column}` missing from {paths_csv}")

        self.file_column = file_column
        self.label_column = label_column
        self.root_dir = Path(root_dir) if root_dir else self.paths_csv.parent
        self.transform = transform
        self.augment = augment
        self.aug_cfg = augment_config or AugmentationConfig()
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        window_path = self._resolve_path(row[self.file_column])
        window = np.load(window_path).astype(np.float32)

        if window.ndim != 2:
            raise ValueError(f"Expected 2D window in {window_path}, got {window.shape}")

        if self.augment:
            window = self._apply_augmentations(window)

        tensor = torch.from_numpy(window).unsqueeze(0)  # (1, S, T)
        if self.transform is not None:
            tensor = self.transform(tensor)

        label = torch.tensor(int(row[self.label_column]), dtype=torch.long)
        return tensor, label

    def _resolve_path(self, rel_path: str) -> Path:
        path = Path(rel_path)
        if path.is_absolute():
            return path
        return self.root_dir / path

    def _apply_augmentations(self, window: np.ndarray) -> np.ndarray:
        if self.aug_cfg.noise_std > 0:
            window = self._add_noise(window, self.aug_cfg.noise_std)
        if 0 < self.aug_cfg.crop_ratio < 1:
            window = self._time_crop(window, self.aug_cfg.crop_ratio)
        if 0 < self.aug_cfg.channel_drop_prob < 1:
            window = self._channel_dropout(window, self.aug_cfg.channel_drop_prob)
        return window

    def _add_noise(self, window: np.ndarray, std: float) -> np.ndarray:
        noise = self.rng.normal(0.0, std, size=window.shape).astype(np.float32)
        return window + noise

    def _time_crop(self, window: np.ndarray, ratio: float) -> np.ndarray:
        _, T = window.shape
        crop_len = max(1, int(T * ratio))
        if crop_len >= T:
            return window
        start = self.rng.integers(0, T - crop_len + 1)
        cropped = window[:, start : start + crop_len]
        pad_width = T - crop_len
        if pad_width > 0:
            cropped = np.pad(
                cropped, pad_width=((0, 0), (0, pad_width)), mode="edge"
            )
        return cropped

    def _channel_dropout(self, window: np.ndarray, prob: float) -> np.ndarray:
        mask = self.rng.random(window.shape[0]) < prob
        if mask.any():
            window = window.copy()
            window[mask, :] = 0.0
        return window


def create_dataloaders(
    csv_train: str | Path,
    csv_val: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    **dataset_kwargs,
) -> Tuple[DataLoader, DataLoader]:
    """
    Instantiate train/validation dataloaders for CSI windows.
    """

    train_dataset = CSIDataset(csv_train, augment=True, **dataset_kwargs)
    val_dataset = CSIDataset(csv_val, augment=False, **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader

