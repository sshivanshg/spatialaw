"""
Preprocessing utilities for WiAR CSI tensors.

Includes helpers to create fixed-length windows, denoise and normalize
them, and persist processed datasets to disk.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Union

import numpy as np


def window_csi(arr: np.ndarray, T: int = 256, stride: int = 64) -> np.ndarray:
    """
    Convert packet-level CSI array into overlapping windows.

    Parameters
    ----------
    arr:
        Array of shape ``(n_packets, n_subcarriers)``.
    T:
        Window length (number of packets/time steps).
    stride:
        Step size between consecutive windows.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_windows, n_subcarriers, T)``.
    """

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {arr.shape}")

    if T <= 0 or stride <= 0:
        raise ValueError("`T` and `stride` must be positive integers.")

    n_packets, n_subcarriers = arr.shape
    if n_packets < T:
        return np.empty((0, n_subcarriers, T))

    windows: List[np.ndarray] = []
    for start in range(0, n_packets - T + 1, stride):
        segment = arr[start : start + T]
        windows.append(segment.T)  # transpose to (subcarriers, T)

    if not windows:
        return np.empty((0, n_subcarriers, T))

    return np.stack(windows, axis=0)


def denoise_window(win: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply a simple moving-average filter along the time axis.

    Parameters
    ----------
    win:
        Window array shaped ``(n_subcarriers, T)``.
    kernel_size:
        Size of the 1D smoothing kernel. Must be odd.
    """

    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
    filtered = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"),
        axis=1,
        arr=win,
    )
    return filtered


def normalize_window(win: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-window z-score normalization across time for each subcarrier.

    Parameters
    ----------
    win:
        Window array shaped ``(n_subcarriers, T)``.
    eps:
        Numerical stability term.
    """

    mean = win.mean(axis=1, keepdims=True)
    std = win.std(axis=1, keepdims=True) + eps
    return (win - mean) / std


def save_windows(
    windows: np.ndarray,
    labels: Sequence[Union[int, float, dict]],
    out_dir: str | Path,
) -> Path:
    """
    Persist processed windows and associated labels.

    Parameters
    ----------
    windows:
        Array of shape ``(n_windows, n_subcarriers, T)``.
    labels:
        Sequence describing each window. Can be integers/floats or dictionaries
        (e.g., ``{"label": 1, "source": "foo.txt"}``).
    out_dir:
        Destination directory.

    Returns
    -------
    pathlib.Path
        Path to the generated ``labels.csv`` file.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, (window, label_info) in enumerate(zip(windows, labels)):
        fname = out_dir / f"window_{idx:06d}.npy"
        np.save(fname, window)

        if isinstance(label_info, dict):
            row = {"window_file": fname.name, **label_info}
        else:
            row = {"window_file": fname.name, "label": label_info}
        records.append(row)

    if not records:
        labels_path = out_dir / "labels.csv"
        labels_path.write_text("window_file,label\n", encoding="utf-8")
        return labels_path

    fieldnames = sorted({key for record in records for key in record.keys()})
    labels_path = out_dir / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return labels_path

