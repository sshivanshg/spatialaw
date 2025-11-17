"""Helpers to find and load WiAR CSI recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
from scipy.io import loadmat

try:
    from .dat_loader import load_dat_file
except ImportError:
    load_dat_file = None

SUPPORTED_EXTENSIONS = (".npy", ".mat", ".csv", ".txt", ".dat")


def load_csi_file(path: str | Path) -> np.ndarray:
    """
    Load a CSI recording as a 2D amplitude array.

    Parameters
    ----------
    path:
        Path to a raw CSI file. Supported formats: ``.npy``, ``.mat``,
        ``.csv`` and plain ``.txt`` tables. Complex arrays are converted
        to magnitudes.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_packets, n_subcarriers)`` with real-valued amplitudes.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file type is unsupported or data cannot be coerced to 2D.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".mat":
        data = _load_mat(path)
    elif suffix == ".csv":
        data = np.loadtxt(path, delimiter=",")
    elif suffix == ".txt":
        data = np.loadtxt(path)
    elif suffix == ".dat":
        if load_dat_file is None:
            raise ImportError(
                "Intel 5300 .dat file support requires dat_loader module. "
                "Ensure src/preprocess/dat_loader.py exists."
            )
        data = load_dat_file(path)
    else:
        raise ValueError(f"Unsupported CSI format: {path.suffix}")

    return _prepare_array(data, path)


def list_recordings(root: str | Path) -> List[Path]:
    """
    Recursively collect CSI recordings underneath ``root``.

    Parameters
    ----------
    root:
        Directory to search.

    Returns
    -------
    list[pathlib.Path]
        Sorted list of file paths that match supported extensions.
    """

    root = Path(root)
    if not root.exists():
        return []

    files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    files = sorted(set(files))
    return files


def _load_mat(path: Path) -> np.ndarray:
    """Return the first numeric array found in a MATLAB file."""

    mat = loadmat(path)
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray):
            return value
    raise ValueError(f"No ndarray found in MATLAB file: {path}")


def _prepare_array(data: np.ndarray, path: Path) -> np.ndarray:
    """Convert arbitrary ndarray to 2D amplitude representation."""

    arr = np.asarray(data)
    if arr.ndim == 0:
        raise ValueError(f"Loaded scalar from {path}; expected array.")

    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    else:
        arr = arr.astype(np.float64, copy=False)

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim > 2:
        first_dim = arr.shape[0]
        arr = arr.reshape(first_dim, -1)

    if arr.ndim != 2:
        raise ValueError(f"Could not coerce data to 2D array for {path}")

    return arr

