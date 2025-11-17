"""
Utilities to ingest the WiFi CSI HAR dataset and convert it into
presence-detection friendly tensors and features.

The original dataset ships CSI magnitude/phase pairs per subcarrier and
per antenna pair in CSV files. Each session folder contains:
    - data.csv       : CSI stream (packets × 1026 columns)
    - label.csv      : activity label per packet (string)
    - label_boxes.csv: bounding box annotations (unused here)

This module exposes helpers to:
    1. Load a session as complex CSI tensors.
    2. Map activity labels to binary presence labels.
    3. Generate sliding-window samples with configurable overlap.
    4. Materialise feature matrices for classical models or raw tensors
       for neural architectures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .csi_features import extract_window_csi_features

SUBCARRIERS = 114
ANTENNA_PAIRS = 4  # 2 Tx × 2 Rx in the dataset

# Activities that correspond to "no human present" in the HAR dataset.
NEGATIVE_KEYWORDS = (
    "no activity",
    "no presence",
    "no_person",
    "empty",
    "background",
    "idle",
    "none",
    "0",
)


@dataclass
class SessionCSI:
    """Container for a single session's CSI stream."""

    room: str
    session: str
    csi_complex: np.ndarray  # shape: (packets, subcarriers, antennas)
    activities: np.ndarray  # raw activity strings per packet
    presence: np.ndarray  # binary labels per packet (0=no human, 1=presence)


def _activity_to_presence(activity: str) -> int:
    """
    Convert a free-form activity string into a binary presence label.
    """
    token = str(activity).strip().lower()
    if token in NEGATIVE_KEYWORDS:
        return 0
    if token.isdigit():
        return int(token != "0")
    for negative in NEGATIVE_KEYWORDS:
        if negative in token:
            return 0
    return 1


def load_csi_data(data_path: Path) -> np.ndarray:
    """
    Load CSI data from a session `data.csv` file.
    Returns a complex-valued array of shape (n_packets, SUBCARRIERS, ANTENNA_PAIRS).
    """
    df = pd.read_csv(data_path, header=0)
    if len(df) < 2:
        raise ValueError(f"File {data_path} does not contain CSI rows.")

    data_rows = df.iloc[1:].to_numpy(dtype=np.float64)
    n_packets = data_rows.shape[0]

    mag_start = SUBCARRIERS
    mag_end = mag_start + SUBCARRIERS * ANTENNA_PAIRS
    phase_start = mag_end
    phase_end = phase_start + SUBCARRIERS * ANTENNA_PAIRS

    if data_rows.shape[1] < phase_end:
        raise ValueError(
            f"Unexpected column count {data_rows.shape[1]} in {data_path}. "
            "Expected at least indices + magnitude + phase."
        )

    magnitudes = data_rows[:, mag_start:mag_end].reshape(n_packets, SUBCARRIERS, ANTENNA_PAIRS)
    phases = data_rows[:, phase_start:phase_end].reshape(n_packets, SUBCARRIERS, ANTENNA_PAIRS)

    return magnitudes * np.exp(1j * phases)


def load_labels(label_path: Path) -> np.ndarray:
    """Load raw activity strings from a session `label.csv` file."""
    df = pd.read_csv(label_path, header=None)
    if df.shape[1] == 0:
        raise ValueError(f"No columns found in {label_path}")
    if df.shape[1] == 1:
        activities = df.iloc[:, 0].astype(str).to_numpy()
    else:
        activities = df.iloc[:, 1].astype(str).to_numpy()
    return activities


def load_session(session_dir: Path) -> SessionCSI:
    """Load a single session directory into a `SessionCSI` object."""
    data_path = session_dir / "data.csv"
    label_path = session_dir / "label.csv"
    if not data_path.exists() or not label_path.exists():
        raise FileNotFoundError(f"Missing data/label files in {session_dir}")

    csi_complex = load_csi_data(data_path)
    activities = load_labels(label_path)

    if len(activities) != len(csi_complex):
        min_len = min(len(activities), len(csi_complex))
        activities = activities[:min_len]
        csi_complex = csi_complex[:min_len]

    presence = np.array([_activity_to_presence(a) for a in activities], dtype=np.int8)

    return SessionCSI(
        room=session_dir.parent.name,
        session=session_dir.name,
        csi_complex=csi_complex,
        activities=activities,
        presence=presence,
    )


def iter_sessions(
    dataset_root: Path,
    rooms: Optional[List[str]] = None,
    sessions: Optional[List[int]] = None,
) -> Iterator[SessionCSI]:
    """
    Yield `SessionCSI` objects for all matching room/session folders.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise ValueError(f"Dataset directory not found: {dataset_root}")
    
    if rooms is None:
        room_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("room_"))
    else:
        room_dirs = [root / room for room in rooms]
    
    for room_dir in room_dirs:
        if not room_dir.exists():
            continue
        
        session_dirs = sorted(d for d in room_dir.iterdir() if d.is_dir() and d.name.isdigit())
        if sessions is not None:
            session_dirs = [d for d in session_dirs if int(d.name) in sessions]
        
        for session_dir in session_dirs:
            try:
                yield load_session(session_dir)
            except Exception as exc:
                print(f"⚠️  Skipping {session_dir} due to error: {exc}")


def _normalise_magnitude(magnitude: np.ndarray) -> np.ndarray:
    """
    Z-score normalisation per subcarrier/antenna across time for a single session.
    """
    mean = magnitude.mean(axis=0, keepdims=True)
    std = magnitude.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (magnitude - mean) / std


def generate_presence_windows(
    session: SessionCSI,
    window_size: int,
    stride: int,
    presence_threshold: float = 0.2,
    normalise: bool = True,
    include_phase: bool = True,
) -> Iterator[Tuple[np.ndarray, int, Dict[str, int]]]:
    """
    Slice a session into sliding windows and emit (window_tensor, label, meta).

    The returned window tensor is shaped (window_size, SUBCARRIERS, ANTENNA_PAIRS, channels)
    where `channels` is 1 (magnitude only) or 2 (magnitude + phase).
    """
    csi_complex = session.csi_complex
    magnitude = np.abs(csi_complex)
    phase = np.angle(csi_complex)

    if normalise:
        magnitude = _normalise_magnitude(magnitude)

    for start in range(0, len(magnitude) - window_size + 1, stride):
        end = start + window_size
        presence_slice = session.presence[start:end]
        ratio = presence_slice.mean()
        label = int(ratio >= presence_threshold)

        mag_window = magnitude[start:end]
        if include_phase:
            phase_window = phase[start:end]
            window = np.stack([mag_window, phase_window], axis=-1)
        else:
            window = mag_window[..., np.newaxis]

        meta = {"start": start, "end": end, "room": session.room, "session": session.session}
        yield window, label, meta


def build_feature_dataset(
    dataset_root: Path,
    window_size: int = 400,
    overlap: float = 0.5,
    rooms: Optional[List[str]] = None,
    sessions: Optional[List[int]] = None,
    presence_threshold: float = 0.2,
    feature_mode: str = "statistics",
    normalise: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int]]]:
    """
    Convert the dataset into a feature matrix ready for classical models.

    feature_mode:
        - "statistics": handcrafted statistics via `extract_window_csi_features`
        - "flatten": flatten magnitude (and optional phase) window into a vector
        - "flatten_magnitude": flatten only the magnitude channel
    """
    stride = max(1, int(window_size * (1.0 - overlap)))
    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    metadata: List[Dict[str, int]] = []

    for session in iter_sessions(dataset_root, rooms=rooms, sessions=sessions):
        for window, label, meta in generate_presence_windows(
            session,
            window_size=window_size,
            stride=stride,
            presence_threshold=presence_threshold,
            normalise=normalise,
            include_phase=(feature_mode != "flatten_magnitude"),
        ):
            if feature_mode == "statistics":
                mag = window[..., 0]
                phase = window[..., 1] if window.shape[-1] > 1 else None
                features = extract_window_csi_features(mag, phase)
            elif feature_mode == "flatten":
                features = window.reshape(-1)
            elif feature_mode == "flatten_magnitude":
                features = window[..., 0].reshape(-1)
            else:
                raise ValueError(f"Unknown feature_mode: {feature_mode}")

            all_features.append(features.astype(np.float32))
            all_labels.append(label)
            metadata.append(meta)

    if not all_features:
        raise ValueError("No windows generated from dataset. Check parameters.")

    feature_lengths = {feat.shape[0] for feat in all_features}
    if len(feature_lengths) != 1:
        raise ValueError(
            f"Inconsistent feature sizes encountered: {sorted(feature_lengths)}. "
            "Try using feature_mode='flatten_magnitude'."
        )

    feature_matrix = np.vstack(all_features)
    labels = np.array(all_labels, dtype=np.int64)
    return feature_matrix, labels, metadata


def build_tensor_dataset(
    dataset_root: Path,
    window_size: int = 400,
    overlap: float = 0.5,
    rooms: Optional[List[str]] = None,
    sessions: Optional[List[int]] = None,
    presence_threshold: float = 0.2,
    normalise: bool = True,
    include_phase: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, int]]]:
    """
    Convert the dataset into raw tensors suitable for deep learning models.

    Returns:
        tensors: (n_windows, channels, window_size, SUBCARRIERS, ANTENNA_PAIRS)
        labels:  (n_windows,)
    """
    stride = max(1, int(window_size * (1.0 - overlap)))
    tensors: List[np.ndarray] = []
    labels: List[int] = []
    metadata: List[Dict[str, int]] = []

    for session in iter_sessions(dataset_root, rooms=rooms, sessions=sessions):
        for window, label, meta in generate_presence_windows(
            session,
            window_size=window_size,
            stride=stride,
            presence_threshold=presence_threshold,
            normalise=normalise,
            include_phase=include_phase,
        ):
            window_tensor = np.transpose(window, (3, 0, 1, 2))
            tensors.append(window_tensor.astype(np.float32))
            labels.append(label)
            metadata.append(meta)

    if not tensors:
        raise ValueError("No windows generated from dataset. Check parameters.")

    tensor_stack = np.stack(tensors)
    label_array = np.array(labels, dtype=np.int64)
    return tensor_stack, label_array, metadata


