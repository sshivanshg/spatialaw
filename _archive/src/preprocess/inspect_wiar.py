#!/usr/bin/env python3
"""
Utility to inspect the WiAR dataset tree downloaded under data/raw/WiAR.

The script prints:
  - File inventory with sizes (first N entries shown to keep output manageable)
  - Unique file extensions detected
  - Counts of recordings, volunteers, and label-related files
  - Sample CSI file stats (shape, dtype, first rows)
It also saves a JSON summary to data/processed/dataset_summary.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

MAX_PRINT_FILES = 40
RECORDING_EXTENSIONS = {".dat", ".txt", ".csv", ".mat", ".npy"}
LABEL_KEYWORDS = {"label", "labels", "annotation"}


def human_readable_size(num_bytes: int) -> str:
    """Convert bytes to human readable string."""
    if num_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    magnitude = min(int(np.log(num_bytes) / np.log(1024)), len(units) - 1)
    value = num_bytes / (1024**magnitude)
    return f"{value:.2f} {units[magnitude]}"


def walk_files(root: Path) -> List[Tuple[Path, int]]:
    """Return list of (path, size) for all files under root."""
    files: List[Tuple[Path, int]] = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            try:
                size = fpath.stat().st_size
            except OSError:
                size = -1
            files.append((fpath, size))
    return files


def detect_subjects(files: List[Tuple[Path, int]]) -> int:
    """Count unique volunteer IDs inferred from filenames/directories."""
    subjects = set()
    for path, _ in files:
        name = path.name.lower()
        if "volunteer" in name:
            base = name.split(".")[0]
            subjects.add(base)
    return len(subjects)


def count_label_files(files: List[Tuple[Path, int]]) -> int:
    """Count files that look like label/annotation metadata."""
    count = 0
    for path, _ in files:
        lower = path.name.lower()
        if any(keyword in lower for keyword in LABEL_KEYWORDS):
            count += 1
    return count


def find_sample_file(root: Path) -> Path | None:
    """Find a human-readable CSI sample text file."""
    candidates = sorted(root.rglob("*.txt"))
    for path in candidates:
        # Skip documentation files
        if "readme" in path.name.lower() or "introduction" in path.name.lower():
            continue
        return path
    # Fallback to .dat if no text sample found
    dat_candidates = sorted(root.rglob("*.dat"))
    return dat_candidates[0] if dat_candidates else None


def load_sample(path: Path) -> Tuple[np.ndarray, str]:
    """Load a sample CSI file and return array + description."""
    if path.suffix.lower() == ".txt":
        data = np.loadtxt(path, delimiter=None)
        description = "np.loadtxt (text)"
    elif path.suffix.lower() == ".csv":
        data = np.loadtxt(path, delimiter=",")
        description = "np.loadtxt (csv)"
    elif path.suffix.lower() == ".npy":
        data = np.load(path)
        description = "np.load (npy)"
    else:
        # Binary fallback: read raw bytes
        raw = np.fromfile(path, dtype=np.uint8)
        # Heuristic: reshape assuming 30 subcarriers * 3 antennas if divisible
        dim = 30 * 3
        rows = raw.size // dim
        if rows > 0:
            data = raw[: rows * dim].reshape(rows, dim)
        else:
            data = raw.reshape(-1, 1)
        description = "np.fromfile (uint8)"
    return data, description


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = repo_root / "data" / "raw" / "WiAR"
    output_dir = repo_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dataset_summary.json"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    files = walk_files(dataset_root)
    total_size = sum(size for _, size in files if size >= 0)
    extensions = sorted({path.suffix.lower() for path, _ in files if path.suffix})
    recording_count = sum(
        1 for path, _ in files if path.suffix.lower() in RECORDING_EXTENSIONS
    )
    subject_count = detect_subjects(files)
    label_file_count = count_label_files(files)

    print("=" * 70)
    print(f"WiAR dataset inventory ({len(files)} files, {human_readable_size(total_size)})")
    print("=" * 70)
    for idx, (path, size) in enumerate(sorted(files)[:MAX_PRINT_FILES]):
        rel = path.relative_to(dataset_root)
        size_str = human_readable_size(size) if size >= 0 else "unknown"
        print(f"[{idx+1:02d}] {rel} ({size_str})")
    if len(files) > MAX_PRINT_FILES:
        print(f"... ({len(files) - MAX_PRINT_FILES} additional files not shown)")
    print()

    print("File extensions detected:")
    print(", ".join(extensions))
    print()

    print(f"Recording-like files: {recording_count}")
    print(f"Volunteer/subject identifiers: {subject_count}")
    print(f"Label/annotation files: {label_file_count}")
    print()

    sample_path = find_sample_file(dataset_root)
    sample_info: Dict[str, object] = {}
    if sample_path:
        print(f"Loading sample CSI file: {sample_path.relative_to(dataset_root)}")
        sample_data, loader_desc = load_sample(sample_path)
        print(f"Loader: {loader_desc}")
        print(f"Shape: {sample_data.shape}, dtype: {sample_data.dtype}")
        rows_to_show = min(5, sample_data.shape[0])
        print("First rows:")
        print(sample_data[:rows_to_show])
        sample_info = {
            "path": str(sample_path.relative_to(dataset_root)),
            "loader": loader_desc,
            "shape": sample_data.shape,
            "dtype": str(sample_data.dtype),
            "preview_rows": sample_data[:rows_to_show].tolist(),
        }
    else:
        print("No sample CSI file found.")

    summary = {
        "dataset_root": str(dataset_root),
        "total_files": len(files),
        "total_size_bytes": total_size,
        "extensions": extensions,
        "recording_file_count": recording_count,
        "subject_count": subject_count,
        "label_file_count": label_file_count,
        "sample": sample_info,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"✓ Summary written to {summary_path}")


if __name__ == "__main__":
    main()

