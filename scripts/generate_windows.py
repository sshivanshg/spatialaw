#!/usr/bin/env python3
"""
CLI utility to convert raw WiAR CSI files into fixed-length windows.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocess.csi_loader import list_recordings, load_csi_file
from src.preprocess.preprocess import (
    denoise_window,
    normalize_window,
    save_windows,
    window_csi,
)

DEFAULT_SEED = 42
DEFAULT_T = 256
DEFAULT_STRIDE = 64
DEFAULT_SUBCARRIERS = 30

ACTIVE_KEYWORDS = [
    "activity",
    "walk",
    "wave",
    "kick",
    "run",
    "throw",
    "jump",
    "motion",
    "hand",
]
IDLE_KEYWORDS = ["idle", "still", "static", "empty", "rest", "noactivity", "silence"]


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_label(path: Path) -> Dict[str, object]:
    text = "_".join(path.parts).lower()
    if any(keyword in text for keyword in ACTIVE_KEYWORDS):
        return {"label": 1, "auto_label": True}
    if any(keyword in text for keyword in IDLE_KEYWORDS):
        return {"label": 0, "auto_label": True}
    return {"label": -1, "auto_label": False}


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/raw/WiAR",
        help="Root directory containing raw CSI files.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/windows",
        help="Directory to store processed windows.",
    )
    parser.add_argument("--T", type=int, default=DEFAULT_T, help="Window length.")
    parser.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE, help="Stride between windows."
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of recordings to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args(argv)


def process_recording(
    rec_path: Path,
    rel_path: Path,
    T: int,
    stride: int,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    csi = load_csi_file(rec_path)
    n_packets, n_subcarriers = csi.shape
    effective_T = min(T, n_packets)
    if effective_T < T:
        print(
            f"  ⚠️  Requested T={T} exceeds packets ({n_packets}); using T={effective_T}"
        )

    windows = window_csi(csi, T=effective_T, stride=stride)
    if windows.size == 0:
        return np.empty((0, n_subcarriers, effective_T)), []

    label_info = infer_label(rel_path)
    records = []
    processed = []
    for win in windows:
        denoised = denoise_window(win)
        normalized = normalize_window(denoised)
        processed.append(normalized.astype(np.float32))
        records.append({**label_info, "source_recording": str(rel_path)})
    return np.stack(processed, axis=0), records


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    set_random_seeds(args.seed)

    input_root = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_dir": str(out_dir),
        "requested_T": args.T,
        "stride": args.stride,
        "seed": args.seed,
        "processed_files": [],
        "total_windows": 0,
    }

    recordings = list_recordings(input_root)
    if args.max_files is not None:
        recordings = recordings[: args.max_files]

    if not recordings:
        print("No CSI recordings found.")
        return 1

    processed_windows = []
    label_records = []
    observed_subcarriers = set()

    for idx, rec_path in enumerate(recordings, start=1):
        rel_path = rec_path.relative_to(input_root)
        print(f"[{idx}/{len(recordings)}] Loading {rel_path}")
        try:
            windows, records = process_recording(rec_path, rel_path, args.T, args.stride)
        except Exception as exc:
            print(f"  ⚠️  Skipping {rec_path.name}: {exc}")
            continue

        if windows.size == 0:
            print("  ⚠️  Insufficient packets for requested window length.")
            continue

        processed_windows.append(windows)
        label_records.extend(records)
        observed_subcarriers.add(windows.shape[1])
        summary["processed_files"].append(
            {"path": str(rel_path), "windows": windows.shape[0]}
        )

    if not processed_windows:
        print("No windows generated.")
        return 1

    all_windows = np.concatenate(processed_windows, axis=0)
    labels_path = save_windows(all_windows, label_records, out_dir)
    summary["total_windows"] = int(all_windows.shape[0])
    summary["subcarrier_counts"] = sorted(observed_subcarriers) or [DEFAULT_SUBCARRIERS]

    summary_path = out_dir / "window_generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved {all_windows.shape[0]} windows to {out_dir}")
    print(f"✓ Labels written to {labels_path}")
    print(f"✓ Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

