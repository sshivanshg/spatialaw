#!/usr/bin/env python3
"""
CLI utility to convert raw WiAR CSI files into fixed-length windows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

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
    parser.add_argument("--T", type=int, default=256, help="Window length.")
    parser.add_argument("--stride", type=int, default=64, help="Stride between windows.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of recordings to process.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    input_root = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recordings = list_recordings(input_root)
    if args.max_files is not None:
        recordings = recordings[: args.max_files]

    if not recordings:
        print("No CSI recordings found.")
        return 1

    processed_windows = []
    label_records = []

    for idx, rec_path in enumerate(recordings, start=1):
        print(f"[{idx}/{len(recordings)}] Loading {rec_path.relative_to(input_root)}")
        try:
            csi = load_csi_file(rec_path)
        except Exception as exc:
            print(f"  ⚠️  Skipping {rec_path.name}: {exc}")
            continue

        windows = window_csi(csi, T=args.T, stride=args.stride)
        if windows.size == 0:
            print("  ⚠️  Insufficient packets for requested window length.")
            continue

        label_info = infer_label(rec_path)
        for win in windows:
            denoised = denoise_window(win)
            normalized = normalize_window(denoised)
            processed_windows.append(normalized)
            label_records.append(
                {
                    **label_info,
                    "source_recording": str(rec_path.relative_to(input_root)),
                }
            )

    if not processed_windows:
        print("No windows generated.")
        return 1

    all_windows = np.stack(processed_windows, axis=0)
    labels_path = save_windows(all_windows, label_records, out_dir)
    print(f"✓ Saved {all_windows.shape[0]} windows to {out_dir}")
    print(f"✓ Labels written to {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

