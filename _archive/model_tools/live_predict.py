#!/usr/bin/env python3
"""
Lightweight "live" demo that streams CSI windows through the trained detector.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.motion_detector import MotionDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "windows",
        help="Directory containing window_*.npy files and labels.csv.",
    )
    parser.add_argument(
        "--binary-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "binary",
        help="Directory containing feature_names.json for ordering.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional substring filter for source_recording.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum number of windows to stream.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Delay (seconds) between window predictions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_path = args.windows_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.csv in {args.windows_dir}")

    labels_df = pd.read_csv(labels_path)
    if args.source:
        mask = labels_df.get("source_recording", "").astype(str).str.contains(
            args.source, na=False
        )
        labels_df = labels_df[mask]

    if labels_df.empty:
        raise ValueError("No windows match the requested filter.")

    feature_names_path = args.binary_dir / "feature_names.json"
    feature_names = json.loads(feature_names_path.read_text())

    detector = MotionDetector(feature_names=feature_names)

    print(f"Streaming {min(args.limit, len(labels_df))} windows (interval={args.interval}s)")
    for idx, (_, row) in enumerate(labels_df.head(args.limit).iterrows(), start=1):
        window_path = args.windows_dir / row["window_file"]
        window = np.load(window_path)
        proba = detector.predict_proba_from_windows([window])[0, 1]
        bar = "█" * int(proba * 20)
        source = row.get("source_recording", "unknown")
        print(f"[{idx:02d}] {source} :: p(activity)={proba:.3f} {bar}")
        time.sleep(args.interval)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

