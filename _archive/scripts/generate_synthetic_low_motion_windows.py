#!/usr/bin/env python3
"""
Create additional low-motion ("empty room") CSI windows via augmentation.

The script samples existing low-motion windows identified in
`data/processed/windows_binary/all_windows.csv`, applies gentle
augmentations (noise, time jitter, scaling), saves them as new .npy
files, and records their metadata in `synthetic_windows.csv`. These
synthetic samples can be included when preparing presence datasets to
balance class 0.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows-dir",
        default="data/processed/windows",
        help="Directory containing window_*.npy files.",
    )
    parser.add_argument(
        "--all-windows-csv",
        default="data/processed/windows_binary/all_windows.csv",
        help="CSV with motion scores and presence labels (produced by prepare_presence_windows.py).",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/windows_binary/synthetic_windows.csv",
        help="Destination CSV describing generated samples.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of synthetic windows to generate.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.02,
        help="Gaussian noise std (relative to normalized CSI values).",
    )
    parser.add_argument(
        "--time-jitter",
        type=int,
        default=8,
        help="Maximum time-axis shift applied to synthetic windows.",
    )
    parser.add_argument(
        "--scale-jitter",
        type=float,
        default=0.05,
        help="Amplitude scaling jitter (e.g., 0.05 -> ±5%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def augment_window(
    window: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
    time_jitter: int,
    scale_jitter: float,
) -> np.ndarray:
    augmented = window.astype(np.float32).copy()

    if time_jitter > 0:
        shift = int(rng.integers(-time_jitter, time_jitter + 1))
        if shift != 0:
            augmented = np.roll(augmented, shift=shift, axis=1)

    if scale_jitter > 0:
        scale = 1.0 + float(rng.uniform(-scale_jitter, scale_jitter))
        augmented *= scale

    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)
        augmented += noise

    return augmented


def compute_motion_score(window: np.ndarray) -> float:
    if window.shape[1] < 2:
        return 0.0
    diff = np.diff(window, axis=1)
    return float(np.mean(np.abs(diff)))


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    windows_dir = Path(args.windows_dir)
    all_windows_csv = Path(args.all_windows_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not all_windows_csv.exists():
        raise FileNotFoundError(
            f"{all_windows_csv} not found. Run prepare_presence_windows.py first."
        )

    df = pd.read_csv(all_windows_csv)
    if "presence_label" not in df.columns:
        raise ValueError("Expected `presence_label` column in all_windows.csv.")

    low_motion_df = df[df["presence_label"] == 0]
    if low_motion_df.empty:
        raise RuntimeError("No low-motion windows available to seed synthetic data.")

    records: List[dict] = []

    for idx in range(args.count):
        base_row = low_motion_df.sample(n=1, random_state=rng.integers(0, 1_000_000)).iloc[0]
        base_path = windows_dir / base_row["window_file"]
        if not base_path.exists():
            raise FileNotFoundError(f"Missing window file: {base_path}")

        window = np.load(base_path).astype(np.float32)
        synthetic = augment_window(
            window,
            rng=rng,
            noise_std=args.noise_std,
            time_jitter=args.time_jitter,
            scale_jitter=args.scale_jitter,
        )

        file_name = f"synthetic_window_{uuid.uuid4().hex[:8]}_{idx:05d}.npy"
        out_path = windows_dir / file_name
        np.save(out_path, synthetic)

        motion_score = compute_motion_score(synthetic)
        record = {
            "window_file": file_name,
            "presence_label": 0,
            "motion_score": motion_score,
            "origin": "synthetic_low_motion",
            "base_window_file": base_row["window_file"],
            "source_recording": base_row.get("source_recording", "synthetic"),
            "activity_id": -1,
            "activity_name": "synthetic_empty",
            "auto_label": False,
            "label": -1,
        }
        records.append(record)

    synth_df = pd.DataFrame(records)
    synth_df.to_csv(output_csv, index=False)

    summary = {
        "generated": args.count,
        "output_csv": str(output_csv),
        "windows_dir": str(windows_dir),
        "noise_std": args.noise_std,
        "time_jitter": args.time_jitter,
        "scale_jitter": args.scale_jitter,
        "seed": args.seed,
        "base_low_motion_count": int(len(low_motion_df)),
        "motion_score_stats": {
            "min": float(synth_df["motion_score"].min()),
            "max": float(synth_df["motion_score"].max()),
            "mean": float(synth_df["motion_score"].mean()),
        },
    }

    summary_path = output_csv.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"✓ Generated {args.count} synthetic low-motion windows.")
    print(f"✓ Metadata saved to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

