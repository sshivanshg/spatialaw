#!/usr/bin/env python3
"""
Create binary presence labels directly from CSI windows using motion scores.

This script scans the processed windows directory, computes a simple
motion score for every window (mean absolute temporal derivative), and
splits the windows into train/val/test CSVs with binary labels:

    label 0 -> low-motion windows (below a quantile threshold)
    label 1 -> high-motion windows (above or equal to the threshold)

These CSVs can be consumed by `_archive/src/train/dataset.py` to train
deep models (e.g., CNNs) directly on CSI tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows-dir",
        default="data/processed/windows",
        help="Directory containing window_*.npy files and labels.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/windows_binary",
        help="Destination directory for CSV splits and summary JSON.",
    )
    parser.add_argument(
        "--motion-quantile",
        type=float,
        default=0.25,
        help="Quantile used to mark low-motion windows as label 0.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=None,
        help="Absolute motion-score threshold. Overrides motion quantile if set.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (portion of total windows).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio (portion of total windows).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--extra-labels-csv",
        nargs="*",
        default=[],
        help="Optional CSV files with additional windows (must include `window_file`). "
        "If they contain `presence_label`, those values are preserved.",
    )
    return parser.parse_args()


def compute_motion_score(window: np.ndarray) -> float:
    """
    Use mean absolute temporal derivative as a proxy for motion energy.
    """

    if window.ndim != 2:
        raise ValueError(f"Expected 2D window, got shape {window.shape}")

    if window.shape[1] < 2:
        return 0.0

    diff = np.diff(window, axis=1)
    return float(np.mean(np.abs(diff)))


def stratified_split(
    df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >=0 and sum to < 1.")

    if val_ratio == 0 and test_ratio == 0:
        return df, pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    temp_size = val_ratio + test_ratio
    if temp_size > 0:
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_size,
            stratify=df["presence_label"],
            random_state=seed,
        )
    else:
        train_df, temp_df = df, pd.DataFrame(columns=df.columns)

    if temp_df.empty:
        return train_df, pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    if val_ratio == 0:
        val_df = pd.DataFrame(columns=df.columns)
        test_df = temp_df
    elif test_ratio == 0:
        val_df = temp_df
        test_df = pd.DataFrame(columns=df.columns)
    else:
        test_fraction = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_fraction,
            stratify=temp_df["presence_label"],
            random_state=seed,
        )

    return train_df, val_df, test_df


def main() -> int:
    args = parse_args()
    windows_dir = Path(args.windows_dir)
    output_dir = Path(args.output_dir)

    labels_path = windows_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    dataframes: List[pd.DataFrame] = [pd.read_csv(labels_path)]

    for extra_path in args.extra_labels_csv:
        extra = Path(extra_path)
        if not extra.exists():
            raise FileNotFoundError(f"Extra labels file not found: {extra}")
        extra_df = pd.read_csv(extra)
        dataframes.append(extra_df)

    labels_df = pd.concat(dataframes, ignore_index=True)
    required_cols = {"window_file"}
    if not required_cols.issubset(labels_df.columns):
        missing = ", ".join(sorted(required_cols - set(labels_df.columns)))
        raise ValueError(f"Missing required columns in combined labels: {missing}")

    if "presence_label" not in labels_df.columns:
        labels_df["presence_label"] = np.nan
    else:
        labels_df["presence_label"] = labels_df["presence_label"].astype(float)

    motion_scores = []
    for idx, row in labels_df.iterrows():
        window_path = windows_dir / row["window_file"]
        if not window_path.exists():
            raise FileNotFoundError(f"Window file missing: {window_path}")
        window = np.load(window_path).astype(np.float32)
        score = compute_motion_score(window)
        motion_scores.append(score)

    labels_df["motion_score"] = motion_scores

    if args.motion_threshold is not None:
        threshold = args.motion_threshold
    else:
        threshold = float(np.quantile(labels_df["motion_score"], args.motion_quantile))

    needs_label = labels_df["presence_label"].isna()
    labels_df.loc[needs_label, "presence_label"] = (
        labels_df.loc[needs_label, "motion_score"] >= threshold
    ).astype(int)

    if labels_df["presence_label"].isna().any():
        raise RuntimeError("Some rows still have undefined presence_label values.")

    label_counts: Dict[int, int] = (
        labels_df["presence_label"].value_counts().sort_index().to_dict()
    )
    if 0 not in label_counts or 1 not in label_counts:
        raise RuntimeError(
            "Quantile threshold failed to produce both classes. "
            "Try adjusting --motion-quantile."
        )

    train_df, val_df, test_df = stratified_split(
        labels_df, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    labels_df.to_csv(output_dir / "all_windows.csv", index=False)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    summary = {
        "windows_dir": str(windows_dir),
        "output_dir": str(output_dir),
        "total_windows": int(len(labels_df)),
        "motion_quantile": args.motion_quantile,
        "motion_threshold": threshold,
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()},
        "split_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "extra_labels_csv": [str(Path(p)) for p in args.extra_labels_csv],
    }

    with (output_dir / "presence_windows_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"✓ Computed motion scores for {len(labels_df)} windows.")
    threshold_label = (
        f"quantile {args.motion_quantile:.2f}"
        if args.motion_threshold is None
        else "manual threshold"
    )
    print(f"✓ Motion threshold ({threshold_label}) = {threshold:.6f}")
    print(
        f"✓ Label counts -> class 0: {label_counts.get(0, 0)}, "
        f"class 1: {label_counts.get(1, 0)}"
    )
    print(f"✓ Splits written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

