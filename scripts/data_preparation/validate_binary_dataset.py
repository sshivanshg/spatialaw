#!/usr/bin/env python3
"""
Analyze the derived binary dataset to sanity-check labels and splits.

Outputs a JSON report with:
- label distribution and motion-score summary statistics
- per-activity breakdown of binary labels
- verification that GroupShuffleSplit keeps source recordings disjoint
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_BINARY_DIR = Path("data/processed/binary")
DEFAULT_REPORT = "validation_report.json"
DEFAULT_TEST_SIZE = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary-dir",
        type=Path,
        default=DEFAULT_BINARY_DIR,
        help="Directory containing binary features, labels, and summary files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional report path. Defaults to <binary-dir>/validation_report.json",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Hold-out size for GroupShuffleSplit validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_binary_assets(binary_dir: Path) -> tuple[np.ndarray, pd.DataFrame, dict]:
    features_path = binary_dir / "features.npy"
    labels_path = binary_dir / "labels.csv"
    summary_path = binary_dir / "binary_dataset_summary.json"

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing binary dataset files under {binary_dir}. "
            "Run scripts/process_binary_dataset.py first."
        )

    features = np.load(features_path)
    labels_df = pd.read_csv(labels_path)
    dataset_summary = {}
    if summary_path.exists():
        dataset_summary = json.loads(summary_path.read_text())

    if len(features) != len(labels_df):
        raise ValueError(
            "features.npy and labels.csv length mismatch: "
            f"{len(features)} vs {len(labels_df)}"
        )
    return features, labels_df, dataset_summary


def summarize_motion_scores(labels_df: pd.DataFrame) -> Dict[str, float]:
    scores = labels_df["motion_score"].to_numpy()
    quantiles = np.quantile(scores, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
    }


def summarize_per_activity(labels_df: pd.DataFrame) -> List[Dict[str, object]]:
    groups = labels_df.groupby("original_label")
    breakdown: List[Dict[str, object]] = []
    for activity, frame in groups:
        counts = frame["label"].value_counts().to_dict()
        breakdown.append(
            {
                "activity": activity,
                "total": int(len(frame)),
                "pct_of_dataset": float(len(frame) / len(labels_df)),
                "label_0": int(counts.get(0, 0)),
                "label_1": int(counts.get(1, 0)),
                "avg_motion_score": float(frame["motion_score"].mean()),
            }
        )
    breakdown.sort(key=lambda entry: entry["total"], reverse=True)
    return breakdown


def validate_split(
    features: np.ndarray,
    labels_df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Dict[str, object]:
    fallback_groups = labels_df.index.to_series().astype(str)
    if "source" in labels_df.columns:
        groups = labels_df["source"].fillna(fallback_groups)
    else:
        groups = fallback_groups
    gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    train_idx, test_idx = next(gss.split(features, labels_df["label"], groups=groups))

    train_sources = set(groups.iloc[train_idx])
    test_sources = set(groups.iloc[test_idx])
    overlap = sorted(train_sources & test_sources)

    return {
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "unique_sources_train": int(len(train_sources)),
        "unique_sources_test": int(len(test_sources)),
        "source_overlap": overlap,
        "leakage_detected": len(overlap) > 0,
    }


def main() -> int:
    args = parse_args()
    binary_dir = args.binary_dir
    report_path = args.output or (binary_dir / DEFAULT_REPORT)

    features, labels_df, dataset_summary = load_binary_assets(binary_dir)

    label_counts = labels_df["label"].value_counts().to_dict()
    motion_stats = summarize_motion_scores(labels_df)
    per_activity = summarize_per_activity(labels_df)
    split_report = validate_split(features, labels_df, args.test_size, args.random_state)

    warnings: List[str] = []
    if split_report["leakage_detected"]:
        warnings.append(
            f"{len(split_report['source_overlap'])} source IDs appear in both train and test splits."
        )
    if dataset_summary:
        threshold_delta = abs(
            motion_stats["q25"] - dataset_summary.get("motion_threshold", 0.0)
        )
        if threshold_delta > 1e-3:
            warnings.append(
                "Recomputed 25th percentile differs from stored motion_threshold "
                f"by {threshold_delta:.4f}."
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "binary_dir": str(binary_dir.resolve()),
        "total_samples": int(len(labels_df)),
        "n_features": int(features.shape[1]),
        "label_distribution": {int(k): int(v) for k, v in label_counts.items()},
        "motion_score_stats": motion_stats,
        "per_activity_breakdown": per_activity,
        "split_validation": split_report,
        "warnings": warnings,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"✓ Validation report written to {report_path}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("No issues detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

