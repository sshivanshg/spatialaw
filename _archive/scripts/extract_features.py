#!/usr/bin/env python3
"""
Extract statistical features from processed CSI windows.

Reads windows from data/processed/windows/ and extracts CSI + RSS features,
saving them as a feature matrix and CSV for training classifiers.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocess.features import extract_fusion_features, features_to_vector

DEFAULT_SEED = 42


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows-dir",
        default="data/processed/windows",
        help="Directory containing processes window .npy files",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional input CSV path (overrides default labels.csv search).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/features",
        help="Directory to save extracted features",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility",
    )
    return parser.parse_args(argv or sys.argv[1:])


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    set_random_seeds(args.seed)

    windows_dir = Path(args.windows_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_csv:
        labels_path = Path(args.input_csv)
        if not labels_path.exists():
            print(f"❌ Input CSV not found: {labels_path}")
            return 1
        labels_df = pd.read_csv(labels_path)
        print(f"Loaded {len(labels_df)} samples from {labels_path}")
        
        # Map presence_label to label if attempting to align with CNN data
        if "presence_label" in labels_df.columns:
            print("Mapping 'presence_label' to 'label' for training...")
            labels_df["label"] = labels_df["presence_label"].astype(int)
        elif "label" in labels_df.columns:
            # Auto-binarize if we see multi-class labels (like 0..16)
            # This matches CNN logic: 0 is empty, >0 is presence
            if labels_df["label"].max() > 1:
                print("Binarizing multi-class labels (0->0, >0->1) to match CNN training...")
                labels_df["label"] = labels_df["label"].apply(lambda x: 0 if x == 0 else 1)
            
    else:
        labels_path = windows_dir / "labels.csv"
        if not labels_path.exists():
            print(f"❌ Labels file not found: {labels_path}")
            return 1

        labels_df = pd.read_csv(labels_path)

    if labels_df.empty:
        print("❌ No windows found in input CSV")
        return 1

    print(f"Extracting features from {len(labels_df)} windows...")

    feature_vectors: List[np.ndarray] = []
    feature_dicts: List[Dict[str, float]] = []
    valid_indices: List[int] = []

    for idx, row in labels_df.iterrows():
        window_path = windows_dir / row["window_file"]
        if not window_path.exists():
            print(f"⚠️  Skipping missing window: {window_path.name}")
            continue

        try:
            window = np.load(window_path).astype(np.float32)
            if window.ndim != 2:
                print(f"⚠️  Skipping invalid window shape: {window.shape}")
                continue

            # Extract CSI features (RSS not available in current windows)
            features = extract_fusion_features(window, rss_series=None)
            feature_dicts.append(features)
            feature_vectors.append(features_to_vector(features))
            valid_indices.append(idx)

        except Exception as exc:
            print(f"⚠️  Error processing {window_path.name}: {exc}")
            continue

    if not feature_vectors:
        print("❌ No valid features extracted")
        return 1

    # Stack into feature matrix
    X = np.stack(feature_vectors, axis=0)
    feature_names = sorted(feature_dicts[0].keys())

    # Filter labels to valid indices
    valid_labels = labels_df.iloc[valid_indices].copy()
    valid_labels["feature_idx"] = range(len(valid_labels))

    # Save feature matrix
    features_npy_path = output_dir / "features.npy"
    np.save(features_npy_path, X)
    print(f"✓ Saved feature matrix ({X.shape}) to {features_npy_path}")

    # Save feature names
    feature_names_path = output_dir / "feature_names.json"
    with feature_names_path.open("w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    print(f"✓ Saved feature names ({len(feature_names)}) to {feature_names_path}")

    # Save labels with feature indices
    labels_output_path = output_dir / "labels.csv"
    valid_labels.to_csv(labels_output_path, index=False)
    print(f"✓ Saved labels to {labels_output_path}")

    # Save summary
    summary = {
        "n_windows": len(feature_vectors),
        "n_features": len(feature_names),
        "feature_shape": list(X.shape),
        "feature_names": feature_names,
        "seed": args.seed,
    }
    summary_path = output_dir / "feature_extraction_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")

    print(f"\n✅ Feature extraction complete: {len(feature_vectors)} windows, {len(feature_names)} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

