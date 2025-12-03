#!/usr/bin/env python3
"""
Build a WiAR-only binary dataset (movement vs. no-movement).

Why this rewrite?
-----------------
Previously the binary dataset mixed **wifi_csi_har** (label 0) and **WiAR**
(label 1). Because those datasets come from different environments/hardware,
the classifier could simply learn "which dataset is this from?" and score
~100% accuracy.

To make the task realistic, we now:
1. Use **only WiAR** feature vectors (14-D CSI statistics).
2. Compute a simple *motion score* from those features.
3. Label the **lowest X% motion score** windows as `0` (no/low movement) and
   the rest as `1` (movement present).

This keeps both labels within the same WiFi environment, so the classifier
has to learn subtle differences rather than dataset identity.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_MOTION_QUANTILE = 0.25  # lowest 25% → label 0
FEATURES_TO_DROP = ["csi_variance_mean", "csi_velocity_mean"]


def load_wiar_features(
    features_dir: Path, motion_quantile: float
) -> Tuple[np.ndarray, List[Dict], List[str], float]:
    """Load WiAR features, derive binary labels, and return metadata."""
    features_path = features_dir / "features.npy"
    labels_path = features_dir / "labels.csv"
    feature_names_path = features_dir / "feature_names.json"

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "WiAR features not found. Run scripts/extract_features.py first."
        )

    features_full = np.load(features_path)
    labels_df = pd.read_csv(labels_path)

    if feature_names_path.exists():
        with open(feature_names_path) as f:
            feature_names = json.load(f)
    else:
        raise FileNotFoundError(
            f"{feature_names_path} missing. Re-run scripts/extract_features.py."
        )

    if len(features_full) != len(labels_df):
        raise ValueError(
            "Feature matrix and labels.csv have mismatched lengths: "
            f"{len(features_full)} vs {len(labels_df)}"
        )

    def idx(name: str) -> int:
        try:
            return feature_names.index(name)
        except ValueError as err:
            raise KeyError(
                f"Feature '{name}' not found in feature_names.json"
            ) from err

    # Motion score = variance_mean + velocity_mean
    score = (
        features_full[:, idx("csi_variance_mean")]
        + features_full[:, idx("csi_velocity_mean")]
    )
    threshold = float(np.quantile(score, motion_quantile))

    binary_labels: List[Dict] = []
    for i, (_, row) in enumerate(labels_df.iterrows()):
        label_val = 0 if score[i] <= threshold else 1
        binary_labels.append(
            {
                "label": label_val,
                "motion_score": float(score[i]),
                "source": row.get("source_recording", "unknown"),
                "original_label": row.get("activity_name", "unknown"),
                "dataset": "wiar",
            }
        )

    # Drop features that were used directly to derive the label to reduce leakage
    keep_indices = [
        i for i, name in enumerate(feature_names) if name not in FEATURES_TO_DROP
    ]
    filtered_features = features_full[:, keep_indices]
    filtered_feature_names = [feature_names[i] for i in keep_indices]

    return filtered_features, binary_labels, filtered_feature_names, threshold


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wiar-features-dir",
        default="data/processed/features",
        help="Directory containing WiAR feature matrix/labels",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/binary",
        help="Output directory for the WiAR-only binary dataset",
    )
    parser.add_argument(
        "--motion-quantile",
        type=float,
        default=DEFAULT_MOTION_QUANTILE,
        help="Fraction of lowest-motion windows to label as 0 (default: 0.25)",
    )

    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building WiAR-only Binary Dataset")
    print("=" * 60)
    print(f"Motion quantile (label 0 cutoff): {args.motion_quantile:.2f}")

    features, labels_meta, feature_names, threshold = load_wiar_features(
        Path(args.wiar_features_dir), args.motion_quantile
    )

    print(f"✓ Loaded {len(features)} WiAR samples")
    label_counts = pd.Series([row["label"] for row in labels_meta]).value_counts()
    for label, count in label_counts.sort_index().items():
        name = "No / Low Movement" if label == 0 else "Movement"
        print(f"  Label {label} ({name}): {count} samples ({count/len(features):.1%})")

    # Save dataset ----------------------------------------------------------
    print("\nSaving dataset...")
    features_path = out_dir / "features.npy"
    np.save(features_path, features)
    print(f"  ✓ Features saved to {features_path}")

    feature_names_path = out_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  ✓ Feature names saved to {feature_names_path}")

    labels_df = pd.DataFrame(labels_meta)
    labels_path = out_dir / "labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"  ✓ Labels saved to {labels_path}")

    summary: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": int(len(features)),
        "n_features": int(features.shape[1]),
        "label_distribution": {
            int(k): int(v) for k, v in label_counts.sort_index().items()
        },
        "motion_quantile": float(args.motion_quantile),
        "motion_threshold": threshold,
    }
    summary_path = out_dir / "binary_dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary saved to {summary_path}")

    print("\nAll done! Re-run model_tools/train_presence_detector.py to retrain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


