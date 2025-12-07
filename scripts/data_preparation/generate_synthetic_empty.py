#!/usr/bin/env python3
"""
Generate synthetic "empty room" feature vectors.

We define "empty room" as having:
1. Extremely low variance, velocity, and MAD (significantly lower than human "low motion").
2. Stable envelope (low std dev).
3. Random but low motion period (noise).
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic empty room features")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/synthetic_empty"))
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--reference-binary-dir", type=Path, default=Path("data/processed/binary"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load reference feature names
    feature_names_path = args.reference_binary_dir / "feature_names.json"
    if not feature_names_path.exists():
        print(f"Error: {feature_names_path} not found.")
        return 1
    
    with open(feature_names_path) as f:
        feature_names = json.load(f)
    
    n_features = len(feature_names)
    print(f"Generating {args.n_samples} samples with {n_features} features...")
    
    # Initialize empty matrix
    X_synth = np.zeros((args.n_samples, n_features), dtype=np.float32)
    
    # Define generation logic per feature
    # We'll use a dictionary to map feature name to column index
    feat_idx = {name: i for i, name in enumerate(feature_names)}
    
    # 1. Entropy:
    if "csi_entropy" in feat_idx:
        X_synth[:, feat_idx["csi_entropy"]] = np.random.uniform(3.0, 5.0, size=args.n_samples)

    # 2. Envelope Mean:
    if "csi_envelope_mean" in feat_idx:
        X_synth[:, feat_idx["csi_envelope_mean"]] = np.random.uniform(0.5, 1.5, size=args.n_samples)

    # 3. Envelope Std:
    if "csi_envelope_std" in feat_idx:
        X_synth[:, feat_idx["csi_envelope_std"]] = np.random.uniform(0.1, 0.5, size=args.n_samples)

    # 4. MAD Mean:
    if "csi_mad_mean" in feat_idx:
        X_synth[:, feat_idx["csi_mad_mean"]] = np.random.uniform(0.05, 0.25, size=args.n_samples)

    # 5. MAD Std:
    if "csi_mad_std" in feat_idx:
        X_synth[:, feat_idx["csi_mad_std"]] = np.random.uniform(0.02, 0.15, size=args.n_samples)

    # 6. Motion Period:
    if "csi_motion_period_mean" in feat_idx:
        X_synth[:, feat_idx["csi_motion_period_mean"]] = np.random.uniform(0, 15, size=args.n_samples)
    
    if "csi_motion_period_std" in feat_idx:
        X_synth[:, feat_idx["csi_motion_period_std"]] = np.random.uniform(0, 3, size=args.n_samples)

    # 7. Norm Std:
    if "csi_norm_std_mean" in feat_idx:
        X_synth[:, feat_idx["csi_norm_std_mean"]] = np.random.uniform(0.1, 0.5, size=args.n_samples)
    
    if "csi_norm_std_std" in feat_idx:
        X_synth[:, feat_idx["csi_norm_std_std"]] = np.random.uniform(0.05, 0.2, size=args.n_samples)

    # 8. Variance:
    if "csi_variance_max" in feat_idx:
        X_synth[:, feat_idx["csi_variance_max"]] = np.random.uniform(0.5, 1.5, size=args.n_samples)
    
    if "csi_variance_std" in feat_idx:
        X_synth[:, feat_idx["csi_variance_std"]] = np.random.uniform(0.1, 0.4, size=args.n_samples)
    
    # 9. Velocity:
    if "csi_velocity_max" in feat_idx:
        X_synth[:, feat_idx["csi_velocity_max"]] = np.random.uniform(0.1, 0.6, size=args.n_samples)

    # Save features
    out_features_path = args.out_dir / "features.npy"
    np.save(out_features_path, X_synth)
    
    # Create labels
    labels = []
    for i in range(args.n_samples):
        # Assign virtual sources to allow GroupShuffleSplit to distribute synthetic data
        # We'll create 50 virtual sources (10 samples each if n=500)
        source_id = i % 50
        labels.append({
            "label": 0, # Empty room is definitely 0
            "motion_score": 0.0, # Synthetic 0
            "source": f"synthetic_empty_{source_id}",
            "original_label": "empty_room",
            "dataset": "synthetic"
        })
    
    labels_df = pd.DataFrame(labels)
    out_labels_path = args.out_dir / "labels.csv"
    labels_df.to_csv(out_labels_path, index=False)
    
    # Save feature names for consistency check
    with open(args.out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"✓ Saved synthetic data to {args.out_dir}")
    print(f"  Features: {out_features_path}")
    print(f"  Labels: {out_labels_path}")

if __name__ == "__main__":
    main()
