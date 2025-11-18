#!/usr/bin/env python3
"""
Process wifi_csi_har_dataset and WiAR data for binary classification.

Maps:
- "standing", "sitting", "lying", "no_person" → label=0 (no activity)
- "walking", "get_down", "get_up" and all WiAR activities → label=1 (activity)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocess.csi_loader import load_csi_file, list_recordings
from src.preprocess.preprocess import denoise_window, normalize_window, window_csi
from src.preprocess.features import extract_csi_features, features_to_vector

DEFAULT_SEED = 42
DEFAULT_T = 256
DEFAULT_STRIDE = 64

# Label mapping for wifi_csi_har_dataset
NO_ACTIVITY_LABELS = {"standing", "sitting", "lying", "no_person"}
ACTIVITY_LABELS = {"walking", "get_down", "get_up"}


def load_wifi_har_data(data_path: Path, label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSI data and labels from wifi_csi_har_dataset format.
    
    Format: Each row has [0,1,2,...,113] (indices) followed by CSI amplitudes (114 values)
    
    Returns:
        (csi_data, labels) where csi_data is (n_packets, n_subcarriers) and labels is (n_packets,)
    """
    # Load data.csv - first 114 columns are subcarrier indices, then CSI amplitudes
    data_df = pd.read_csv(data_path, header=None)
    
    # Extract CSI amplitudes (columns 114-227 are amplitudes for 114 subcarriers)
    n_subcarriers = 114
    csi_data = data_df.iloc[:, n_subcarriers:n_subcarriers*2].values.astype(np.float64)
    
    # Load labels
    label_df = pd.read_csv(label_path, header=None)
    if len(label_df.columns) >= 2:
        labels = label_df.iloc[:, 1].values  # Second column has label names
    else:
        labels = label_df.iloc[:, 0].values
    
    # Ensure same length
    min_len = min(len(csi_data), len(labels))
    csi_data = csi_data[:min_len]
    labels = labels[:min_len]
    
    return csi_data, labels


def map_to_binary_label(label: str) -> int:
    """Map activity label to binary (0=no activity, 1=activity)."""
    label_lower = str(label).lower().strip()
    if label_lower in NO_ACTIVITY_LABELS:
        return 0
    elif label_lower in ACTIVITY_LABELS:
        return 1
    else:
        # Default: treat unknown as activity
        return 1


def process_wifi_har_dataset(
    dataset_dir: Path,
    T: int,
    stride: int,
    target_subcarriers: int = 30,
    only_no_activity: bool = True,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Process all sessions in wifi_csi_har_dataset.
    
    If only_no_activity=True, only processes samples with "no activity" labels
    (standing, sitting, lying, no_person) and excludes activity labels.
    """
    all_windows = []
    all_labels = []
    
    # Find all room directories
    for room_dir in sorted(dataset_dir.glob("room_*")):
        for session_dir in sorted(room_dir.glob("*")):
            if not session_dir.is_dir():
                continue
            
            data_path = session_dir / "data.csv"
            label_path = session_dir / "label.csv"
            
            if not (data_path.exists() and label_path.exists()):
                continue
            
            print(f"Processing {session_dir.relative_to(dataset_dir)}...")
            try:
                csi_data, labels = load_wifi_har_data(data_path, label_path)
                
                # Filter to only "no activity" labels if requested
                if only_no_activity:
                    mask = np.array([str(l).lower().strip() in NO_ACTIVITY_LABELS for l in labels])
                    if not mask.any():
                        print(f"  ⚠️  No 'no activity' samples found, skipping...")
                        continue
                    csi_data = csi_data[mask]
                    labels = labels[mask]
                    print(f"  ✓ Filtered to {len(csi_data)} 'no activity' samples")
                
                # Map labels to binary (should all be 0 for no_activity)
                binary_labels = np.array([map_to_binary_label(l) for l in labels])
                
                # Create windows
                effective_T = min(T, len(csi_data))
                windows = window_csi(csi_data, T=effective_T, stride=stride)
                
                if windows.size == 0:
                    continue
                
                # Standardize subcarriers
                if windows.shape[1] != target_subcarriers:
                    if windows.shape[1] > target_subcarriers:
                        windows = windows[:, :target_subcarriers, :]
                    else:
                        pad_width = target_subcarriers - windows.shape[1]
                        windows = np.pad(
                            windows, ((0, 0), (0, pad_width), (0, 0)), mode="constant"
                        )
                
                # Standardize time dimension
                if windows.shape[2] != T:
                    if windows.shape[2] > T:
                        windows = windows[:, :, :T]
                    else:
                        pad_width = T - windows.shape[2]
                        windows = np.pad(
                            windows, ((0, 0), (0, 0), (0, pad_width)), mode="edge"
                        )
                
                # Process windows
                processed_windows = []
                window_labels = []
                
                for i, win in enumerate(windows):
                    # Get label for this window (use middle timestamp)
                    win_start = i * stride
                    win_mid = win_start + effective_T // 2
                    if win_mid < len(binary_labels):
                        win_label = int(binary_labels[win_mid])
                    else:
                        win_label = int(binary_labels[-1])
                    
                    # Denoise and normalize
                    denoised = denoise_window(win)
                    normalized = normalize_window(denoised)
                    processed_windows.append(normalized.astype(np.float32))
                    
                    window_labels.append({
                        "label": win_label,
                        "source": str(session_dir.relative_to(dataset_dir)),
                        "original_label": str(labels[win_mid]) if win_mid < len(labels) else "unknown",
                        "dataset": "wifi_csi_har",
                    })
                
                all_windows.append(np.stack(processed_windows, axis=0))
                all_labels.extend(window_labels)
                
            except Exception as e:
                print(f"  ⚠️  Error processing {session_dir}: {e}")
                continue
    
    if not all_windows:
        return np.empty((0, target_subcarriers, T)), []
    
    return np.concatenate(all_windows, axis=0), all_labels


def process_wiar_data(
    wiar_features_dir: Path,
) -> Tuple[np.ndarray, List[Dict]]:
    """Load existing WiAR features and relabel as activity=1."""
    features_path = wiar_features_dir / "features.npy"
    labels_path = wiar_features_dir / "labels.csv"
    
    if not (features_path.exists() and labels_path.exists()):
        print(f"⚠️  WiAR features not found in {wiar_features_dir}")
        return np.empty((0, 0)), []
    
    features = np.load(features_path)
    labels_df = pd.read_csv(labels_path)
    
    # Relabel all WiAR activities as 1 (activity present)
    binary_labels = []
    for _, row in labels_df.iterrows():
        binary_labels.append({
            "label": 1,  # All WiAR activities → activity present
            "source": row.get("source_recording", "unknown"),
            "original_label": row.get("activity_name", "unknown"),
            "dataset": "wiar",
        })
    
    return features, binary_labels


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wifi-har-dir",
        default="wifi_csi_har_dataset",
        help="Directory containing wifi_csi_har_dataset",
    )
    parser.add_argument(
        "--wiar-features-dir",
        default="data/processed/features",
        help="Directory containing WiAR extracted features",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/binary",
        help="Output directory for binary classification dataset",
    )
    parser.add_argument("--T", type=int, default=DEFAULT_T, help="Window length")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Stride")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    
    args = parser.parse_args(argv or sys.argv[1:])
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Processing Binary Classification Dataset")
    print("=" * 60)
    
    # Process wifi_csi_har_dataset (only "no activity" samples)
    wifi_har_dir = Path(args.wifi_har_dir)
    if wifi_har_dir.exists():
        print(f"\n1. Processing wifi_csi_har_dataset from {wifi_har_dir}...")
        print(f"   (Only processing 'no activity' samples: standing, sitting, lying, no_person)")
        wifi_windows, wifi_labels = process_wifi_har_dataset(
            wifi_har_dir, args.T, args.stride, only_no_activity=True
        )
        print(f"   ✓ Generated {len(wifi_windows)} windows from wifi_csi_har (no activity only)")
    else:
        print(f"\n⚠️  wifi_csi_har_dataset not found at {wifi_har_dir}")
        wifi_windows = np.empty((0, 30, args.T))
        wifi_labels = []
    
    # Process WiAR data
    print(f"\n2. Loading WiAR features from {args.wiar_features_dir}...")
    wiar_features, wiar_labels = process_wiar_data(Path(args.wiar_features_dir))
    print(f"   ✓ Loaded {len(wiar_features)} WiAR feature vectors")
    
    # Extract features from wifi_csi_har windows
    if len(wifi_windows) > 0:
        print(f"\n3. Extracting features from wifi_csi_har windows...")
        wifi_features_list = []
        for win in wifi_windows:
            feats = extract_csi_features(win)
            wifi_features_list.append(feats)
        
        # Get feature names from first window
        feature_names = sorted(wifi_features_list[0].keys())
        wifi_features = np.array([
            features_to_vector(f, feature_names) for f in wifi_features_list
        ])
        print(f"   ✓ Extracted {len(wifi_features)} feature vectors")
    else:
        wifi_features = np.empty((0, 0))
        feature_names = []
    
    # Combine datasets
    print(f"\n4. Combining datasets...")
    if len(wifi_features) > 0 and len(wiar_features) > 0:
        # Ensure same feature dimensions
        if wifi_features.shape[1] != wiar_features.shape[1]:
            print(f"   ⚠️  Feature dimension mismatch: wifi={wifi_features.shape[1]}, wiar={wiar_features.shape[1]}")
            # Use WiAR feature names if available
            if len(wiar_features) > 0:
                feature_names_path = Path(args.wiar_features_dir) / "feature_names.json"
                if feature_names_path.exists():
                    with open(feature_names_path) as f:
                        feature_names = json.load(f)
                    # Re-extract wifi features with same feature set
                    wifi_features_list = []
                    for win in wifi_windows:
                        feats = extract_csi_features(win)
                        wifi_features_list.append(feats)
                    wifi_features = np.array([
                        features_to_vector(f, feature_names) for f in wifi_features_list
                    ])
        
        all_features = np.vstack([wifi_features, wiar_features])
        all_labels = wifi_labels + wiar_labels
    elif len(wifi_features) > 0:
        all_features = wifi_features
        all_labels = wifi_labels
    elif len(wiar_features) > 0:
        all_features = wiar_features
        all_labels = wiar_labels
        # Load feature names
        feature_names_path = Path(args.wiar_features_dir) / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path) as f:
                feature_names = json.load(f)
    else:
        print("   ❌ No data to combine!")
        return 1
    
    print(f"   ✓ Combined dataset: {len(all_features)} samples")
    
    # Save combined dataset
    print(f"\n5. Saving binary classification dataset...")
    
    # Save features
    features_path = out_dir / "features.npy"
    np.save(features_path, all_features)
    print(f"   ✓ Saved features: {features_path} ({all_features.shape})")
    
    # Save feature names
    if feature_names:
        feature_names_path = out_dir / "feature_names.json"
        with open(feature_names_path, "w") as f:
            json.dump(feature_names, f, indent=2)
        print(f"   ✓ Saved feature names: {feature_names_path}")
    
    # Save labels
    labels_df = pd.DataFrame(all_labels)
    labels_path = out_dir / "labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"   ✓ Saved labels: {labels_path}")
    
    # Print summary
    print(f"\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total samples: {len(all_features)}")
    print(f"Features per sample: {all_features.shape[1]}")
    print(f"\nLabel distribution:")
    label_counts = labels_df["label"].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "No Activity" if label == 0 else "Activity Present"
        print(f"  {label} ({label_name}): {count} samples ({100*count/len(all_features):.1f}%)")
    
    print(f"\nDataset sources:")
    if "dataset" in labels_df.columns:
        dataset_counts = labels_df["dataset"].value_counts()
        for dataset, count in dataset_counts.items():
            print(f"  {dataset}: {count} samples")
    
    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": int(len(all_features)),
        "n_features": int(all_features.shape[1]),
        "label_distribution": label_counts.to_dict(),
        "dataset_sources": dataset_counts.to_dict() if "dataset" in labels_df.columns else {},
        "T": args.T,
        "stride": args.stride,
        "seed": args.seed,
    }
    summary_path = out_dir / "binary_dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary: {summary_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

