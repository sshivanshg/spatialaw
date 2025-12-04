#!/usr/bin/env python3
"""
Run presence detection on a raw CSI file (.dat or .npy).

Usage:
    python model_tools/predict_from_raw.py path/to/recording.dat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.dat_loader import load_dat_file
from src.preprocess.preprocess import window_csi, denoise_window, normalize_window
from src.models.motion_detector import MotionDetector

def main():
    parser = argparse.ArgumentParser(description="Run presence detection on raw CSI file")
    parser.add_argument("input_file", type=Path, help="Path to .dat or .npy file")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models", help="Directory containing trained model")
    parser.add_argument("--binary-dir", type=Path, default=ROOT / "data" / "processed" / "binary", help="Directory containing feature names")
    parser.add_argument("--T", type=int, default=256, help="Window size")
    parser.add_argument("--stride", type=int, default=64, help="Window stride")
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: File {args.input_file} not found.")
        return 1

    # 1. Load Raw Data
    print(f"Loading {args.input_file}...")
    if args.input_file.suffix == ".npy":
        csi = np.load(args.input_file)
    else:
        try:
            csi = load_dat_file(args.input_file)
        except ImportError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Error loading file: {e}")
            return 1

    print(f"  Raw CSI shape: {csi.shape}")
    if csi.ndim != 2:
        print("  Error: Expected 2D CSI array (packets, subcarriers).")
        # If 3D (packets, subcarriers, antennas), we might need to average?
        # dat_loader already handles this, so if we get here, something is odd.
        return 1

    # 2. Windowing
    print("Windowing...")
    windows = window_csi(csi, T=args.T, stride=args.stride)
    print(f"  Generated {len(windows)} windows.")

    if len(windows) == 0:
        print("  Not enough data for a single window.")
        return 0

    # 3. Preprocessing (Denoise + Normalize)
    print("Preprocessing...")
    processed_windows = []
    for w in windows:
        # w shape: (subcarriers, T)
        w_denoised = denoise_window(w)
        w_norm = normalize_window(w_denoised)
        processed_windows.append(w_norm)
    
    processed_windows = np.stack(processed_windows)

    # 4. Load Model
    feature_names_path = args.binary_dir / "feature_names.json"
    if not feature_names_path.exists():
        print(f"Error: {feature_names_path} not found. Have you trained the model?")
        return 1
    
    with open(feature_names_path) as f:
        feature_names = json.load(f)

    detector = MotionDetector(
        feature_names=feature_names,
        model_path=args.model_dir / "presence_detector_rf.joblib",
        scaler_path=args.model_dir / "presence_detector_scaler.joblib"
    )

    # 5. Inference
    print("Running inference...")
    probs = detector.predict_proba_from_windows(processed_windows)[:, 1]
    preds = (probs > 0.5).astype(int)

    # 6. Report
    print("\n" + "="*40)
    print("PREDICTION RESULTS")
    print("="*40)
    
    # Group consecutive predictions
    # We'll just print a timeline
    
    # Calculate time per window (approx)
    # If we assume 100Hz sampling (typical for Intel 5300 default)
    # Stride 64 = 0.64s step
    # Window 256 = 2.56s duration
    # This is rough estimation.
    
    fs = 100.0 # Hz (Assumption)
    step_time = args.stride / fs
    
    print(f"{'Time (approx)':<15} | {'Prob':<6} | {'Prediction'}")
    print("-" * 40)
    
    for i, p in enumerate(probs):
        start_time = i * step_time
        label = "PRESENCE" if p > 0.5 else "EMPTY"
        bar = "█" * int(p * 10)
        print(f"{start_time:6.1f}s - {start_time+2.5:4.1f}s | {p:.2f}   | {label} {bar}")

    avg_prob = np.mean(probs)
    final_verdict = "PRESENCE DETECTED" if avg_prob > 0.5 else "ROOM EMPTY"
    print("="*40)
    print(f"FINAL VERDICT: {final_verdict} (Avg Prob: {avg_prob:.2f})")
    print("="*40)

    return 0

if __name__ == "__main__":
    main()
