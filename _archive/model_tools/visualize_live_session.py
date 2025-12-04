#!/usr/bin/env python3
"""
Visualize a live session from a raw CSI file.

Generates a report image showing:
1. CSI Heatmap
2. Noise Level (Variance)
3. Presence Probability

Usage:
    python model_tools/visualize_live_session.py path/to/recording.dat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.dat_loader import load_dat_file
from src.preprocess.preprocess import window_csi, denoise_window, normalize_window
from src.models.motion_detector import MotionDetector
from src.preprocess.features import extract_fusion_features

def main():
    parser = argparse.ArgumentParser(description="Visualize live session from raw CSI")
    parser.add_argument("input_file", type=Path, help="Path to .dat or .npy file")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models", help="Directory containing trained model")
    parser.add_argument("--binary-dir", type=Path, default=ROOT / "data" / "processed" / "binary", help="Directory containing feature names")
    parser.add_argument("--T", type=int, default=256, help="Window size")
    parser.add_argument("--stride", type=int, default=64, help="Window stride")
    parser.add_argument("--output", type=Path, default=Path("session_report.png"), help="Output image path")
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
        except Exception as e:
            print(f"Error loading file: {e}")
            return 1

    print(f"  Raw CSI shape: {csi.shape}")
    
    # 2. Windowing
    print("Windowing...")
    windows = window_csi(csi, T=args.T, stride=args.stride)
    
    if len(windows) == 0:
        print("  Not enough data for a single window.")
        return 0

    # 3. Preprocessing & Feature Extraction
    print("Processing & Extracting Features...")
    processed_windows = []
    noise_levels = []
    
    # We need to extract features manually to get "noise level" (variance)
    # AND to pass to the model.
    # MotionDetector usually does extraction internally, but we can do it here too.
    
    # Actually, let's use MotionDetector for inference, but extract variance manually for plotting.
    # Or better: Preprocess windows for inference, and calculate variance on the PREPROCESSED windows?
    # No, variance should be on the raw-ish window (maybe just denoised).
    # Let's calculate variance on the denoised window.
    
    for w in windows:
        # Denoise
        w_denoised = denoise_window(w)
        
        # Calculate Noise Level (Mean Variance across subcarriers)
        # Variance of the signal amplitude over time
        var_per_sub = np.var(w_denoised, axis=1)
        noise_level = np.mean(var_per_sub)
        noise_levels.append(noise_level)
        
        # Normalize for Inference
        w_norm = normalize_window(w_denoised)
        processed_windows.append(w_norm)
    
    processed_windows = np.stack(processed_windows)
    noise_levels = np.array(noise_levels)

    # 4. Inference
    feature_names_path = args.binary_dir / "feature_names.json"
    with open(feature_names_path) as f:
        feature_names = json.load(f)

    detector = MotionDetector(
        feature_names=feature_names,
        model_path=args.model_dir / "presence_detector_rf.joblib",
        scaler_path=args.model_dir / "presence_detector_scaler.joblib"
    )

    print("Running inference...")
    probs = detector.predict_proba_from_windows(processed_windows)[:, 1]
    preds = (probs > 0.5).astype(int)

    # 5. Visualization
    print("Generating plot...")
    
    # Concatenate windows for heatmap (first 20 or so if too long, or downsample?)
    # If session is long, heatmap might be huge. Let's limit to first 100 windows for clarity?
    # Or just plot everything.
    
    # Construct full heatmap from windows (overlapping)
    # A simple way is to just concat them, but that repeats data due to overlap.
    # For visualization, let's just concat them side-by-side (discontinuous time) or 
    # just plot the raw CSI if it's not too huge.
    # Let's plot the raw CSI (transposed) for the heatmap.
    
    heatmap_data = csi.T # (subcarriers, time)
    # Normalize heatmap for display
    heatmap_data = (heatmap_data - np.mean(heatmap_data)) / (np.std(heatmap_data) + 1e-6)
    
    # Time axis for predictions
    # Each window corresponds to a specific time range.
    # We'll plot predictions at the center of each window.
    fs = 100.0 # Hz (Assumption)
    window_time = args.T / fs
    stride_time = args.stride / fs
    
    time_points = np.arange(len(probs)) * stride_time + window_time / 2
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

    # Subplot 1: CSI Heatmap
    ax1 = fig.add_subplot(gs[0])
    # Downsample heatmap if too large
    if heatmap_data.shape[1] > 5000:
        heatmap_data_view = heatmap_data[:, ::5] # Downsample x5
        extent = [0, heatmap_data.shape[1]/fs, 0, 30]
    else:
        heatmap_data_view = heatmap_data
        extent = [0, heatmap_data.shape[1]/fs, 0, 30]
        
    im1 = ax1.imshow(heatmap_data_view, aspect='auto', origin='lower', 
                    cmap='viridis', extent=extent, interpolation='nearest')
    ax1.set_title('Raw CSI Signal (Heatmap)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Subcarriers', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Amplitude (z-score)')
    
    # Overlay predictions on heatmap
    # We can draw colored bars at the top
    for i, (p, prob) in enumerate(zip(preds, probs)):
        start_t = i * stride_time
        width = stride_time
        color = 'red' if p == 1 else 'blue'
        alpha = 0.3 if p == 1 else 0.1
        rect = Rectangle((start_t, 28), width, 2, facecolor=color, alpha=alpha)
        ax1.add_patch(rect)

    # Subplot 2: Noise Level
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(time_points, noise_levels, color='orange', linewidth=2, label='Signal Variance')
    ax2.set_title('Noise Level (Signal Variance)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Variance', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Subplot 3: Presence Probability
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(time_points, probs, color='green', linewidth=2, label='Presence Probability')
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
    ax3.fill_between(time_points, 0, probs, alpha=0.3, color='green')
    ax3.set_title('Human Presence Detection', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"✓ Session report saved to: {args.output}")

    return 0

if __name__ == "__main__":
    main()
