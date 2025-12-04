#!/usr/bin/env python
# coding: utf-8

# # Visualize Activity Detection with Heatmap
# 
# This notebook creates a heatmap visualization showing:
# 1. **CSI Amplitude Heatmap**: Raw CSI signal over time (subcarriers × time)
# 2. **Activity Probability Overlay**: Model predictions overlaid on the heatmap
# 3. **Time-Series Activity Plot**: Binary predictions and probabilities over time
# 
# This helps visualize where and when human activity is detected in the CSI signal.
# 

# In[8]:


# Load model and data
from pathlib import Path
import numpy as np
import pandas as pd
import json
import joblib

# Find paths
root_candidates = [Path().resolve(), Path().resolve().parent]
BINARY_DIR = None
WINDOWS_DIR = None
MODELS_DIR = None

for root in root_candidates:
    if (root / "data" / "processed" / "binary").exists():
        BINARY_DIR = root / "data" / "processed" / "binary"
    if (root / "data" / "processed" / "windows").exists():
        WINDOWS_DIR = root / "data" / "processed" / "windows"
    if (root / "models").exists():
        MODELS_DIR = root / "models"

if BINARY_DIR is None or WINDOWS_DIR is None:
    raise FileNotFoundError("Could not locate data directories")

# Load model
if MODELS_DIR and (MODELS_DIR / "presence_detector_rf.joblib").exists():
    model = joblib.load(MODELS_DIR / "presence_detector_rf.joblib")
    scaler = joblib.load(MODELS_DIR / "presence_detector_scaler.joblib")
    print("✓ Model loaded")
else:
    print("⚠️  Model not found. Train the model first using train_presence_detector.ipynb")
    model = None
    scaler = None


# In[9]:


# Load a sample recording to visualize
import sys
from pathlib import Path
import os

# Add project root to path (more robust detection)
current_dir = Path().resolve()
parent_dir = current_dir.parent

# Check if we're in notebooks folder or project root
if (current_dir / "src").exists():
    project_root = current_dir
elif (parent_dir / "src").exists():
    project_root = parent_dir
else:
    # Fallback: try to find project root by looking for src directory
    project_root = None
    for path in [current_dir, parent_dir, parent_dir.parent]:
        if (path / "src").exists():
            project_root = path
            break

if project_root and str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"✓ Added project root to path: {project_root}")

# Now import
try:
    from src.preprocess.features import extract_fusion_features, features_to_vector
    print("✓ Successfully imported features module")
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}")
    raise

# Load windows and labels
windows_labels = pd.read_csv(WINDOWS_DIR / "labels.csv")
binary_labels = pd.read_csv(BINARY_DIR / "labels.csv")

# Get WiAR samples (they have activity and matching windows)
wiar_samples = binary_labels[binary_labels["dataset"] == "wiar"]
if len(wiar_samples) > 0:
    # Get first WiAR sample
    sample_row = wiar_samples.iloc[0]
    print(f"Visualizing WiAR sample: {sample_row.get('source', 'unknown')}")

    # Find corresponding window file by matching source_recording
    if "source" in sample_row:
        # Try exact match first
        matching_windows = windows_labels[
            windows_labels["source_recording"] == sample_row["source"]
        ]

        # If no exact match, try partial match
        if len(matching_windows) == 0:
            source_parts = str(sample_row["source"]).split("/")
            if len(source_parts) > 0:
                last_part = source_parts[-1]
                matching_windows = windows_labels[
                    windows_labels["source_recording"].str.contains(last_part, na=False, regex=False)
                ]

        if len(matching_windows) > 0:
            window_file = WINDOWS_DIR / matching_windows.iloc[0]["window_file"]
            if window_file.exists():
                window_data = np.load(window_file)
                print(f"✓ Loaded window: {window_file.name}, shape: {window_data.shape}")

                # Get prediction for this sample
                if model is not None:
                    features = np.load(BINARY_DIR / "features.npy")
                    # Find the index in binary_labels that matches this sample
                    sample_idx = wiar_samples.index[0]
                    if sample_idx < len(features):
                        X_sample = features[sample_idx:sample_idx+1]
                        X_scaled = scaler.transform(X_sample)
                        proba = model.predict_proba(X_scaled)[0, 1]
                        pred = model.predict(X_scaled)[0]
                        print(f"Prediction: {pred} (Activity={'Yes' if pred==1 else 'No'})")
                        print(f"Probability: {proba:.3f}")
            else:
                print(f"⚠️  Window file not found: {window_file}")
                window_data = None
        else:
            print("⚠️  Could not find matching window file")
            window_data = None
    else:
        window_data = None
else:
    print("No WiAR samples found")
    window_data = None


# In[10]:


# Create comprehensive visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

if window_data is not None and model is not None:
    # Get predictions for multiple windows from the same source
    source_name = str(sample_row.get("source", "unknown"))

    # Find all windows from the same source recording
    matching_windows_all = windows_labels[
        windows_labels["source_recording"] == source_name
    ]

    # If no exact match, try partial
    if len(matching_windows_all) == 0:
        source_parts = source_name.split("/")
        if len(source_parts) > 0:
            last_part = source_parts[-1]
            matching_windows_all = windows_labels[
                windows_labels["source_recording"].str.contains(last_part, na=False, regex=False)
            ]

    # Get first 20 windows from this source
    matching_windows_subset = matching_windows_all.head(20)

    if len(matching_windows_subset) > 0:
        # Get corresponding binary label indices
        matching_indices = []
        for _, win_row in matching_windows_subset.iterrows():
            # Find matching binary label by source
            matching_binary = binary_labels[
                binary_labels["source"] == win_row["source_recording"]
            ]
            if len(matching_binary) > 0:
                matching_indices.append(matching_binary.index[0])

        if len(matching_indices) == 0:
            # Fallback: use first 20 WiAR samples
            matching_indices = wiar_samples.head(20).index.tolist()

    if len(matching_indices) > 0:
        # Load features and get predictions
        features = np.load(BINARY_DIR / "features.npy")
        X_batch = features[matching_indices]
        X_batch_scaled = scaler.transform(X_batch)
        predictions = model.predict(X_batch_scaled)
        probabilities = model.predict_proba(X_batch_scaled)[:, 1]

        # Load corresponding windows
        window_files = []
        window_arrays = []
        for idx in matching_indices:
            row = binary_labels.iloc[idx]
            if "source" in row:
                matching = windows_labels[
                    windows_labels.get("source_recording", "").str.contains(
                        str(row["source"]).split("/")[-1], na=False
                    )
                ]
                if len(matching) > 0:
                    wf = WINDOWS_DIR / matching.iloc[0]["window_file"]
                    if wf.exists():
                        window_files.append(wf)
                        window_arrays.append(np.load(wf))

        if len(window_arrays) > 0:
            # Concatenate windows horizontally to create a time-series
            # Each window is (subcarriers, time_steps)
            combined_heatmap = np.hstack(window_arrays)  # (subcarriers, total_time)

            # Create figure with subplots
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

            # 1. CSI Amplitude Heatmap
            ax1 = fig.add_subplot(gs[0])
            im1 = ax1.imshow(combined_heatmap, aspect='auto', origin='lower', 
                            cmap='viridis', interpolation='nearest')
            ax1.set_xlabel('Time (packet index)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Subcarrier Index', fontsize=12, fontweight='bold')
            ax1.set_title('CSI Amplitude Heatmap (Normalized)', fontsize=14, fontweight='bold')
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Amplitude (z-score)', fontsize=10)

            # Overlay activity predictions as colored bars at the top
            window_size = window_arrays[0].shape[1]  # Time dimension of each window
            for i, (pred, prob) in enumerate(zip(predictions[:len(window_arrays)], 
                                                  probabilities[:len(window_arrays)])):
                x_start = i * window_size
                x_end = (i + 1) * window_size
                color = 'red' if pred == 1 else 'blue'
                alpha = prob if pred == 1 else (1 - prob)
                rect = Rectangle((x_start, combined_heatmap.shape[0]), 
                               window_size, combined_heatmap.shape[0] * 0.1,
                               facecolor=color, alpha=alpha*0.5, edgecolor='black', linewidth=1)
                ax1.add_patch(rect)

            # 2. Activity Probability Over Time
            ax2 = fig.add_subplot(gs[1])
            time_points = np.arange(len(probabilities[:len(window_arrays)])) * window_size + window_size // 2
            ax2.plot(time_points, probabilities[:len(window_arrays)], 
                    'o-', linewidth=2, markersize=6, color='green', label='Activity Probability')
            ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Decision Threshold (0.5)')
            ax2.fill_between(time_points, 0, probabilities[:len(window_arrays)], 
                           alpha=0.3, color='green', where=(probabilities[:len(window_arrays)] > 0.5))
            ax2.fill_between(time_points, 0, probabilities[:len(window_arrays)], 
                           alpha=0.3, color='red', where=(probabilities[:len(window_arrays)] <= 0.5))
            ax2.set_ylabel('Probability', fontsize=12, fontweight='bold')
            ax2.set_title('Activity Detection Probability Over Time', fontsize=14, fontweight='bold')
            ax2.set_ylim([0, 1])
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. Binary Predictions Over Time
            ax3 = fig.add_subplot(gs[2])
            ax3.step(time_points, predictions[:len(window_arrays)], 
                    where='mid', linewidth=2, color='darkblue', label='Predicted Activity')
            ax3.fill_between(time_points, 0, predictions[:len(window_arrays)], 
                            alpha=0.4, color='green', step='mid')
            ax3.set_ylabel('Activity (0=No, 1=Yes)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Time (packet index)', fontsize=12, fontweight='bold')
            ax3.set_title('Binary Activity Predictions Over Time', fontsize=14, fontweight='bold')
            ax3.set_ylim([-0.1, 1.1])
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(['No Activity', 'Activity'])
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.suptitle(f'Activity Detection Visualization: {source_name}', 
                        fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plot_path = MODELS_DIR / "activity_heatmap.png"
            plt.savefig(plot_path)
            print(f"✓ Visualization saved to: {plot_path}")
            # plt.show()
        else:
            print("⚠️  Could not load window files for visualization")
    else:
        print("⚠️  No matching windows found for visualization")
else:
    print("⚠️  Missing data or model. Please ensure:")
    print("   1. Model is trained (run train_presence_detector.ipynb)")
    print("   2. Window data is available in data/processed/windows/")

