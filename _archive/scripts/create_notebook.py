import json
import re
from pathlib import Path

def read_and_clean(filepath):
    """Reads a file and removes import lines."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        # Remove local imports (from src...)
        if line.strip().startswith("from src.") or line.strip().startswith("import src."):
            continue
        # Remove standard imports (we'll add them to the top cell)
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            # Keep some specific imports if needed, but generally safe to remove if we consolidate
            # Let's be aggressive and remove all, then add back what we need in the first cell.
            continue
        cleaned_lines.append(line)
    return "".join(cleaned_lines)

def create_notebook():
    cells = []

    # 1. Header (Markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Spatial Awareness through Ambient Wireless Signals\n",
            "\n",
            "**Authors:** Rishabh (230178), Shivansh (230054)  \n",
            "**Institution:** Newton School of Technology  \n",
            "\n",
            "## Abstract\n",
            "This project implements a privacy-preserving presence detection system using WiFi Channel State Information (CSI). ",
            "It leverages synthetic data generation to address the lack of 'empty room' samples in public datasets and uses a Random Forest classifier to achieve robust detection."
        ]
    })

    # 2. Imports (Code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import os\n",
            "import json\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from pathlib import Path\n",
            "from scipy.signal import butter, lfilter, hilbert\n",
            "from scipy.stats import entropy, median_abs_deviation, zscore\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.model_selection import GroupShuffleSplit, cross_val_score\n",
            "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from imblearn.over_sampling import SMOTE\n",
            "import joblib\n",
            "import csiread\n",
            "\n",
            "%matplotlib inline"
        ]
    })

    # 3. Data Loading (Code)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Data Loading and Preprocessing\n", "Functions to load raw `.dat` files and preprocess the CSI signals."]
    })
    
    dat_loader_code = read_and_clean("src/preprocess/dat_loader.py")
    preprocess_code = read_and_clean("src/preprocess/preprocess.py")
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [dat_loader_code + "\n\n" + preprocess_code]
    })

    # 4. Feature Extraction (Code)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Feature Extraction\n", "Extracting 14 statistical features from CSI windows."]
    })
    
    features_code = read_and_clean("src/preprocess/features.py")
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [features_code]
    })

    # 5. Synthetic Data (Code)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Synthetic Data Generation\n", "Generating synthetic 'Empty Room' data to solve class imbalance."]
    })
    
    # We need to adapt the script to be a function
    synth_code = read_and_clean("scripts/generate_synthetic_empty.py")
    # Wrap it in a function if it's not already (it has a main, but we want the logic)
    # Actually, let's just extract the generation logic.
    # For now, just pasting the code is fine, but we need to make sure it runs.
    # The script uses argparse. We should replace main() with a function `generate_synthetic_data()`.
    
    # Let's manually write a clean version of the synthetic generator for the notebook
    synth_clean = """
def generate_synthetic_data(n_samples=500, n_subcarriers=30):
    print(f"Generating {n_samples} synthetic empty samples...")
    
    # Feature names (must match what we use in extraction)
    feature_names = [
        "csi_variance_max", "csi_variance_std",
        "csi_velocity_max",
        "csi_entropy",
        "csi_envelope_mean", "csi_envelope_std",
        "csi_mad_mean", "csi_mad_std",
        "csi_motion_period_mean", "csi_motion_period_std",
        "csi_norm_std_mean", "csi_norm_std_std"
    ]
    
    n_features = len(feature_names)
    X_synth = np.zeros((n_samples, n_features))
    feat_idx = {name: i for i, name in enumerate(feature_names)}
    
    # 1. Entropy: Uniform(3.0, 5.0)
    if "csi_entropy" in feat_idx:
        X_synth[:, feat_idx["csi_entropy"]] = np.random.uniform(3.0, 5.0, size=n_samples)

    # 2. Envelope Mean:
    if "csi_envelope_mean" in feat_idx:
        X_synth[:, feat_idx["csi_envelope_mean"]] = np.random.uniform(0.5, 1.5, size=n_samples)

    # 3. Envelope Std:
    if "csi_envelope_std" in feat_idx:
        X_synth[:, feat_idx["csi_envelope_std"]] = np.random.uniform(0.1, 0.5, size=n_samples)

    # 4. MAD Mean:
    if "csi_mad_mean" in feat_idx:
        X_synth[:, feat_idx["csi_mad_mean"]] = np.random.uniform(0.05, 0.25, size=n_samples)

    # 5. MAD Std:
    if "csi_mad_std" in feat_idx:
        X_synth[:, feat_idx["csi_mad_std"]] = np.random.uniform(0.02, 0.15, size=n_samples)

    # 6. Motion Period:
    if "csi_motion_period_mean" in feat_idx:
        X_synth[:, feat_idx["csi_motion_period_mean"]] = np.random.uniform(0, 15, size=n_samples)
    
    if "csi_motion_period_std" in feat_idx:
        X_synth[:, feat_idx["csi_motion_period_std"]] = np.random.uniform(0, 3, size=n_samples)

    # 7. Norm Std:
    if "csi_norm_std_mean" in feat_idx:
        X_synth[:, feat_idx["csi_norm_std_mean"]] = np.random.uniform(0.1, 0.5, size=n_samples)
    
    if "csi_norm_std_std" in feat_idx:
        X_synth[:, feat_idx["csi_norm_std_std"]] = np.random.uniform(0.05, 0.2, size=n_samples)

    # 8. Variance:
    if "csi_variance_max" in feat_idx:
        X_synth[:, feat_idx["csi_variance_max"]] = np.random.uniform(0.5, 1.5, size=n_samples)
    
    if "csi_variance_std" in feat_idx:
        X_synth[:, feat_idx["csi_variance_std"]] = np.random.uniform(0.1, 0.4, size=n_samples)
    
    # 9. Velocity:
    if "csi_velocity_max" in feat_idx:
        X_synth[:, feat_idx["csi_velocity_max"]] = np.random.uniform(0.1, 0.6, size=n_samples)
        
    return X_synth, feature_names
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [synth_clean]
    })

    # 6. Model Training (Code)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 4. Model Training\n", "Loading data, training Random Forest, and evaluating performance."]
    })
    
    training_code = """
# Load WiAR Data (assuming it's already processed into features, or we load raw?)
# For simplicity, let's assume the user has run the scripts and we load the PROCESSED data.
# OR, we can load the raw features from `data/processed/wiar/features.npy` if it exists.

# Let's try to load the binary dataset we created.
try:
    X = np.load("data/processed/binary/features.npy")
    y = pd.read_csv("data/processed/binary/labels.csv")['label'].values
    with open("data/processed/binary/feature_names.json", 'r') as f:
        feature_names = json.load(f)
    print("Loaded existing binary dataset.")
except:
    print("Binary dataset not found. Please run the data processing scripts first or ensure data is in `data/processed/binary`.")
    # Fallback: Generate synthetic only for demo
    X, feature_names = generate_synthetic_data()
    y = np.zeros(len(X))
    print("Generated synthetic data only (Demo Mode).")

# Split Data
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# We need groups. If we loaded from disk, we might not have groups easily available unless we saved them.
# Let's assume random split for now if groups are missing.
train_idx, test_idx = next(splitter.split(X, y, groups=np.arange(len(X)))) # Dummy groups

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
clf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [training_code]
    })

    # 7. Live Visualization (Code)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 5. Live Session Visualization\n", "Visualizing a raw recording with the trained model."]
    })
    
    # We need the MotionDetector class for this.
    # Let's define a simplified MotionDetector class here or just use the code inline.
    # We can copy the class definition from src/models/motion_detector.py
    
    motion_detector_code = read_and_clean("src/models/motion_detector.py")
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [motion_detector_code]
    })
    
    viz_code = """
def visualize_session(file_path):
    if not Path(file_path).exists():
        print(f"File {file_path} not found.")
        return

    print(f"Loading {file_path}...")
    if str(file_path).endswith(".npy"):
        csi = np.load(file_path)
    else:
        csi = load_dat_file(file_path)
        
    # Windowing
    windows = window_csi(csi, T=256, stride=64)
    
    # Process
    processed_windows = []
    noise_levels = []
    
    for w in windows:
        w_denoised = denoise_window(w)
        var_per_sub = np.var(w_denoised, axis=1)
        noise_levels.append(np.mean(var_per_sub))
        processed_windows.append(normalize_window(w_denoised))
        
    processed_windows = np.stack(processed_windows)
    
    # Predict
    # We need to use the trained 'clf' and 'scaler' from the previous cell
    # We can manually extract features here since we have the functions
    
    # Extract features for all windows
    X_feats = []
    for w in processed_windows:
        f = extract_csi_features(w)
        # Convert dict to vector using feature_names order
        vec = [f.get(name, 0) for name in feature_names]
        X_feats.append(vec)
    X_feats = np.array(X_feats)
    
    X_feats_scaled = scaler.transform(X_feats)
    probs = clf.predict_proba(X_feats_scaled)[:, 1]
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    # 1. Heatmap
    plt.subplot(3, 1, 1)
    plt.imshow(csi.T, aspect='auto', cmap='viridis', origin='lower')
    plt.title("Raw CSI Heatmap")
    plt.ylabel("Subcarrier")
    
    # 2. Noise
    plt.subplot(3, 1, 2)
    plt.plot(noise_levels, color='orange')
    plt.title("Signal Variance (Noise Level)")
    plt.grid(True, alpha=0.3)
    
    # 3. Probability
    plt.subplot(3, 1, 3)
    plt.plot(probs, color='green')
    plt.axhline(0.5, color='red', linestyle='--')
    plt.title("Presence Probability")
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example usage (commented out)
# visualize_session("data/raw/test_session.dat")
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [viz_code]
    })

    # Save Notebook
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open("Spatial_Awareness_Project.ipynb", "w") as f:
        json.dump(notebook, f, indent=2)
    
    print("Notebook created: Spatial_Awareness_Project.ipynb")

if __name__ == "__main__":
    create_notebook()
