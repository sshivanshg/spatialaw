import sys
from pathlib import Path
import numpy as np
import joblib
import json
import pandas as pd

# Setup path
ROOT = Path(__file__).resolve().parent
_ARCHIVE = ROOT / "_archive"
sys.path.insert(0, str(_ARCHIVE))

from src.preprocess.dat_loader import load_dat_file
from src.preprocess.preprocess import window_csi, denoise_window, normalize_window
from src.preprocess.features import extract_fusion_features, features_to_vector

def main():
    # 1. Load File
    path = ROOT / "data/raw/WiAR/distance_factor_activity_data/1_alldata/csi_a12_2.dat"
    print(f"Loading {path}...")
    try:
        csi = load_dat_file(path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"CSI Shape: {csi.shape}")

    # 2. Process like App
    if csi.shape[0] in [30, 60] and csi.shape[1] > 256:
        csi = csi.T
    
    windows = window_csi(csi, T=256, stride=64)
    print(f"Generated {len(windows)} windows")

    # 3. Load Model & Scaler
    model_dir = ROOT / "models"
    binary_dir = ROOT / "data/processed/binary"
    
    rf_model = joblib.load(model_dir / "presence_detector_rf.joblib")
    scaler = joblib.load(model_dir / "presence_detector_scaler.joblib")
    
    with open(binary_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    print(f"\nFeature Names: {feature_names}")
    print(f"Scaler Mean: {scaler.mean_}")
    print(f"Scaler Scale: {scaler.scale_}")

    # 4. Extract Features
    vectors = []
    for i, w in enumerate(windows):
        w_denoised = denoise_window(w)
        w_norm = normalize_window(w_denoised)
        w_norm = w_norm.astype(np.float32)  # Simulate .npy serialization
        
        feats = extract_fusion_features(w_norm)
        vec = features_to_vector(feats, feature_names)
        vectors.append(vec)
        
        if i == 0:
            print(f"\nWindow 0 Features:")
            for name, val, mean, scale in zip(feature_names, vec, scaler.mean_, scaler.scale_):
                z_score = (val - mean) / scale
                print(f"  {name}: {val:.4f} (Avg: {mean:.4f}, Z: {z_score:.2f})")

    X = np.stack(vectors)
    X_scaled = scaler.transform(X)
    
    probs = rf_model.predict_proba(X_scaled)[:, 1]
    print(f"\nPredictions (Prob Activity): {probs}")

if __name__ == "__main__":
    main()
