# Spatial Awareness through Ambient Wireless Signals

WiFi Channel State Information (CSI) can double as a privacy-preserving motion sensor.  
This repo contains everything we used to turn the WiAR dataset (Intel 5300 CSI captures) into a binary **presence detector** with visualizations, an interactive dashboard, and a reproducible notebook.

## Highlights
- **Two Models**:
    - **Random Forest**: Fast, feature-based (96% accuracy).
    - **1D-CNN (Deep Learning)**: End-to-end learning on raw CSI (91%+ accuracy).
- **Interactive Dashboard**: A Streamlit app for live demonstrations, featuring real-time simulation and model switching.
- **End-to-End Notebook**: `Spatial_Awareness_Project.ipynb` walks through the entire pipeline (Data Loading -> Preprocessing -> Training -> Evaluation).
- **Synthetic Data**: Generates "Empty Room" samples to handle class imbalance.

## Repository Tour
```
.
├── Spatial_Awareness_Project.ipynb      # Main Project Notebook (Report)
├── app.py                               # Interactive Streamlit Dashboard
├── models/                              # Saved Models
│   ├── presence_detector_rf.joblib      # Random Forest Model
│   ├── presence_detector_scaler.joblib  # Scaler for RF
│   └── presence_detector_cnn.pth        # CNN Model (PyTorch)
├── model_tools/                         # Model Utilities
│   ├── train_cnn.py                     # Script to train/retrain the CNN
│   └── predict_from_raw.py              # CLI tool for single-file prediction
├── data/                                # Dataset (Git-ignored)
├── requirements.txt                     # Python dependencies
└── _archive/                            # Legacy/Helper scripts
```

## Getting Started

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard (Demo)
The dashboard allows you to visualize the system in action.
```bash
streamlit run app.py
```
**Features:**
- **Model Selector**: Switch between Random Forest and CNN.
- **Simulate Live Mode**: Replays a file as if it were a live stream.
- **Stability Filter**: Smooths predictions over time.

### 3. Run the Notebook
Open `Spatial_Awareness_Project.ipynb` in Jupyter to see the full training and evaluation report.

### 4. Train the CNN (Optional)
If you want to retrain the Deep Learning model:
```bash
python model_tools/train_cnn.py
```
# SpatialAw: Presence Detection via Ambient Wi‑Fi Signals

Human presence detection using Channel State Information (CSI) from commodity Wi‑Fi. This repo implements an end‑to‑end pipeline: ingest raw CSI, window and preprocess, extract robust statistical features, build a binary dataset (activity vs no‑activity), train a Random Forest detector, and visualize predictions.

Authors: Rishabh (230178) and Shivansh (230054) — Newton School of Technology

## What You Get
- End‑to‑end reproducible pipeline for Wi‑Fi sensing
- Multi‑format CSI loaders (`.dat`, `.mat`, `.csv`, `.txt`, `.npy`)
- Preprocessing: windowing, denoising, normalization
- Feature engineering: 14 CSI features + optional RSS features
- Binary dataset builder combining WiAR (activity) + wifi_csi_har (idle)
- Presence detector (Random Forest) with metrics & artifacts
- Visualizations: heatmaps, feature importance, ROC, confusion matrix
- Unit tests for loaders and dataset

## Project Layout

```
spatialaw/
├── scripts/
│   ├── fetch_wiar.sh                 # Download WiAR dataset
│   ├── generate_windows.py           # Create windows from raw CSI streams
│   ├── extract_features.py           # Compute 14‑feature vectors per window
│   └── process_binary_dataset.py     # Build activity/no‑activity dataset
├── src/
│   ├── preprocess/
│   │   ├── csi_loader.py             # Robust CSI file loader (2D amplitudes)
│   │   ├── dat_loader.py             # Intel 5300 (.dat) parser (via csiread)
│   │   ├── preprocess.py             # Windowing, moving‑average, z‑score
│   │   ├── features.py               # Feature extraction (FFT, Hilbert, etc.)
│   │   └── inspect_wiar.py           # Utilities to inspect WiAR recordings
│   └── train/
│       └── dataset.py                # PyTorch dataset helpers (future work)
├── model_tools/
│   ├── train_presence_detector.py    # Train, evaluate, save RF + scaler
│   ├── visualize_activity_heatmap.py # Heatmap with prediction overlay
│   ├── visualize_samples.py          # Random window viewer
│   └── view_data.py                  # Inspect `.npy` features/windows
├── models/                           # Saved models & scalers
├── data/
│   ├── raw/WiAR/                     # Raw WiAR dataset (via fetch_wiar.sh)
│   └── processed/
│       ├── windows/                  # Windowed CSI tensors
│       ├── features/                 # Extracted features + labels
│       └── binary/                   # Combined binary dataset (X, y)
├── tests/                            # Unit tests (PyTest)
├── requirements.txt                  # Python dependencies
└── README.md                         # This guide
```

## Quick Start (macOS/Linux)

```bash
# 1) Create & activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Fetch WiAR dataset
./scripts/fetch_wiar.sh

# 4) Generate windows from raw WiAR CSI
python scripts/generate_windows.py \
    --input-dir data/raw/WiAR \
    --out-dir data/processed/windows \
    --T 256 \
    --stride 64

# 5) Extract 14‑feature vectors from windows
python scripts/extract_features.py \
    --windows-dir data/processed/windows \
    --output-dir data/processed/features

# 6) Build binary dataset (WiAR activity + wifi_csi_har idle)
python scripts/process_binary_dataset.py \
    --wiar-features-dir data/processed/features \
    --out-dir data/processed/binary

# 7) Train & evaluate presence detector
python model_tools/train_presence_detector.py

# 8) Visualize predictions
python model_tools/visualize_activity_heatmap.py
```

## Pipeline Details

### Data Sources
- **WiAR** (activity): 16 motion/gesture classes treated as activity=1
- **wifi_csi_har_dataset** (idle): idle states (standing/sitting/lying/no_person) → activity=0

### Loading (`src/preprocess/csi_loader.py`)
- Supports `.npy`, `.mat`, `.csv`, `.txt`, `.dat` via `csiread`
- Outputs 2D amplitudes: `(n_packets, n_subcarriers)`; complex values → magnitudes

### Windowing & Preprocessing (`src/preprocess/preprocess.py`)
- Window shape: `(n_windows, n_subcarriers, T)` with `T=256`, `stride=64`
- Denoising: moving‑average (kernel=5)
- Normalization: per‑subcarrier z‑score per window
- Standardization: pads/truncates to fixed `T` and target subcarrier count (e.g., 30)

### Feature Engineering (`src/preprocess/features.py`)
Per window (CSI):
- Variance (mean/std/max)
- Hilbert envelope (mean/std)
- Shannon entropy (binned amplitudes)
- Velocity of change (|first derivative| mean/max)
- Median Absolute Deviation (mean/std)
- Motion period: dominant FFT frequency (mean/std)
- Normalized std (coefficient of variation)

Optional RSS features (if available): peaks, local variance, range. Features are serialized to vectors with stable ordering.

### Binary Dataset Builder (`scripts/process_binary_dataset.py`)
- Extracts features for wifi_csi_har idle windows; relabels all WiAR activities to 1
- Harmonizes feature dimensions and writes:
    - `data/processed/binary/features.npy` (matrix `N x F`)
    - `data/processed/binary/feature_names.json`
    - `data/processed/binary/labels.csv` (with metadata: source, dataset, original_label)
    - `data/processed/binary/binary_dataset_summary.json`

### Training & Evaluation (`model_tools/train_presence_detector.py`)
- Splits: stratified train/test (80/20), scales via `StandardScaler`
- Model: `RandomForestClassifier(n_estimators=150, class_weight='balanced')`
- Metrics: Accuracy, Precision, Recall, F1, ROC‑AUC + classification report
- Artifacts:
    - `models/presence_detector_rf.joblib`
    - `models/presence_detector_scaler.joblib`
    - `models/presence_detector_metrics.json`
- Visuals: confusion matrix, ROC curve, top‑10 feature importance, probability histograms

### Visualization
- `model_tools/visualize_activity_heatmap.py`: CSI heatmap with prediction overlay
- `model_tools/visualize_samples.py`: sample random windows for quick inspection
- `model_tools/view_data.py`: print dataset shapes and label distribution

## Running Tests

```bash
pytest -q
```

Key tests:
- `tests/test_csi_loader.py`: format coverage and 2D coercion
- `tests/test_dataset.py`: dataset shape/consistency checks

## Reproducibility & Configs
- Default seed: 42 (windowing/stratification)
- Window length `T=256`, stride `=64` (tune via CLI)
- Target subcarriers: 30 (pad/truncate if needed)
- All paths configurable via CLI args; artifacts written under `data/processed/` and `models/`

## Troubleshooting
- Virtualenv activation (macOS):
    - Create: `python3 -m venv .venv`
    - Activate: `source .venv/bin/activate`
- Intel 5300 `.dat` parsing:
    - Ensure `csiread` is installed; `dat_loader.py` is present under `src/preprocess/`
- Feature dimension mismatch during `process_binary_dataset.py`:
    - The script auto‑reorders features using `feature_names.json` (WiAR) to align datasets
- Empty windows:
    - If `n_packets < T`, adjust `--T` or aggregate longer sessions

## Requirements
- Python 3.8+
- NumPy, SciPy, Pandas, scikit‑learn
- matplotlib, seaborn
- csiread (for Intel 5300 `.dat` files)
- PyTorch (future work; not required for RF pipeline)

Install via:

```bash
pip install -r requirements.txt
```

## References
- WiAR: Guo, L., et al., “A Novel Benchmark on Human Activity Recognition Using WiFi Signals,” IEEE Healthcom, 2017
- Intel 5300 CSI Tool: http://dhalperi.github.io/linux-80211n-csitool/
- csiread: Python library for Intel 5300 CSI parsing

## License
Specify your licensing terms here.

## Acknowledgements
Project by Rishabh (230178) and Shivansh (230054), Newton School of Technology.
