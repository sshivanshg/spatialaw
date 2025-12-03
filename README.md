## Spatial Awareness through Ambient Wireless Signals

Project by Rishabh (230178) and Shivansh (230054) – Newton School of Technology  
Course: Computer Networks + AI/ML

---

## Overview

This project is a **WiFi-based human activity recognition (HAR) system** built on top of the **WiAR** dataset.  
It uses **Channel State Information (CSI)** from standard **802.11n WiFi** to detect whether **human activity is present or not**.

- **Computer Networks focus**:  
  - Analyze **802.11n physical layer data** (CSI) from an Intel 5300 NIC  
  - Parse binary CSI `.dat` files and treat them as **network traffic traces**  
  - Study how the **wireless channel response** changes due to human motion (multipath)

- **AI/ML focus**:  
  - Convert CSI packet streams into fixed-length **windows**  
  - Extract **14 statistical features** from each window  
  - Train a **Random Forest binary classifier** for **activity vs no-activity**  
  - Visualize results using **CSI heatmaps, probabilities, and confusion matrices**

**Current status**: a working **binary presence detector** (activity vs no activity) with clear **network-signal visualizations** for presentations and demos.

---

## How this is a Computer Networks project

- Works directly with **802.11n** WiFi, using:
  - **OFDM subcarriers** (30 per channel)
  - **MIMO** configuration (multiple antennas)
  - **Physical layer (Layer 1)** channel estimates (CSI)
- Uses the **Intel 5300 CSI Tool** to extract CSI from WiFi frames:
  - Timestamp, RSSI, rate, antenna configuration, CSI matrix
  - Parsed using the `csiread` Python library
- Treats CSI as **network traffic**:
  - Each CSI entry corresponds to a WiFi packet
  - We perform **packet-level analysis** and **time-windowing** (like traffic analysis)
- Applies **network signal processing**:
  - Denoising and normalization (handling interference and level shifts)
  - Channel statistics analogous to **SNR / channel quality metrics**

You can describe it as:  
**“Repurposing 802.11n physical-layer protocol data (CSI) for ambient sensing.”**

---

## Project Structure

```text
spatialaw/
├── scripts/
│   ├── fetch_wiar.sh              # Download WiAR dataset
│   ├── generate_windows.py        # Process CSI recordings into windows
│   ├── extract_features.py        # Extract 14 statistical CSI features
│   └── process_binary_dataset.py  # Build binary (activity / no-activity) dataset
├── src/
│   ├── preprocess/                # Data loading & preprocessing
│   │   ├── csi_loader.py          # Load CSI files (.dat, .txt, .csv, .npy)
│   │   ├── dat_loader.py          # Intel 5300 .dat file parser (uses csiread)
│   │   ├── preprocess.py          # Windowing, denoising, normalization
│   │   ├── features.py            # Feature extraction (14 CSI features)
│   │   └── inspect_wiar.py        # Inspect WiAR recordings / metadata
│   ├── models/
│   │   ├── __init__.py            # Exposes motion detector helpers
│   │   └── motion_detector.py     # Runtime wrapper for trained detector
│   └── train/
│       └── dataset.py             # PyTorch Dataset for CSI windows (for future deep models)
├── data/                          # Not tracked in git (.gitignore)
│   ├── raw/
│   │   └── WiAR/                  # Cloned WiAR dataset (original CSI recordings)
│   └── processed/
│       ├── windows/               # Windowed CSI signals + labels
│       ├── features/              # 14‑dim feature vectors (multi-class WiAR)
│       └── binary/                # Binary presence dataset (activity / no-activity)
├── model_tools/
│   ├── train_presence_detector.py     # Train + evaluate Random Forest presence detector
│   ├── visualize_activity_heatmap.py  # CSI heatmap with prediction overlay
│   ├── visualize_samples.py           # Random CSI window viewer
│   └── view_data.py                   # Inspect .npy feature/window files
├── models/
│   ├── presence_detector_rf.joblib        # Trained Random Forest model
│   ├── presence_detector_scaler.joblib    # StandardScaler for features
│   └── presence_detector_metrics.json     # Saved metrics & config
├── tests/                           # Unit tests
└── logs/                            # Training logs and checkpoints
```

## Quick Start

### 1. Setup Environment

```bash
cd spatialaw

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download WiAR Dataset

```bash
# Clone WiAR dataset into data/raw/WiAR
./scripts/fetch_wiar.sh
```

> The `data/` folder is **git-ignored** to avoid committing large files.

### 3. Process raw CSI into windows and features

```bash
# 1) Generate fixed-length windows from raw CSI recordings
python scripts/generate_windows.py \
    --input-dir data/raw/WiAR \
    --out-dir data/processed/windows \
    --T 256 \
    --stride 64 \
    --seed 42

# 2) Extract 14 CSI features from each window (multi-class WiAR)
python scripts/extract_features.py \
    --windows-dir data/processed/windows \
    --output-dir data/processed/features
```

### 4. Build the binary presence dataset

```bash
python scripts/process_binary_dataset.py
```

This script:
- Loads **activity** features from WiAR
- Loads **no-activity** samples from an additional CSI dataset
- Creates `data/processed/binary/` with:
  - `features.npy` – feature matrix for binary classification
  - `labels.csv` – labels (`0` = no activity, `1` = activity)
  - `feature_names.json` – names of the 14 features

### 5. Validate the binary dataset

```bash
python scripts/validate_binary_dataset.py
```

This generates `data/processed/binary/validation_report.json` with label distributions, motion-score percentiles, per-activity breakdowns, and a leakage check for the GroupShuffleSplit configuration.

### 6. (Optional) Hyperparameter tuning

```bash
python model_tools/tune_presence_detector.py
```

This performs 5-fold GroupKFold cross-validation across several Random Forest configurations, saves the tuned model/scaler/pipeline under `models/`, and records leaderboard metrics in `models/tuning_results.json` plus `models/presence_detector_metrics.json`.

### 7. Train the presence detector (end-to-end)

```bash
python model_tools/train_presence_detector.py
```

This script:
- Loads binary dataset from `data/processed/binary/`
- Splits into **80% train / 20% test** (stratified)
- Scales features with `StandardScaler`
- Trains a **RandomForestClassifier** with `class_weight='balanced'`
- Prints metrics (accuracy, precision, recall, F1, ROC‑AUC)
- Shows:
  - Confusion matrix
  - ROC curve
  - Feature importance
  - Prediction probability distribution
- Saves:
  - `models/presence_detector_rf.joblib`
  - `models/presence_detector_scaler.joblib`
  - `models/presence_detector_metrics.json`

### 8. Visualize activity over time (heatmaps)

```bash
python model_tools/visualize_activity_heatmap.py
```

This script:
- Loads a CSI recording and its windows
- Applies the trained model to every window
- Plots:
  1. **CSI heatmap** (subcarriers × time) with activity overlay
  2. **Activity probability vs time**
  3. **Binary predictions vs time**

Perfect for **demo videos** or **class presentations**.

### 9. (Optional) Run the entire pipeline with one command

```bash
python scripts/run_pipeline.py
```

Use `--steps` to run a subset (e.g., `--steps validate tune`) and the other flags to override directories or hyperparameters.

---

---

## Dataset

### WiAR Dataset

- **Source**: [WiAR GitHub Repository](https://github.com/linteresa/WiAR)
- **Activities**: 16 human activities (gestures + body motions)
- **Format**: Intel 5300 CSI binary files (.dat) + text files (.txt)
- **Processed**: 2,092 windows from 1,932 recordings

### Activity Classes (16)

1. horizontal_arm_wave
2. high_arm_wave
3. two_hands_wave
4. high_throw
5. draw_x
6. draw_tick
7. toss_paper
8. forward_kick
9. side_kick
10. bend
11. hand_clap
12. walk
13. phone_call
14. drink_water
15. sit_down
16. squat

### Current Dataset Statistics (Multi-Class WiAR)

- **Total Windows**: 2,092  
- **Features per Window**: 14 CSI statistical features  
- **Feature Matrix**: `data/processed/features/features.npy` with shape `(2092, 14)`  
- **Labels**: `data/processed/features/labels.csv` (IDs 0–15 mapped to 16 activities)

### Binary Presence Dataset (Final Training Data)

Built by `scripts/process_binary_dataset.py` (WiAR-only):

- **Label 1 (Movement)**:
  - WiAR windows whose motion score (variance + velocity) is above a threshold
- **Label 0 (No / Low Movement)**:
  - WiAR windows in the bottom `motion_quantile` (default 25%) of motion scores

Outputs:
- `data/processed/binary/features.npy` – WiAR-only features with motion-leakage features removed
- `data/processed/binary/labels.csv` – derived binary labels with motion scores
- `data/processed/binary/feature_names.json`

---

---

## Data processing pipeline (end to end)

### 1. CSI Loading
- Supports multiple formats: `.dat` (Intel 5300), `.txt`, `.csv`, `.npy`
- Uses `csiread` library for binary .dat files
- Extracts CSI amplitudes: shape `(n_packets, 30_subcarriers)`

### 2. Windowing
- Converts continuous CSI streams to fixed-length windows
- Window size: T=256 time steps (or available packets)
- Stride: 64 (overlapping windows)
- Output: `(n_windows, 30_subcarriers, 256_timesteps)`

### 3. Preprocessing
- **Denoising**: Moving average filter
- **Normalization**: Per-window z-score normalization
- **Standardization**: Handles variable packet/subcarrier counts

### 4. Feature Extraction

Implemented in `src/preprocess/features.py` and used by `scripts/extract_features.py`.

From each window, it extracts 14 statistical CSI features, including:
- CSI variance (mean, std, max)
- Envelope statistics (Hilbert transform)
- Signal entropy
- Velocity of change (first derivative)
- Median Absolute Deviation (MAD)
- Motion period (dominant frequency via FFT)
- Normalized standard deviation

Result:
- `data/processed/features/features.npy` – `(n_windows, 14)`
- `data/processed/features/labels.csv` – multiclass WiAR labels

### 5. Binary Fusion & Model Training

1. **Binary fusion** – `scripts/process_binary_dataset.py`  
   Combines:
   - All **activity windows** → label `1`
   - **No-activity windows** → label `0`  
   and writes the final binary dataset under `data/processed/binary/`.

2. **Model training** – `model_tools/train_presence_detector.py`  
   - Loads binary features and labels  
   - Splits into train/test, scales features  
   - Trains **Random Forest** with class weighting  
   - Evaluates and saves model + metrics

---

---

## Scripts

### `scripts/generate_windows.py`
Process raw CSI recordings into fixed-length windows.

```bash
python scripts/generate_windows.py \
    --input-dir data/raw/WiAR \
    --out-dir data/processed/windows \
    --T 256 \
    --stride 64 \
    --seed 42
```

### `scripts/extract_features.py`
Extract 14 statistical CSI features from processed windows.

```bash
python scripts/extract_features.py \
    --windows-dir data/processed/windows \
    --output-dir data/processed/features
```

### `scripts/process_binary_dataset.py`
Build the final **binary presence dataset** (activity vs no-activity).

```bash
python scripts/process_binary_dataset.py
```

### `model_tools/*.py`
Python scripts (converted from notebooks) for training and visualization:

| Script | Purpose |
| --- | --- |
| `model_tools/train_presence_detector.py` | Train + evaluate the Random Forest presence detector |
| `model_tools/visualize_activity_heatmap.py` | Render CSI heatmap with prediction overlay |
| `model_tools/visualize_samples.py` | View random CSI windows as heatmaps |
| `model_tools/live_predict.py` | Stream CSI windows and print live prediction probabilities |
| `model_tools/view_data.py` | Print contents of `.npy` feature/window files |

Run training with:

```bash
python model_tools/train_presence_detector.py
```

---

---

## Requirements

- Python 3.8+
- **Core**: `numpy`, `scipy`, `pandas`
- **ML**: `scikit-learn`
- **Visualization**: `matplotlib`, `seaborn`
- **CSI parsing**: `csiread` (for Intel 5300 `.dat` files)
- **Optional**: `jupyterlab` (only if you want to recreate notebooks)

See `requirements.txt` for the complete list and versions.

---

---

## Data files (summary)

- **Windows**: `data/processed/windows/window_*.npy`  
- **Window labels**: `data/processed/windows/labels.csv`  
- **Multi-class features**: `data/processed/features/features.npy`  
- **Multi-class labels**: `data/processed/features/labels.csv`  
- **Binary features**: `data/processed/binary/features.npy`  
- **Binary labels**: `data/processed/binary/labels.csv`

## Quality reports & automation

- **Dataset validation**: `data/processed/binary/validation_report.json` captures motion-score stats, per-activity label splits, and confirms train/test source separation.
- **Model tuning leaderboard**: `models/tuning_results.json` logs every evaluated Random Forest configuration plus the selected winner; cross-validation means/std values are mirrored in `models/presence_detector_metrics.json`.
- **Saved artifacts**: `models/presence_detector_rf.joblib`, `presence_detector_scaler.joblib`, and the combined `presence_detector_pipeline.joblib`.
- **CI coverage**: `.github/workflows/ci.yml` installs dependencies and runs `pytest` (including the CLI integration tests) on every push/PR.
- **End-to-end CLI**: `scripts/run_pipeline.py` chains all preprocessing/training stages with reproducible arguments.

---

---

## References

- **WiAR Dataset**: Guo, L., et al. *"A Novel Benchmark on Human Activity Recognition Using WiFi Signals"* (IEEE Healthcom, 2017)
- **Intel 5300 CSI Tool**: <http://dhalperi.github.io/linux-80211n-csitool/>
- **csiread Library**: Python library for parsing Intel 5300 CSI files

---

---

## License

[Your License Here]

---

---

## Authors

- Rishabh (230178)  
- Shivansh (230054)

Newton School of Technology
