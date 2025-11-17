# Spatial Awareness through Ambient Wireless Signals

Project by Rishabh (230178) and Shivansh (230054) - Newton School of Technology

## Overview

WiFi-based human activity recognition system using Channel State Information (CSI) from the WiAR dataset. The system processes CSI signals to classify human activities in indoor environments.

**Current Focus**: Binary classification (activity present vs no activity) with visualization.

## Project Structure

```
spatialaw/
├── scripts/
│   ├── fetch_wiar.sh              # Download WiAR dataset
│   ├── generate_windows.py         # Process CSI recordings into windows
│   ├── extract_features.py         # Extract statistical features
│   └── train_motion_detector.py    # Train classifiers (legacy, to be updated)
├── src/
│   ├── preprocess/                 # Data preprocessing
│   │   ├── csi_loader.py          # Load CSI files (.dat, .txt, .csv, .npy)
│   │   ├── dat_loader.py          # Intel 5300 .dat file parser
│   │   ├── preprocess.py          # Windowing, denoising, normalization
│   │   ├── features.py            # Feature extraction (14 CSI features)
│   │   └── inspect_wiar.py        # Dataset inspection utility
│   └── train/
│       └── dataset.py             # PyTorch Dataset for CSI windows
├── data/
│   ├── raw/WiAR/                  # WiAR dataset (cloned from GitHub)
│   └── processed/
│       ├── windows/               # Processed CSI windows
│       └── features/              # Extracted features
├── notebooks/
│   └── visualize_samples.ipynb    # Visualize CSI windows
├── tests/                          # Unit tests
└── logs/                          # Training logs and checkpoints
```

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download WiAR Dataset

```bash
# Clone WiAR dataset
./scripts/fetch_wiar.sh
```

### 3. Process Dataset

```bash
# Generate windows from raw CSI recordings
python scripts/generate_windows.py \
    --input-dir data/raw/WiAR \
    --out-dir data/processed/windows \
    --T 256 \
    --stride 64

# Extract features from windows
python scripts/extract_features.py \
    --windows-dir data/processed/windows \
    --output-dir data/processed/features
```

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

### Current Dataset Statistics

- **Total Windows**: 2,092
- **Features per Window**: 14 CSI statistical features
- **Feature Matrix**: `(2092, 14)` saved as `data/processed/features/features.npy`
- **Labels**: Activity IDs 0-15 (0-indexed) mapped to 16 activities

## Data Processing Pipeline

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
Extracts 14 statistical features from each window:
- CSI variance (mean, std, max)
- CSI envelope (Hilbert transform)
- Signal entropy
- Velocity of change (first derivative)
- Median Absolute Deviation (MAD)
- Motion period (dominant frequency via FFT)
- Normalized standard deviation

## Current Task: Binary Classification

**Goal**: Classify "activity present" (1) vs "no activity" (0)

### Next Steps

1. **Relabel Data**: Convert 16-class labels → binary (all activities → 1)
2. **Create "No Activity" Samples**: Generate baseline/idle samples (label → 0)
3. **Train Binary Classifier**: SVM, Random Forest, Logistic Regression
4. **Visualization**: 
   - Confusion matrix
   - ROC curve
   - Time-series plot with activity regions highlighted
   - Confidence scores

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
Extract statistical features from processed windows.

```bash
python scripts/extract_features.py \
    --windows-dir data/processed/windows \
    --output-dir data/processed/features
```

### `scripts/inspect_wiar.py`
Inspect WiAR dataset structure and sample files.

```bash
python src/preprocess/inspect_wiar.py
```

## Requirements

- Python 3.8+
- NumPy, SciPy, Pandas
- scikit-learn
- PyTorch, torchvision
- matplotlib, seaborn
- csiread (for Intel 5300 .dat files)
- jupyterlab (for notebooks)

See `requirements.txt` for full list.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Data Files

### Processed Data Location

- **Windows**: `data/processed/windows/window_*.npy`
- **Labels**: `data/processed/windows/labels.csv`
- **Features**: `data/processed/features/features.npy`
- **Feature Labels**: `data/processed/features/labels.csv`

## References

- **WiAR Dataset**: Guo, L., et al. "A Novel Benchmark on Human Activity Recognition Using WiFi Signals" (IEEE Healthcom, 2017)
- **Intel 5300 CSI Tool**: [dhalperi.github.io/linux-80211n-csitool](http://dhalperi.github.io/linux-80211n-csitool/)
- **csiread Library**: Python library for parsing Intel 5300 CSI files

## License

[Your License Here]

## Authors

- Rishabh (230178)
- Shivansh (230054)

Newton School of Technology
