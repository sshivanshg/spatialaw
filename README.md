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

Human presence detection using Channel State Information (CSI) from commodity Wi‑Fi. This repo contains the complete pipeline: preprocessing, feature extraction, model training, evaluation, and documentation.

**Authors:** Rishabh (230158) and Shivansh (230054) — Newton School of Technology

## Project Layout

```
spatialaw/
├── paper/                       # Research documentation
│   ├── spatialaw_paper.tex      # Full research paper (LaTeX)
│   ├── spatialaw_paper.pdf      # Compiled research paper
│   └── (spatialaw_short_report) # Plain-language explainer (not in git)
├── model_tools/                 # Model training and visualization
│   ├── train_presence_detector.py  # Train RF/CNN presence detector
│   ├── visualize_samples.py     # Data visualization tools
│   ├── visualize_activity_heatmap.py  # Activity heatmap generation
│   └── html/                    # Generated reports and metrics
├── scripts/                     # Data preparation pipeline
│   ├── fetch_wiar.sh            # Download WiAR dataset
│   ├── process_binary_dataset.py  # Process .dat files
│   ├── generate_windows.py      # Sliding window generation
│   └── extract_features.py      # Feature extraction
├── src/                         # Core preprocessing modules
│   ├── preprocess/              # CSI loaders and preprocessing
│   │   ├── csi_loader.py        # Intel 5300 CSI parser
│   │   ├── dat_loader.py        # .dat file loader
│   │   ├── features.py          # Feature computation
│   │   └── preprocess.py        # Cleaning and normalization
│   └── models/                  # (Future) model definitions
├── models/                      # Saved models and scalers
│   ├── presence_detector_rf.joblib     # Random Forest model
│   └── presence_detector_scaler.joblib # Feature scaler
├── tests/                       # Unit tests
├── .texmf/                      # Local LaTeX packages (titlesec)
├── Makefile                     # Convenience commands
├── setup.sh                     # One-shot environment setup
├── requirements.txt             # Python dependencies
└── pyproject.toml               # Packaging metadata
```

## Quick Start

```bash
# 1) Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Prepare data (WiAR or custom Intel 5300 .dat files)
bash scripts/fetch_wiar.sh              # Download WiAR dataset (optional)
python scripts/process_binary_dataset.py --input <path-to-dat-files>

# 3) Generate windows and extract features
python scripts/generate_windows.py --window 256 --stride 64
python scripts/extract_features.py --features all

# 4) Train and evaluate models
python model_tools/train_presence_detector.py --model rf
python model_tools/train_presence_detector.py --model cnn

# 5) View results
ls models/                              # Saved models
ls model_tools/html/                    # Metrics, ROC curves, confusion matrices
```

## Pipeline Overview

### 1. Data Preparation
- **Input:** Intel 5300 CSI `.dat` files (WiAR dataset or custom recordings)
- **Processing:** Parse binary CSI data, extract amplitudes, handle antenna configurations
- **Output:** Processed CSI streams ready for windowing

### 2. Feature Extraction
We compute 14 physics-informed features per window:
- **Variability:** variance, standard deviation, MAD
- **Envelope:** Hilbert envelope statistics
- **Motion cues:** spectral entropy, velocity, motion period
- **Frequency:** dominant frequency from FFT

### 3. Model Training

**Random Forest (Recommended)**
- Feature-based classifier
- 150 trees, group-aware train/test split
- Accuracy: 92%, ROC-AUC: 0.96
- Fast inference: ~0.08 ms per window

**1D-CNN (Alternative)**
- End-to-end learning from raw CSI windows
- Slightly lower accuracy but learns patterns automatically

### 4. Evaluation
- Confusion matrices, ROC curves, precision-recall
- Group-aware splits prevent data leakage
- SMOTE for class balancing

## Documentation

### Research Paper
Full technical details in [`paper/spatialaw_paper.pdf`](paper/spatialaw_paper.pdf):
- Problem formulation and related work
- Dataset description and preprocessing
- Feature engineering rationale
- Model architecture and training
- Results and analysis

### Plain-Language Explainer
A simplified 4-page explainer is available locally (not tracked in git). To generate:
```bash
cd paper
TEXINPUTS="$PWD/../.texmf/tex/latex//:" pdflatex spatialaw_short_report.tex
```

## Environment & Commands

**Virtual Environment Setup:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Makefile Shortcuts:**
```bash
make setup          # Create .venv and install dependencies
make clean          # Remove caches and build artifacts
```

## Requirements
- Python 3.8+
- **Core:** NumPy, SciPy, Pandas, scikit-learn
- **Visualization:** matplotlib, seaborn
- **Deep Learning:** PyTorch (for CNN model)
- **LaTeX:** BasicTeX or similar (for compiling papers)

Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Key Features

✅ **Privacy-Preserving:** No cameras or microphones—only radio signals  
✅ **Device-Free:** No wearables or user intervention required  
✅ **Real-Time Ready:** Fast inference (~0.08 ms per window on laptop CPU)  
✅ **Well-Documented:** Research paper + plain-language explainer  
✅ **Reproducible:** Complete pipeline from raw data to trained models

## Applications
- **Energy Efficiency:** Smart HVAC/lighting control
- **Elder Care:** Non-intrusive presence monitoring
- **Security:** Intrusion detection without cameras
- **Space Utilization:** Occupancy analytics for offices

## Tests

Run unit tests:
```bash
pytest -q
```

Current coverage includes CSI loaders and dataset utilities.

## Limitations & Future Work

**Current Limitations:**
- Single lab environment (WiAR dataset)
- Binary labels derived from multi-class activities
- Single-person scenarios only

**Planned Improvements:**
- Multi-environment training for better generalization
- Transfer learning and domain adaptation
- Multi-person counting and tracking
- Hybrid models (hand-crafted + learned features)
- Self-supervised pretraining on unlabeled CSI

## Citation

If you use this work, please cite:
```
Rishabh & Shivansh (2024). SpatialAw: Device-Free Human Presence Detection
Using WiFi CSI. Newton School of Technology.
```

## License
[Specify your licensing terms]

## Acknowledgements
- **Authors:** Rishabh (230158) and Shivansh (230054)
- **Institution:** Newton School of Technology
- **Dataset:** WiAR (WiFi-based Activity Recognition benchmark)
