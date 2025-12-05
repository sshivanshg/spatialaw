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

Human presence detection using Channel State Information (CSI) from commodity Wi‑Fi. This repo currently focuses on training models from prepared datasets and saving/evaluating results.

Authors: Rishabh (230178) and Shivansh (230054) — Newton School of Technology

## Project Layout (current)

```
spatialaw/
├── model_tools/
│   ├── train_random_forest.py   # Train & evaluate Random Forest presence detector
│   └── train_cnn.py             # Train a CNN model (if dataset and code configured)
├── models/                      # Saved models, scalers, and metrics
├── tests/                       # Unit tests (to be expanded)
├── Makefile                     # Convenience commands (uses .venv)
├── setup.sh                     # One‑shot environment setup (uses .venv)
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Packaging metadata & deps
└── app.py                       # (Optional) entrypoint/demo, not used in training
```

Note: Documentation in older versions referenced `scripts/` and `src/` pipelines. These are not present in this workspace snapshot. The README is aligned to existing files.

## Quick Start (macOS/Linux)

```bash
# 1) Create & activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Train the Random Forest detector
python model_tools/train_random_forest.py

# 4) Optional: Train CNN model
python model_tools/train_cnn.py
```

Artifacts are written to the `models/` directory (model files, scaler, metrics JSON), if implemented by the training scripts.

## Training & Evaluation

### Random Forest (`model_tools/train_random_forest.py`)
- Recommended: fit `StandardScaler` on train split only; evaluate on test split
- Metrics: Accuracy, Precision, Recall, F1, ROC‑AUC; save a `metrics.json`
- Artifacts: `models/` directory for `.joblib` model and scaler (if used)

### CNN (`model_tools/train_cnn.py`)
- Ensure dataset paths and loaders are configured inside the script
- Consider early stopping, learning rate scheduling, and saving best checkpoints

## Environment & Commands

Use `.venv` consistently:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Makefile shortcuts:

```bash
make setup      # create .venv and install deps
make train-rf   # run Random Forest training
make train-cnn  # run CNN training
make clean      # remove caches
```

## Requirements
- Python 3.8+
- NumPy, SciPy, Pandas, scikit‑learn
- matplotlib, seaborn
- torch/torchvision (for CNN)

Install via:

```bash
pip install -r requirements.txt
```

## Tests

Run tests (expand as needed):

```bash
pytest -q
```

## Notes & Next Steps
- If you re‑introduce preprocessing and feature extraction modules (`src/`, `scripts/`), update this README and Makefile accordingly.
- Consider reducing dependency footprint (e.g., make torch optional if only RF is used).
- Standardize saved artifacts: `models/model.joblib`, `models/scaler.joblib`, `models/metrics.json`.

## License
Specify your licensing terms here.

## Acknowledgements
Project by Rishabh (230178) and Shivansh (230054), Newton School of Technology.
