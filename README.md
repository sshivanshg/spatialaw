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
This will train the model on `data/processed/windows` and save it to `models/presence_detector_cnn.pth`.

## Data Layout
All data lives under `data/` (ignored by git).
```
data/
├── raw/WiAR/           # Original Dataset
└── processed/          # Generated Windows & Features
```

## Authors
- Rishabh (230178)
- Shivansh (230054)
Newton School of Technology — Computer Networks + AI/ML Capstone
