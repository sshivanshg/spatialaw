# Spatial Awareness through Ambient Wireless Signals

WiFi Channel State Information (CSI) can double as a privacy-preserving motion sensor.  
This repo turns the WiAR dataset (Intel 5300 CSI captures) into a binary **presence detector** with reproducible data pipelines, classical + deep models, and a Streamlit dashboard for live demos.

## Highlights
- Parses raw 802.11n CSI traces, producing fixed-length windows and (optionally) 14 handcrafted features.
- **Two complementary detectors**
  - **Random Forest** (feature-based, ~96 % accuracy) for quick baselines.
  - **CNN on windows** (~91 % accuracy) for end-to-end learning.
- **Streamlit Dashboard** (`app.py`) with model switcher, simulated live replay, and smoothing.
- **Notebook report** (`Spatial_Awareness_Project.ipynb`) covering data loading → preprocessing → synthetic data → training → evaluation.
- Synthetic “empty room” generation plus motion-score thresholding balances class 0 for both feature and window pipelines.

## Repository Tour
```
.
├── app.py                            # Streamlit dashboard
├── src/spatialaw/                    # Main package
│   ├── data/                         # Dataset utilities
│   ├── models/                       # Model implementations
│   ├── preprocessing/                # CSI data preprocessing
│   └── utils/                        # Helper utilities
├── training/                         # Model training scripts
│   ├── train_random_forest.py        # Feature-based baseline
│   ├── train_cnn.py                  # CNN training
│   ├── train_presence_cnn.py         # Advanced CNN training
│   ├── live_predict.py               # Live prediction
│   └── tune_presence_detector.py     # Hyperparameter tuning
├── scripts/                          # Data pipeline scripts
│   ├── data_preparation/             # Data prep (windowing, features, synthetic generation)
│   ├── visualization/                # Visualization utilities
│   └── run_pipeline.py               # End-to-end pipeline
├── config/                           # Configuration files
│   └── settings.py                   # Project settings
├── models/                           # Saved model artifacts (git-ignored)
├── data/                             # Raw + processed datasets (git-ignored)
├── docs/                             # Documentation & papers
│   ├── spatialaw_paper.tex           # LaTeX paper
│   ├── spatialaw_short_report.tex    # Short report
│   ├── VivaPrep.txt                  # Viva Q&A cheat sheet
│   └── original_README.md            # Original documentation
├── notebooks/                        # Jupyter notebooks
├── tests/                            # Unit tests
├── requirements.txt / setup.sh       # Environment helpers
└── Makefile                          # Convenience targets
```

> All raw/processed data lives in `data/` and is ignored by git. Regenerate using the scripts below if you clone the repo from scratch.

## Data Pipeline (CLI-Friendly)

1. **Download WiAR**
   ```bash
   bash scripts/data_preparation/fetch_wiar.sh   # or follow README in data/raw/WiAR
   ```
2. **Generate CSI windows**
   ```bash
   python scripts/data_preparation/generate_windows.py \
       --input-dir data/raw/WiAR \
       --out-dir data/processed/windows
   ```
3. **Extract features (optional, for RandomForest)**
   ```bash
   python scripts/data_preparation/extract_features.py \
       --windows-dir data/processed/windows \
       --output-dir data/processed/features
   python scripts/data_preparation/process_binary_dataset.py \
       --features-dir data/processed/features \
       --output-dir data/processed/binary
   ```
4. **Prepare raw-window presence splits (for CNN)**
   ```bash
   python scripts/data_preparation/generate_synthetic_low_motion_windows.py --count 1000
   python scripts/data_preparation/prepare_presence_windows.py \
       --windows-dir data/processed/windows \
       --output-dir data/processed/windows_binary \
       --extra-labels-csv data/processed/windows_binary/synthetic_windows.csv
   ```
   This creates `train.csv`, `val.csv`, `test.csv`, and `presence_windows_summary.json` under `data/processed/windows_binary/`.

5. **Validate (optional)**
   ```bash
   python scripts/data_preparation/validate_binary_dataset.py
   ```

## Training & Evaluation

| Model | Command | Inputs | Artifacts |
| --- | --- | --- | --- |
| RandomForestClassifier | `python training/train_random_forest.py` | `data/processed/windows_binary/all_windows.csv` via feature extraction | `models/presence_detector_rf.joblib`, `presence_detector_scaler.joblib`, `presence_detector_metrics.json`, plots under `models/` |
| CNN (raw windows) | `python training/train_cnn.py` (local helper) or `training/train_presence_cnn.py` for advanced CLI | `data/processed/windows_binary/train/val/test.csv` + window tensors | `models/presence_detector_cnn.pth`, `presence_detector_cnn_metrics.json` |

Both scripts print accuracy/precision/recall/F1/ROC-AUC and save metrics for reproducibility.

"source .venv/bin/activate   " 

## Streamlit Dashboard
```bash
streamlit run app.py
```
Prerequisites:
- `models/presence_detector_rf.joblib` + `presence_detector_scaler.joblib`
- `models/presence_detector_cnn.pth`
- `data/processed/windows_binary/train.csv` (or a folder of windows to replay)

Features:
- Model selector (RF vs CNN)
- Simulated live mode that replays recorded CSI windows
- Stability filter to smooth predictions
- Probability timeline & current decision indicator
- **Compatibility**: Requires input files in `.dat` (Intel 5300 binary) or `.npy` format.

## Using Custom Data
To test with your own recordings or new datasets:
1.  **Hardware Requirement**: Data **must** be captured using an **Intel 5300 NIC** with the [Linux 802.11n CSI Tool](http://dhalperi.github.io/linux-80211n-csitool/).
    *   *Why?* The model is trained on 30 subcarriers. Other cards (Atheros, ESP32, AX210) produce different shapes (56, 64, 256 subcarriers) and will not work.
2.  **Format**: Save the raw output as a `.dat` file.
3.  **Testing**:
    *   Open the Dashboard (`streamlit run app.py`).
    *   Drag & drop your `.dat` file.
    *   The app automatically handles windowing, partial denoising, and normalization.

## Notebook Workflow
- Jupyter notebooks in `notebooks/` directory mirror the CLI pipeline but in a single report: parsing `csiread`, windowing/feature extraction, synthetic empty-room generation, RandomForest training, and evaluation plots. Extend it with the CNN cells if you want a notebook-only submission.

## Data Layout
```
data/
├── raw/WiAR/                  # Original CSI recordings, MATLAB tools, videos, papers
└── processed/
    ├── windows/               # window_*.npy + labels.csv + summary
    ├── features/              # 14-feature matrices + labels
    ├── binary/                # Feature-based presence dataset
    └── windows_binary/        # Raw-window splits + synthetic metadata
```

## References
- WiAR dataset: Guo et al., IEEE Healthcom 2017 – “A Novel Benchmark on Human Activity Recognition Using WiFi Signals”
- Intel 5300 CSI Tool: <http://dhalperi.github.io/linux-80211n-csitool/>
- csiread library: <https://github.com/citywu/csiread>

## Authors
- Rishabh (230158)
- Shivansh (230054)
Newton School of Technology — Computer Networks + AI/ML Capstone


