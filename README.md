# Spatial Awareness through Ambient Wireless Signals - Phase-1 Baseline

Project by Rishabh (230178) and Shivansh (230054) - Newton School of Technology

## Overview

Phase-1 baseline implementation for WiFi signal spatial mapping. This project creates WiFi signal heatmaps and builds simple machine learning models to predict signal strength across indoor positions.

## Project Structure

```
spatialaw/
├── notebooks/
│   └── baseline_analysis.ipynb    # Jupyter notebook with analysis pipeline
├── scripts/
│   ├── generate_synthetic_wifi_data.py  # Generate synthetic WiFi data
│   ├── train_baseline_model.py          # Train baseline models
│   ├── combine_multi_device_data.py     # Combine data from multiple devices
│   ├── validate_collected_data.py       # Validate collected data
│   ├── collect_time_series_data.py      # Collect time-series data for motion detection
│   ├── train_motion_detector.py         # Train motion detection models
│   ├── visualize_motion_detection.py    # Visualize motion detection results
│   └── generate_synthetic_motion_data.py # Generate synthetic motion data
├── src/
│   ├── data_collection/          # WiFi data collection utilities
│   ├── preprocessing/            # Data preprocessing pipelines
│   │   └── time_series_features.py  # Time-series feature extraction
│   ├── models/                   # Model definitions
│   │   └── motion_detector.py    # Motion detection models
│   └── training/                 # Training utilities
├── data/                         # Dataset storage
├── visualizations/               # Visualization outputs
├── checkpoints/                  # Model checkpoints
└── configs/                      # Configuration files
```

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
# Generate synthetic WiFi data (200 samples)
python scripts/generate_synthetic_wifi_data.py \
    --num_samples 200 \
    --room_width 10.0 \
    --room_height 8.0 \
    --output data/synthetic_wifi_data.json
```

### 3. Run Baseline Analysis

**Option A: Jupyter Notebook (Recommended)**
```bash
# Start Jupyter notebook
jupyter notebook notebooks/baseline_analysis.ipynb
```

**Option B: Command Line**
```bash
# Train baseline model
python scripts/train_baseline_model.py \
    --data_path data/synthetic_wifi_data.json \
    --model_type random_forest \
    --predict_signal
```

### 4. View Visualizations

Check the `visualizations/` directory for:
- Position vs signal strength scatter plots
- Signal distribution histograms
- Correlation matrices
- PCA visualizations
- Model prediction plots

## Detailed Workflow

### Step 1: Generate Synthetic Data

```bash
python scripts/generate_synthetic_wifi_data.py \
    --num_samples 200 \
    --room_width 10.0 \
    --room_height 8.0 \
    --num_aps 3 \
    --noise_level 5.0 \
    --output data/synthetic_wifi_data.json
```

**Parameters:**
- `--num_samples`: Number of data samples (default: 200)
- `--room_width`: Room width in meters (default: 10.0)
- `--room_height`: Room height in meters (default: 8.0)
- `--num_aps`: Number of access points (default: 3)
- `--noise_level`: Noise level in dB (default: 5.0)
- `--output`: Output file path

### Step 2: Train Baseline Model

```bash
python scripts/train_baseline_model.py \
    --data_path data/synthetic_wifi_data.json \
    --model_type random_forest \
    --predict_signal \
    --output_dir checkpoints
```

**Model Types:**
- `random_forest`: Random Forest Regressor (default)
- `linear`: Linear Regression

**Prediction Modes:**
- `--predict_signal`: Predict signal from position (default)
- `--predict_position`: Predict position from signal

### Step 3: Analyze Results

Open the Jupyter notebook:
```bash
jupyter notebook notebooks/baseline_analysis.ipynb
```

The notebook includes:
1. **Data Loading**: Load and inspect data
2. **Preprocessing**: Clean and normalize data
3. **Feature Extraction**: Extract features for modeling
4. **Visualizations**: 
   - Position vs signal strength scatter plots
   - Signal distribution histograms
   - Correlation matrices
   - PCA visualizations
5. **Model Training**: Train Random Forest and Linear Regression models
6. **Evaluation**: Evaluate models with metrics (RMSE, MAE, R²)

## Baseline Analysis Notebook

The `notebooks/baseline_analysis.ipynb` notebook provides a complete analysis pipeline:

1. **Data Preprocessing**
   - Load synthetic or real WiFi data
   - Filter outliers
   - Normalize features

2. **Feature Extraction**
   - Extract position and signal features
   - Statistical feature extraction
   - Feature scaling

3. **Visualizations**
   - Position vs signal strength heatmaps
   - Signal distribution histograms
   - Correlation matrices
   - PCA scatter plots

4. **Model Training**
   - Random Forest Regressor
   - Linear Regression
   - Model comparison

5. **Evaluation**
   - RMSE, MAE, R² metrics
   - Prediction vs actual plots
   - Model performance comparison

## Data Format

The synthetic data generator creates JSON files with the following structure:

```json
{
  "position_x": 2.5,
  "position_y": 3.0,
  "rssi": -65.2,
  "snr": 30.5,
  "signal_strength": 75,
  "channel": 44,
  "noise": -95,
  "timestamp": "2025-11-09T10:00:00",
  "location": "synthetic_room"
}
```

## Model Performance

### Random Forest Model
- **Input**: Position coordinates (x, y)
- **Output**: RSSI (signal strength in dBm)
- **Typical Performance**: R² > 0.85, RMSE < 5 dBm

### Linear Regression Model
- **Input**: Position coordinates (x, y)
- **Output**: RSSI (signal strength in dBm)
- **Typical Performance**: R² > 0.70, RMSE < 8 dBm

## Visualizations

The baseline analysis generates several visualizations:

1. **Position vs Signal Strength Scatter Plot**: Shows signal distribution across room
2. **Signal Distribution Histograms**: Distribution of RSSI, SNR, signal strength
3. **Correlation Matrix**: Correlation between features
4. **PCA Visualization**: Dimensionality reduction visualization
5. **Model Predictions**: Predicted vs actual signal strength

## Requirements

- Python 3.8+
- NumPy, Pandas
- scikit-learn
- matplotlib, seaborn
- jupyter (for notebook)
- joblib (for model saving)

## Installation

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter joblib
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Usage Examples

### Generate Data and Train Model

```bash
# 1. Generate synthetic data
python scripts/generate_synthetic_wifi_data.py --num_samples 200

# 2. Train model
python scripts/train_baseline_model.py \
    --data_path data/synthetic_wifi_data.json \
    --model_type random_forest

# 3. View results
# Check visualizations/ directory for plots
# Check checkpoints/ directory for saved models
```

### Run Notebook Analysis

```bash
# Start Jupyter notebook
jupyter notebook notebooks/baseline_analysis.ipynb

# Run all cells to see complete analysis
```

## Output Files

After running the baseline:

- `data/synthetic_wifi_data.json`: Generated synthetic data
- `checkpoints/baseline_*.pkl`: Trained models
- `checkpoints/baseline_*_metrics.json`: Model metrics
- `visualizations/baseline_*.png`: Visualization plots

## Troubleshooting

### Data Generation Issues
- Ensure output directory exists: `mkdir -p data`
- Check file permissions
- Verify Python version (3.8+)

### Model Training Issues
- Ensure data file exists and is valid JSON
- Check data format matches expected structure
- Verify enough samples (50+ recommended)

### Visualization Issues
- Ensure matplotlib backend is configured
- Check visualization directory exists: `mkdir -p visualizations`
- Verify data is loaded correctly

## Motion Detection

The project now supports motion detection from WiFi time-series data. This task classifies movement vs no movement from CSI amplitude time series, providing robust signal analysis and great visualization capabilities.

### Overview

**Task**: Classify movement vs no movement from WiFi time-series data  
**Input**: WiFi signal time series (RSSI, SNR, signal strength over time)  
**Output**: Binary classification (movement / no movement)  
**Features**: Statistical features (mean, variance, range, etc.), frequency domain features (FFT), time-domain features  
**Models**: Random Forest, Logistic Regression, SVM

### Quick Start

```bash
# Run complete pipeline (synthetic data)
./quick_start_motion_detection.sh
```

### Collect Motion Data

```bash
# Interactive mode (recommended)
python scripts/collect_time_series_data.py \
    --location room1 \
    --interactive \
    --duration 60 \
    --sampling_rate 10

# Collect with movement label
python scripts/collect_time_series_data.py \
    --location room1 \
    --movement \
    --duration 60 \
    --sampling_rate 10

# Collect without movement label
python scripts/collect_time_series_data.py \
    --location room1 \
    --no_movement \
    --duration 60 \
    --sampling_rate 10
```

### Train Motion Detector

```bash
# Train on collected data
python scripts/train_motion_detector.py \
    --data_paths data/room1/time_series/*.json \
    --model_type random_forest \
    --window_size 20 \
    --output_dir checkpoints

# Model types: random_forest, logistic, svm
```

### Visualize Motion Detection

```bash
# Generate time-series plot with motion regions
python scripts/visualize_motion_detection.py \
    --data_path data/room1/time_series/room1_movement_*.json \
    --model_path checkpoints/motion_detector_random_forest.pkl \
    --scaler_path checkpoints/motion_detector_random_forest_scaler.pkl \
    --output visualizations/motion_detection_timeseries.png
```

### Generate Synthetic Motion Data (for testing)

```bash
# Generate synthetic time-series data
python scripts/generate_synthetic_motion_data.py \
    --num_samples 2000 \
    --sampling_rate 10.0 \
    --movement_ratio 0.5 \
    --output data/synthetic_motion_data.json
```

### Motion Detection Outputs

After training and evaluation:

- **Confusion Matrix**: `visualizations/motion_detection_confusion_matrix.png`
- **ROC Curve**: `visualizations/motion_detection_roc_curve.png`
- **Time-Series Plot**: `visualizations/motion_detection_timeseries.png` (with motion regions highlighted)
- **Metrics**: `checkpoints/motion_detector_*_metrics.json` (accuracy, precision, recall, F1, ROC AUC)

### Model Performance

Typical performance on synthetic data:
- **Accuracy**: 70-90% (depending on data quality)
- **Precision**: 0.7-0.9
- **Recall**: 0.6-0.9
- **F1 Score**: 0.7-0.9
- **ROC AUC**: 0.7-0.95

## Next Steps

After completing Phase-1 baseline:

1. **Collect Real Data**: Use `collect_with_position.py` to collect real WiFi data for spatial mapping
2. **Collect Motion Data**: Use `collect_time_series_data.py` to collect time-series data for motion detection
3. **Improve Models**: Experiment with different models and features
4. **Advanced Analysis**: Add more sophisticated preprocessing and feature engineering
5. **Motion Detection**: Train and evaluate motion detection models

## References

- Inspired by "Human Identification Using WiFi Signal" paper
- Uses Butterworth filter concepts for preprocessing
- RandomForest baseline similar to reference paper
- Adapted for spatial WiFi signal mapping

## License

[Your License Here]

## Authors

- Rishabh (230178)
- Shivansh (230054)

Newton School of Technology
