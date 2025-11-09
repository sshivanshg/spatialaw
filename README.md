# Spatial Awareness through Ambient Wireless Signals - Motion Detection

Project by Rishabh (230178) and Shivansh (230054) - Newton School of Technology

## Overview

WiFi-based motion detection system that classifies movement vs no movement from WiFi signal time-series data. The system uses real-time RSSI, SNR, and signal strength measurements to detect human motion in indoor environments.

## Project Structure

```
spatialaw/
├── scripts/
│   ├── collect_time_series_data.py      # Collect time-series data for motion detection
│   ├── train_motion_detector.py         # Train motion detection models
│   ├── visualize_motion_detection.py    # Visualize motion detection results
│   ├── generate_synthetic_motion_data.py # Generate synthetic motion data
│   ├── live_prediction.py               # Live motion detection
│   └── check_trained_models.py          # Check which models are trained
├── src/
│   ├── data_collection/          # WiFi data collection utilities
│   │   └── wifi_collector.py     # WiFi data collector for macOS
│   ├── preprocessing/            # Data preprocessing pipelines
│   │   └── time_series_features.py  # Time-series feature extraction
│   └── models/                   # Model definitions
│       └── motion_detector.py    # Motion detection models
├── data/                         # Dataset storage
├── visualizations/               # Visualization outputs
└── checkpoints/                  # Model checkpoints
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

### 2. Collect Motion Data

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

### 3. Train Motion Detector

```bash
# Train on collected data
python scripts/train_motion_detector.py \
    --data_paths data/room1/time_series/*.json \
    --model_type random_forest \
    --window_size 20 \
    --output_dir checkpoints
```

### 4. Run Live Prediction

```bash
# Real-time motion detection
python scripts/live_prediction.py
```

## Motion Detection

### Overview

**Task**: Classify movement vs no movement from WiFi time-series data  
**Input**: WiFi signal time series (RSSI, SNR, signal strength over time)  
**Output**: Binary classification (movement / no movement) with confidence score  
**Features**: 9 statistical features extracted from 20-sample windows:
- Mean, std, variance of RSSI, SNR, signal strength
- Range (max - min) of RSSI
- Mean absolute change of RSSI

**Models**: Random Forest (default), Logistic Regression, SVM

### How It Works

1. **Data Collection**: Collects real-time WiFi signals (RSSI, SNR, signal strength) at 10 Hz sampling rate
2. **Feature Extraction**: Extracts statistical features from sliding windows (20 samples = 2 seconds)
3. **Model Training**: Trains Random Forest classifier on labeled data (movement vs no movement)
4. **Prediction**: Classifies new samples and outputs confidence scores

### Data Collection

```bash
# Interactive mode (asks for movement label)
python scripts/collect_time_series_data.py \
    --location room1 \
    --interactive \
    --duration 60 \
    --sampling_rate 10

# Collect with movement label
python scripts/collect_time_series_data.py \
    --location room1 \
    --movement \
    --duration 60

# Collect without movement label
python scripts/collect_time_series_data.py \
    --location room1 \
    --no_movement \
    --duration 60
```

**Parameters:**
- `--location`: Location name (e.g., "room1", "office")
- `--interactive`: Interactive mode (asks for movement label)
- `--movement`: Label data as "movement"
- `--no_movement`: Label data as "no movement"
- `--duration`: Collection duration in seconds (default: 60)
- `--sampling_rate`: Sampling rate in Hz (default: 10.0)

### Training

```bash
# Train on collected data
python scripts/train_motion_detector.py \
    --data_paths data/room1/time_series/*.json \
    --model_type random_forest \
    --window_size 20 \
    --output_dir checkpoints

# Model types: random_forest, logistic, svm
# Window size: Number of samples per window (default: 20)
```

**Output:**
- `checkpoints/motion_detector_*.pkl`: Trained model
- `checkpoints/motion_detector_*_scaler.pkl`: Feature scaler
- `checkpoints/motion_detector_*_metrics.json`: Performance metrics

### Live Prediction

```bash
# Real-time motion detection
python scripts/live_prediction.py
```

**Output:**
- Real-time predictions every 0.1 seconds (10 Hz)
- Confidence scores for each prediction
- RSSI and SNR values

### Visualization

```bash
# Generate time-series plot with motion regions
python scripts/visualize_motion_detection.py \
    --data_path data/room1/time_series/room1_movement_*.json \
    --model_path checkpoints/motion_detector_random_forest.pkl \
    --scaler_path checkpoints/motion_detector_random_forest_scaler.pkl \
    --output visualizations/motion_detection_timeseries.png
```

**Outputs:**
- **Confusion Matrix**: `visualizations/motion_detection_confusion_matrix.png`
- **ROC Curve**: `visualizations/motion_detection_roc_curve.png`
- **Time-Series Plot**: `visualizations/motion_detection_timeseries.png` (with motion regions highlighted)
- **Metrics**: `checkpoints/motion_detector_*_metrics.json` (accuracy, precision, recall, F1, ROC AUC)

### Model Performance

Typical performance metrics:
- **Accuracy**: 70-90% (depending on data quality and environment)
- **Precision**: 0.7-0.9
- **Recall**: 0.6-0.9
- **F1 Score**: 0.7-0.9
- **ROC AUC**: 0.7-0.95

**Note**: Performance depends on:
- WiFi router configuration (2.4GHz vs 5GHz)
- Environmental factors (walls, furniture, interference)
- Signal quality and noise levels
- Data collection quality and labeling accuracy

### Generate Synthetic Motion Data (for testing)

```bash
# Generate synthetic time-series data
python scripts/generate_synthetic_motion_data.py \
    --num_samples 2000 \
    --sampling_rate 10.0 \
    --movement_ratio 0.5 \
    --output data/synthetic_motion_data.json
```

## Data Format

Time-series data is stored in JSON format:

```json
{
  "rssi": -65,
  "snr": 30,
  "signal_strength": 75,
  "channel": 44,
  "ssid": "MyWiFi",
  "timestamp": "2025-01-15T10:00:00",
  "unix_timestamp": 1705312800.0,
  "movement": true,
  "movement_label": 1,
  "location": "room1",
  "device_id": "macbook-pro",
  "device_hostname": "MacBook-Pro",
  "device_platform": "macOS"
}
```

## Requirements

- Python 3.8+
- NumPy, Pandas
- scikit-learn
- matplotlib, seaborn
- joblib (for model saving)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Check Trained Models

```bash
# Check which models are trained
python scripts/check_trained_models.py
```

## Limitations

1. **RSSI/SNR Only**: Currently uses RSSI and SNR, not full CSI (Channel State Information)
2. **5GHz Limitations**: 5GHz signals have less penetration and multipath effects, making motion detection harder
3. **Environmental Factors**: Performance depends on room layout, furniture, and interference
4. **Labeling**: Requires manual labeling of movement vs no movement during data collection

## Next Steps

1. **Collect Real Data**: Collect more real-world data with accurate labels
2. **Improve Features**: Add frequency-domain features (FFT) and more sophisticated feature engineering
3. **Better Models**: Experiment with deep learning models (LSTM, CNN)
4. **CSI Integration**: Integrate full CSI data if available from compatible hardware
5. **Multi-Room Detection**: Extend to multiple rooms and locations

## References

- Inspired by "Human Identification Using WiFi Signal" paper
- WiFi CSI-Based motion detection research
- Random Forest baseline for time-series classification

## License

[Your License Here]

## Authors

- Rishabh (230178)
- Shivansh (230054)

Newton School of Technology
