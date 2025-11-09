# Spatial Awareness through Ambient Wireless Signals

Project by Rishabh (230178) and Shivansh (230054) - Newton School of Technology

## Overview

This project creates WiFi signal heatmaps for indoor environments by collecting WiFi data from multiple positions within a room and training a model to predict signal distribution across the space.

## Project Structure

```
spatialaw/
├── collect_with_position.py    # WiFi data collection with position tracking
├── data/                        # Dataset storage
├── scripts/                     # Utility scripts
│   ├── train_heatmap.py        # Train heatmap model
│   ├── generate_heatmap.py     # Generate heatmap visualization
│   ├── combine_multi_device_data.py  # Combine data from multiple devices
│   └── validate_collected_data.py    # Validate collected data
├── src/                         # Source code
│   ├── data_collection/        # WiFi data collection utilities
│   ├── preprocessing/          # Data preprocessing pipelines
│   ├── models/                 # Model definitions (heatmap models)
│   └── training/               # Training utilities
├── configs/                     # Configuration files
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs
└── visualizations/              # Visualization outputs
```

## Quick Start

### 1. Setup

```bash
# Create virtual environment and install dependencies
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Collect Data with Positions

**Interactive Mode** (Recommended):
```bash
python collect_with_position.py --location Room_101 --interactive --duration 30
```

This will prompt you to enter positions:
```
Enter position (x, y) or 'q' to quit: 0, 0
Enter position (x, y) or 'q' to quit: 2.5, 0
Enter position (x, y) or 'q' to quit: 5, 0
Enter position (x, y) or 'q' to quit: q
```

**Single Position Mode**:
```bash
python collect_with_position.py --location Room_101 --x 2.5 --y 3.0 --duration 30
```

**Coordinate System Setup**:
- Choose a corner of the room as origin (0, 0)
- Measure room dimensions (meters)
- Use consistent units
- Plan collection points in a grid pattern (1-2 meters apart)

### 3. Train Heatmap Model

```bash
python scripts/train_heatmap.py \
    --data_path data/Room_101/Room_101_all_positions_*.json \
    --predict_signal \
    --num_epochs 100 \
    --batch_size 32
```

### 4. Generate Heatmap

```bash
python scripts/generate_heatmap.py \
    --model_path checkpoints/signal_heatmap_model.pth \
    --data_path data/Room_101/Room_101_all_positions_*.json \
    --feature 0 \
    --feature_name "RSSI" \
    --show_points \
    --output visualizations/room_101_heatmap.png
```

## Data Collection

### Setup Coordinates

1. **Choose origin**: Pick a corner of the room as (0, 0)
2. **Measure room**: Note dimensions (e.g., 5m × 4m)
3. **Plan points**: Create a grid of collection points (1-2m apart)
4. **Collect data**: Stay at each position for 30-60 seconds

### Collection Tips

- Collect from at least 10-20 different positions
- Space positions evenly across the room
- Keep device stationary during collection
- Cover entire room area
- Use consistent coordinate system

### Example Room Layout

```
Room (5m × 4m):
(0,4) ──────────── (5,4)
  │                 │
  │      Room       │
  │                 │
(0,0) ──────────── (5,0)
```

**Collection points** (1m grid):
- (0,0), (1,0), (2,0), (3,0), (4,0), (5,0)
- (0,1), (1,1), (2,1), (3,1), (4,1), (5,1)
- ... and so on

## Model Training

### Signal Prediction (Position → Signal)

Predict WiFi signal features from position coordinates:
- **Input**: (x, y) coordinates
- **Output**: WiFi signal features (RSSI, SNR, signal_strength, channel)
- **Use case**: WiFi coverage mapping

```bash
python scripts/train_heatmap.py \
    --data_path data/Room_101/Room_101_all_positions_*.json \
    --predict_signal \
    --num_epochs 100
```

### Position Prediction (Signal → Position)

Predict position from WiFi signal features:
- **Input**: WiFi signal features
- **Output**: (x, y) coordinates
- **Use case**: Indoor positioning

```bash
python scripts/train_heatmap.py \
    --data_path data/Room_101/Room_101_all_positions_*.json \
    --predict_position \
    --num_epochs 100
```

## Heatmap Generation

Generate heatmap visualization of WiFi signal distribution:

```bash
python scripts/generate_heatmap.py \
    --model_path checkpoints/signal_heatmap_model.pth \
    --data_path data/Room_101/Room_101_all_positions_*.json \
    --feature 0 \
    --feature_name "RSSI" \
    --show_points
```

**Features to visualize**:
- `--feature 0`: RSSI (signal strength in dBm)
- `--feature 1`: Signal strength (0-100 scale)
- `--feature 2`: SNR (Signal-to-Noise Ratio)
- `--feature 3`: Channel

## Data Validation

Validate collected data:

```bash
python scripts/validate_collected_data.py data/Room_101/
```

## Combine Data from Multiple Devices

If collecting from multiple devices:

```bash
python scripts/combine_multi_device_data.py --data_dir data/Room_101
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas
- matplotlib (for visualization)
- scikit-learn (for data preprocessing)

## Files

### Essential Files

- `collect_with_position.py` - Data collection with position tracking
- `scripts/train_heatmap.py` - Train heatmap model
- `scripts/generate_heatmap.py` - Generate heatmap visualization
- `scripts/validate_collected_data.py` - Validate data
- `scripts/combine_multi_device_data.py` - Combine data
- `src/models/heatmap_model.py` - Heatmap model architecture
- `src/data_collection/wifi_collector.py` - WiFi data collection
- `src/preprocessing/data_loader.py` - Data loading
- `src/preprocessing/wifi_to_csi.py` - WiFi to CSI conversion

## Usage Examples

### Complete Workflow

```bash
# 1. Collect data
python collect_with_position.py --location Room_101 --interactive --duration 30

# 2. Train model
python scripts/train_heatmap.py \
    --data_path data/Room_101/Room_101_all_positions_*.json \
    --num_epochs 100

# 3. Generate heatmap
python scripts/generate_heatmap.py \
    --model_path checkpoints/signal_heatmap_model.pth \
    --data_path data/Room_101/Room_101_all_positions_*.json \
    --feature 0 \
    --feature_name "RSSI" \
    --show_points
```

## Troubleshooting

### WiFi Connection Issues

- Ensure you are connected to WiFi
- Check System Preferences > Network > WiFi
- Verify `system_profiler SPAirPortDataType` works

### Data Collection Issues

- Check WiFi connection before starting
- Verify device has network permissions
- Use `--duration` to control collection time

### Training Issues

- Ensure you have enough data (recommended: 50+ samples)
- Check data format with validation script
- Verify data path is correct
- Collect from multiple positions (10-20 minimum)

### Heatmap Quality

- Collect more data points (20+ positions)
- Ensure good coverage across room
- Train for more epochs
- Check data quality (signal variation)

## Notes

- **Real Data Only**: This project only collects and uses real WiFi data
- **Position-Based**: Requires position coordinates for each data sample
- **Heatmap Focus**: Generates WiFi signal heatmaps for spatial analysis
- **Portable**: Can collect data from multiple devices/locations

## License

[Your License Here]

## Authors

- Rishabh (230178)
- Shivansh (230054)

Newton School of Technology
