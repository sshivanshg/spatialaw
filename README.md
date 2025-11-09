# Spatial Awareness through Ambient Wireless Signals

Project by Rishabh (230178) and Shivansh (230054) - Newton School of Technology

## Overview

This project aims to reconstruct, monitor, and infer spatial layouts and dynamics of indoor environments using only ubiquitous WiFi signals, creating a scalable, privacy-preserving "sixth sense."

## Project Structure

```
spatialaw/
├── data/                  # Dataset storage
├── notebooks/             # Jupyter notebooks for exploration
├── scripts/               # Utility scripts
│   ├── train_baseline.py  # Main training script
│   ├── collect_wifi_data.py  # WiFi data collection
│   └── quick_start.py     # Quick test script
├── wifi-csi-mapping/      # WiFi CSI data collection and processing
├── src/                   # Source code
│   ├── data_collection/   # WiFi data collection utilities
│   ├── preprocessing/     # Data preprocessing pipelines
│   ├── models/            # Model definitions
│   ├── training/          # Training scripts
│   └── evaluation/        # Evaluation and visualization
├── configs/               # Configuration files
├── checkpoints/           # Model checkpoints (created during training)
├── logs/                  # Tensorboard logs (created during training)
└── visualizations/        # Visualization outputs
```

## Setup

### Quick Setup (Recommended)

We've created a setup script that handles everything:

```bash
# Run the setup script
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Test the setup

### Manual Setup

1. **Create and activate virtual environment** (required on Mac):
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **For WiFi monitoring on Mac**, you may need to grant Terminal/Python network permissions in System Preferences > Security & Privacy > Privacy > Full Disk Access

### Using the Virtual Environment

**Important**: Always activate the virtual environment before running scripts:

```bash
# Activate virtual environment
source venv/bin/activate

# Now you can run scripts
python scripts/quick_start.py
python scripts/train_baseline.py

# Deactivate when done
deactivate
```

See `SETUP.md` for detailed setup instructions and troubleshooting.

## Quick Start

### 1. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 2. Test the Baseline Model with Mock Data

```bash
python scripts/quick_start.py
```

This will:
- Generate mock CSI data
- Create and train a baseline model
- Evaluate the model
- Generate visualizations

### 3. Collect WiFi Data (Mac)

```bash
# Collect WiFi data for 60 seconds at 2 samples/second
python scripts/collect_wifi_data.py --duration 60 --sampling_rate 2.0

# Save to specific file
python scripts/collect_wifi_data.py --duration 120 --output data/my_wifi_data.json

# Use mock data (if airport utility is not available)
python scripts/collect_wifi_data.py --duration 60 --use_mock
```

**Note**: On newer Macs, the `airport` utility may not be available. The script will automatically use mock data if real WiFi collection fails. You can also explicitly use mock data with the `--use_mock` flag.

### 4. Train Baseline Model

```bash
# Train with mock data (default)
python scripts/train_baseline.py --num_epochs 50 --batch_size 8

# Train with your own CSI data
python scripts/train_baseline.py --data_path data/csi_data.npy --images_path data/images/ --num_epochs 100

# Resume from checkpoint
python scripts/train_baseline.py --resume checkpoints/best_model.pth --num_epochs 50
```

### 5. Monitor Training

```bash
# Start tensorboard
tensorboard --logdir logs

# Open browser to http://localhost:6006
```

## Baseline Model

The baseline model processes WiFi CSI data (amplitude and phase vectors) and reconstructs spatial information (e.g., images). It uses a CNN-based encoder-decoder architecture:

- **Encoder**: Processes CSI data into latent representations
- **Decoder**: Reconstructs spatial information from latent codes

### Model Architecture

- Input: CSI data (amplitude + phase) of shape (2, num_antennas, num_subcarriers)
- Output: Spatial reconstruction (e.g., RGB image) of shape (3, height, width)
- Loss: Combined MSE, SSIM, and spatial loss

## Data Collection

### Mac WiFi Monitoring

The `WiFiCollector` class collects WiFi signal information using Mac's `airport` utility:

- RSSI (Received Signal Strength Indicator)
- Signal strength percentage
- SNR (Signal-to-Noise Ratio)
- Channel information
- SSID and BSSID

### CSI Data

For full CSI (Channel State Information) data collection on Mac, you may need:
- Compatible WiFi hardware
- Special drivers or tools (e.g., nexmon for certain cards)
- Or use the mock data generator for development

The codebase is structured to work with both real CSI data and mock data for development.

## Configuration

Edit `configs/baseline_config.yaml` to customize:
- Model architecture
- Training parameters
- Data paths
- Loss function weights

## Usage Examples

### Collect WiFi Data

```python
from src.data_collection.wifi_collector import WiFiCollector

collector = WiFiCollector(sampling_rate=2.0)
samples = collector.collect_sample(duration=60.0)
collector.save_data("data/wifi_data.json")
```

### Process CSI Data

```python
from src.data_collection.csi_processor import CSIProcessor, generate_mock_csi

# Generate mock CSI
csi_data = generate_mock_csi(num_samples=100, num_antennas=3, num_subcarriers=64)

# Process CSI
processor = CSIProcessor(num_subcarriers=64, num_antennas=3)
amplitude, phase = processor.process_csi(csi_data[0])
features = processor.extract_features(csi_data[0])
```

### Train Model

```python
from src.models.baseline_model import BaselineSpatialModel
from src.preprocessing.data_loader import CSIDataset
from src.training.trainer import Trainer
from torch.utils.data import DataLoader

# Create model
model = BaselineSpatialModel(
    input_channels=2,
    num_subcarriers=64,
    num_antennas=3,
    latent_dim=128,
    output_channels=3,
    output_size=(64, 64)
)

# Create dataset
dataset = CSIDataset(generate_mock=True, num_mock_samples=1000)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Train
trainer = Trainer(model=model, train_loader=train_loader)
trainer.train(num_epochs=50)
```

## Next Steps

1. **Collect Real Data**: Use the WiFi collector to gather real WiFi signals from your Ruckus setup
2. **Extend Model**: Add text conditioning, improve architecture, or add diffusion models
3. **Evaluation**: Use the evaluation utilities to assess model performance
4. **Visualization**: Generate visualizations to understand model outputs

## Notes

- The baseline model currently works with mock CSI data for development
- For production use with real CSI data, you may need to adapt the data collection based on your hardware
- Mac WiFi CSI collection is limited compared to Linux systems with compatible hardware
- The model architecture can be extended to support text-guided generation as mentioned in your project goals

## Troubleshooting

### Mac WiFi Collection Issues

If `airport` command is not found (common on newer Macs):
-  **This is normal** - The code automatically falls back to mock data
- Use `--use_mock` flag to explicitly use mock data: `python scripts/collect_wifi_data.py --duration 60 --use_mock`
- Mock data is sufficient for development and testing
- For real CSI data, consider using Linux systems or Ruckus API access
- See `WIFI_COLLECTION.md` for detailed information about WiFi data collection

### Training Issues

- Reduce batch size if you run out of memory
- Adjust learning rate in config file
- Use fewer mock samples for faster testing

## References

See the project poster for literature review and references.

