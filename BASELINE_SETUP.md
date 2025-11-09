# Baseline Model Setup Summary

## What Was Created

A complete baseline model implementation for your spatial awareness project using WiFi signals. The implementation includes:

### 1. **Data Collection** (`src/data_collection/`)
   - `wifi_collector.py`: Collects WiFi signal data on Mac (RSSI, signal strength, SNR, channel info)
   - `csi_processor.py`: Processes CSI data (amplitude, phase) with normalization and feature extraction
   - Mock data generator for development and testing

### 2. **Model Architecture** (`src/models/`)
   - `baseline_model.py`: CNN-based encoder-decoder architecture
     - Encoder: Processes CSI data (amplitude + phase) into latent representations
     - Decoder: Reconstructs spatial information (e.g., RGB images) from latent codes
   - Supports configurable input/output dimensions

### 3. **Data Preprocessing** (`src/preprocessing/`)
   - `data_loader.py`: PyTorch datasets for WiFi and CSI data
   - `transforms.py`: Data augmentation and normalization transforms

### 4. **Training** (`src/training/`)
   - `trainer.py`: Complete training loop with validation, checkpointing, and TensorBoard logging
   - `losses.py`: Multiple loss functions (MSE, SSIM, spatial loss, combined loss)

### 5. **Evaluation** (`src/evaluation/`)
   - `evaluator.py`: Model evaluation utilities
   - `metrics.py`: Evaluation metrics (MSE, MAE, PSNR, SSIM)
   - `visualizer.py`: Visualization utilities for CSI data and predictions

### 6. **Scripts** (`scripts/`)
   - `train_baseline.py`: Main training script with command-line arguments
   - `collect_wifi_data.py`: WiFi data collection script
   - `quick_start.py`: Quick test script to verify everything works
   - `test_setup.py`: Setup verification script

### 7. **Configuration** (`configs/`)
   - `baseline_config.yaml`: Configuration file for model and training parameters

## Quick Start

### 1. Install Dependencies
```bash
# On Mac, use pip3 (or python3 -m pip)
pip3 install -r requirements.txt

# Alternatively:
python3 -m pip install -r requirements.txt
```

### 2. Test Setup
```bash
python scripts/test_setup.py
```

### 3. Quick Test with Mock Data
```bash
python scripts/quick_start.py
```

This will:
- Generate mock CSI data
- Train a baseline model for 5 epochs
- Evaluate the model
- Generate visualizations

### 4. Collect Real WiFi Data
```bash
# Collect WiFi data for 60 seconds
python scripts/collect_wifi_data.py --duration 60 --sampling_rate 2.0
```

### 5. Train Baseline Model
```bash
# Train with mock data (default)
python scripts/train_baseline.py --num_epochs 50 --batch_size 8

# Train with your own data
python scripts/train_baseline.py --data_path data/csi_data.npy --images_path data/images/
```

### 6. Monitor Training
```bash
tensorboard --logdir logs
# Open http://localhost:6006 in browser
```

## Model Architecture

The baseline model is a CNN-based encoder-decoder:

**Input**: CSI data of shape `(2, num_antennas, num_subcarriers)`
- Channel 0: Amplitude
- Channel 1: Phase

**Encoder**: 
- 3 convolutional blocks with BatchNorm and ReLU
- Adaptive average pooling
- Latent projection to `latent_dim` dimensions

**Decoder**:
- Transposed convolutions to upsample
- Reconstructs spatial information (e.g., RGB images)
- Output shape: `(3, height, width)`

**Loss Function**: Combined MSE, SSIM, and spatial loss

## Data Flow

1. **WiFi Collection** → Collect RSSI, signal strength, channel info
2. **CSI Processing** → Extract amplitude and phase from CSI data
3. **Preprocessing** → Normalize and prepare data for training
4. **Model Training** → Train encoder-decoder to reconstruct spatial information
5. **Evaluation** → Assess model performance with metrics
6. **Visualization** → Visualize predictions and CSI data

## Current Limitations & Next Steps

### Current State
- ✅ Baseline model architecture implemented
- ✅ WiFi data collection for Mac (RSSI, signal strength)
- ✅ Mock CSI data generation for development
- ✅ Training and evaluation pipeline
- ⚠️ Real CSI collection on Mac is limited (hardware-dependent)

### Next Steps for Your Project

1. **Collect Real CSI Data**
   - For Mac: May need compatible hardware or specialized tools
   - Consider using Linux systems with Intel 5300 NIC for full CSI access
   - Or work with Ruckus access points that provide CSI data via API

2. **Extend Model Architecture**
   - Add text conditioning for text-guided generation
   - Implement diffusion models (as mentioned in your project goals)
   - Add attention mechanisms
   - Improve decoder for better spatial reconstruction

3. **Data Collection**
   - Collect ground truth images corresponding to WiFi signals
   - Create dataset with CSI-image pairs
   - Augment data for better generalization

4. **Evaluation**
   - Collect real-world test data
   - Evaluate on actual spatial reconstruction tasks
   - Compare with ground truth images

5. **Applications**
   - Security applications
   - Health monitoring
   - Assistive technology
   - Urban infrastructure monitoring

## File Structure

```
spatialaw/
├── src/
│   ├── data_collection/    # WiFi and CSI data collection
│   ├── models/             # Model architectures
│   ├── preprocessing/      # Data loaders and transforms
│   ├── training/           # Training utilities
│   └── evaluation/         # Evaluation and visualization
├── scripts/                # Executable scripts
├── configs/                # Configuration files
├── data/                   # Data storage (created when collecting data)
├── checkpoints/            # Model checkpoints (created during training)
├── logs/                   # TensorBoard logs (created during training)
└── visualizations/         # Visualization outputs
```

## Notes for Mac Users

1. **WiFi Collection**: The `airport` utility is used for WiFi monitoring. If it's not found:
   - Full path: `/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport`
   - Grant Terminal network permissions in System Preferences
   - The code falls back to mock data if collection fails

2. **CSI Collection**: Full CSI collection on Mac is challenging. Options:
   - Use mock data for development (already implemented)
   - Use Linux systems with compatible hardware
   - Work with Ruckus access points that provide CSI via API
   - Consider using external CSI collection tools

3. **Permissions**: You may need to grant Terminal/Python network permissions in System Preferences > Security & Privacy

## Troubleshooting

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (requires Python 3.8+)

### WiFi Collection Issues
- Verify airport utility is accessible
- Check network permissions
- Code will use mock data if real collection fails

### Training Issues
- Reduce batch size if out of memory
- Adjust learning rate in config
- Use fewer samples for faster testing

### Model Performance
- Start with mock data to verify pipeline works
- Collect real data for better results
- Adjust model architecture based on your data

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Run `python scripts/test_setup.py` to verify setup
3. Check tensorboard logs for training progress
4. Review configuration in `configs/baseline_config.yaml`

Good luck with your project! 🚀

