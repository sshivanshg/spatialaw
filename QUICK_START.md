# Quick Start Guide

## Step 1: Setup (One-time)

### Option A: Use Setup Script (Easiest)
```bash
./setup.sh
```

### Option B: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Activate Virtual Environment

**Every time you open a new terminal**, activate the virtual environment:

```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

## Step 3: Test the Setup

```bash
python scripts/test_setup.py
```

This will verify that everything is installed correctly.

## Step 4: Quick Test with Mock Data

```bash
python scripts/quick_start.py
```

This will:
- Generate mock CSI data
- Train a baseline model for 5 epochs
- Evaluate the model
- Generate visualizations

## Step 5: Collect Real WiFi Data

```bash
# Collect WiFi data for 60 seconds
python scripts/collect_wifi_data.py --duration 60 --sampling_rate 2.0
```

## Step 6: Train Your Model

```bash
# Train with mock data (for testing)
python scripts/train_baseline.py --num_epochs 50 --batch_size 8

# Train with your own data
python scripts/train_baseline.py --data_path data/csi_data.npy --images_path data/images/
```

## Step 7: Monitor Training

In a new terminal (with venv activated):

```bash
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser.

## Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run quick test
python scripts/quick_start.py

# Collect WiFi data
python scripts/collect_wifi_data.py --duration 60

# Train model
python scripts/train_baseline.py --num_epochs 50

# Test setup
python scripts/test_setup.py

# View tensorboard
tensorboard --logdir logs

# Deactivate virtual environment (when done)
deactivate
```

## Troubleshooting

### "pip: command not found"
- Make sure you activated the virtual environment: `source venv/bin/activate`
- Use `pip3` if `pip` doesn't work

### "python: command not found"
- Use `python3` instead of `python`
- Make sure virtual environment is activated

### "Module not found" errors
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### Virtual environment not activating
- Make sure you're in the project directory
- Check that `venv` folder exists
- Try: `python3 -m venv venv` to recreate it

## Next Steps

1. Read `README.md` for detailed documentation
2. Check `BASELINE_SETUP.md` for architecture details
3. See `SETUP.md` for troubleshooting
4. Customize `configs/baseline_config.yaml` for your needs

Happy coding! 🚀

