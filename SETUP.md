# Setup Guide for Mac

## Python and pip Setup

On Mac, you typically need to use `python3` and `pip3` instead of `python` and `pip`.

### Check Your Python Installation

```bash
# Check Python version
python3 --version

# Check pip version
pip3 --version

# Check if they're installed
which python3
which pip3
```

### Install Dependencies

```bash
# Install all required packages
pip3 install -r requirements.txt

# Or use python3 -m pip
python3 -m pip install -r requirements.txt
```

### If pip3 is not found

1. **Install Python via Homebrew** (recommended):
   ```bash
   brew install python3
   ```

2. **Or use the system Python** (if available):
   ```bash
   python3 -m ensurepip --upgrade
   ```

3. **Or install pip manually**:
   ```bash
   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   python3 get-pip.py
   ```

## Running Scripts

All scripts should be run with `python3`:

```bash
# Test setup
python3 scripts/test_setup.py

# Quick start
python3 scripts/quick_start.py

# Collect WiFi data
python3 scripts/collect_wifi_data.py --duration 60

# Train model
python3 scripts/train_baseline.py --num_epochs 50
```

## Creating a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Now you can use 'python' and 'pip' directly
python scripts/test_setup.py
```

## Troubleshooting

### "pip: command not found"
- Use `pip3` instead of `pip`
- Or use `python3 -m pip`

### "python: command not found"
- Use `python3` instead of `python`

### Permission Errors
- Use `pip3 install --user -r requirements.txt` to install for your user only
- Or use a virtual environment (recommended)

### Module Not Found Errors
- Make sure you're using `python3` to run scripts
- Verify dependencies are installed: `pip3 list`
- Reinstall dependencies: `pip3 install -r requirements.txt --upgrade`

## Quick Start Commands

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Test setup
python3 scripts/test_setup.py

# 3. Quick test with mock data
python3 scripts/quick_start.py

# 4. (Optional) Create virtual environment for isolated environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

