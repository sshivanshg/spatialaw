# Quick Commands Reference

## Collect Real WiFi Data

```bash
# Activate virtual environment
source venv/bin/activate

# Collect real WiFi data (60 seconds)
python scripts/collect_wifi_data.py --duration 60
```

## Check Collected Data

```bash
# Check all collected data files
python scripts/check_collected_data.py

# Check a specific file
python -c "import json; data = json.load(open('data/test_real_wifi.json')); print(f'Method: {data[0][\"collection_method\"]}, RSSI: {data[0][\"rssi\"]}')"
```

## Check WiFi Status

```bash
# Check if WiFi is connected and what data you can collect
python scripts/check_wifi_status.py
```

## Train Baseline Model

```bash
# Train with mock CSI data
python scripts/train_baseline.py --num_epochs 50

# Train with your WiFi data (when ready)
python scripts/train_baseline.py --data_path data/your_wifi_data.json
```

## Quick Test

```bash
# Test the entire setup
python scripts/test_setup.py

# Quick start with mock data
python scripts/quick_start.py
```

## Find Your Latest Data File

```bash
# List all WiFi data files
ls -lt data/wifi_data_*.json | head -1

# Or use Python
python -c "import glob, os; files = glob.glob('data/wifi_data_*.json'); print(max(files, key=os.path.getmtime) if files else 'No files')"
```

