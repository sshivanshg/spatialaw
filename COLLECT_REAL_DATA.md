# How to Collect Real WiFi Data

## Quick Command

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Collect real WiFi data for 60 seconds
python scripts/collect_wifi_data.py --duration 60
```

## Command Options

### Basic Collection
```bash
# Collect for 60 seconds at 2 samples per second (default)
python scripts/collect_wifi_data.py --duration 60

# Collect for 2 minutes
python scripts/collect_wifi_data.py --duration 120

# Collect with custom sampling rate (more samples per second)
python scripts/collect_wifi_data.py --duration 60 --sampling_rate 5.0
```

### Save to Specific File
```bash
# Save to custom filename
python scripts/collect_wifi_data.py --duration 60 --output data/my_wifi_data.json

# Save as CSV
python scripts/collect_wifi_data.py --duration 60 --output data/my_wifi_data.csv --format csv
```

### All Options
```bash
python scripts/collect_wifi_data.py \
  --duration 60 \           # Collection duration in seconds
  --sampling_rate 2.0 \     # Samples per second
  --output data/wifi.json \ # Output file path
  --format json \           # json or csv
  --interface en0           # Network interface (usually en0 for WiFi)
```

## Example: Collect Real Data

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Collect real WiFi data for 1 minute
python scripts/collect_wifi_data.py --duration 60 --output data/real_wifi_data.json

# 3. Check the collected data
python -c "import json; data = json.load(open('data/real_wifi_data.json')); print(f'Collected {len(data)} samples'); print(f'RSSI range: {min(d[\"rssi\"] for d in data)} to {max(d[\"rssi\"] for d in data)}'); print(f'Channel: {data[0][\"channel\"]}')"
```

## What Data You'll Get

The collected data includes:
- **RSSI**: Real signal strength in dBm (e.g., -53, -54, -52)
- **Signal Strength**: Percentage (0-100%)
- **Channel**: WiFi channel (e.g., 44 for 5GHz)
- **SNR**: Signal-to-noise ratio
- **Noise**: Noise level in dBm
- **SSID**: Network name (may be redacted for privacy)
- **Timestamp**: When each sample was collected
- **Collection Method**: `system_profiler` (real data!)

## Verify You're Getting Real Data

**Easy way - Use the check script:**
```bash
python scripts/check_collected_data.py
```

**Or check a specific file:**
```bash
# Replace with your actual filename
python -c "
import json
data = json.load(open('data/wifi_data_20251109_150557.json'))
print(f'Method: {data[0][\"collection_method\"]}')
print(f'RSSI: {data[0][\"rssi\"]} dBm')
print(f'Channel: {data[0][\"channel\"]}')
print(f'Samples: {len(data)}')
"
```

**Or use glob to find latest file:**
```bash
python -c "
import json, glob, os
files = glob.glob('data/wifi_data_*.json')
latest = max(files, key=os.path.getmtime) if files else None
if latest:
    data = json.load(open(latest))
    print(f'Latest file: {latest}')
    print(f'Method: {data[0][\"collection_method\"]}')
    print(f'RSSI range: {min(d[\"rssi\"] for d in data)} to {max(d[\"rssi\"] for d in data)}')
"
```

## Requirements

1. **Connected to WiFi**: Make sure you're connected to a WiFi network
2. **Virtual Environment**: Activate it first (`source venv/bin/activate`)
3. **Mac System**: The `system_profiler` method works on Mac

## Troubleshooting

### Not Getting Real Data?
```bash
# Check WiFi status
python scripts/check_wifi_status.py

# If not connected, connect to WiFi first
# System Preferences > Network > Wi-Fi
```

### Want Mock Data Instead?
```bash
# Use mock data flag
python scripts/collect_wifi_data.py --duration 60 --use_mock
```

## Full Example Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Check WiFi status
python scripts/check_wifi_status.py

# 3. Collect real data (60 seconds)
python scripts/collect_wifi_data.py --duration 60 --output data/real_wifi_60s.json

# 4. Verify data
python -c "import json; d=json.load(open('data/real_wifi_60s.json')); print(f'Samples: {len(d)}, Method: {d[0][\"collection_method\"]}, RSSI: {d[0][\"rssi\"]}')"

# 5. Use data for training (when ready)
# python scripts/train_baseline.py --data_path data/real_wifi_60s.json
```

## Notes

- Real data collection uses `system_profiler` on Mac
- Data varies naturally over time (RSSI changes)
- Collection method will be `system_profiler` (not `mock`)
- Make sure you're connected to WiFi for best results

