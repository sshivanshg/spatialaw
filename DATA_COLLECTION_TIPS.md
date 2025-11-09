# WiFi Data Collection Tips

## Understanding Your Collection Results

### Issue: All Values Are Identical

If you see data where all RSSI, signal_strength, channel, and SNR values are the same (like in your collection), this typically means:

1. **Not Connected to WiFi**: Your Mac is not connected to a WiFi network
2. **Limited Data Available**: The `networksetup` method only provides basic information
3. **Static Defaults**: When real data isn't available, the code uses default values

### Solutions

#### Option 1: Use Mock Data (Recommended for Development)

Mock data provides realistic variation and is perfect for testing:

```bash
python scripts/collect_wifi_data.py --duration 60 --use_mock
```

**Benefits**:
- ✅ Realistic variation in RSSI, signal strength, channels
- ✅ Perfect for model development and testing
- ✅ Works regardless of WiFi connection status
- ✅ Consistent data structure

#### Option 2: Connect to WiFi

If you want real WiFi data:

1. Connect to your Ruckus WiFi network
2. Run the collection script:
   ```bash
   python scripts/collect_wifi_data.py --duration 60
   ```
3. The collector will try to get real signal information

**Note**: Even when connected, `networksetup` provides limited information. For full CSI data, you'll need:
- Linux systems with compatible hardware
- Ruckus API access
- Specialized WiFi monitoring tools

#### Option 3: Use Collected Data for Testing

The data you collected (even with static values) can still be useful:
- Testing the data pipeline
- Verifying data loading and processing
- Understanding the data structure
- For baseline model development, use mock data instead

## Data Collection Methods Comparison

| Method | RSSI | Signal Strength | Channel | SSID | Availability |
|--------|------|-----------------|---------|------|--------------|
| **airport** | ✅ Real | ✅ Real | ✅ Real | ✅ Real | ❌ Rare on newer Macs |
| **networksetup** | ⚠️ Limited/Static | ⚠️ Limited/Static | ❌ Not Available | ✅ Real | ✅ Available |
| **mock** | ✅ Varied | ✅ Varied | ✅ Varied | ✅ Simulated | ✅ Always |

## Recommendations

### For Development & Testing
**Use mock data** - It's perfect for:
- Developing the baseline model
- Testing data processing pipelines
- Training models
- Debugging code

```bash
python scripts/collect_wifi_data.py --duration 60 --use_mock
```

### For Production
**You'll need**:
1. Real CSI data from Ruckus access points (via API or specialized tools)
2. Or use Linux systems with compatible WiFi hardware
3. Or work with network administrators for CSI data access

### For Your Baseline Model
**Mock data is sufficient** because:
- ✅ The model architecture doesn't depend on real vs. mock data
- ✅ You can test the entire pipeline
- ✅ You can evaluate model performance
- ✅ You can develop and debug your code

## Understanding Your Collected Data

Your collection shows:
```
rssi: -75 (all same)
signal_strength: 50 (all same)
channel: 0 (not available)
SSID: "Unknown" (not connected)
```

This indicates:
- ❌ Not connected to WiFi
- ⚠️ Using `networksetup` with limited capabilities
- 💡 Should use `--use_mock` for development

## Next Steps

1. **For Development**: Use mock data
   ```bash
   python scripts/collect_wifi_data.py --duration 60 --use_mock
   ```

2. **Verify Data**: Check that mock data has variation
   ```bash
   python -c "import json; data = json.load(open('data/wifi_data.json')); print('RSSI range:', min(d['rssi'] for d in data), 'to', max(d['rssi'] for d in data))"
   ```

3. **Train Model**: Use the collected data (mock or real) to train your baseline model
   ```bash
   python scripts/train_baseline.py --data_path data/wifi_data.json
   ```

4. **For Real CSI**: Contact your network administrator about accessing Ruckus CSI data

## Quick Reference

```bash
# Collect with mock data (recommended)
python scripts/collect_wifi_data.py --duration 60 --use_mock

# Try to collect real data (requires WiFi connection)
python scripts/collect_wifi_data.py --duration 60

# Check your data
python -c "import json, pandas as pd; data = json.load(open('data/wifi_data.json')); df = pd.DataFrame(data); print(df.describe())"
```

## Summary

- ✅ **Mock data is fine for development** - Use `--use_mock` flag
- ✅ **Your collection worked** - The pipeline is functioning correctly
- ✅ **Static values are expected** - When not connected to WiFi
- 💡 **For development**: Use mock data for realistic variation
- 💡 **For production**: You'll need real CSI data from Ruckus or compatible hardware

