# WiFi Data Collection Guide

## Overview

The WiFi data collection utility collects WiFi signal information including RSSI, signal strength, SNR, and channel information. On Mac systems, there are limitations due to Apple's restrictions on WiFi utilities.

## Collection Methods

### 1. Airport Utility (Preferred - Limited Availability)

The `airport` utility is the best method for collecting detailed WiFi information, but it's not always available on newer Macs.

**Status**: ❌ Not available on many newer Mac systems

**What it provides**:
- RSSI (Received Signal Strength Indicator)
- Signal strength percentage
- SNR (Signal-to-Noise Ratio)
- Channel information
- SSID and BSSID

### 2. Networksetup (Fallback - Limited Information)

The `networksetup` command is available on most Macs but provides limited information.

**Status**: ⚠️ Available but limited

**What it provides**:
- SSID (Network name)
- Limited signal information

### 3. Mock Data (Always Available)

Mock data generation is always available and useful for:
- Development and testing
- When real WiFi collection is not possible
- Testing the baseline model

**Status**: ✅ Always available

**What it provides**:
- Simulated RSSI values
- Simulated signal strength
- Simulated channel information
- Realistic data structure matching real WiFi data

## Usage

### Automatic Detection (Recommended)

The WiFi collector automatically tries to find the best available method:

```python
from src.data_collection.wifi_collector import WiFiCollector

# Automatically detects available method
collector = WiFiCollector(sampling_rate=2.0)
samples = collector.collect_sample(duration=60.0)
```

### Explicit Mock Data

If you want to explicitly use mock data:

```python
# Use mock data explicitly
collector = WiFiCollector(sampling_rate=2.0, use_mock=True)
samples = collector.collect_sample(duration=60.0)
```

### Command Line

```bash
# Try to collect real data (falls back to mock if unavailable)
python scripts/collect_wifi_data.py --duration 60

# Explicitly use mock data
python scripts/collect_wifi_data.py --duration 60 --use_mock
```

## Data Structure

The collected data has the following structure:

```python
{
    'rssi': -75,              # Received Signal Strength Indicator (dBm)
    'signal_strength': 50,    # Signal strength percentage (0-100)
    'snr': 25,                # Signal-to-Noise Ratio (dB)
    'channel': 6,             # WiFi channel
    'ssid': 'RUCKUS_NETWORK', # Network name
    'bssid': '00:11:22:33:44:55', # Access point MAC address
    'timestamp': '2024-11-09T14:47:50', # ISO format timestamp
    'unix_timestamp': 1699542470.123,    # Unix timestamp
    'collection_method': 'mock'  # Method used: 'airport', 'networksetup', or 'mock'
}
```

## Troubleshooting

### Airport Utility Not Found

**Problem**: `airport` command not found

**Solution**: 
- The collector will automatically fall back to mock data
- Use `--use_mock` flag to explicitly use mock data
- This is normal on newer Mac systems

### No Real WiFi Data Available

**Problem**: Can't collect real WiFi data on Mac

**Solutions**:
1. **Use mock data** (recommended for development):
   ```bash
   python scripts/collect_wifi_data.py --duration 60 --use_mock
   ```

2. **Use Linux system** with compatible WiFi hardware for full CSI collection

3. **Use Ruckus API** if your access points provide CSI data via API

4. **External tools**: Consider using specialized WiFi monitoring tools

### Permission Errors

**Problem**: Permission denied when trying to access WiFi information

**Solution**:
- Grant Terminal/Python network permissions in System Preferences > Security & Privacy > Privacy > Full Disk Access
- Note: Even with permissions, `airport` may not be available on newer Macs

## Mock Data for Development

Mock data is perfectly suitable for:
- ✅ Testing the baseline model
- ✅ Developing and debugging code
- ✅ Training models when real data is not available
- ✅ Demonstrating the pipeline

Mock data characteristics:
- Realistic RSSI values (-90 to -50 dBm)
- Realistic signal strength (20-80%)
- Random but realistic channel assignments
- Proper data structure matching real WiFi data

## Real CSI Data Collection

For full Channel State Information (CSI) data collection, you'll need:

1. **Compatible Hardware**:
   - Linux systems with Intel 5300 NIC (for full CSI access)
   - Specialized WiFi monitoring hardware

2. **Software Tools**:
   - nexmon (for certain WiFi cards)
   - Custom drivers for CSI extraction
   - Ruckus API access (if available)

3. **Alternative Approaches**:
   - Work with Ruckus access points that provide CSI via API
   - Use external CSI collection devices
   - Collaborate with network administrators for CSI data access

## Recommendations

1. **For Development**: Use mock data - it's sufficient for testing and development
2. **For Production**: Work with your network administrator to get CSI data from Ruckus access points
3. **For Research**: Consider using Linux systems with compatible hardware for full CSI access

## Next Steps

1. Test with mock data: `python scripts/collect_wifi_data.py --duration 60 --use_mock`
2. Verify data structure: Check the collected JSON/CSV files
3. Use collected data: Feed it into the baseline model for training
4. Contact network admin: Discuss CSI data access from Ruckus access points

## Example: Collect and Use WiFi Data

```python
from src.data_collection.wifi_collector import WiFiCollector
import pandas as pd

# Collect WiFi data (automatically uses best available method)
collector = WiFiCollector(sampling_rate=2.0, use_mock=True)
collector.collect_continuous(duration=60.0, save_path="data/wifi_data.json")

# Load and analyze data
df = collector.get_dataframe()
print(df.describe())
print(f"\nCollection method: {df['collection_method'].value_counts()}")

# Use data for training
# (Integrate with your baseline model)
```

## Notes

- Mock data is sufficient for baseline model development
- Real CSI data will be needed for production deployment
- The data structure is consistent between mock and real data
- All collection methods produce the same data format

