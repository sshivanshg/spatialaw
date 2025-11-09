# How to Get Real WiFi Data

## Understanding the Challenge

On Mac, getting **real WiFi data** (especially full CSI - Channel State Information) is limited due to:
1. Apple's restrictions on WiFi utilities
2. Hardware limitations
3. Security/privacy restrictions

## Option 1: Basic WiFi Data (RSSI, Signal Strength) - Mac

### Step 1: Connect to WiFi

First, make sure you're connected to a WiFi network:

```bash
# Check if connected
/usr/sbin/networksetup -getairportnetwork en0

# If not connected, connect via System Preferences or:
# System Preferences > Network > Wi-Fi > Select your network
```

### Step 2: Try to Find Airport Utility

The `airport` utility provides the best WiFi information, but it's often not available on newer Macs:

```bash
# Try to find airport utility
ls -la /System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport

# If it exists, test it:
/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -I
```

**If airport works**, you'll get:
- Real RSSI values
- Signal strength
- Channel information
- SSID and BSSID
- Noise levels

### Step 3: Collect Data While Connected

Once connected to WiFi:

```bash
# Activate virtual environment
source venv/bin/activate

# Collect real WiFi data
python scripts/collect_wifi_data.py --duration 60 --sampling_rate 2.0
```

**What you'll get:**
- Real SSID (network name)
- Real RSSI (if airport utility works)
- Real signal strength (if airport utility works)
- Real channel information (if airport utility works)

**Limitations:**
- Even with airport, you won't get full CSI (amplitude/phase vectors)
- networksetup only provides basic info (SSID, limited signal data)

## Option 2: Full CSI Data - Requires Special Setup

### For Full CSI (Channel State Information) Data:

#### A. Linux System with Compatible Hardware

**Best option for full CSI:**

1. **Use Linux** (Ubuntu recommended)
2. **Compatible WiFi card**: Intel 5300 NIC or similar
3. **Tools needed**:
   - nexmon (for certain WiFi cards)
   - Custom CSI extraction tools
   - Modified WiFi drivers

**Setup:**
```bash
# On Linux system
# Install nexmon (if compatible)
git clone https://github.com/seemoo-lab/nexmon.git
# Follow nexmon installation instructions

# Or use other CSI tools like:
# - CSI Tool (for Intel 5300)
# - ESP32-based CSI collection
```

#### B. Ruckus Access Point API

**If you have access to Ruckus management:**

1. **Ruckus API Access**:
   - Contact your network administrator
   - Ruckus access points can provide CSI data via API
   - Requires admin credentials

2. **Ruckus Unleashed/Cloud**:
   - Some Ruckus systems expose WiFi analytics
   - May include CSI data in management interface
   - Check Ruckus documentation for your model

3. **Ruckus API Example** (if available):
```python
import requests

# Ruckus API endpoint (varies by model)
ruckus_api = "https://your-ruckus-ap-ip/api/v1/csi"

# Get CSI data
response = requests.get(ruckus_api, auth=('username', 'password'))
csi_data = response.json()
```

#### C. External CSI Collection Devices

**Hardware solutions:**

1. **ESP32 with CSI Support**:
   - ESP32 can collect CSI data
   - Requires custom firmware
   - Cheaper option

2. **Specialized WiFi Monitoring Tools**:
   - Commercial WiFi analyzers
   - Research-grade CSI collection devices

## Option 3: Hybrid Approach (Recommended for Your Project)

### For Your Baseline Model Development:

**Use mock data for now**, but structure it to match real CSI data:

```bash
# Collect mock CSI data (realistic structure)
python scripts/collect_wifi_data.py --duration 60 --use_mock

# The mock data has the same structure as real CSI data
# Your model will work with both
```

**Then, when you get real CSI data:**
- Replace mock data with real data
- Model architecture stays the same
- Just swap the data source

## Step-by-Step: Getting Real Basic WiFi Data on Mac

### Complete Process:

1. **Connect to WiFi**:
   ```
   System Preferences > Network > Wi-Fi
   Select your Ruckus network
   Enter password if needed
   ```

2. **Verify Connection**:
   ```bash
   /usr/sbin/networksetup -getairportnetwork en0
   # Should show: "Current Wi-Fi Network: YourNetworkName"
   ```

3. **Check Airport Utility**:
   ```bash
   /System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -I
   ```

4. **If Airport Works**:
   ```bash
   source venv/bin/activate
   python scripts/collect_wifi_data.py --duration 60
   # You'll get real RSSI, signal strength, channel info
   ```

5. **If Airport Doesn't Work**:
   - You'll only get SSID and basic info
   - For full CSI, you need Linux or Ruckus API access

## For Your Ruckus Setup Specifically

### Contact Your Network Administrator:

1. **Ask about**:
   - Ruckus API access
   - CSI data availability
   - WiFi analytics features
   - Management interface access

2. **Ruckus Models that Support CSI**:
   - Some Ruckus access points support CSI via API
   - Check Ruckus documentation for your specific model
   - Ruckus Unleashed/Cloud may have analytics

3. **Alternative**:
   - Use a Linux laptop/computer near the Ruckus AP
   - Install CSI collection tools
   - Collect data from there

## Quick Test: Check What You Can Get

Run this to see what's available:

```bash
source venv/bin/activate
python -c "
from src.data_collection.wifi_collector import WiFiCollector
import subprocess

# Check WiFi connection
result = subprocess.run(['/usr/sbin/networksetup', '-getairportnetwork', 'en0'], 
                       capture_output=True, text=True)
print('WiFi Status:', result.stdout.strip())

# Try to collect one sample
collector = WiFiCollector()
sample = collector.get_wifi_info()
print('\nCollected Sample:')
print(f'  Method: {sample.get(\"collection_method\")}')
print(f'  SSID: {sample.get(\"ssid\")}')
print(f'  RSSI: {sample.get(\"rssi\")}')
print(f'  Signal: {sample.get(\"signal_strength\")}')
print(f'  Channel: {sample.get(\"channel\")}')
"
```

## Summary

### What You CAN Get on Mac:
- ✅ Basic WiFi info (SSID) - via networksetup
- ✅ Real RSSI/Signal (if airport utility works) - rare on newer Macs
- ❌ Full CSI (amplitude/phase vectors) - NOT available on Mac

### What You NEED for Full CSI:
- ✅ Linux system with compatible WiFi card
- ✅ OR Ruckus API access
- ✅ OR External CSI collection device

### For Your Project NOW:
- ✅ Use mock data for baseline model development
- ✅ Model will work with real CSI when you get it
- ✅ Focus on architecture, not data source

## Next Steps

1. **Try connecting to WiFi** and collecting data
2. **Contact network admin** about Ruckus API access
3. **Use mock data** for now to develop your baseline model
4. **Plan for real CSI** collection later (Linux system or Ruckus API)

Would you like me to help you:
- Test WiFi connection and collection?
- Set up mock data collection?
- Create a script to check what data you can get?

