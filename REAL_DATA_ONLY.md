# Real Data Only - No Mock Data

## Overview

This project **ONLY** collects and uses **REAL WiFi data**. Mock data functionality has been completely removed.

## Why Real Data Only?

- **Accurate Training**: Real data ensures your model learns from actual WiFi signal patterns
- **Real-World Performance**: Models trained on real data perform better in real scenarios
- **Data Integrity**: No confusion between real and synthetic data
- **Research Quality**: Real data is essential for valid research results

## Requirements

### WiFi Connection

**You MUST be connected to WiFi** before starting data collection:

1. ✅ Connect to a WiFi network
2. ✅ Ensure WiFi is enabled on your Mac
3. ✅ Verify connection is stable

### System Requirements

- Mac with WiFi capability
- `system_profiler` or `networksetup` available (usually built-in)
- Network permissions enabled

## Error Messages

If you see errors like:

```
❌ ERROR: No real WiFi data collection method available!
```

This means:
- You are not connected to WiFi, OR
- Your system cannot access WiFi data collection tools, OR
- Network permissions are not granted

**Solutions:**
1. Connect to WiFi
2. Check System Preferences > Security & Privacy > Privacy > Location Services
3. Ensure you have network permissions
4. Try a different network interface (use `--interface` flag)

## Data Collection

### Basic Collection

```bash
# Collect real WiFi data (requires WiFi connection)
python scripts/collect_wifi_data.py --duration 60
```

### Location-Based Collection

```bash
# Collect at specific location (requires WiFi connection)
python scripts/collect_with_location.py --location "Room 101" --duration 60
```

### Multi-Location Collection

```bash
# Collect from multiple locations (requires WiFi connection at each location)
python scripts/collect_multi_location.py --locations "Room 101" "Room 102" --duration 60
```

## What Happens If WiFi Is Not Available?

The system will **FAIL** with a clear error message. It will **NOT** generate mock data.

**Example Error:**
```
❌ ERROR: No real WiFi data collection method available!
   This system cannot collect real WiFi data.
   Please ensure:
   1. You are connected to WiFi
   2. Your Mac has system_profiler or networksetup available
   3. You have appropriate permissions
   
   Note: Mock data is not supported. Only real WiFi data collection is allowed.
```

## Verification

### Check WiFi Status

Before collecting data, verify your WiFi connection:

```bash
# Check WiFi status
python scripts/check_wifi_status.py
```

### Validate Collected Data

After collection, validate that data is real:

```bash
# Validate collected data
python scripts/validate_collected_data.py data/wifi_data.json
```

The validation will show:
- ✅ Real data: `collection_method: system_profiler` or `airport` or `networksetup`
- ❌ Mock data: Will not appear (mock data is not supported)

## Training

All training scripts require real data:

```bash
# Train with real data
python scripts/train_baseline.py --data_path data/wifi_data.json
```

**Note**: Training scripts will fail if you try to use mock data or if no real data is available.

## Troubleshooting

### Problem: "No real WiFi data collection method available"

**Solution:**
1. Connect to WiFi
2. Check System Preferences > Network > WiFi
3. Verify `system_profiler` is available: `system_profiler SPAirPortDataType`
4. Try different network interface: `--interface en1`

### Problem: "Not connected to WiFi"

**Solution:**
1. Connect to a WiFi network
2. Wait for connection to stabilize
3. Verify connection: `ping -c 1 google.com`
4. Try again

### Problem: "Collection method: networksetup" but no data

**Solution:**
1. This means you're using `networksetup` which has limited capabilities
2. Try to use `system_profiler` instead (usually better)
3. Check if `system_profiler SPAirPortDataType` works
4. Ensure you're connected to WiFi

## Best Practices

1. **Always verify WiFi connection** before starting collection
2. **Use location-based collection** for organized data
3. **Validate data** after collection to ensure it's real
4. **Check collection method** in metadata to verify real data
5. **Collect at multiple locations** for diverse training data

## Summary

- ✅ **Real data only** - No mock data
- ✅ **Clear error messages** - System fails clearly if real data unavailable
- ✅ **WiFi connection required** - Must be connected to WiFi
- ✅ **Data validation** - Verify data is real after collection
- ✅ **Training on real data** - All models trained on real WiFi data

## Migration from Mock Data

If you previously used mock data:

1. **Remove `--use_mock` flags** - These no longer exist
2. **Connect to WiFi** - Required for all collection
3. **Update scripts** - Remove any mock data references
4. **Validate data** - Ensure all data is real

## Support

If you encounter issues:

1. Check WiFi connection
2. Verify system permissions
3. Check error messages for specific issues
4. Validate collected data
5. Review this document

---

**Remember**: This project is designed for real-world WiFi data collection. Mock data is not supported to ensure research quality and real-world performance.

