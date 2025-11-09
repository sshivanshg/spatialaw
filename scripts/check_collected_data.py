#!/usr/bin/env python3
"""
Check Collected WiFi Data
Shows summary of collected WiFi data files
"""

import sys
import os
import json
import glob
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def check_data_file(filepath):
    """Check a single data file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not data:
            print(f"  ⚠️  File is empty")
            return
        
        df = pd.DataFrame(data)
        
        print(f"\n  File: {os.path.basename(filepath)}")
        print(f"  Samples: {len(data)}")
        print(f"  Collection method: {data[0].get('collection_method', 'unknown')}")
        print(f"  SSID: {data[0].get('ssid', 'N/A')}")
        print(f"  RSSI range: {df['rssi'].min()} to {df['rssi'].max()} dBm")
        print(f"  Channel: {data[0].get('channel', 'N/A')}")
        print(f"  Signal strength: {df['signal_strength'].min():.1f}% to {df['signal_strength'].max():.1f}%")
        print(f"  SNR: {df['snr'].min():.1f} to {df['snr'].max():.1f} dB")
        print(f"  Unique RSSI values: {df['rssi'].nunique()}")
        
        # Check if it's real data
        if data[0].get('collection_method') == 'system_profiler':
            print(f"   REAL WiFi data!")
        elif data[0].get('collection_method') == 'mock':
            print(f"   Mock data (not real WiFi)")
        else:
            print(f"  Limited data")
        
    except Exception as e:
        print(f"  Error reading file: {e}")


def main():
    print("=" * 60)
    print("WiFi Data Files Check")
    print("=" * 60)
    
    # Find all WiFi data files
    wifi_files = glob.glob('data/wifi_data_*.json') + glob.glob('data/test_real_wifi.json')
    wifi_files = sorted(set(wifi_files))  # Remove duplicates and sort
    
    if not wifi_files:
        print("\n No WiFi data files found in data/ directory")
        print("\nTo collect data, run:")
        print("  python scripts/collect_wifi_data.py --duration 60")
        return
    
    print(f"\nFound {len(wifi_files)} WiFi data file(s):")
    
    for filepath in wifi_files:
        check_data_file(filepath)
    
    # Show latest file in detail
    if wifi_files:
        latest_file = max(wifi_files, key=os.path.getmtime)
        print(f"\n" + "=" * 60)
        print(f"Latest File: {os.path.basename(latest_file)}")
        print("=" * 60)
        check_data_file(latest_file)
        
        # Show sample data
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            print(f"\nSample entry:")
            print(json.dumps(data[0], indent=2))
        except Exception as e:
            print(f"Error showing sample: {e}")


if __name__ == "__main__":
    main()

