#!/usr/bin/env python3
"""
Check WiFi Status and Data Collection Capabilities
"""

import sys
import os
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.wifi_collector import WiFiCollector


def check_wifi_status():
    """Check current WiFi connection status."""
    print("=" * 60)
    print("WiFi Status Check")
    print("=" * 60)
    print()
    
    # Check networksetup
    print("1. Checking WiFi Connection Status...")
    try:
        result = subprocess.run(
            ["/usr/sbin/networksetup", "-getairportnetwork", "en0"],
            capture_output=True,
            text=True,
            timeout=2
        )
        output = result.stdout.strip()
        print(f"   Status: {output}")
        
        if "not associated" in output.lower() or "not connected" in output.lower():
            print("    NOT CONNECTED to WiFi")
            print("   → Connect to WiFi to get real data")
        elif ":" in output:
            ssid = output.split(":", 1)[1].strip()
            print(f"    CONNECTED to: {ssid}")
        else:
            print("     Unknown status")
    except Exception as e:
        print(f"   Error: {e}")
    
    print()
    
    # Check airport utility
    print("2. Checking Airport Utility...")
    airport_paths = [
        "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
        "/System/Library/PrivateFrameworks/Apple80211.framework/Resources/airport",
    ]
    
    airport_found = False
    for path in airport_paths:
        if os.path.exists(path):
            print(f"    Found: {path}")
            try:
                result = subprocess.run(
                    [path, "-I"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0 and result.stdout:
                    print("    Airport utility WORKS!")
                    print("   → You can get real RSSI, signal strength, channel info")
                    # Show sample output
                    lines = result.stdout.strip().split('\n')[:5]
                    print("   Sample output:")
                    for line in lines:
                        print(f"     {line}")
                    airport_found = True
                    break
                else:
                    print(f"     Airport found but not working")
            except Exception as e:
                print(f"     Airport found but error: {e}")
    
    if not airport_found:
        print("    Airport utility NOT FOUND or NOT WORKING")
        print("   → This is normal on newer Macs")
        print("   → You'll only get basic WiFi info (SSID)")
    
    print()
    
    # Test data collection
    print("3. Testing Data Collection...")
    try:
        collector = WiFiCollector()
        sample = collector.get_wifi_info()
        
        print(f"   Collection method: {sample.get('collection_method', 'unknown')}")
        print(f"   SSID: {sample.get('ssid', 'N/A')}")
        print(f"   RSSI: {sample.get('rssi', 'N/A')}")
        print(f"   Signal Strength: {sample.get('signal_strength', 'N/A')}")
        print(f"   Channel: {sample.get('channel', 'N/A')}")
        
        if sample.get('collection_method') == 'mock':
            print("     Using mock data (not real WiFi data)")
        elif sample.get('collection_method') == 'networksetup':
            if sample.get('ssid') == 'Unknown':
                print("     Not connected to WiFi - using placeholder values")
            else:
                print("    Connected to WiFi - but limited data available")
        elif sample.get('collection_method') == 'airport':
            print("    Using airport utility - real WiFi data available!")
    except Exception as e:
        print(f"    Error: {e}")
    
    print()
    print("=" * 60)
    print("Summary & Recommendations")
    print("=" * 60)
    print()
    
    # Provide recommendations
    if not airport_found:
        print("📋 For Real WiFi Data:")
        print("   1. Connect to WiFi network")
        print("   2. You'll get SSID and basic info")
        print("   3. For full CSI data, you need:")
        print("      - Linux system with compatible WiFi card")
        print("      - OR Ruckus API access")
        print("      - OR External CSI collection device")
        print()
        print("💡 For Development:")
        print("   Use mock data: python scripts/collect_wifi_data.py --duration 60 --use_mock")
        print("   This provides realistic variation for model development")
    else:
        print(" You can collect real WiFi data!")
        print("   Run: python scripts/collect_wifi_data.py --duration 60")
        print("   Note: You'll get RSSI/signal info, but not full CSI")
    
    print()


if __name__ == "__main__":
    check_wifi_status()

