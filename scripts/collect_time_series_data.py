#!/usr/bin/env python3
"""
Time-Series WiFi Data Collection for Motion Detection
Collects WiFi signals over time with movement/no-movement labels
"""

import subprocess
import json
import time
import numpy as np
import os
import sys
import socket
import platform
import re
from datetime import datetime
from typing import Dict, Optional, List
import argparse

# Configuration
DEFAULT_DURATION = 60.0  # seconds
DEFAULT_SAMPLING_RATE = 10.0  # Hz (higher for motion detection)
DEFAULT_INTERFACE = "en0"
OUTPUT_DIR = "data"


class WiFiCollector:
    """WiFi collector for time-series data."""
    
    def __init__(self, interface: str = "en0", sampling_rate: float = 10.0):
        self.interface = interface
        self.sampling_rate = sampling_rate
        self.data = []
        self.collection_method = None
        self._detect_collection_method()
    
    def _detect_collection_method(self):
        """Detect available WiFi collection method."""
        # Try system_profiler first (most reliable on macOS)
        try:
            result = subprocess.run(
                ['system_profiler', 'SPAirPortDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and len(result.stdout) > 0:
                # Check if we can actually get WiFi info
                test_info = self._get_info_from_system_profiler()
                if test_info and test_info.get('ssid'):
                    self.collection_method = 'system_profiler'
                    return
        except:
            pass
        
        # Try airport utility as fallback
        airport_paths = [
            '/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport',
            '/usr/local/bin/airport',
            'airport'
        ]
        
        for path in airport_paths:
            try:
                result = subprocess.run(
                    [path, '-I'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and 'SSID' in result.stdout:
                    self.collection_method = 'airport'
                    self.airport_path = path
                    return
            except:
                continue
        
        raise RuntimeError("No WiFi collection method available. Please check your WiFi connection.")
    
    def _get_info_from_system_profiler(self) -> Optional[Dict]:
        """Get WiFi info using system_profiler."""
        try:
            result = subprocess.run(
                ['system_profiler', 'SPAirPortDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return None
            
            output = result.stdout
            
            # Extract SSID - try multiple patterns
            # Pattern 1: Current Network Information: SSID_NAME: ...
            ssid_match = re.search(r'Current Network Information:\s*\n\s*([^\n:]+?):', output)
            if ssid_match:
                ssid = ssid_match.group(1).strip()
                # Skip if it's redacted or empty
                if ssid and ssid != '<redacted>' and ssid != 'Unknown':
                    pass  # Use this SSID
                else:
                    ssid = None
            else:
                ssid = None
            
            # Pattern 2: Look for SSID field directly
            if not ssid or ssid == '<redacted>':
                ssid_match = re.search(r'SSID:\s*([^\n]+)', output, re.IGNORECASE)
                if ssid_match:
                    ssid = ssid_match.group(1).strip()
            
            # Pattern 3: Network Name field
            if not ssid or ssid == '<redacted>':
                ssid_match = re.search(r'Network Name:\s*([^\n]+)', output, re.IGNORECASE)
                if ssid_match:
                    ssid = ssid_match.group(1).strip()
            
            # If still no valid SSID, check if we're connected by looking for "Status: Connected"
            if not ssid or ssid == '<redacted>' or ssid == 'Unknown':
                # Check if connected - if yes, use a placeholder
                if 'Status: Connected' in output:
                    # We're connected but SSID is redacted - use a placeholder
                    ssid = "Connected_Network"
                else:
                    # Not connected
                    return None
            
            # Extract Signal/Noise - try multiple patterns
            # Pattern 1: "Signal / Noise: -68 dBm / -93 dBm" (with spaces)
            signal_noise_match = re.search(r'Signal\s*[/\\]\s*Noise:\s*(-?\d+)\s*dBm\s*[/\\]\s*(-?\d+)\s*dBm', output)
            if signal_noise_match:
                signal = int(signal_noise_match.group(1))
                noise = int(signal_noise_match.group(2))
                rssi = signal
                snr = signal - noise
            else:
                # Pattern 2: "Signal/Noise: -68 dBm / -93 dBm" (no spaces around /)
                signal_noise_match = re.search(r'Signal[/\\]Noise:\s*(-?\d+)\s*dBm\s*[/\\]\s*(-?\d+)\s*dBm', output)
                if signal_noise_match:
                    signal = int(signal_noise_match.group(1))
                    noise = int(signal_noise_match.group(2))
                    rssi = signal
                    snr = signal - noise
                else:
                    # Pattern 3: Separate Signal and Noise fields
                    signal_match = re.search(r'Signal:\s*(-?\d+)', output)
                    noise_match = re.search(r'Noise:\s*(-?\d+)', output)
                    signal = int(signal_match.group(1)) if signal_match else -70
                    noise = int(noise_match.group(1)) if noise_match else -90
                    rssi = signal
                    snr = signal - noise
            
            # Extract Channel
            channel_match = re.search(r'Channel:\s*(\d+)', output)
            channel = int(channel_match.group(1)) if channel_match else 44
            
            # Extract PHY Mode
            phy_match = re.search(r'PHY Mode:\s*([^\n]+)', output)
            phy_mode = phy_match.group(1).strip() if phy_match else "802.11ac"
            
            # Extract MAC Address (BSSID)
            mac_match = re.search(r'MAC Address:\s*([0-9a-fA-F:]{17})', output)
            bssid = mac_match.group(1) if mac_match else "unknown"
            
            # Calculate signal strength
            signal_strength = max(0, min(100, ((rssi + 100) * 100 / 70)))
            
            return {
                'ssid': ssid,
                'rssi': rssi,
                'noise': noise,
                'snr': snr,
                'signal_strength': int(signal_strength),
                'channel': channel,
                'phy_mode': phy_mode,
                'bssid': bssid,
                'num_antennas': 1,
                'signals': [rssi],
                'noises': [noise],
                'collection_method': 'system_profiler'
            }
        except Exception as e:
            return None
    
    def collect_time_series(self, duration: float, movement_label: bool) -> List[Dict]:
        """Collect WiFi samples over time with movement label."""
        samples = []
        num_samples = int(duration * self.sampling_rate)
        sample_interval = 1.0 / self.sampling_rate
        
        label_text = "MOVEMENT" if movement_label else "NO_MOVEMENT"
        print(f"Collecting {num_samples} samples ({label_text})...")
        print(f"Duration: {duration} seconds, Sampling rate: {self.sampling_rate} Hz")
        print(f"Please {'move around' if movement_label else 'stay still'} during collection...")
        
        start_time = time.time()
        
        for i in range(num_samples):
            wifi_info = self.get_wifi_info()
            if wifi_info:
                # Add timestamp and movement label
                wifi_info['timestamp'] = datetime.now().isoformat()
                wifi_info['unix_timestamp'] = time.time()
                wifi_info['time_index'] = i
                wifi_info['elapsed_time'] = time.time() - start_time
                wifi_info['movement'] = movement_label
                wifi_info['movement_label'] = 1 if movement_label else 0
                wifi_info['device_id'] = socket.gethostname()
                wifi_info['device_hostname'] = socket.gethostname()
                wifi_info['device_platform'] = platform.platform()
                
                samples.append(wifi_info)
            
            if i < num_samples - 1:
                time.sleep(sample_interval)
            
            # Progress indicator
            if (i + 1) % int(self.sampling_rate) == 0:
                elapsed = time.time() - start_time
                print(f"  Collected {i+1}/{num_samples} samples ({elapsed:.1f}s elapsed)")
        
        return samples
    
    def get_wifi_info(self) -> Optional[Dict]:
        """Get current WiFi information."""
        if self.collection_method == 'system_profiler':
            return self._get_info_from_system_profiler()
        elif self.collection_method == 'airport':
            return self._get_info_from_airport()
        return None
    
    def _get_info_from_airport(self) -> Optional[Dict]:
        """Get WiFi info using airport utility."""
        try:
            result = subprocess.run(
                [self.airport_path, '-I'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return None
            
            output = result.stdout
            
            # Parse airport output
            ssid_match = re.search(r'SSID:\s*([^\n]+)', output)
            ssid = ssid_match.group(1).strip() if ssid_match else "Unknown"
            
            rssi_match = re.search(r'agrCtlRSSI:\s*(-?\d+)', output)
            rssi = int(rssi_match.group(1)) if rssi_match else -70
            
            noise_match = re.search(r'agrCtlNoise:\s*(-?\d+)', output)
            noise = int(noise_match.group(1)) if noise_match else -90
            
            snr = rssi - noise
            
            channel_match = re.search(r'channel:\s*(\d+)', output, re.IGNORECASE)
            channel = int(channel_match.group(1)) if channel_match else 44
            
            bssid_match = re.search(r'BSSID:\s*([0-9a-fA-F:]{17})', output)
            bssid = bssid_match.group(1) if bssid_match else "unknown"
            
            signal_strength = max(0, min(100, ((rssi + 100) * 100 / 70)))
            
            return {
                'ssid': ssid,
                'rssi': rssi,
                'noise': noise,
                'snr': snr,
                'signal_strength': int(signal_strength),
                'channel': channel,
                'phy_mode': '802.11ac',  # Default
                'bssid': bssid,
                'num_antennas': 1,
                'signals': [rssi],
                'noises': [noise],
                'collection_method': 'airport'
            }
        except Exception as e:
            return None
    
    def is_wifi_connected(self) -> bool:
        """Check if WiFi is connected."""
        wifi_info = self.get_wifi_info()
        return wifi_info is not None and wifi_info.get('ssid') and wifi_info.get('ssid') != "Unknown"


def make_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    return obj


def collect_motion_data(
    location: str,
    movement_label: bool,
    duration: float = DEFAULT_DURATION,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    notes: Optional[str] = None
):
    """Collect time-series WiFi data with movement label."""
    collector = WiFiCollector(interface=DEFAULT_INTERFACE, sampling_rate=sampling_rate)
    
    if not collector.is_wifi_connected():
        print("❌ ERROR: WiFi is not connected. Please connect to a WiFi network and try again.")
        return None
    
    samples = collector.collect_time_series(duration, movement_label)
    
    if not samples:
        print("❌ No samples collected. Check your WiFi connection.")
        return None
    
    # Add location and notes to all samples
    for sample in samples:
        sample['location'] = location
        if notes:
            sample['notes'] = notes
    
    # Save to file
    label_text = "movement" if movement_label else "no_movement"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{location}_{label_text}_{timestamp}.json"
    filepath = os.path.join(OUTPUT_DIR, location, "time_series", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Make serializable
    samples = make_serializable(samples)
    
    with open(filepath, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✅ Collected {len(samples)} samples")
    print(f"✅ Label: {label_text}")
    print(f"✅ Saved to: {filepath}")
    
    return filepath


def interactive_collection(location: str, duration: float = DEFAULT_DURATION, sampling_rate: float = DEFAULT_SAMPLING_RATE):
    """Interactive data collection - prompt for movement labels."""
    print(f"\n{'='*60}")
    print(f"Interactive Time-Series Data Collection for: {location}")
    print(f"{'='*60}\n")
    print("Instructions:")
    print("  - Collect data with movement (someone moving in room)")
    print("  - Collect data without movement (empty room, stay still)")
    print("  - Enter 'q' to quit\n")
    
    all_samples = []
    
    while True:
        try:
            movement_str = input("Enter movement label (1=movement, 0=no_movement) or 'q' to quit: ").strip()
            if movement_str.lower() == 'q':
                break
            
            movement_label = movement_str == '1' or movement_str.lower() == 'movement'
            
            print(f"\nCollecting data with label: {'MOVEMENT' if movement_label else 'NO_MOVEMENT'}...")
            filepath = collect_motion_data(location, movement_label, duration, sampling_rate)
            
            if filepath:
                # Load and add to all_samples
                with open(filepath, 'r') as f:
                    samples = json.load(f)
                all_samples.extend(samples)
                print(f"✅ Total samples collected: {len(all_samples)}\n")
        
        except ValueError:
            print("❌ Invalid input. Use: 1 (movement) or 0 (no_movement)")
        except KeyboardInterrupt:
            print("\n\nCollection interrupted.")
            break
    
    # Save combined data
    if all_samples:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filepath = os.path.join(OUTPUT_DIR, location, "time_series", f"{location}_all_time_series_{timestamp}.json")
        
        with open(combined_filepath, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        print(f"\n✅ Combined data saved to: {combined_filepath}")
        print(f"✅ Total samples: {len(all_samples)}")
        print(f"✅ Movement samples: {sum(1 for s in all_samples if s.get('movement', False))}")
        print(f"✅ No-movement samples: {sum(1 for s in all_samples if not s.get('movement', False))}")


def main():
    parser = argparse.ArgumentParser(description='Collect Time-Series WiFi Data for Motion Detection')
    parser.add_argument('--location', type=str, required=True, help='Location/room name')
    parser.add_argument('--movement', action='store_true', help='Collect data with movement')
    parser.add_argument('--no_movement', action='store_true', help='Collect data without movement')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION, help='Collection duration (seconds)')
    parser.add_argument('--sampling_rate', type=float, default=DEFAULT_SAMPLING_RATE, help='Sampling rate (Hz)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--notes', type=str, help='Notes about this collection')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_collection(args.location, args.duration, args.sampling_rate)
    elif args.movement or args.no_movement:
        movement_label = args.movement
        collect_motion_data(
            args.location,
            movement_label,
            args.duration,
            args.sampling_rate,
            args.notes
        )
    else:
        print("❌ Error: Either provide --movement or --no_movement, or use --interactive mode")
        parser.print_help()


if __name__ == "__main__":
    main()

