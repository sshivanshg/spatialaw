#!/usr/bin/env python3
"""
WiFi Data Collection with Position Coordinates
Collects WiFi data from different positions within a single room
Usage: python collect_with_position.py --location Room_101 --x 2.5 --y 3.0
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
DEFAULT_DURATION = 30.0  # seconds (shorter for position-based collection)
DEFAULT_SAMPLING_RATE = 2.0  # Hz
DEFAULT_INTERFACE = "en0"
OUTPUT_DIR = "data"


class WiFiCollector:
    """WiFi collector with position tracking."""
    
    def __init__(self, interface: str = "en0", sampling_rate: float = 1.0):
        self.interface = interface
        self.sampling_rate = sampling_rate
        self.data = []
        self.collection_method = None
        self._detect_collection_method()
    
    def _detect_collection_method(self):
        """Detect available WiFi collection method."""
        # Try system_profiler first (most reliable on Mac)
        try:
            result = subprocess.run(
                ['system_profiler', 'SPAirPortDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and 'IEEE' in result.stdout:
                self.collection_method = 'system_profiler'
                return
        except:
            pass
        
        # Try airport utility
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
                if result.returncode == 0:
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
            
            # Extract SSID
            ssid_match = re.search(r'Current Network Information:\s*\n\s*([^\n:]+):\s*([^\n]+)', output)
            if not ssid_match:
                # Try alternative pattern
                ssid_match = re.search(r'([^\n:]+):\s*(.+)', output)
            
            if not ssid_match:
                return None
            
            ssid = ssid_match.group(2).strip() if ssid_match else "Unknown"
            
            # Extract Signal/Noise
            signal_noise_match = re.search(r'Signal[/\\]Noise:\s*(\d+)\s*dBm[/\\](\d+)\s*dBm', output)
            if signal_noise_match:
                signal = int(signal_noise_match.group(1))
                noise = int(signal_noise_match.group(2))
                rssi = signal
                snr = signal - noise
            else:
                # Try alternative patterns
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
            
            # Calculate signal strength (0-100 scale, approximate)
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
                'num_antennas': 1,  # Default, can't detect on Mac
                'signals': [rssi],
                'noises': [noise],
                'collection_method': 'system_profiler'
            }
        except Exception as e:
            return None
    
    def get_wifi_info(self) -> Optional[Dict]:
        """Get current WiFi information."""
        if self.collection_method == 'system_profiler':
            return self._get_info_from_system_profiler()
        return None
    
    def collect_sample(self, duration: float, position: Dict[str, float]) -> List[Dict]:
        """Collect WiFi samples for a given duration at a specific position."""
        samples = []
        num_samples = int(duration * self.sampling_rate)
        sample_interval = 1.0 / self.sampling_rate
        
        print(f"Collecting {num_samples} samples at position ({position['x']}, {position['y']})...")
        
        for i in range(num_samples):
            wifi_info = self.get_wifi_info()
            if wifi_info:
                # Add position and metadata
                wifi_info['position_x'] = position['x']
                wifi_info['position_y'] = position['y']
                wifi_info['timestamp'] = datetime.now().isoformat()
                wifi_info['unix_timestamp'] = time.time()
                wifi_info['device_id'] = socket.gethostname()
                wifi_info['device_hostname'] = socket.gethostname()
                wifi_info['device_platform'] = platform.platform()
                
                samples.append(wifi_info)
            
            if i < num_samples - 1:
                time.sleep(sample_interval)
        
        return samples


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


def collect_data_at_position(
    location: str,
    x: float,
    y: float,
    duration: float = DEFAULT_DURATION,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    notes: Optional[str] = None
):
    """Collect WiFi data at a specific position."""
    collector = WiFiCollector(interface=DEFAULT_INTERFACE, sampling_rate=sampling_rate)
    
    position = {'x': x, 'y': y}
    samples = collector.collect_sample(duration, position)
    
    if not samples:
        print("❌ No samples collected. Check your WiFi connection.")
        return None
    
    # Add location and notes to all samples
    for sample in samples:
        sample['location'] = location
        if notes:
            sample['notes'] = notes
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{location}_x{x}_y{y}_{timestamp}.json"
    filepath = os.path.join(OUTPUT_DIR, location, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Make serializable
    samples = make_serializable(samples)
    
    with open(filepath, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"✅ Collected {len(samples)} samples")
    print(f"✅ Saved to: {filepath}")
    
    return filepath


def interactive_collection(location: str, duration: float = DEFAULT_DURATION):
    """Interactive data collection - prompt for positions."""
    print(f"\n{'='*60}")
    print(f"Interactive WiFi Data Collection for: {location}")
    print(f"{'='*60}\n")
    print("Instructions:")
    print("  - Enter position coordinates (x, y) for each collection point")
    print("  - Units: meters (or any consistent unit)")
    print("  - Origin (0, 0) should be a corner of the room")
    print("  - Enter 'q' to quit\n")
    
    all_samples = []
    
    while True:
        try:
            position_str = input("Enter position (x, y) or 'q' to quit: ").strip()
            if position_str.lower() == 'q':
                break
            
            parts = position_str.split(',')
            if len(parts) != 2:
                print("❌ Invalid format. Use: x, y (e.g., 2.5, 3.0)")
                continue
            
            x = float(parts[0].strip())
            y = float(parts[1].strip())
            
            print(f"\nCollecting at position ({x}, {y})...")
            filepath = collect_data_at_position(location, x, y, duration)
            
            if filepath:
                # Load and add to all_samples
                with open(filepath, 'r') as f:
                    samples = json.load(f)
                all_samples.extend(samples)
                print(f"✅ Total samples collected: {len(all_samples)}\n")
        
        except ValueError:
            print("❌ Invalid number format. Use: x, y (e.g., 2.5, 3.0)")
        except KeyboardInterrupt:
            print("\n\nCollection interrupted.")
            break
    
    # Save combined data
    if all_samples:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filepath = os.path.join(OUTPUT_DIR, location, f"{location}_all_positions_{timestamp}.json")
        
        with open(combined_filepath, 'w') as f:
            json.dump(all_samples, f, indent=2)
        
        print(f"\n✅ Combined data saved to: {combined_filepath}")
        print(f"✅ Total samples: {len(all_samples)}")
        print(f"✅ Unique positions: {len(set((s['position_x'], s['position_y']) for s in all_samples))}")


def main():
    parser = argparse.ArgumentParser(description='Collect WiFi data with position coordinates')
    parser.add_argument('--location', type=str, required=True, help='Location/room name')
    parser.add_argument('--x', type=float, help='X coordinate (meters)')
    parser.add_argument('--y', type=float, help='Y coordinate (meters)')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION, help='Collection duration (seconds)')
    parser.add_argument('--sampling_rate', type=float, default=DEFAULT_SAMPLING_RATE, help='Sampling rate (Hz)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode (prompt for positions)')
    parser.add_argument('--notes', type=str, help='Notes about this collection')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_collection(args.location, args.duration)
    elif args.x is not None and args.y is not None:
        collect_data_at_position(
            args.location,
            args.x,
            args.y,
            args.duration,
            args.sampling_rate,
            args.notes
        )
    else:
        print("❌ Error: Either provide --x and --y, or use --interactive mode")
        parser.print_help()


if __name__ == "__main__":
    main()

