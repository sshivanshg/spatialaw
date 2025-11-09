#!/usr/bin/env python3
"""
Standalone WiFi Data Collection Script
Single file that can be copied to any laptop and run directly
Usage: python collect.py --location Room_101
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

# Configuration
DEFAULT_DURATION = 600.0  # seconds
DEFAULT_SAMPLING_RATE = 2.0  # Hz
DEFAULT_INTERFACE = "en0"
OUTPUT_DIR = "data"


class WiFiCollector:
    """Simple WiFi collector for standalone use."""
    
    def __init__(self, interface: str = "en0", sampling_rate: float = 1.0):
        self.interface = interface
        self.sampling_rate = sampling_rate
        self.data = []
        self.collection_method = None
        self._detect_collection_method()
    
    def _detect_collection_method(self):
        """Detect available WiFi collection method."""
        # Try system_profiler (best on newer Macs)
        try:
            result = subprocess.run(
                ["system_profiler", "SPAirPortDataType"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and "Status: Connected" in result.stdout:
                self.collection_method = "system_profiler"
                return
        except:
            pass
        
        # Try networksetup
        try:
            result = subprocess.run(
                ["/usr/sbin/networksetup", "-getairportnetwork", self.interface],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                self.collection_method = "networksetup"
                return
        except:
            pass
        
        raise RuntimeError(
            "❌ ERROR: Cannot collect real WiFi data!\n"
            "   Please ensure:\n"
            "   1. You are connected to WiFi\n"
            "   2. WiFi is enabled on your Mac"
        )
    
    def _get_info_from_system_profiler(self) -> Optional[Dict]:
        """Get WiFi info from system_profiler."""
        try:
            result = subprocess.run(
                ["/usr/sbin/system_profiler", "SPAirPortDataType"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return None
            
            output = result.stdout
            
            # Check if connected
            if "Status: Connected" not in output and "Current Network Information" not in output:
                return None
            
            info = {}
            info['ssid'] = 'Connected'  # Default - SSID might be redacted
            
            # Split output to get only the current network section
            current_network_section = ""
            if "Current Network Information:" in output:
                sections = output.split("Current Network Information:")
                if len(sections) > 1:
                    current_network_section = sections[1].split("Other Local")[0] if "Other Local" in sections[1] else sections[1]
            
            # Parse Signal/Noise values
            signal_pattern = r'Signal / Noise: (-?\d+) dBm / (-?\d+) dBm'
            signal_matches = re.findall(signal_pattern, current_network_section if current_network_section else output)
            
            if signal_matches:
                signals = [int(s[0]) for s in signal_matches]
                noises = [int(s[1]) for s in signal_matches]
                info['rssi'] = max(signals)
                info['noise'] = max(noises) if noises else -91
                info['snr'] = info['rssi'] - info['noise']
                info['signal_strength'] = max(0, min(100, int(100 * (info['rssi'] + 100) / 70)))
                info['signals'] = signals
                info['noises'] = noises
                info['num_antennas'] = len(signals)
            else:
                # Try alternative pattern
                signal_match = re.search(r'Signal:\s*(-?\d+)', output, re.IGNORECASE)
                if signal_match:
                    info['rssi'] = int(signal_match.group(1))
                    info['signal_strength'] = max(0, min(100, 2 * (info['rssi'] + 100)))
                    info['noise'] = -91
                    info['snr'] = info['rssi'] - info['noise']
                    info['signals'] = [info['rssi']]
                    info['noises'] = [-91]
                    info['num_antennas'] = 1
                else:
                    return None
            
            # Parse Channel
            channel_match = re.search(r'Channel:\s*(\d+)', output)
            if channel_match:
                info['channel'] = int(channel_match.group(1))
            else:
                info['channel'] = 0
            
            # Parse PHY Mode
            phy_match = re.search(r'PHY Mode:\s*(802\.11\w+)', output)
            if phy_match:
                info['phy_mode'] = phy_match.group(1)
            else:
                info['phy_mode'] = '802.11ac'
            
            # Parse MAC Address
            mac_match = re.search(r'MAC Address:\s*([0-9A-Fa-f:]{17})', output)
            if mac_match:
                info['bssid'] = mac_match.group(1)
            else:
                info['bssid'] = 'Unknown'
            
            return info
            
        except Exception:
            return None
    
    def get_wifi_info(self) -> Dict:
        """Get current WiFi information."""
        if self.collection_method == "system_profiler":
            wifi_info = self._get_info_from_system_profiler()
            if wifi_info:
                wifi_info['timestamp'] = datetime.now().isoformat()
                wifi_info['unix_timestamp'] = time.time()
                wifi_info['collection_method'] = 'system_profiler'
                return wifi_info
        
        raise RuntimeError("❌ Failed to collect WiFi data. Please check your connection.")
    
    def collect_continuous(self, duration: float):
        """Collect WiFi data continuously."""
        print(f"Starting collection for {duration} seconds...")
        print(f"Collection method: {self.collection_method}\n")
        
        start_time = time.time()
        sample_count = 0
        
        try:
            while time.time() - start_time < duration:
                try:
                    sample = self.get_wifi_info()
                    self.data.append(sample)
                    sample_count += 1
                    
                    if sample_count % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"✓ Collected {sample_count} samples (elapsed: {elapsed:.1f}s)")
                    
                except RuntimeError as e:
                    print(f"❌ Error: {e}")
                    raise
                except Exception as e:
                    print(f"⚠️  Error: {e}")
                    continue
                
                time.sleep(1.0 / self.sampling_rate)
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Collection interrupted. Collected {sample_count} samples.")
    
    def save_data(self, filepath: str):
        """Save collected data to JSON file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Convert numpy types to Python native types
        def make_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: make_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            return obj
        
        serializable_data = make_serializable(self.data)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"✅ Data saved to {filepath}")


def get_device_id():
    """Get unique device identifier."""
    try:
        return socket.gethostname()
    except:
        return "unknown_device"


def collect_data(location: str, duration: float = 60.0, sampling_rate: float = 2.0, device_id: str = None, notes: str = "", output_dir: str = "data"):
    """Main collection function."""
    print("=" * 70)
    print("WiFi Data Collection")
    print("=" * 70)
    print()
    
    # Get device info
    device_info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'device_id': device_id or get_device_id(),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"Device: {device_info['hostname']}")
    print(f"Device ID: {device_info['device_id']}")
    print(f"Location: {location}")
    print(f"Duration: {duration} seconds")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Expected samples: ~{int(duration * sampling_rate)}")
    if notes:
        print(f"Notes: {notes}")
    print()
    
    # Create collector
    try:
        collector = WiFiCollector(sampling_rate=sampling_rate)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
    
    # Verify connection
    try:
        test_sample = collector.get_wifi_info()
        print(f"✅ WiFi connection verified: {test_sample.get('ssid', 'Unknown')}")
        print(f"   RSSI: {test_sample.get('rssi', 'N/A')} dBm")
        print()
    except RuntimeError as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)
    
    # Collect data
    print("Starting collection...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        collector.collect_continuous(duration)
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user.")
    except RuntimeError as e:
        print(f"\n❌ ERROR: {str(e)}")
        sys.exit(1)
    
    if len(collector.data) == 0:
        print("❌ No data collected!")
        sys.exit(1)
    
    # Add metadata to each sample
    for sample in collector.data:
        sample['device_id'] = device_info['device_id']
        sample['device_hostname'] = device_info['hostname']
        sample['device_platform'] = device_info['platform']
        sample['location'] = location
        if notes:
            sample['collection_notes'] = notes
    
    # Create output directory structure
    location_clean = location.replace(" ", "_").replace("/", "_")
    device_clean = device_info['device_id'].replace(":", "-").replace("/", "_")
    device_dir = os.path.join(output_dir, location_clean, device_clean)
    os.makedirs(device_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{location_clean}_{device_clean}_{timestamp}.json"
    filepath = os.path.join(device_dir, filename)
    
    # Save data
    collector.save_data(filepath)
    
    # Print summary
    print()
    print("=" * 70)
    print("Collection Summary")
    print("=" * 70)
    print(f"Device: {device_info['hostname']} ({device_info['device_id']})")
    print(f"Location: {location}")
    print(f"Samples collected: {len(collector.data)}")
    print(f"Data file: {filepath}")
    print()
    
    # Data statistics
    if collector.data:
        rssi_values = [s.get('rssi', 0) for s in collector.data if 'rssi' in s]
        if rssi_values:
            print("Data Statistics:")
            print(f"  RSSI range: {min(rssi_values)} to {max(rssi_values)} dBm")
            print(f"  RSSI mean: {sum(rssi_values)/len(rssi_values):.1f} dBm")
        print()
    
    return filepath


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Standalone WiFi Data Collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data at a location
  python collect.py --location Room_101

  # Collect for longer duration
  python collect.py --location Library --duration 120

  # Collect with custom device ID
  python collect.py --location Lab --device_id Laptop_01

  # Collect with notes
  python collect.py --location Room_102 --notes "Testing WiFi signal"

Usage on Multiple Laptops:
  1. Copy this file (collect.py) to each laptop
  2. On each laptop, run: python collect.py --location <LOCATION>
  3. Data is saved to: data/<location>/<device_id>/
  4. Combine data later using combine_multi_device_data.py
        """
    )
    
    parser.add_argument('--location', type=str, required=True,
                       help='Location name (e.g., "Room_101", "Library")')
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION,
                       help=f'Collection duration in seconds (default: {DEFAULT_DURATION})')
    parser.add_argument('--sampling_rate', type=float, default=DEFAULT_SAMPLING_RATE,
                       help=f'Sampling rate in Hz (default: {DEFAULT_SAMPLING_RATE})')
    parser.add_argument('--device_id', type=str, default=None,
                       help='Device identifier (auto-detected if not provided)')
    parser.add_argument('--notes', type=str, default='',
                       help='Additional notes about the collection')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help=f'Output directory (default: {OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    try:
        collect_data(
            location=args.location,
            duration=args.duration,
            sampling_rate=args.sampling_rate,
            device_id=args.device_id,
            notes=args.notes,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        print("\n\nCollection cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

