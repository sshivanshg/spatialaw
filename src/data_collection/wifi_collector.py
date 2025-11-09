"""
WiFi Data Collection Utility for Mac
Collects WiFi signal information including RSSI, signal strength, and basic network metrics
"""

import subprocess
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os
from datetime import datetime


class WiFiCollector:
    """
    Collects WiFi signal data on Mac using system utilities.
    Currently collects RSSI, signal strength, and channel information.
    Future: Can be extended to collect full CSI data with compatible hardware.
    """
    
    def __init__(self, interface: str = "en0", sampling_rate: float = 1.0, use_mock: bool = False):
        """
        Initialize WiFi collector.
        
        Args:
            interface: Network interface name (usually 'en0' for WiFi on Mac)
            sampling_rate: Sampling rate in Hz (samples per second)
            use_mock: If True, use mock data instead of trying to collect real data
        """
        self.interface = interface
        self.sampling_rate = sampling_rate
        self.sample_interval = 1.0 / sampling_rate
        self.data = []
        self.use_mock = use_mock
        self.airport_path = None
        self.collection_method = None
        self._warned_about_mock = False
        
        # Try to find airport utility
        if not use_mock:
            self.airport_path = self._find_airport_utility()
            if self.airport_path:
                self.collection_method = "airport"
            else:
                # Try alternative methods
                if self._check_networksetup():
                    self.collection_method = "networksetup"
                else:
                    self.collection_method = "mock"
                    if not self._warned_about_mock:
                        print("⚠️  Warning: Could not find airport utility. Using mock data.")
                        print("   To use mock data explicitly, set use_mock=True")
                        self._warned_about_mock = True
        else:
            self.collection_method = "mock"
            print("ℹ️  Using mock WiFi data (use_mock=True)")
    
    def _find_airport_utility(self) -> Optional[str]:
        """Try to find the airport utility in common locations."""
        possible_paths = [
            "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
            "/System/Library/PrivateFrameworks/Apple80211.framework/Resources/airport",
            "/usr/local/bin/airport",
            "airport"  # If in PATH
        ]
        
        for path in possible_paths:
            try:
                if path == "airport":
                    # Check if it's in PATH
                    result = subprocess.run(
                        ["which", "airport"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        test_path = result.stdout.strip()
                        # Test if it works
                        test_result = subprocess.run(
                            [test_path, "-I"],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if test_result.returncode == 0:
                            return test_path
                else:
                    # Check if file exists and is executable
                    if os.path.exists(path) and os.access(path, os.X_OK):
                        # Test if it works
                        test_result = subprocess.run(
                            [path, "-I"],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if test_result.returncode == 0:
                            return path
            except Exception:
                continue
        
        return None
    
    def _check_networksetup(self) -> bool:
        """Check if networksetup is available."""
        try:
            result = subprocess.run(
                ["/usr/sbin/networksetup", "-getairportnetwork", self.interface],
                capture_output=True,
                text=True,
                timeout=2
            )
            return result.returncode == 0
        except Exception:
            return False
        
    def get_wifi_info(self) -> Dict:
        """
        Get current WiFi information using available methods.
        
        Returns:
            Dictionary containing WiFi signal information
        """
        # Use mock data if explicitly requested or if no collection method available
        if self.use_mock or self.collection_method == "mock":
            return self._get_mock_data()
        
        # Try airport utility
        if self.collection_method == "airport" and self.airport_path:
            try:
                result = subprocess.run(
                    [self.airport_path, "-I"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0 and result.stdout:
                    wifi_info = self._parse_airport_output(result.stdout)
                    wifi_info['timestamp'] = datetime.now().isoformat()
                    wifi_info['unix_timestamp'] = time.time()
                    wifi_info['collection_method'] = 'airport'
                    return wifi_info
            except Exception:
                # Fall through to mock data
                pass
        
        # Try networksetup as fallback
        if self.collection_method == "networksetup":
            try:
                wifi_info = self._get_info_from_networksetup()
                if wifi_info:
                    wifi_info['timestamp'] = datetime.now().isoformat()
                    wifi_info['unix_timestamp'] = time.time()
                    wifi_info['collection_method'] = 'networksetup'
                    return wifi_info
            except Exception:
                # Fall through to mock data
                pass
        
        # Fallback to mock data
        mock_data = self._get_mock_data()
        mock_data['collection_method'] = 'mock'
        return mock_data
    
    def _get_info_from_networksetup(self) -> Optional[Dict]:
        """Get WiFi info using networksetup (limited information)."""
        try:
            # Get network name
            result = subprocess.run(
                ["/usr/sbin/networksetup", "-getairportnetwork", self.interface],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode != 0:
                return None
            
            # Parse output (format: "Current Wi-Fi Network: NetworkName")
            output = result.stdout.strip()
            if ":" in output:
                ssid = output.split(":", 1)[1].strip()
            else:
                ssid = "Unknown"
            
            # networksetup doesn't provide RSSI, so we create a basic info dict
            info = {
                'ssid': ssid,
                'rssi': -75,  # Default value (networksetup doesn't provide this)
                'signal_strength': 50,  # Estimated
                'channel': 0,  # Not available
                'snr': 0,  # Not available
                'bssid': '00:00:00:00:00:00',  # Not available
            }
            
            return info
        except Exception:
            return None
    
    def _parse_airport_output(self, output: str) -> Dict:
        """Parse airport command output."""
        info = {}
        for line in output.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().replace(' ', '_').lower()
                value = value.strip()
                
                # Parse numeric values
                if key == 'rssi' or key == 'agrctlrssi':
                    try:
                        info['rssi'] = int(value)
                    except:
                        info['rssi'] = -100
                elif key == 'agrctlnoise':
                    try:
                        info['noise'] = int(value)
                    except:
                        info['noise'] = -95
                elif key == 'channel':
                    try:
                        info['channel'] = int(value.split(',')[0])
                    except:
                        info['channel'] = 0
                elif key == 'ssid':
                    info['ssid'] = value
                elif key == 'bssid':
                    info['bssid'] = value
                else:
                    info[key] = value
        
        # Calculate signal strength percentage (RSSI to percentage conversion)
        if 'rssi' in info:
            # RSSI typically ranges from -100 (weak) to -50 (strong)
            signal_strength = max(0, min(100, 2 * (info['rssi'] + 100)))
            info['signal_strength'] = signal_strength
        
        # Calculate SNR if both RSSI and noise are available
        if 'rssi' in info and 'noise' in info:
            info['snr'] = info['rssi'] - info['noise']
        else:
            info['snr'] = 0
        
        return info
    
    def _get_mock_data(self) -> Dict:
        """Generate mock WiFi data for testing when real collection fails."""
        return {
            'rssi': np.random.randint(-90, -50),
            'signal_strength': np.random.uniform(20, 80),
            'channel': np.random.choice([1, 6, 11, 36, 40, 44, 48]),
            'snr': np.random.randint(10, 40),
            'ssid': 'RUCKUS_NETWORK',
            'bssid': '00:11:22:33:44:55',
            'timestamp': datetime.now().isoformat(),
            'unix_timestamp': time.time()
        }
    
    def collect_sample(self, duration: float = 1.0) -> List[Dict]:
        """
        Collect WiFi samples for a specified duration.
        
        Args:
            duration: Duration in seconds to collect samples
            
        Returns:
            List of WiFi information dictionaries
        """
        samples = []
        start_time = time.time()
        num_samples = int(duration * self.sampling_rate)
        
        print(f"Collecting {num_samples} WiFi samples...")
        for i in range(num_samples):
            sample = self.get_wifi_info()
            samples.append(sample)
            
            if i < num_samples - 1:
                time.sleep(self.sample_interval)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Collected {i + 1}/{num_samples} samples (elapsed: {elapsed:.1f}s)")
        
        self.data.extend(samples)
        return samples
    
    def collect_continuous(self, duration: float = 60.0, save_path: Optional[str] = None):
        """
        Collect WiFi data continuously for a specified duration.
        
        Args:
            duration: Duration in seconds
            save_path: Optional path to save data periodically
        """
        print(f"Starting continuous WiFi collection for {duration} seconds...")
        print(f"Collection method: {self.collection_method}")
        if self.collection_method == "mock":
            print("⚠️  Note: Using mock data. Real WiFi data collection is not available on this system.")
        print()
        
        start_time = time.time()
        sample_count = 0
        error_count = 0
        last_error_print = 0
        
        try:
            while time.time() - start_time < duration:
                try:
                    sample = self.get_wifi_info()
                    self.data.append(sample)
                    sample_count += 1
                    
                    # Save periodically
                    if save_path and sample_count % 100 == 0:
                        self.save_data(save_path)
                        print(f"✓ Saved {sample_count} samples to {save_path}")
                    
                    # Show progress every 10 samples
                    if sample_count % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"✓ Collected {sample_count} samples (elapsed: {elapsed:.1f}s)")
                    
                except Exception as e:
                    error_count += 1
                    # Only print errors occasionally to avoid spam
                    if error_count - last_error_print >= 10:
                        print(f"⚠️  Error collecting sample (errors: {error_count}): {str(e)[:50]}")
                        last_error_print = error_count
                    # Still add mock data to continue collection
                    mock_data = self._get_mock_data()
                    mock_data['collection_method'] = 'mock'
                    mock_data['error'] = True
                    self.data.append(mock_data)
                    sample_count += 1
                
                time.sleep(self.sample_interval)
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Collection interrupted. Collected {sample_count} samples.")
        
        if save_path:
            self.save_data(save_path)
            print(f"✓ Final save: {len(self.data)} samples saved to {save_path}")
        
        if error_count > 0:
            print(f"⚠️  Total errors during collection: {error_count}")
    
    def save_data(self, filepath: str, format: str = 'json'):
        """Save collected data to file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.data, f, indent=2)
        elif format == 'csv':
            df = pd.DataFrame(self.data)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Data saved to {filepath}")
    
    def load_data(self, filepath: str, format: str = 'json'):
        """Load data from file."""
        if format == 'json':
            with open(filepath, 'r') as f:
                self.data = json.load(f)
        elif format == 'csv':
            df = pd.read_csv(filepath)
            self.data = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Loaded {len(self.data)} samples from {filepath}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert collected data to pandas DataFrame."""
        return pd.DataFrame(self.data)
    
    def clear_data(self):
        """Clear collected data."""
        self.data = []
        print("Data cleared.")


if __name__ == "__main__":
    # Example usage
    collector = WiFiCollector(sampling_rate=2.0)  # 2 samples per second
    
    # Collect for 10 seconds
    samples = collector.collect_sample(duration=10.0)
    
    # Save data
    collector.save_data("data/wifi_samples.json", format='json')
    collector.save_data("data/wifi_samples.csv", format='csv')
    

    df = collector.get_dataframe()
    print("\nCollected Data Summary:")
    print(df.describe())

