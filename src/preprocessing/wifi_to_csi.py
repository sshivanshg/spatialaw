"""
Convert Real WiFi Data to CSI Format for Model Training
Converts collected WiFi signals (RSSI, signal strength) to CSI-like matrices
"""

import numpy as np
import json
from typing import List, Dict, Tuple
from pathlib import Path


def wifi_to_csi(
    wifi_data: Dict,
    num_antennas: int = 3,
    num_subcarriers: int = 64,
    method: str = 'realistic'
) -> np.ndarray:
    """
    Convert real WiFi data to CSI matrix format.
    
    Args:
        wifi_data: Dictionary with WiFi signal data (RSSI, signal_strength, etc.)
        num_antennas: Number of antennas to simulate
        num_subcarriers: Number of subcarriers (typically 64)
        method: Conversion method ('realistic', 'simple', 'multipath')
        
    Returns:
        CSI matrix of shape (num_antennas, num_subcarriers) as complex array
    """
    # Extract signal information
    rssi = wifi_data.get('rssi', -60)
    signal_strength = wifi_data.get('signal_strength', 50) / 100.0  # Normalize to [0, 1]
    snr = wifi_data.get('snr', 30)
    channel = wifi_data.get('channel', 36)
    noise = wifi_data.get('noise', -90)
    
    # Convert RSSI to power (dBm to linear scale)
    # RSSI ranges from -100 (weak) to -30 (strong)
    # Convert to amplitude scale
    rssi_normalized = (rssi + 100) / 70.0  # Normalize to roughly [0, 1]
    rssi_normalized = np.clip(rssi_normalized, 0, 1)
    
    if method == 'simple':
        # Simple method: Use signal strength directly
        base_amplitude = signal_strength
        amplitude = np.full((num_antennas, num_subcarriers), base_amplitude)
        # Add small random variations
        amplitude += np.random.normal(0, 0.05, (num_antennas, num_subcarriers))
        amplitude = np.clip(amplitude, 0, 1)
        
        # Phase: Random phase with some correlation
        phase = np.random.uniform(-np.pi, np.pi, (num_antennas, num_subcarriers))
        
    elif method == 'realistic':
        # Realistic method: Simulate multipath and frequency response
        base_amplitude = rssi_normalized * signal_strength
        
        # Create frequency-dependent amplitude (subcarriers have different gains)
        frequency_response = np.ones(num_subcarriers)
        # Add frequency-selective fading
        for i in range(num_subcarriers):
            # Simulate frequency-dependent attenuation
            freq_factor = 1.0 + 0.1 * np.sin(2 * np.pi * i / num_subcarriers)
            frequency_response[i] = freq_factor
        
        # Create antenna-dependent amplitude (MIMO effects)
        antenna_gains = np.ones(num_antennas)
        # Add antenna correlation (similar but not identical)
        for a in range(num_antennas):
            antenna_gains[a] = 1.0 + 0.05 * np.sin(a * np.pi / num_antennas)
        
        # Combine: amplitude = base * frequency_response * antenna_gain
        amplitude = np.outer(antenna_gains, frequency_response) * base_amplitude
        
        # Add realistic noise based on SNR
        noise_level = 1.0 / (10 ** (snr / 10.0)) if snr > 0 else 0.1
        amplitude += np.random.normal(0, noise_level, (num_antennas, num_subcarriers))
        amplitude = np.clip(amplitude, 0.01, 1.0)  # Ensure non-zero
        
        # Phase: Realistic phase with correlation
        # Phase should vary smoothly across subcarriers (frequency domain)
        base_phase = np.random.uniform(-np.pi, np.pi)
        phase = np.zeros((num_antennas, num_subcarriers))
        
        for a in range(num_antennas):
            # Phase varies smoothly across subcarriers
            phase_shift = base_phase + a * 0.1  # Antenna-dependent phase shift
            for s in range(num_subcarriers):
                # Frequency-dependent phase
                freq_phase = 2 * np.pi * s / num_subcarriers
                phase[a, s] = phase_shift + freq_phase + np.random.normal(0, 0.1)
        
        # Wrap phase to [-pi, pi]
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        
    elif method == 'multipath':
        # Multipath method: Simulate multiple signal paths
        base_amplitude = rssi_normalized * signal_strength
        
        # Simulate multiple paths
        num_paths = 3
        amplitude = np.zeros((num_antennas, num_subcarriers))
        phase = np.zeros((num_antennas, num_subcarriers))
        
        for path in range(num_paths):
            path_amplitude = base_amplitude / (path + 1)  # Weaker paths
            path_delay = path * 0.1  # Delay in samples
            
            for a in range(num_antennas):
                for s in range(num_subcarriers):
                    # Frequency-dependent phase shift due to delay
                    freq = 2 * np.pi * s / num_subcarriers
                    path_phase = freq * path_delay + np.random.uniform(-0.1, 0.1)
                    
                    amplitude[a, s] += path_amplitude
                    phase[a, s] += path_phase
        
        # Normalize amplitude
        amplitude = amplitude / amplitude.max() if amplitude.max() > 0 else amplitude
        amplitude = np.clip(amplitude, 0.01, 1.0)
        
        # Wrap phase
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to complex CSI
    csi_complex = amplitude * np.exp(1j * phase)
    
    return csi_complex


def convert_wifi_json_to_csi(
    wifi_json_path: str,
    output_path: str = None,
    num_antennas: int = 3,
    num_subcarriers: int = 64,
    method: str = 'realistic'
) -> np.ndarray:
    """
    Convert WiFi JSON data file to CSI numpy array.
    
    Args:
        wifi_json_path: Path to WiFi data JSON file
        output_path: Optional path to save CSI data as .npy file
        num_antennas: Number of antennas
        num_subcarriers: Number of subcarriers
        method: Conversion method
        
    Returns:
        CSI data array of shape (num_samples, num_antennas, num_subcarriers)
    """
    # Load WiFi data
    with open(wifi_json_path, 'r') as f:
        wifi_data_list = json.load(f)
    
    # Convert each sample to CSI
    csi_data = []
    for wifi_sample in wifi_data_list:
        csi_matrix = wifi_to_csi(
            wifi_sample,
            num_antennas=num_antennas,
            num_subcarriers=num_subcarriers,
            method=method
        )
        csi_data.append(csi_matrix)
    
    csi_array = np.array(csi_data)
    
    # Save if output path provided
    if output_path:
        np.save(output_path, csi_array)
        print(f"✅ Converted {len(csi_data)} WiFi samples to CSI format")
        print(f"   Saved to: {output_path}")
        print(f"   Shape: {csi_array.shape}")
    
    return csi_array


if __name__ == "__main__":
    # Test conversion
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    # Convert real WiFi data
    wifi_file = "data/wifi_data_20251109_161754.json"
    if os.path.exists(wifi_file):
        csi_data = convert_wifi_json_to_csi(
            wifi_file,
            output_path="data/csi_from_wifi.npy",
            method='realistic'
        )
        print(f"✅ Conversion complete: {csi_data.shape}")

