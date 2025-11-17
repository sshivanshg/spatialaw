"""
CSI Feature Extraction
Extracts features from CSI data for activity recognition
"""

import numpy as np
from typing import Tuple
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks


def extract_csi_features(
    csi_data: np.ndarray,
    window_size: int = 50,
    overlap: float = 0.5
) -> np.ndarray:
    """
    Extract features from CSI data using sliding windows.
    
    Args:
        csi_data: CSI data of shape (n_packets, n_subcarriers, n_antennas)
        window_size: Size of sliding window (number of packets)
        overlap: Overlap ratio between windows (0.0 to 1.0)
    
    Returns:
        Feature matrix of shape (n_windows, n_features)
    """
    n_packets, n_subcarriers, n_antennas = csi_data.shape
    
    # Convert to magnitude if complex
    if np.iscomplexobj(csi_data):
        csi_magnitude = np.abs(csi_data)
        csi_phase = np.angle(csi_data)
    else:
        csi_magnitude = csi_data
        csi_phase = None
    
    # Create sliding windows
    step_size = int(window_size * (1 - overlap))
    features_list = []
    
    for i in range(0, n_packets - window_size + 1, step_size):
        window_mag = csi_magnitude[i:i+window_size, :, :]  # (window, subcarriers, antennas)
        window_phase = csi_phase[i:i+window_size, :, :] if csi_phase is not None else None
        
        # Extract features from this window
        features = extract_window_csi_features(window_mag, window_phase)
        features_list.append(features)
    
    return np.array(features_list)


def extract_window_csi_features(
    window_mag: np.ndarray,
    window_phase: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Extract features from a CSI window.
    
    Args:
        window_mag: Magnitude data (window_size, n_subcarriers, n_antennas)
        window_phase: Phase data (optional)
    
    Returns:
        Feature vector
    """
    features = []
    window_size, n_subcarriers, n_antennas = window_mag.shape
    
    # Flatten across antennas for each subcarrier
    # Shape: (window_size, n_subcarriers * n_antennas)
    window_flat = window_mag.reshape(window_size, -1)
    
    # Statistical features per subcarrier
    for subcarrier_idx in range(n_subcarriers):
        subcarrier_data = window_mag[:, subcarrier_idx, :].flatten()
        
        # Basic statistics
        features.append(np.mean(subcarrier_data))
        features.append(np.std(subcarrier_data))
        features.append(np.var(subcarrier_data))
        features.append(np.max(subcarrier_data) - np.min(subcarrier_data))  # Range
        features.append(stats.skew(subcarrier_data))
        features.append(stats.kurtosis(subcarrier_data))
        
        # Percentiles
        features.append(np.percentile(subcarrier_data, 25))
        features.append(np.percentile(subcarrier_data, 50))  # Median
        features.append(np.percentile(subcarrier_data, 75))
    
    # Cross-subcarrier features
    # Mean magnitude per packet (averaged across subcarriers and antennas)
    mean_per_packet = np.mean(window_mag, axis=(1, 2))  # (window_size,)
    features.append(np.mean(mean_per_packet))
    features.append(np.std(mean_per_packet))
    features.append(np.max(mean_per_packet) - np.min(mean_per_packet))
    
    # Time-domain features
    if window_size > 1:
        diff = np.diff(mean_per_packet)
        features.append(np.mean(np.abs(diff)))  # Mean absolute change
        features.append(np.std(diff))  # Std of changes
    
    # Frequency-domain features (FFT on mean magnitude)
    if window_size > 4:
        fft_vals = np.abs(fft(mean_per_packet))
        # Dominant frequency
        features.append(np.argmax(fft_vals[1:len(fft_vals)//2]) + 1)
        # Total power
        features.append(np.sum(fft_vals[1:len(fft_vals)//2]))
    
    # Phase features (if available)
    if window_phase is not None:
        # Mean phase per subcarrier
        for subcarrier_idx in range(min(10, n_subcarriers)):  # Limit to first 10 for feature size
            phase_data = window_phase[:, subcarrier_idx, :].flatten()
            features.append(np.mean(phase_data))
            features.append(np.std(phase_data))
    
    # Antenna correlation features
    # Correlation between antennas for each subcarrier
    for subcarrier_idx in range(min(10, n_subcarriers)):
        antenna_data = window_mag[:, subcarrier_idx, :]  # (window_size, n_antennas)
        if n_antennas > 1:
            # Correlation between first two antennas
            corr = np.corrcoef(antenna_data[:, 0], antenna_data[:, 1])[0, 1]
            features.append(corr if not np.isnan(corr) else 0.0)
    
    return np.array(features)


def extract_csi_features_simple(
    csi_data: np.ndarray,
    window_size: int = 50
) -> np.ndarray:
    """
    Extract simplified features from CSI data (faster, fewer features).
    
    Args:
        csi_data: CSI data of shape (n_packets, n_subcarriers, n_antennas)
        window_size: Size of sliding window
    
    Returns:
        Feature matrix
    """
    n_packets, n_subcarriers, n_antennas = csi_data.shape
    
    # Convert to magnitude
    if np.iscomplexobj(csi_data):
        csi_magnitude = np.abs(csi_data)
    else:
        csi_magnitude = csi_data
    
    features_list = []
    
    for i in range(0, n_packets - window_size + 1, window_size):
        window = csi_magnitude[i:i+window_size, :, :]
        
        # Simple features: mean magnitude per subcarrier (averaged across time and antennas)
        mean_per_subcarrier = np.mean(window, axis=(0, 2))  # (n_subcarriers,)
        
        # Statistical features on subcarrier means
        features = [
            np.mean(mean_per_subcarrier),
            np.std(mean_per_subcarrier),
            np.var(mean_per_subcarrier),
            np.max(mean_per_subcarrier) - np.min(mean_per_subcarrier),
        ]
        
        # Mean magnitude per packet (temporal variation)
        mean_per_packet = np.mean(window, axis=(1, 2))  # (window_size,)
        features.extend([
            np.mean(mean_per_packet),
            np.std(mean_per_packet),
            np.max(mean_per_packet) - np.min(mean_per_packet),
        ])
        
        # Time-domain change
        if window_size > 1:
            diff = np.diff(mean_per_packet)
            features.append(np.mean(np.abs(diff)))
        
        features_list.append(features)
    
    return np.array(features_list)

