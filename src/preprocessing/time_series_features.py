"""
Time Series Feature Extraction for Motion Detection
Extracts features from WiFi time series data for motion detection
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy import stats
from scipy.fft import fft
from scipy.signal import find_peaks


def extract_time_series_features(
    time_series_data: List[Dict],
    window_size: int = 10,
    overlap: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from time series WiFi data.
    
    Args:
        time_series_data: List of WiFi data dictionaries with timestamps
        window_size: Size of sliding window for feature extraction
        overlap: Overlap ratio between windows (0.0 to 1.0)
    
    Returns:
        Tuple of (features, labels) where:
        - features: (n_samples, n_features) array
        - labels: (n_samples,) array (0=no_movement, 1=movement)
    """
    # Extract signal values over time
    rssi_values = [item['rssi'] for item in time_series_data]
    snr_values = [item['snr'] for item in time_series_data]
    signal_strength_values = [item['signal_strength'] for item in time_series_data]
    movement_labels = [item.get('movement_label', 0) for item in time_series_data]
    
    # Convert to numpy arrays
    rssi_array = np.array(rssi_values)
    snr_array = np.array(snr_values)
    signal_array = np.array(signal_strength_values)
    labels_array = np.array(movement_labels)
    
    # Create sliding windows
    step_size = int(window_size * (1 - overlap))
    features_list = []
    labels_list = []
    
    for i in range(0, len(rssi_array) - window_size + 1, step_size):
        # Extract window
        rssi_window = rssi_array[i:i+window_size]
        snr_window = snr_array[i:i+window_size]
        signal_window = signal_array[i:i+window_size]
        label_window = labels_array[i:i+window_size]
        
        # Use majority label for window
        window_label = int(np.round(np.mean(label_window)))
        
        # Extract features from window
        features = extract_window_features(rssi_window, snr_window, signal_window)
        features_list.append(features)
        labels_list.append(window_label)
    
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    return features_array, labels_array


def extract_window_features(
    rssi_window: np.ndarray,
    snr_window: np.ndarray,
    signal_window: np.ndarray
) -> np.ndarray:
    """
    Extract features from a time window.
    
    Args:
        rssi_window: RSSI values in window
        snr_window: SNR values in window
        signal_window: Signal strength values in window
    
    Returns:
        Feature vector
    """
    features = []
    
    # Statistical features for RSSI
    features.append(np.mean(rssi_window))  # Mean
    features.append(np.std(rssi_window))   # Standard deviation
    features.append(np.var(rssi_window))   # Variance
    features.append(np.max(rssi_window) - np.min(rssi_window))  # Range
    features.append(stats.skew(rssi_window))  # Skewness
    features.append(stats.kurtosis(rssi_window))  # Kurtosis
    
    # Statistical features for SNR
    features.append(np.mean(snr_window))
    features.append(np.std(snr_window))
    
    # Statistical features for signal strength
    features.append(np.mean(signal_window))
    features.append(np.std(signal_window))
    
    # Time domain features
    # Rate of change
    if len(rssi_window) > 1:
        rssi_diff = np.diff(rssi_window)
        features.append(np.mean(np.abs(rssi_diff)))  # Mean absolute change
        features.append(np.std(rssi_diff))  # Std of changes
    else:
        features.extend([0.0, 0.0])
    
    # Frequency domain features (FFT)
    if len(rssi_window) > 4:
        fft_vals = np.abs(fft(rssi_window))
        # Dominant frequency component
        features.append(np.argmax(fft_vals[1:len(fft_vals)//2]) + 1)  # Dominant frequency
        features.append(np.sum(fft_vals[1:len(fft_vals)//2]))  # Total frequency power
    else:
        features.extend([0.0, 0.0])
    
    # Peak detection
    try:
        peaks, _ = find_peaks(np.abs(rssi_window - np.mean(rssi_window)), height=0)
        features.append(len(peaks))  # Number of peaks
    except:
        features.append(0.0)
    
    return np.array(features)


def extract_simple_features(time_series_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract simple features from time series data.
    Simplified version for quick testing.
    
    Args:
        time_series_data: List of WiFi data dictionaries
    
    Returns:
        Tuple of (features, labels)
    """
    # Extract values
    rssi_values = np.array([item['rssi'] for item in time_series_data])
    snr_values = np.array([item['snr'] for item in time_series_data])
    signal_values = np.array([item['signal_strength'] for item in time_series_data])
    labels = np.array([item.get('movement_label', 0) for item in time_series_data])
    
    # Simple features: variance and mean
    # Motion typically causes higher variance in signal
    window_size = min(20, len(rssi_values) // 2)
    if window_size < 5:
        window_size = len(rssi_values)
    
    features_list = []
    labels_list = []
    
    for i in range(0, len(rssi_values) - window_size + 1, window_size):
        rssi_window = rssi_values[i:i+window_size]
        snr_window = snr_values[i:i+window_size]
        signal_window = signal_values[i:i+window_size]
        label_window = labels[i:i+window_size]
        
        # Simple features
        features = [
            np.mean(rssi_window),
            np.std(rssi_window),
            np.var(rssi_window),
            np.mean(snr_window),
            np.std(snr_window),
            np.mean(signal_window),
            np.std(signal_window),
            np.max(rssi_window) - np.min(rssi_window),  # Range
            np.mean(np.abs(np.diff(rssi_window))) if len(rssi_window) > 1 else 0.0,  # Mean change
        ]
        
        # Use majority label
        window_label = int(np.round(np.mean(label_window)))
        
        features_list.append(features)
        labels_list.append(window_label)
    
    return np.array(features_list), np.array(labels_list)


if __name__ == "__main__":
    # Test feature extraction
    print("Testing Time Series Feature Extraction...")
    
    # Create dummy time series data
    dummy_data = []
    for i in range(100):
        dummy_data.append({
            'rssi': -60 + np.random.normal(0, 2),
            'snr': 30 + np.random.normal(0, 1),
            'signal_strength': 70 + np.random.normal(0, 5),
            'movement_label': 1 if i > 50 else 0
        })
    
    # Extract features
    features, labels = extract_simple_features(dummy_data)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of features: {features.shape[1]}")
    print(f"Movement samples: {np.sum(labels == 1)}")
    print(f"No-movement samples: {np.sum(labels == 0)}")
    
    print("\n✅ Feature extraction works!")

