"""
CSI (Channel State Information) Processor
Processes WiFi CSI data including amplitude and phase vectors
"""

import numpy as np
from typing import Tuple, Optional, List
import scipy.signal as signal
from scipy.fft import fft, ifft


class CSIProcessor:
    """
    Processes Channel State Information (CSI) data.
    Handles amplitude, phase, and complex CSI matrices.
    """
    
    def __init__(self, num_subcarriers: int = 64, num_antennas: int = 3):
        """
        Initialize CSI processor.
        
        Args:
            num_subcarriers: Number of OFDM subcarriers (typically 64 for 20MHz, 128 for 40MHz)
            num_antennas: Number of receive antennas
        """
        self.num_subcarriers = num_subcarriers
        self.num_antennas = num_antennas
    
    def process_csi(self, csi_complex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process complex CSI data to extract amplitude and phase.
        
        Args:
            csi_complex: Complex CSI matrix of shape (num_antennas, num_subcarriers)
            
        Returns:
            Tuple of (amplitude, phase) arrays
        """
        amplitude = np.abs(csi_complex)
        phase = np.angle(csi_complex)
        
        return amplitude, phase
    
    def normalize_amplitude(self, amplitude: np.ndarray, method: str = 'min_max') -> np.ndarray:
        """
        Normalize amplitude values.
        
        Args:
            amplitude: Amplitude array
            method: Normalization method ('min_max', 'z_score', 'unit_norm')
            
        Returns:
            Normalized amplitude array
        """
        if method == 'min_max':
            min_val = np.min(amplitude)
            max_val = np.max(amplitude)
            if max_val - min_val > 0:
                return (amplitude - min_val) / (max_val - min_val)
            return amplitude
        elif method == 'z_score':
            mean = np.mean(amplitude)
            std = np.std(amplitude)
            if std > 0:
                return (amplitude - mean) / std
            return amplitude
        elif method == 'unit_norm':
            norm = np.linalg.norm(amplitude)
            if norm > 0:
                return amplitude / norm
            return amplitude
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def unwrap_phase(self, phase: np.ndarray) -> np.ndarray:
        """
        Unwrap phase to remove discontinuities.
        
        Args:
            phase: Phase array in radians
            
        Returns:
            Unwrapped phase array
        """
        return np.unwrap(phase)
    
    def sanitize_phase(self, phase: np.ndarray, amplitude: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Sanitize phase by removing noise from low-amplitude components.
        
        Args:
            phase: Phase array
            amplitude: Amplitude array
            threshold: Amplitude threshold below which phase is set to 0
            
        Returns:
            Sanitized phase array
        """
        sanitized_phase = phase.copy()
        low_amplitude_mask = amplitude < threshold
        sanitized_phase[low_amplitude_mask] = 0
        return sanitized_phase
    
    def extract_features(self, csi_complex: np.ndarray) -> dict:
        """
        Extract features from CSI data.
        
        Args:
            csi_complex: Complex CSI matrix
            
        Returns:
            Dictionary of extracted features
        """
        amplitude, phase = self.process_csi(csi_complex)
        
        features = {
            'amplitude': amplitude,
            'phase': phase,
            'amplitude_mean': np.mean(amplitude),
            'amplitude_std': np.std(amplitude),
            'amplitude_max': np.max(amplitude),
            'amplitude_min': np.min(amplitude),
            'phase_mean': np.mean(phase),
            'phase_std': np.std(phase),
            'power': np.sum(amplitude ** 2),
            'snr': np.mean(amplitude) / (np.std(amplitude) + 1e-10),
        }
        
        # Frequency domain features
        fft_amplitude = np.abs(fft(amplitude, axis=-1))
        features['fft_amplitude'] = fft_amplitude
        features['dominant_frequency'] = np.argmax(fft_amplitude, axis=-1)
        
        return features
    
    def create_csi_matrix(self, amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Reconstruct complex CSI matrix from amplitude and phase.
        
        Args:
            amplitude: Amplitude array
            phase: Phase array in radians
            
        Returns:
            Complex CSI matrix
        """
        return amplitude * np.exp(1j * phase)
    
    def apply_filter(self, csi_complex: np.ndarray, filter_type: str = 'median', kernel_size: int = 3) -> np.ndarray:
        """
        Apply filtering to CSI data.
        
        Args:
            csi_complex: Complex CSI matrix
            filter_type: Type of filter ('median', 'gaussian', 'moving_average')
            kernel_size: Size of filter kernel
            
        Returns:
            Filtered CSI matrix
        """
        if filter_type == 'median':
            # Apply median filter to real and imaginary parts separately
            real_filtered = signal.medfilt2d(csi_complex.real, kernel_size=(kernel_size, kernel_size))
            imag_filtered = signal.medfilt2d(csi_complex.imag, kernel_size=(kernel_size, kernel_size))
            return real_filtered + 1j * imag_filtered
        elif filter_type == 'gaussian':
            from scipy.ndimage import gaussian_filter
            real_filtered = gaussian_filter(csi_complex.real, sigma=kernel_size/3)
            imag_filtered = gaussian_filter(csi_complex.imag, sigma=kernel_size/3)
            return real_filtered + 1j * imag_filtered
        else:
            return csi_complex
    
    def downsample(self, csi_complex: np.ndarray, factor: int = 2) -> np.ndarray:
        """
        Downsample CSI data.
        
        Args:
            csi_complex: Complex CSI matrix
            factor: Downsampling factor
            
        Returns:
            Downsampled CSI matrix
        """
        return csi_complex[:, ::factor]
    
    def augment_csi(self, csi_complex: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """
        Augment CSI data with noise for training.
        
        Args:
            csi_complex: Complex CSI matrix
            noise_level: Standard deviation of Gaussian noise
            
        Returns:
            Augmented CSI matrix
        """
        noise = np.random.normal(0, noise_level, csi_complex.shape) + 1j * np.random.normal(0, noise_level, csi_complex.shape)
        return csi_complex + noise


def generate_mock_csi(num_samples: int = 100, num_antennas: int = 3, num_subcarriers: int = 64) -> np.ndarray:
    """
    Generate mock CSI data for testing and development.
    
    Args:
        num_samples: Number of CSI samples
        num_antennas: Number of antennas
        num_subcarriers: Number of subcarriers
        
    Returns:
        Array of shape (num_samples, num_antennas, num_subcarriers) with complex CSI
    """
    # Generate realistic CSI with multipath effects
    csi_data = []
    
    for _ in range(num_samples):
        # Base channel response
        base = np.random.normal(0, 1, (num_antennas, num_subcarriers)) + \
               1j * np.random.normal(0, 1, (num_antennas, num_subcarriers))
        
        # Add frequency-selective fading
        freq_response = np.exp(1j * 2 * np.pi * np.arange(num_subcarriers) / num_subcarriers)
        base = base * freq_response[np.newaxis, :]
        
        # Add noise
        noise = 0.1 * (np.random.normal(0, 1, (num_antennas, num_subcarriers)) + \
                      1j * np.random.normal(0, 1, (num_antennas, num_subcarriers)))
        
        csi_data.append(base + noise)
    
    return np.array(csi_data)


if __name__ == "__main__":
    # Example usage
    processor = CSIProcessor(num_subcarriers=64, num_antennas=3)
    
    # Generate mock CSI
    mock_csi = generate_mock_csi(num_samples=10, num_antennas=3, num_subcarriers=64)
    print(f"Generated mock CSI shape: {mock_csi.shape}")
    
    # Process CSI
    csi_sample = mock_csi[0]
    amplitude, phase = processor.process_csi(csi_sample)
    print(f"Amplitude shape: {amplitude.shape}, Phase shape: {phase.shape}")
    
    # Extract features
    features = processor.extract_features(csi_sample)
    print(f"\nExtracted features: {list(features.keys())}")
    print(f"Amplitude mean: {features['amplitude_mean']:.4f}")
    print(f"SNR: {features['snr']:.4f}")

