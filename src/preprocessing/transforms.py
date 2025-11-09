"""
Data Transforms for CSI and WiFi Data
"""

import torch
import numpy as np
from typing import Dict, Optional
import torchvision.transforms as transforms


class CSITransforms:
    """Transforms for CSI data."""
    
    @staticmethod
    def normalize_amplitude_phase(csi_data: torch.Tensor) -> torch.Tensor:
        """Normalize amplitude and phase channels."""
        amplitude = csi_data[0]
        phase = csi_data[1]
        
        # Normalize amplitude to [0, 1]
        amplitude_min = amplitude.min()
        amplitude_max = amplitude.max()
        if amplitude_max - amplitude_min > 0:
            amplitude = (amplitude - amplitude_min) / (amplitude_max - amplitude_min)
        
        # Normalize phase to [0, 1]
        phase_min = phase.min()
        phase_max = phase.max()
        if phase_max - phase_min > 0:
            phase = (phase - phase_min) / (phase_max - phase_min)
        
        return torch.stack([amplitude, phase], dim=0)
    
    @staticmethod
    def add_noise(csi_data: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to CSI data."""
        noise = torch.randn_like(csi_data) * noise_level
        return csi_data + noise
    
    @staticmethod
    def random_scale(csi_data: torch.Tensor, scale_range: tuple = (0.9, 1.1)) -> torch.Tensor:
        """Randomly scale CSI data."""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return csi_data * scale


class NormalizeCSI:
    """Normalize CSI data transform."""
    
    def __init__(self, method: str = 'min_max'):
        self.method = method
    
    def __call__(self, sample: Dict) -> Dict:
        """Apply normalization to CSI data."""
        if 'csi' in sample:
            csi = sample['csi']
            if self.method == 'min_max':
                # Normalize each channel separately
                for i in range(csi.shape[0]):
                    channel = csi[i]
                    min_val = channel.min()
                    max_val = channel.max()
                    if max_val - min_val > 0:
                        csi[i] = (channel - min_val) / (max_val - min_val)
            elif self.method == 'z_score':
                for i in range(csi.shape[0]):
                    channel = csi[i]
                    mean = channel.mean()
                    std = channel.std()
                    if std > 0:
                        csi[i] = (channel - mean) / std
            sample['csi'] = csi
        return sample


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, sample: Dict) -> Dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


if __name__ == "__main__":
    # Test transforms
    print("Testing CSI transforms...")
    
    # Create dummy CSI data
    csi = torch.randn(2, 3, 64)  # (channels, antennas, subcarriers)
    
    # Test normalization
    normalize = NormalizeCSI(method='min_max')
    sample = {'csi': csi}
    normalized_sample = normalize(sample)
    print(f"Original CSI range: [{csi.min():.3f}, {csi.max():.3f}]")
    print(f"Normalized CSI range: [{normalized_sample['csi'].min():.3f}, {normalized_sample['csi'].max():.3f}]")

