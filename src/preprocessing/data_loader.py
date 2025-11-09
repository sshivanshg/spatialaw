"""
Data Loaders for WiFi and CSI Data
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
import os
from PIL import Image
import json


class WiFiDataset(Dataset):
    """
    Dataset for WiFi signal data (RSSI, signal strength, etc.)
    """
    
    def __init__(
        self,
        data_path: str,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize WiFi dataset.
        
        Args:
            data_path: Path to CSV or JSON file with WiFi data
            feature_columns: List of column names to use as features
            target_column: Column name for target (if supervised learning)
            transform: Optional transform to apply to samples
        """
        self.transform = transform
        
        # Load data
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data_list = json.load(f)
            self.data = pd.DataFrame(data_list)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Select feature columns
        if feature_columns is None:
            # Default features
            self.feature_columns = ['rssi', 'signal_strength', 'snr', 'channel']
            # Filter to available columns
            self.feature_columns = [col for col in self.feature_columns if col in self.data.columns]
        else:
            self.feature_columns = feature_columns
        
        self.target_column = target_column
        
        # Remove rows with missing values
        self.data = self.data.dropna(subset=self.feature_columns)
        
        print(f"Loaded {len(self.data)} samples with features: {self.feature_columns}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        sample = self.data.iloc[idx]
        
        # Extract features
        features = torch.tensor(
            [sample[col] for col in self.feature_columns],
            dtype=torch.float32
        )
        
        result = {'features': features}
        
        # Add target if available
        if self.target_column and self.target_column in self.data.columns:
            target = sample[self.target_column]
            if isinstance(target, (int, float)):
                result['target'] = torch.tensor(target, dtype=torch.float32)
            else:
                result['target'] = target
        
        # Apply transform
        if self.transform:
            result = self.transform(result)
        
        return result


class CSIDataset(Dataset):
    """
    Dataset for CSI (Channel State Information) data.
    Supports both real CSI data and mock data generation.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        csi_data: Optional[np.ndarray] = None,
        images_path: Optional[str] = None,
        num_subcarriers: int = 64,
        num_antennas: int = 3,
        image_size: Tuple[int, int] = (64, 64),
        transform: Optional[callable] = None,
        generate_mock: bool = False,
        num_mock_samples: int = 100
    ):
        """
        Initialize CSI dataset.
        
        Args:
            data_path: Path to CSI data file (numpy array or pickle)
            csi_data: Optional pre-loaded CSI data array
            images_path: Path to corresponding ground truth images
            num_subcarriers: Number of OFDM subcarriers
            num_antennas: Number of antennas
            image_size: Size of output images (height, width)
            transform: Optional transform to apply
            generate_mock: Whether to generate mock CSI data
            num_mock_samples: Number of mock samples to generate
        """
        self.num_subcarriers = num_subcarriers
        self.num_antennas = num_antennas
        self.image_size = image_size
        self.transform = transform
        
        # Load or generate CSI data
        if generate_mock:
            from src.data_collection.csi_processor import generate_mock_csi
            self.csi_data = generate_mock_csi(
                num_samples=num_mock_samples,
                num_antennas=num_antennas,
                num_subcarriers=num_subcarriers
            )
            print(f"Generated {num_mock_samples} mock CSI samples")
        elif csi_data is not None:
            self.csi_data = csi_data
        elif data_path and os.path.exists(data_path):
            if data_path.endswith('.npy'):
                self.csi_data = np.load(data_path)
            elif data_path.endswith('.pkl'):
                import pickle
                with open(data_path, 'rb') as f:
                    self.csi_data = pickle.load(f)
            elif data_path.endswith('.json'):
                # Convert WiFi JSON data to CSI format
                from src.preprocessing.wifi_to_csi import convert_wifi_json_to_csi
                print(f"Converting WiFi JSON data to CSI format...")
                self.csi_data = convert_wifi_json_to_csi(
                    data_path,
                    num_antennas=num_antennas,
                    num_subcarriers=num_subcarriers,
                    method='realistic'
                )
                print(f"Converted {len(self.csi_data)} WiFi samples to CSI format")
            else:
                raise ValueError(f"Unsupported CSI data format: {data_path}")
        else:
            # Generate mock data if no data provided
            from src.data_collection.csi_processor import generate_mock_csi
            self.csi_data = generate_mock_csi(
                num_samples=num_mock_samples,
                num_antennas=num_antennas,
                num_subcarriers=num_subcarriers
            )
            print(f"Generated {num_mock_samples} mock CSI samples (no data provided)")
        
        # Load images if provided
        self.images = None
        if images_path and os.path.exists(images_path):
            self.images = self._load_images(images_path)
            if len(self.images) != len(self.csi_data):
                print(f"Warning: {len(self.images)} images but {len(self.csi_data)} CSI samples")
        
        print(f"CSI Dataset: {len(self.csi_data)} samples")
    
    def _load_images(self, images_path: str) -> List[np.ndarray]:
        """Load images from directory or file."""
        images = []
        
        if os.path.isdir(images_path):
            # Load from directory
            image_files = sorted([f for f in os.listdir(images_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for img_file in image_files:
                img_path = os.path.join(images_path, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(self.image_size)
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                images.append(img_array)
        else:
            # Assume single file with images
            # This would need to be implemented based on specific format
            pass
        
        return images
    
    def __len__(self) -> int:
        return len(self.csi_data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        # Get CSI data
        csi_complex = self.csi_data[idx]  # Shape: (num_antennas, num_subcarriers)
        
        # Process CSI to get amplitude and phase
        from src.data_collection.csi_processor import CSIProcessor
        processor = CSIProcessor(
            num_subcarriers=self.num_subcarriers,
            num_antennas=self.num_antennas
        )
        
        amplitude, phase = processor.process_csi(csi_complex)
        
        # Normalize
        amplitude = processor.normalize_amplitude(amplitude, method='min_max')
        phase = processor.unwrap_phase(phase)
        phase = (phase + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        
        # Stack amplitude and phase as channels
        csi_input = np.stack([amplitude, phase], axis=0)  # Shape: (2, num_antennas, num_subcarriers)
        csi_input = torch.tensor(csi_input, dtype=torch.float32)
        
        result = {
            'csi': csi_input,
            'amplitude': torch.tensor(amplitude, dtype=torch.float32),
            'phase': torch.tensor(phase, dtype=torch.float32)
        }
        
        # Add image if available
        if self.images is not None and idx < len(self.images):
            image = self.images[idx]
            # Convert to tensor and normalize to [-1, 1]
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
            image_tensor = image_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            result['image'] = image_tensor
        
        # Apply transform
        if self.transform:
            result = self.transform(result)
        
        return result


if __name__ == "__main__":
    # Test datasets
    print("Testing WiFiDataset...")
    # This would require actual data file
    # dataset = WiFiDataset("data/wifi_samples.csv")
    
    print("\nTesting CSIDataset...")
    dataset = CSIDataset(generate_mock=True, num_mock_samples=10)
    sample = dataset[0]
    print(f"CSI shape: {sample['csi'].shape}")
    print(f"Amplitude shape: {sample['amplitude'].shape}")
    print(f"Phase shape: {sample['phase'].shape}")

