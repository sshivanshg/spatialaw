"""
Heatmap Model for WiFi Signal Prediction
Predicts WiFi signal features at given (x, y) positions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class SignalHeatmapModel(nn.Module):
    """
    Model that predicts WiFi signal features at given positions.
    Input: (x, y) coordinates
    Output: WiFi signal features (RSSI, SNR, signal_strength, etc.)
    """
    
    def __init__(
        self,
        output_features: int = 4,  # RSSI, SNR, signal_strength, channel
        hidden_dims: Optional[list] = None,
        dropout: float = 0.2
    ):
        """
        Initialize signal heatmap model.
        
        Args:
            output_features: Number of output features to predict
            hidden_dims: List of hidden layer dimensions (default: [128, 256, 128])
            dropout: Dropout rate
        """
        super(SignalHeatmapModel, self).__init__()
        
        self.output_features = output_features
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 128]
        
        layers = []
        in_dim = 2  # Input: (x, y) coordinates
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, output_features))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            positions: Position coordinates (batch, 2) where 2 = (x, y)
        
        Returns:
            Signal features: (batch, output_features)
        """
        return self.model(positions)
    
    def predict_at_position(self, x: float, y: float) -> np.ndarray:
        """Predict signal features at a single position."""
        self.eval()
        with torch.no_grad():
            pos = torch.tensor([[x, y]], dtype=torch.float32)
            if next(self.parameters()).is_cuda:
                pos = pos.cuda()
            features = self.forward(pos)
            return features.cpu().numpy()[0]


class PositionHeatmapModel(nn.Module):
    """
    Model that predicts position from WiFi signal features.
    Input: WiFi signal features
    Output: (x, y) coordinates
    """
    
    def __init__(
        self,
        input_features: int = 4,  # RSSI, SNR, signal_strength, channel
        hidden_dims: Optional[list] = None,
        dropout: float = 0.2
    ):
        """
        Initialize position heatmap model.
        
        Args:
            input_features: Number of input WiFi features
            hidden_dims: List of hidden layer dimensions (default: [128, 256, 128])
            dropout: Dropout rate
        """
        super(PositionHeatmapModel, self).__init__()
        
        self.input_features = input_features
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 128]
        
        layers = []
        in_dim = input_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer (2D position: x, y)
        layers.append(nn.Linear(in_dim, 2))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            signals: WiFi signal features (batch, input_features)
        
        Returns:
            Positions: (batch, 2) where 2 = (x, y)
        """
        return self.model(signals)
    
    def predict_position(self, signals: np.ndarray) -> np.ndarray:
        """Predict position from signal features."""
        self.eval()
        with torch.no_grad():
            sig = torch.tensor(signals, dtype=torch.float32)
            if len(sig.shape) == 1:
                sig = sig.unsqueeze(0)
            if next(self.parameters()).is_cuda:
                sig = sig.cuda()
            position = self.forward(sig)
            return position.cpu().numpy()[0]


if __name__ == "__main__":
    # Test models
    print("Testing Heatmap Models...")
    
    # Test Signal Prediction Model
    print("\n1. Signal Prediction Model (Position → Signal):")
    model = SignalHeatmapModel(output_features=4)
    positions = torch.randn(8, 2)  # (batch, x, y)
    signals = model(positions)
    print(f"   Input shape: {positions.shape}")
    print(f"   Output shape: {signals.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test Position Prediction Model
    print("\n2. Position Prediction Model (Signal → Position):")
    model_pos = PositionHeatmapModel(input_features=4)
    wifi_signals = torch.randn(8, 4)  # (batch, features)
    positions_pred = model_pos(wifi_signals)
    print(f"   Input shape: {wifi_signals.shape}")
    print(f"   Output shape: {positions_pred.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_pos.parameters()):,}")
    
    print("\n✅ Models work correctly!")

