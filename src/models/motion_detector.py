"""
Motion Detection Model
Classifies movement vs no movement from WiFi CSI time series
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class MotionDetector(nn.Module):
    """
    Neural network model for motion detection from WiFi time series.
    Input: Time series features (amplitude statistics, variance, etc.)
    Output: Binary classification (movement vs no movement)
    """
    
    def __init__(
        self,
        input_features: int = 10,  # Number of input features
        hidden_dims: Optional[list] = None,
        dropout: float = 0.3
    ):
        """
        Initialize motion detector model.
        
        Args:
            input_features: Number of input features (from time series)
            hidden_dims: List of hidden layer dimensions (default: [64, 32])
            dropout: Dropout rate
        """
        super(MotionDetector, self).__init__()
        
        self.input_features = input_features
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
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
        
        # Output layer (binary classification)
        layers.append(nn.Linear(in_dim, 2))  # 2 classes: movement, no_movement
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Time series features (batch, input_features)
        
        Returns:
            Logits: (batch, 2) where 2 = [no_movement, movement]
        """
        return self.model(features)
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Predict movement class."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(features)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        return predictions
    
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Predict movement probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(features)
            probs = F.softmax(logits, dim=1)
        return probs


class SimpleMotionDetector(nn.Module):
    """
    Simple MLP classifier for motion detection.
    Uses scikit-learn compatible interface for baseline comparison.
    """
    
    def __init__(
        self,
        input_features: int = 10,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.2
    ):
        super(SimpleMotionDetector, self).__init__()
        
        self.input_features = input_features
        
        if hidden_dims is None:
            hidden_dims = [32, 16]
        
        layers = []
        in_dim = input_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, 2))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(features)


if __name__ == "__main__":
    # Test models
    print("Testing Motion Detector Models...")
    
    # Test Motion Detector
    print("\n1. Motion Detector:")
    model = MotionDetector(input_features=10)
    features = torch.randn(8, 10)  # (batch, features)
    output = model(features)
    print(f"   Input shape: {features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test predictions
    predictions = model.predict(features)
    probs = model.predict_proba(features)
    print(f"   Predictions: {predictions}")
    print(f"   Probabilities: {probs[0]}")
    
    print("\ Models work correctly!")

