"""
Baseline Model for Spatial Awareness from WiFi Signals
CNN-based architecture for processing WiFi CSI data and reconstructing spatial information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BaselineSpatialModel(nn.Module):
    """
    Baseline CNN model for spatial awareness from WiFi CSI data.
    
    Architecture:
    - Encoder: Processes CSI data (amplitude/phase) into latent representations
    - Decoder: Reconstructs spatial information (e.g., images, spatial features)
    """
    
    def __init__(
        self,
        input_channels: int = 2,  # Amplitude and Phase
        num_subcarriers: int = 64,
        num_antennas: int = 3,
        latent_dim: int = 128,
        output_channels: int = 3,  # For image reconstruction (RGB)
        output_size: Tuple[int, int] = (64, 64)
    ):
        """
        Initialize baseline model.
        
        Args:
            input_channels: Number of input channels (2 for amplitude+phase)
            num_subcarriers: Number of OFDM subcarriers
            num_antennas: Number of receive antennas
            latent_dim: Dimension of latent representation
            output_channels: Number of output channels (e.g., 3 for RGB image)
            output_size: Output spatial dimensions (height, width)
        """
        super(BaselineSpatialModel, self).__init__()
        
        self.input_channels = input_channels
        self.num_subcarriers = num_subcarriers
        self.num_antennas = num_antennas
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.output_size = output_size
        
        # Encoder: Process CSI data
        self.encoder = self._build_encoder()
        
        # Latent projection
        self.latent_projection = nn.Sequential(
            nn.Linear(self._get_encoder_output_size(), latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Decoder: Reconstruct spatial information
        self.decoder = self._build_decoder()
        
    def _build_encoder(self) -> nn.Module:
        """Build encoder network for CSI processing."""
        layers = []
        
        # Input: (batch, input_channels, num_antennas, num_subcarriers)
        # First conv block
        layers.extend([
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ])
        
        # Second conv block
        layers.extend([
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ])
        
        # Third conv block
        layers.extend([
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        ])
        
        return nn.Sequential(*layers)
    
    def _get_encoder_output_size(self) -> int:
        """Calculate encoder output size."""
        # After conv layers and adaptive pooling to (4, 4)
        return 128 * 4 * 4
    
    def _build_decoder(self) -> nn.Module:
        """Build decoder network for spatial reconstruction."""
        # Start with a small spatial dimension
        start_size = (self.output_size[0] // 16, self.output_size[1] // 16)
        start_channels = self.latent_dim // (start_size[0] * start_size[1])
        start_channels = max(512, start_channels)
        
        layers = []
        
        # Initial projection to spatial dimensions
        layers.append(
            nn.ConvTranspose2d(
                self.latent_dim,
                start_channels,
                kernel_size=start_size,
                stride=1,
                padding=0
            )
        )
        layers.extend([
            nn.BatchNorm2d(start_channels),
            nn.ReLU()
        ])
        
        # Upsampling blocks
        current_channels = start_channels
        current_size = start_size
        
        while current_size[0] < self.output_size[0]:
            next_channels = max(self.output_channels, current_channels // 2)
            layers.extend([
                nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(next_channels),
                nn.ReLU()
            ])
            current_channels = next_channels
            current_size = (current_size[0] * 2, current_size[1] * 2)
        
        # Final output layer
        layers.append(
            nn.Conv2d(current_channels, self.output_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.Tanh())  # Output in [-1, 1] range
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_channels, num_antennas, num_subcarriers)
            
        Returns:
            Reconstructed output of shape (batch, output_channels, height, width)
        """
        # Encode CSI
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        
        # Flatten and project to latent space
        encoded_flat = encoded.view(batch_size, -1)
        latent = self.latent_projection(encoded_flat)
        
        # Reshape latent for decoder
        latent_spatial = latent.view(batch_size, self.latent_dim, 1, 1)
        
        # Decode to spatial representation
        output = self.decoder(latent_spatial)
        
        # Resize to target output size if needed
        if output.size()[2:] != self.output_size:
            output = F.interpolate(output, size=self.output_size, mode='bilinear', align_corners=False)
        
        return output
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        encoded_flat = encoded.view(batch_size, -1)
        latent = self.latent_projection(encoded_flat)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        batch_size = latent.size(0)
        latent_spatial = latent.view(batch_size, self.latent_dim, 1, 1)
        output = self.decoder(latent_spatial)
        if output.size()[2:] != self.output_size:
            output = F.interpolate(output, size=self.output_size, mode='bilinear', align_corners=False)
        return output


class CSIEncoder(nn.Module):
    """
    Simple encoder for CSI data that can be used as a feature extractor.
    """
    
    def __init__(
        self,
        input_channels: int = 2,
        num_subcarriers: int = 64,
        num_antennas: int = 3,
        output_dim: int = 256
    ):
        super(CSIEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.num_subcarriers = num_subcarriers
        self.num_antennas = num_antennas
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_channels, num_antennas, num_subcarriers)
            
        Returns:
            Feature vector of shape (batch, output_dim)
        """
        # Convolutional encoding
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = BaselineSpatialModel(
        input_channels=2,
        num_subcarriers=64,
        num_antennas=3,
        latent_dim=128,
        output_channels=3,
        output_size=(64, 64)
    ).to(device)
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 2, 3, 64).to(device)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test encoder
    encoder = CSIEncoder(
        input_channels=2,
        num_subcarriers=64,
        num_antennas=3,
        output_dim=256
    ).to(device)
    
    features = encoder(dummy_input)
    print(f"Encoder output shape: {features.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

