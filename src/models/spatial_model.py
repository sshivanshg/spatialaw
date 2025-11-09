"""
Unified Spatial Awareness Model
Single model file with configurable architecture for different model sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from enum import Enum


class ModelSize(Enum):
    """Predefined model sizes."""
    SMALL = "small"      # ~3-5M parameters
    MEDIUM = "medium"    # ~50M parameters
    LARGE = "large"      # ~100M+ parameters
    # CUSTOM = "custom"    # Custom configuration


class SpatialModel(nn.Module):
    """
    Unified CNN model for spatial awareness from WiFi CSI data.
    Can be configured for different model sizes (small, medium, large, custom).
    
    This replaces the need for separate baseline_model.py, medium_model.py, and large_model.py
    by using a single configurable architecture.
    """
    
    # Predefined configurations
    CONFIGS = {
        ModelSize.SMALL: {
            'base_channels': 32,
            'latent_dim': 128,
            'num_encoder_blocks': 3,
            'num_decoder_blocks': 4,
            'pooling_size': (4, 4),
        },
        ModelSize.MEDIUM: {
            'base_channels': 112,
            'latent_dim': 448,
            'num_encoder_blocks': 4,
            'num_decoder_blocks': 6,
            'pooling_size': (6, 6),
        },
        ModelSize.LARGE: {
            'base_channels': 256,
            'latent_dim': 1024,
            'num_encoder_blocks': 5,
            'num_decoder_blocks': 8,
            'pooling_size': (8, 8),
        }
    }
    
    def __init__(
        self,
        input_channels: int = 2,
        num_subcarriers: int = 64,
        num_antennas: int = 3,
        output_channels: int = 3,
        output_size: Tuple[int, int] = (64, 64),
        model_size: ModelSize = ModelSize.SMALL,
        # Custom parameters (used if model_size=CUSTOM)
        base_channels: Optional[int] = None,
        latent_dim: Optional[int] = None,
        num_encoder_blocks: Optional[int] = None,
        num_decoder_blocks: Optional[int] = None,
        pooling_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize spatial model with configurable size.
        
        Args:
            input_channels: Number of input channels (2 for amplitude+phase)
            num_subcarriers: Number of OFDM subcarriers
            num_antennas: Number of receive antennas
            output_channels: Number of output channels (e.g., 3 for RGB)
            output_size: Output spatial dimensions (height, width)
            model_size: Predefined model size (SMALL, MEDIUM, LARGE, or CUSTOM)
            base_channels: Base number of channels (for CUSTOM)
            latent_dim: Latent dimension (for CUSTOM)
            num_encoder_blocks: Number of encoder blocks (for CUSTOM)
            num_decoder_blocks: Number of decoder blocks (for CUSTOM)
            pooling_size: Pooling size (for CUSTOM)
        """
        super(SpatialModel, self).__init__()
        
        self.input_channels = input_channels
        self.num_subcarriers = num_subcarriers
        self.num_antennas = num_antennas
        self.output_channels = output_channels
        self.output_size = output_size
        self.model_size = model_size
        
        # Get configuration
        if False:  # ModelSize.CUSTOM not implemented
            # Use provided custom parameters
            self.base_channels = base_channels or 32
            self.latent_dim = latent_dim or 128
            self.num_encoder_blocks = num_encoder_blocks or 3
            self.num_decoder_blocks = num_decoder_blocks or 4
            self.pooling_size = pooling_size or (4, 4)
        else:
            # Use predefined configuration
            config = self.CONFIGS[model_size]
            self.base_channels = config['base_channels']
            self.latent_dim = config['latent_dim']
            self.num_encoder_blocks = config['num_encoder_blocks']
            self.num_decoder_blocks = config['num_decoder_blocks']
            self.pooling_size = config['pooling_size']
        
        # Build model
        self.encoder = self._build_encoder()
        self.latent_projection = self._build_latent_projection()
        self.decoder = self._build_decoder()
        
    def _build_encoder(self) -> nn.Module:
        """Build encoder with configurable number of blocks."""
        layers = []
        
        # First block
        in_channels = self.input_channels
        out_channels = self.base_channels
        
        for i in range(self.num_encoder_blocks):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])
            in_channels = out_channels
            out_channels = min(out_channels * 2, self.base_channels * 8)  # Cap at 8x
        
        # Adaptive pooling
        layers.append(nn.AdaptiveAvgPool2d(self.pooling_size))
        
        return nn.Sequential(*layers)
    
    def _get_encoder_output_size(self) -> int:
        """Calculate encoder output size."""
        # Final channels after encoder
        final_channels = min(self.base_channels * (2 ** (self.num_encoder_blocks - 1)), 
                           self.base_channels * 8)
        return final_channels * self.pooling_size[0] * self.pooling_size[1]
    
    def _build_latent_projection(self) -> nn.Module:
        """Build latent projection layers."""
        encoder_output_size = self._get_encoder_output_size()
        
        # Scale projection size based on latent_dim
        if self.model_size == ModelSize.SMALL:
            hidden_dims = [self.latent_dim]
        elif self.model_size == ModelSize.MEDIUM:
            hidden_dims = [self.latent_dim * 4, self.latent_dim * 2, self.latent_dim]
        else:  # LARGE
            hidden_dims = [self.latent_dim * 8, self.latent_dim * 4, self.latent_dim * 2, self.latent_dim]
        
        layers = []
        in_dim = encoder_output_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2 if self.model_size == ModelSize.SMALL else 0.3),
            ])
            in_dim = hidden_dim
        
        # Final projection to latent_dim
        if hidden_dims[-1] != self.latent_dim:
            layers.append(nn.Linear(hidden_dims[-1], self.latent_dim))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Module:
        """Build decoder with configurable number of blocks."""
        # Calculate start size
        start_size = (self.output_size[0] // (2 ** self.num_decoder_blocks), 
                     self.output_size[1] // (2 ** self.num_decoder_blocks))
        start_size = (max(1, start_size[0]), max(1, start_size[1]))
        
        # Start channels
        start_channels = max(512, self.latent_dim // (start_size[0] * start_size[1]))
        if self.model_size == ModelSize.SMALL:
            start_channels = 512
        elif self.model_size == ModelSize.MEDIUM:
            start_channels = 1024
        else:  # LARGE
            start_channels = 2048
        
        layers = []
        
        # Initial projection
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
        
        # Calculate channel progression
        num_upsamples = self.num_decoder_blocks
        channel_reductions = [2] * num_upsamples  # Halve channels each time
        
        for reduction in channel_reductions:
            if current_size[0] < self.output_size[0]:
                next_channels = max(self.output_channels, current_channels // reduction)
                layers.extend([
                    nn.ConvTranspose2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU()
                ])
                current_channels = next_channels
                current_size = (current_size[0] * 2, current_size[1] * 2)
        
        # Refinement layers (more for larger models)
        num_refinement = 1 if self.model_size == ModelSize.SMALL else (2 if self.model_size == ModelSize.MEDIUM else 3)
        for _ in range(num_refinement):
            layers.extend([
                nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(current_channels),
                nn.ReLU()
            ])
        
        # Final output layer
        layers.append(
            nn.Conv2d(current_channels, self.output_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encode
        encoded = self.encoder(x)
        batch_size = encoded.size(0)
        encoded_flat = encoded.view(batch_size, -1)
        
        # Project to latent
        latent = self.latent_projection(encoded_flat)
        latent_spatial = latent.view(batch_size, self.latent_dim, 1, 1)
        
        # Decode
        output = self.decoder(latent_spatial)
        
        # Resize if needed
        if output.size()[2:] != self.output_size:
            output = F.interpolate(output, size=self.output_size, mode='bilinear', align_corners=False)
        
        return output
    
    def get_model_info(self) -> dict:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_memory_mb = total_params * 4 / 1024 / 1024
        
        return {
            'model_size': self.model_size.value,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_memory_mb': model_memory_mb,
            'base_channels': self.base_channels,
            'latent_dim': self.latent_dim,
            'num_encoder_blocks': self.num_encoder_blocks,
            'num_decoder_blocks': self.num_decoder_blocks,
            'pooling_size': self.pooling_size,
        }


# CSIEncoder - Simple encoder for CSI data
class CSIEncoder(nn.Module):
    """
    Simple encoder for CSI data that can be used as a feature extractor.
    Moved from baseline_model.py for unified model structure.
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
            Encoded features of shape (batch, output_dim)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


# Convenience functions for backward compatibility
def BaselineSpatialModel(*args, **kwargs):
    """Create baseline (small) model - for backward compatibility."""
    kwargs['model_size'] = ModelSize.SMALL
    return SpatialModel(*args, **kwargs)


def MediumSpatialModel(*args, **kwargs):
    """Create medium model - for backward compatibility."""
    kwargs['model_size'] = ModelSize.MEDIUM
    return SpatialModel(*args, **kwargs)


def LargeSpatialModel(*args, **kwargs):
    """Create large model - for backward compatibility."""
    kwargs['model_size'] = ModelSize.LARGE
    return SpatialModel(*args, **kwargs)


if __name__ == "__main__":
    # Test different model sizes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    for model_size in [ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE]:
        print(f"Testing {model_size.value.upper()} model...")
        model = SpatialModel(
            input_channels=2,
            num_subcarriers=64,
            num_antennas=3,
            output_channels=3,
            output_size=(64, 64),
            model_size=model_size
        ).to(device)
        
        info = model.get_model_info()
        print(f"  Parameters: {info['total_parameters']:,} ({info['total_parameters']/1e6:.2f}M)")
        print(f"  Memory: {info['model_memory_mb']:.2f} MB")
        print(f"  Config: base_channels={info['base_channels']}, latent_dim={info['latent_dim']}")
        print()
        
        # Test forward pass (set to eval mode to avoid BatchNorm issues with batch_size=1)
        model.eval()
        dummy_input = torch.randn(1, 2, 3, 64).to(device)
        try:
            with torch.no_grad():
                output = model(dummy_input)
            print(f"   Forward pass successful: {dummy_input.shape} → {output.shape}")
        except Exception as e:
            print(f"   Error: {e}")
        print()

