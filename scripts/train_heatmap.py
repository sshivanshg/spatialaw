#!/usr/bin/env python3
"""
Training Script for WiFi Signal Heatmap Model
Trains a model to predict WiFi signal features at given positions
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.heatmap_model import SignalHeatmapModel, PositionHeatmapModel


class HeatmapDataset(Dataset):
    """Dataset for heatmap training."""
    
    def __init__(
        self,
        data_path: str,
        predict_signal: bool = True,
        feature_columns: list = None,
        normalize_positions: bool = True,
        normalize_features: bool = True
    ):
        """
        Initialize heatmap dataset.
        
        Args:
            data_path: Path to JSON file with WiFi data and positions
            predict_signal: If True, predict signal from position (else predict position from signal)
            feature_columns: List of signal feature columns to use
            normalize_positions: Whether to normalize positions to [0, 1]
            normalize_features: Whether to normalize signal features
        """
        self.predict_signal = predict_signal
        
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Extract positions and signals
        positions = []
        signals = []
        
        for item in data:
            # Get position
            if 'position_x' in item and 'position_y' in item:
                x = float(item['position_x'])
                y = float(item['position_y'])
                positions.append([x, y])
            else:
                continue
            
            # Get signal features
            if feature_columns is None:
                feature_columns = ['rssi', 'signal_strength', 'snr', 'channel']
            
            signal_vector = []
            for col in feature_columns:
                value = item.get(col, 0)
                if isinstance(value, (list, dict)):
                    value = len(value) if isinstance(value, list) else 0
                signal_vector.append(float(value))
            
            signals.append(signal_vector)
        
        self.positions = np.array(positions)
        self.signals = np.array(signals)
        
        # Normalize positions
        if normalize_positions:
            self.position_scaler = StandardScaler()
            self.positions = self.position_scaler.fit_transform(self.positions)
        else:
            self.position_scaler = None
        
        # Normalize features
        if normalize_features:
            self.feature_scaler = StandardScaler()
            self.signals = self.feature_scaler.fit_transform(self.signals)
        else:
            self.feature_scaler = None
        
        print(f"Loaded {len(self.positions)} samples")
        print(f"Position range: x=[{self.positions[:, 0].min():.2f}, {self.positions[:, 0].max():.2f}], "
              f"y=[{self.positions[:, 1].min():.2f}, {self.positions[:, 1].max():.2f}]")
        print(f"Signal features: {feature_columns}")
        print(f"Signal range: [{self.signals.min():.2f}, {self.signals.max():.2f}]")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        if self.predict_signal:
            # Predict signal from position
            input_data = torch.FloatTensor(self.positions[idx])
            target_data = torch.FloatTensor(self.signals[idx])
        else:
            # Predict position from signal
            input_data = torch.FloatTensor(self.signals[idx])
            target_data = torch.FloatTensor(self.positions[idx])
        
        return input_data, target_data


def main():
    parser = argparse.ArgumentParser(description='Train WiFi Signal Heatmap Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to WiFi data JSON file with positions')
    parser.add_argument('--predict_signal', action='store_true', default=True, help='Predict signal from position (default: True)')
    parser.add_argument('--predict_position', action='store_true', help='Predict position from signal (overrides --predict_signal)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Determine prediction mode
    predict_signal = not args.predict_position
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Prediction mode: {'Signal from Position' if predict_signal else 'Position from Signal'}")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = HeatmapDataset(
        data_path=args.data_path,
        predict_signal=predict_signal
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    if predict_signal:
        model = SignalHeatmapModel(
            output_features=dataset.signals.shape[1]
        )
    else:
        model = PositionHeatmapModel(
            input_features=dataset.signals.shape[1]
        )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            model_name = 'signal_heatmap_model.pth' if predict_signal else 'position_heatmap_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'predict_signal': predict_signal,
                'position_scaler': dataset.position_scaler,
                'feature_scaler': dataset.feature_scaler,
            }, os.path.join(args.save_dir, model_name))
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(args.save_dir, model_name)}")


if __name__ == "__main__":
    main()

