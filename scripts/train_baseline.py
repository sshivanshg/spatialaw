#!/usr/bin/env python3
"""
Training Script for Baseline Spatial Awareness Model
Main entry point for training the baseline model
"""

import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.spatial_model import SpatialModel, ModelSize
from src.preprocessing.data_loader import CSIDataset
from src.training.trainer import Trainer
from src.training.losses import CombinedLoss


def main():
    parser = argparse.ArgumentParser(description='Train Baseline Spatial Awareness Model')
    parser.add_argument('--data_path', type=str, default=None, help='Path to CSI data')
    parser.add_argument('--images_path', type=str, default=None, help='Path to ground truth images')
    parser.add_argument('--num_mock_samples', type=int, default=1000, help='Number of mock samples to generate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_subcarriers', type=int, default=64, help='Number of OFDM subcarriers')
    parser.add_argument('--num_antennas', type=int, default=3, help='Number of antennas')
    parser.add_argument('--output_size', type=int, nargs=2, default=[64, 64], help='Output image size (height width)')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Arguments: {args}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = CSIDataset(
        data_path=args.data_path,
        images_path=args.images_path,
        num_subcarriers=args.num_subcarriers,
        num_antennas=args.num_antennas,
        image_size=tuple(args.output_size),
        generate_mock=args.data_path is None,
        num_mock_samples=args.num_mock_samples
    )
    
    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = SpatialModel(
        input_channels=2,
        num_subcarriers=args.num_subcarriers,
        num_antennas=args.num_antennas,
        output_channels=3,
        output_size=tuple(args.output_size),
        model_size=ModelSize.SMALL  # Baseline model = small size
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create loss function
    criterion = CombinedLoss(
        mse_weight=1.0,
        ssim_weight=0.1,
        spatial_weight=0.1
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        save_best=True
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=args.num_epochs, save_freq=10)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

