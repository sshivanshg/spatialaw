#!/usr/bin/env python3
"""
Training Script for 50M Parameter Model
"""

import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.spatial_model import SpatialModel, ModelSize
from src.preprocessing.data_loader import CSIDataset
from src.training.trainer import Trainer
from src.training.losses import CombinedLoss


def main():
    parser = argparse.ArgumentParser(description='Train 50M Parameter Model')
    parser.add_argument('--data_path', type=str, default=None, help='Path to combined dataset')
    parser.add_argument('--num_mock_samples', type=int, default=10000, 
                       help='Number of mock samples if no data provided')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (reduce if OOM)')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (lower for large model)')
    parser.add_argument('--output_size', type=int, nargs=2, default=[64, 64], help='Output size')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension')
    parser.add_argument('--base_channels', type=int, default=128, help='Base channels')
    parser.add_argument('--save_dir', type=str, default='checkpoints/50m_model', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs/50m_model', help='Log directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 70)
    print("Training 50M Parameter Model")
    print("=" * 70)
    print(f"Using device: {device}")
    print(f"Arguments: {args}")
    print()
    
    # Create dataset
    print("Creating dataset...")
    if args.data_path and os.path.exists(args.data_path):
        # Load from combined dataset
        dataset = CSIDataset(
            data_path=args.data_path,
            num_subcarriers=64,
            num_antennas=3,
            image_size=tuple(args.output_size),
            generate_mock=False
        )
    else:
        # Generate mock data
        dataset = CSIDataset(
            generate_mock=True,
            num_mock_samples=args.num_mock_samples,
            num_subcarriers=64,
            num_antennas=3,
            image_size=tuple(args.output_size)
        )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print()
    
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
    
    # Create model
    print("Creating 50M parameter model...")
    model = SpatialModel(
        input_channels=2,
        num_subcarriers=64,
        num_antennas=3,
        output_channels=3,
        output_size=tuple(args.output_size),
        model_size=ModelSize.MEDIUM  # Medium model = ~50M parameters
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_memory_mb = total_params * 4 / 1024 / 1024
    
    print(f"✅ Model created!")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f} million)")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model memory: {model_memory_mb:.2f} MB ({model_memory_mb/1024:.2f} GB)")
    print()
    
    # Check memory requirements
    if device.type == 'cpu':
        # Estimate training memory
        training_memory_gb = (model_memory_mb * 4) / 1024  # Model + gradients + optimizer
        print(f"⚠️  Estimated training memory: ~{training_memory_gb:.2f} GB")
        print(f"⚠️  Make sure you have enough RAM!")
        print()
    
    # Create loss function
    criterion = CombinedLoss(
        mse_weight=1.0,
        ssim_weight=0.1,
        spatial_weight=0.1
    )
    
    # Create optimizer (lower learning rate for larger model)
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
    
    # Resume if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("Starting training...")
    print(f"⚠️  This may take a while on CPU. Consider using GPU (Colab) for faster training.")
    print()
    
    trainer.train(num_epochs=args.num_epochs, save_freq=10)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

