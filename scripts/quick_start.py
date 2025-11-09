#!/usr/bin/env python3
"""
Quick Start Script for Baseline Model
This script provides a quick way to test the baseline model with mock data
"""

import sys
import os
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.spatial_model import SpatialModel, ModelSize
from src.preprocessing.data_loader import CSIDataset
from src.training.trainer import Trainer
from src.training.losses import CombinedLoss
from src.evaluation.evaluator import Evaluator
from src.evaluation.visualizer import Visualizer


def main():
    print("=" * 60)
    print("Spatial Awareness Baseline Model - Quick Start")
    print("=" * 60)
    print()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Create mock dataset
    print("1. Creating mock CSI dataset...")
    dataset = CSIDataset(
        generate_mock=True,
        num_mock_samples=200,
        num_subcarriers=64,
        num_antennas=3,
        image_size=(64, 64)
    )
    print(f"   Created {len(dataset)} samples")
    print()
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"   Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    print()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model
    print("2. Creating baseline model...")
    model = SpatialModel(
        input_channels=2,
        num_subcarriers=64,
        num_antennas=3,
        output_channels=3,
        output_size=(64, 64),
        model_size=ModelSize.SMALL  # Baseline model = small size
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print()
    
    # Test forward pass
    print("3. Testing forward pass...")
    sample = dataset[0]
    csi_input = sample['csi'].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(csi_input)
    print(f"   Input shape: {csi_input.shape}")
    print(f"   Output shape: {output.shape}")
    print()
    
    # Create trainer
    print("4. Setting up trainer...")
    criterion = CombinedLoss(mse_weight=1.0, ssim_weight=0.1, spatial_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir="checkpoints",
        log_dir="logs"
    )
    print()
    
    # Train for a few epochs
    print("5. Training model (5 epochs)...")
    print("   (This is a quick test - train longer for better results)")
    print()
    trainer.train(num_epochs=5, save_freq=5)
    print()
    
    # Evaluate
    print("6. Evaluating model...")
    evaluator = Evaluator(model, device=device)
    metrics = evaluator.evaluate(val_loader)
    print("   Evaluation metrics:")
    for key, value in metrics.items():
        print(f"     {key}: {value:.4f}")
    print()
    
    # Visualize
    print("7. Creating visualizations...")
    visualizer = Visualizer(save_dir="visualizations")
    
    # Get some samples
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        csi_batch = sample_batch['csi'].to(device)
        predictions = model(csi_batch)
        
        pred_np = predictions.cpu().numpy()
        if 'image' in sample_batch:
            target_np = sample_batch['image'].cpu().numpy()
            visualizer.visualize_predictions(
                pred_np[:8],
                target_np[:8],
                save_path="visualizations/predictions.png"
            )
        else:
            visualizer.visualize_predictions(
                pred_np[:8],
                save_path="visualizations/predictions.png"
            )
    
    print("   Visualizations saved to visualizations/")
    print()
    
    print("=" * 60)
    print("Quick start completed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Collect real WiFi data: python scripts/collect_wifi_data.py")
    print("  2. Train with real data: python scripts/train_baseline.py --data_path <path>")
    print("  3. View tensorboard: tensorboard --logdir logs")
    print()


if __name__ == "__main__":
    main()

