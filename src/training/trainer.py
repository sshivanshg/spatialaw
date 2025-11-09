"""
Training Script for Baseline Spatial Awareness Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from typing import Dict, Optional
import numpy as np
from datetime import datetime


class Trainer:
    """
    Trainer class for training the baseline spatial awareness model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        save_dir: str = "checkpoints",
        log_dir: str = "logs",
        save_best: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            save_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            save_best: Whether to save best model based on validation loss
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
        
        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        else:
            self.optimizer = optimizer
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Directories
        self.save_dir = save_dir
        self.log_dir = log_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.save_best = save_best
        
        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            csi_input = batch['csi'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(csi_input)
            
            # Calculate loss
            if 'image' in batch:
                target = batch['image'].to(self.device)
                loss = self.criterion(output, target)
            else:
                # If no target, use reconstruction loss on CSI
                # This is a placeholder - you might want to use a different loss
                loss = torch.mean(torch.abs(output))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                csi_input = batch['csi'].to(self.device)
                output = self.model(csi_input)
                
                if 'image' in batch:
                    target = batch['image'].to(self.device)
                    loss = self.criterion(output, target)
                else:
                    loss = torch.mean(torch.abs(output))
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_checkpoint(self, filename: str = None, is_best: bool = False):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pth"
        
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] - 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Checkpoint loaded from {filepath}")
    
    def train(self, num_epochs: int, save_freq: int = 10):
        """Train the model for specified number of epochs."""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss if val_loss > 0 else train_loss)
            
            # Log metrics
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            if val_loss > 0:
                self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
            self.writer.add_scalar('Epoch/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            if val_loss > 0:
                print(f"  Val Loss: {val_loss:.6f}")
            
            # Save checkpoint
            is_best = False
            if val_loss > 0 and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
            
            if (epoch + 1) % save_freq == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        print("Training completed!")
        self.writer.close()


if __name__ == "__main__":
    # Example usage
    from src.models.baseline_model import BaselineSpatialModel
    from src.preprocessing.data_loader import CSIDataset
    
    # Create model
    model = BaselineSpatialModel(
        input_channels=2,
        num_subcarriers=64,
        num_antennas=3,
        latent_dim=128,
        output_channels=3,
        output_size=(64, 64)
    )
    
    # Create dataset
    dataset = CSIDataset(generate_mock=True, num_mock_samples=100)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        save_dir="checkpoints",
        log_dir="logs"
    )
    
    # Train
    trainer.train(num_epochs=10)

