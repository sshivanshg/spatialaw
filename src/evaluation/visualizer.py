"""
Visualization Utilities for Spatial Awareness Model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List
import os


class Visualizer:
    """
    Visualizer for model predictions and CSI data.
    """
    
    def __init__(self, save_dir: str = "visualizations"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_csi(self, csi_data: np.ndarray, save_path: Optional[str] = None):
        """
        Visualize CSI data (amplitude and phase).
        
        Args:
            csi_data: CSI data array of shape (2, num_antennas, num_subcarriers) or (num_antennas, num_subcarriers)
            save_path: Path to save visualization
        """
        if len(csi_data.shape) == 2:
            # Single antenna, create amplitude and phase from complex data
            amplitude = np.abs(csi_data)
            phase = np.angle(csi_data)
            num_antennas = 1
        else:
            amplitude = csi_data[0] if csi_data.shape[0] == 2 else np.abs(csi_data)
            phase = csi_data[1] if csi_data.shape[0] == 2 else np.angle(csi_data)
            if len(amplitude.shape) > 2:
                num_antennas = amplitude.shape[0]
            else:
                num_antennas = 1
                amplitude = amplitude[np.newaxis, :]
                phase = phase[np.newaxis, :]
        
        fig, axes = plt.subplots(2, num_antennas, figsize=(5 * num_antennas, 10))
        if num_antennas == 1:
            axes = axes[:, np.newaxis]
        
        for i in range(num_antennas):
            # Amplitude
            axes[0, i].imshow(amplitude[i], aspect='auto', cmap='viridis')
            axes[0, i].set_title(f'Amplitude - Antenna {i+1}')
            axes[0, i].set_xlabel('Subcarrier')
            axes[0, i].set_ylabel('Time/Sample')
            
            # Phase
            axes[1, i].imshow(phase[i], aspect='auto', cmap='hsv')
            axes[1, i].set_title(f'Phase - Antenna {i+1}')
            axes[1, i].set_xlabel('Subcarrier')
            axes[1, i].set_ylabel('Time/Sample')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_predictions(
        self,
        predictions: np.ndarray,
        targets: Optional[np.ndarray] = None,
        num_samples: int = 8,
        save_path: Optional[str] = None
    ):
        """
        Visualize model predictions.
        
        Args:
            predictions: Predicted images array
            targets: Ground truth images array (optional)
            num_samples: Number of samples to visualize
            save_path: Path to save visualization
        """
        num_samples = min(num_samples, predictions.shape[0])
        
        # Normalize images to [0, 1] for visualization
        pred_norm = predictions.copy()
        if pred_norm.min() < 0:
            pred_norm = (pred_norm + 1) / 2
        pred_norm = np.clip(pred_norm, 0, 1)
        
        if targets is not None:
            target_norm = targets.copy()
            if target_norm.min() < 0:
                target_norm = (target_norm + 1) / 2
            target_norm = np.clip(target_norm, 0, 1)
            
            fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))
            for i in range(num_samples):
                # Predictions
                if len(pred_norm.shape) == 4:  # (batch, channels, height, width)
                    pred_img = pred_norm[i].transpose(1, 2, 0)
                else:
                    pred_img = pred_norm[i]
                axes[0, i].imshow(pred_img)
                axes[0, i].set_title(f'Prediction {i+1}')
                axes[0, i].axis('off')
                
                # Targets
                if len(target_norm.shape) == 4:
                    target_img = target_norm[i].transpose(1, 2, 0)
                else:
                    target_img = target_norm[i]
                axes[1, i].imshow(target_img)
                axes[1, i].set_title(f'Target {i+1}')
                axes[1, i].axis('off')
        else:
            fig, axes = plt.subplots(1, num_samples, figsize=(3 * num_samples, 3))
            if num_samples == 1:
                axes = [axes]
            for i in range(num_samples):
                if len(pred_norm.shape) == 4:
                    pred_img = pred_norm[i].transpose(1, 2, 0)
                else:
                    pred_img = pred_norm[i]
                axes[i].imshow(pred_img)
                axes[i].set_title(f'Prediction {i+1}')
                axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot training history.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses (optional)
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', marker='o')
        if val_losses:
            plt.plot(val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    # Test visualizer
    visualizer = Visualizer()
    
    # Test CSI visualization
    from src.data_collection.csi_processor import generate_mock_csi
    mock_csi = generate_mock_csi(num_samples=1, num_antennas=3, num_subcarriers=64)
    csi_sample = mock_csi[0]
    
    from src.data_collection.csi_processor import CSIProcessor
    processor = CSIProcessor()
    amplitude, phase = processor.process_csi(csi_sample)
    csi_viz = np.stack([amplitude, phase], axis=0)
    
    visualizer.visualize_csi(csi_viz, save_path="test_csi_visualization.png")
    
    # Test prediction visualization
    pred = np.random.rand(8, 3, 64, 64)
    target = np.random.rand(8, 3, 64, 64)
    visualizer.visualize_predictions(pred, target, save_path="test_predictions.png")

