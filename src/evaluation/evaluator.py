"""
Evaluation Utilities for Spatial Awareness Model
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm
import os


class Evaluator:
    """
    Evaluator for the spatial awareness model.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        from src.evaluation.metrics import calculate_metrics
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        criterion = torch.nn.MSELoss()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                csi_input = batch['csi'].to(self.device)
                output = self.model(csi_input)
                
                if 'image' in batch:
                    target = batch['image'].to(self.device)
                    loss = criterion(output, target)
                    total_loss += loss.item()
                    
                    all_predictions.append(output.cpu().numpy())
                    all_targets.append(target.cpu().numpy())
                
                num_batches += 1
        
        # Calculate metrics
        metrics = {}
        if all_predictions and all_targets:
            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            
            metrics = calculate_metrics(predictions, targets)
            metrics['mse_loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        
        return metrics
    
    def predict(self, csi_data: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions for CSI data.
        
        Args:
            csi_data: CSI input tensor
            
        Returns:
            Model predictions
        """
        self.model.eval()
        with torch.no_grad():
            csi_data = csi_data.to(self.device)
            if len(csi_data.shape) == 3:
                csi_data = csi_data.unsqueeze(0)  # Add batch dimension
            output = self.model(csi_data)
            if output.shape[0] == 1:
                output = output.squeeze(0)  # Remove batch dimension if single sample
        return output


if __name__ == "__main__":
    # Example usage
    from src.models.baseline_model import BaselineSpatialModel
    from src.preprocessing.data_loader import CSIDataset
    from torch.utils.data import DataLoader
    
    # Load model
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
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Evaluate
    evaluator = Evaluator(model)
    metrics = evaluator.evaluate(data_loader)
    print(f"Evaluation metrics: {metrics}")

