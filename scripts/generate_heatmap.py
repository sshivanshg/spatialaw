#!/usr/bin/env python3
"""
Generate WiFi Signal Heatmap
Creates a heatmap visualization of WiFi signal distribution
"""

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import griddata

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.heatmap_model import SignalHeatmapModel


def load_model(checkpoint_path: str, predict_signal: bool = True, output_features: int = 4):
    """Load trained heatmap model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if predict_signal:
        model = SignalHeatmapModel(output_features=output_features)
    else:
        raise ValueError("Position prediction not supported for heatmap generation")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def generate_heatmap(
    model: SignalHeatmapModel,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    resolution: int = 50,
    feature_index: int = 0,  # Which feature to visualize (0=RSSI, 1=signal_strength, etc.)
    position_scaler=None
):
    """Generate heatmap for a given feature."""
    # Create grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid
    positions = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    # Normalize positions if scaler provided
    if position_scaler is not None:
        positions = position_scaler.transform(positions)
    
    # Predict signals
    positions_tensor = torch.FloatTensor(positions)
    with torch.no_grad():
        signals = model(positions_tensor)
    
    # Extract feature of interest
    feature_values = signals[:, feature_index].numpy()
    
    # Reshape to grid
    Z = feature_values.reshape(resolution, resolution)
    
    return X, Y, Z


def visualize_heatmap(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    feature_name: str = "RSSI",
    save_path: str = "visualizations/heatmap.png",
    show_points: bool = True,
    data_points: np.ndarray = None
):
    """Visualize heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label=feature_name)
    
    # Plot data points if provided
    if show_points and data_points is not None:
        plt.scatter(data_points[:, 0], data_points[:, 1], 
                   c='red', s=50, marker='x', label='Data Points', linewidths=2)
        plt.legend()
    
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.title(f'WiFi Signal Heatmap: {feature_name}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Heatmap saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate WiFi Signal Heatmap')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, help='Path to data file (for determining bounds and showing points)')
    parser.add_argument('--x_min', type=float, help='Minimum X coordinate')
    parser.add_argument('--x_max', type=float, help='Maximum X coordinate')
    parser.add_argument('--y_min', type=float, help='Minimum Y coordinate')
    parser.add_argument('--y_max', type=float, help='Maximum Y coordinate')
    parser.add_argument('--resolution', type=int, default=50, help='Heatmap resolution')
    parser.add_argument('--feature', type=int, default=0, help='Feature to visualize (0=RSSI, 1=signal_strength, 2=SNR, 3=channel)')
    parser.add_argument('--feature_name', type=str, default='RSSI', help='Feature name for visualization')
    parser.add_argument('--output', type=str, default='visualizations/heatmap.png', help='Output image path')
    parser.add_argument('--show_points', action='store_true', help='Show data collection points on heatmap')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, checkpoint = load_model(args.model_path, predict_signal=True, output_features=4)
    position_scaler = checkpoint.get('position_scaler', None)
    
    # Determine bounds
    if args.data_path and os.path.exists(args.data_path):
        with open(args.data_path, 'r') as f:
            data = json.load(f)
        
        positions = []
        for item in data:
            if 'position_x' in item and 'position_y' in item:
                positions.append([float(item['position_x']), float(item['position_y'])])
        
        if positions:
            positions = np.array(positions)
            x_min = args.x_min if args.x_min is not None else positions[:, 0].min() - 0.5
            x_max = args.x_max if args.x_max is not None else positions[:, 0].max() + 0.5
            y_min = args.y_min if args.y_min is not None else positions[:, 1].min() - 0.5
            y_max = args.y_max if args.y_max is not None else positions[:, 1].max() + 0.5
            data_points = positions
        else:
            raise ValueError("No position data found in file")
    else:
        if args.x_min is None or args.x_max is None or args.y_min is None or args.y_max is None:
            raise ValueError("Must provide bounds (--x_min, --x_max, --y_min, --y_max) or --data_path")
        x_min, x_max = args.x_min, args.x_max
        y_min, y_max = args.y_min, args.y_max
        data_points = None
    
    print(f"Generating heatmap for region: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    # Generate heatmap
    print("Generating heatmap...")
    X, Y, Z = generate_heatmap(
        model, x_min, x_max, y_min, y_max,
        resolution=args.resolution,
        feature_index=args.feature,
        position_scaler=position_scaler
    )
    
    # Visualize
    print("Creating visualization...")
    visualize_heatmap(
        X, Y, Z,
        feature_name=args.feature_name,
        save_path=args.output,
        show_points=args.show_points,
        data_points=data_points
    )
    
    print("✅ Done!")


if __name__ == "__main__":
    main()

