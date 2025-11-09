#!/usr/bin/env python3
"""
Visualize Localization Data
Creates heatmaps and scatter plots showing signal distribution across positions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from scipy.interpolate import griddata

def load_localization_data(data_path: str):
    """Load processed localization data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def plot_signal_heatmap(data, signal_type='rssi', save_path=None):
    """Plot signal strength heatmap across positions."""
    # Extract position and signal data
    angles = []
    distances = []
    signals = []
    
    for sample in data:
        angle = sample.get('angle')
        distance = sample.get('distance')
        signal = sample.get(signal_type, 0)
        
        if angle is not None and distance is not None:
            angles.append(angle)
            distances.append(distance)
            signals.append(signal)
    
    angles = np.array(angles)
    distances = np.array(distances)
    signals = np.array(signals)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Create grid for interpolation
    angle_range = np.linspace(angles.min(), angles.max(), 50)
    distance_range = np.linspace(distances.min(), distances.max(), 50)
    angle_grid, distance_grid = np.meshgrid(angle_range, distance_range)
    
    # Interpolate signal values
    signal_grid = griddata(
        (angles, distances), signals,
        (angle_grid, distance_grid),
        method='cubic',
        fill_value=signals.mean()
    )
    
    # Plot heatmap
    plt.contourf(angle_grid, distance_grid, signal_grid, levels=50, cmap='viridis')
    plt.colorbar(label=f'{signal_type.upper()} (dBm)' if signal_type == 'rssi' else f'{signal_type.upper()}')
    
    # Overlay data points
    scatter = plt.scatter(angles, distances, c=signals, cmap='viridis', 
                         edgecolors='black', linewidths=0.5, s=20, alpha=0.7)
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Distance (meters)')
    plt.title(f'Signal {signal_type.upper()} Heatmap by Position')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved heatmap to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_signal_distribution(data, save_path=None):
    """Plot signal strength distribution by position."""
    # Group by angle and distance
    positions = {}
    
    for sample in data:
        angle = sample.get('angle')
        distance = sample.get('distance')
        rssi = sample.get('rssi', 0)
        
        if angle is not None and distance is not None:
            pos_key = f"{angle}deg_{distance}m"
            if pos_key not in positions:
                positions[pos_key] = []
            positions[pos_key].append(rssi)
    
    # Create box plot
    plt.figure(figsize=(14, 6))
    
    pos_labels = list(positions.keys())
    pos_data = [positions[key] for key in pos_labels]
    
    # Create boxplot (without labels parameter)
    bp = plt.boxplot(pos_data)
    plt.xticks(range(1, len(pos_labels) + 1), pos_labels, rotation=45, ha='right')
    plt.xlabel('Position (Angle_Distance)')
    plt.ylabel('RSSI (dBm)')
    plt.title('Signal Strength Distribution by Position')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved distribution plot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_position_scatter(data, save_path=None):
    """Plot position scatter with color-coded signal strength."""
    angles = []
    distances = []
    rssi_values = []
    
    for sample in data:
        angle = sample.get('angle')
        distance = sample.get('distance')
        rssi = sample.get('rssi', 0)
        
        if angle is not None and distance is not None:
            angles.append(angle)
            distances.append(distance)
            rssi_values.append(rssi)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(angles, distances, c=rssi_values, cmap='viridis', 
                         s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    plt.colorbar(scatter, label='RSSI (dBm)')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Distance (meters)')
    plt.title('Signal Strength at Different Positions')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved scatter plot to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize Localization Data')
    parser.add_argument('--data_path', type=str, default='data/localization_data.json',
                       help='Path to processed localization data')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Visualizing Localization Data")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print()
    
    # Load data
    print("Loading data...")
    data = load_localization_data(args.data_path)
    print(f"Loaded {len(data)} samples")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Signal heatmap
    print("1. Creating signal heatmap...")
    plot_signal_heatmap(data, signal_type='rssi', 
                       save_path=f'{args.output_dir}/localization_rssi_heatmap.png')
    
    # 2. Signal distribution
    print("2. Creating signal distribution plot...")
    plot_signal_distribution(data, 
                           save_path=f'{args.output_dir}/localization_signal_distribution.png')
    
    # 3. Position scatter
    print("3. Creating position scatter plot...")
    plot_position_scatter(data, 
                         save_path=f'{args.output_dir}/localization_position_scatter.png')
    
    print("\n✅ Visualization completed!")
    print(f"   Check {args.output_dir}/ for output files")

if __name__ == "__main__":
    main()

