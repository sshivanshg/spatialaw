#!/usr/bin/env python3
"""
Baseline Visualization Script
Creates visualizations for Phase-1 baseline analysis
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_data(data_path: str) -> pd.DataFrame:
    """Load WiFi data from JSON or CSV file."""
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    return df


def plot_position_signal_heatmap(df: pd.DataFrame, output_path: str = None):
    """Plot position vs signal strength heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # RSSI heatmap
    scatter1 = axes[0].scatter(df['position_x'], df['position_y'], 
                              c=df['rssi'], cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(scatter1, ax=axes[0], label='RSSI (dBm)')
    axes[0].set_xlabel('Position X (meters)')
    axes[0].set_ylabel('Position Y (meters)')
    axes[0].set_title('Signal Strength Heatmap (RSSI)')
    axes[0].grid(True, alpha=0.3)
    
    # Signal strength heatmap
    scatter2 = axes[1].scatter(df['position_x'], df['position_y'], 
                              c=df['signal_strength'], cmap='plasma', s=50, alpha=0.6)
    plt.colorbar(scatter2, ax=axes[1], label='Signal Strength (0-100)')
    axes[1].set_xlabel('Position X (meters)')
    axes[1].set_ylabel('Position Y (meters)')
    axes[1].set_title('Signal Strength Heatmap (0-100 scale)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved heatmap to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_signal_distributions(df: pd.DataFrame, output_path: str = None):
    """Plot signal distribution histograms."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(df['rssi'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('RSSI (dBm)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('RSSI Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(df['snr'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('SNR (dB)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('SNR Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(df['signal_strength'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Signal Strength (0-100)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Signal Strength Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(df['channel'], bins=20, edgecolor='black', alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Channel')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Channel Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved distributions to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_path: str = None):
    """Plot correlation matrix."""
    numeric_cols = ['position_x', 'position_y', 'rssi', 'snr', 'signal_strength', 'channel', 'noise']
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved correlation matrix to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_pca_visualization(df: pd.DataFrame, output_path: str = None):
    """Plot PCA visualization."""
    pca_features = ['rssi', 'snr', 'signal_strength', 'channel', 'noise']
    X_pca = df[pca_features].values
    
    scaler = StandardScaler()
    X_pca_scaled = scaler.fit_transform(X_pca)
    
    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(X_pca_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                         c=df['rssi'], cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(scatter, label='RSSI (dBm)')
    plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA Visualization of WiFi Signals')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved PCA visualization to {output_path}")
    else:
        plt.show()
    plt.close()
    
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")


def main():
    parser = argparse.ArgumentParser(description='Create Baseline Visualizations')
    parser.add_argument('--data_path', type=str, required=True, help='Path to WiFi data file')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Baseline Visualization - Phase-1")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print()
    
    # Load data
    print("Loading data...")
    df = load_data(args.data_path)
    print(f"Loaded {len(df)} samples")
    print()
    
    # Clean data
    df_clean = df.dropna()
    df_clean = df_clean[(df_clean['rssi'] >= -100) & (df_clean['rssi'] <= -30)]
    print(f"After cleaning: {len(df_clean)} samples")
    print()
    
    # Create visualizations
    print("Creating visualizations...")
    plot_position_signal_heatmap(df_clean, 
                                 os.path.join(args.output_dir, 'baseline_position_signal_scatter.png'))
    plot_signal_distributions(df_clean, 
                             os.path.join(args.output_dir, 'baseline_signal_distributions.png'))
    plot_correlation_matrix(df_clean, 
                           os.path.join(args.output_dir, 'baseline_correlation_matrix.png'))
    plot_pca_visualization(df_clean, 
                          os.path.join(args.output_dir, 'baseline_pca.png'))
    
    print("\n✅ All visualizations created!")


if __name__ == "__main__":
    main()

