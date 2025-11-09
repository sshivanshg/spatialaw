#!/usr/bin/env python3
"""
Generate Synthetic Time-Series Data for Motion Detection
Creates realistic WiFi time series with movement/no-movement labels
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
import argparse


def generate_synthetic_motion_data(
    num_samples: int = 600,
    sampling_rate: float = 10.0,
    movement_ratio: float = 0.5,
    noise_level: float = 2.0,
    movement_variance: float = 5.0,
    seed: int = 42
) -> List[Dict]:
    """
    Generate synthetic time-series WiFi data with movement labels.
    
    Args:
        num_samples: Number of time samples
        sampling_rate: Sampling rate in Hz
        movement_ratio: Ratio of samples with movement (0.0 to 1.0)
        noise_level: Base noise level (dB)
        movement_variance: Additional variance during movement (dB)
        seed: Random seed
    
    Returns:
        List of WiFi data dictionaries with movement labels
    """
    np.random.seed(seed)
    
    # Generate movement labels
    num_movement = int(num_samples * movement_ratio)
    movement_labels = np.zeros(num_samples, dtype=int)
    movement_labels[:num_movement] = 1
    np.random.shuffle(movement_labels)
    
    # Generate time series
    base_time = datetime.now()
    data = []
    
    base_rssi = -60.0  # Base RSSI value
    
    for i in range(num_samples):
        movement = movement_labels[i] == 1
        
        # Base signal with noise
        if movement:
            # Movement causes higher variance
            rssi = base_rssi + np.random.normal(0, movement_variance)
            # Add periodic variations (simulating person walking)
            periodic = 2.0 * np.sin(2 * np.pi * i / (sampling_rate * 2))  # ~2 second period
            rssi += periodic
        else:
            # No movement: lower variance
            rssi = base_rssi + np.random.normal(0, noise_level)
        
        # Clamp RSSI to reasonable range
        rssi = np.clip(rssi, -100, -30)
        
        # Calculate SNR and signal strength
        noise_floor = -95
        snr = rssi - noise_floor
        signal_strength = max(0, min(100, ((rssi + 100) * 100 / 70)))
        
        # Create data record
        record = {
            'rssi': float(rssi),
            'snr': float(snr),
            'signal_strength': float(signal_strength),
            'channel': 44,
            'noise': float(noise_floor),
            'num_antennas': 1,
            'phy_mode': '802.11ac',
            'timestamp': (base_time + timedelta(seconds=i/sampling_rate)).isoformat(),
            'unix_timestamp': (base_time + timedelta(seconds=i/sampling_rate)).timestamp(),
            'time_index': i,
            'elapsed_time': float(i / sampling_rate),
            'movement': bool(movement),
            'movement_label': int(movement),
            'location': 'synthetic_room',
            'collection_method': 'synthetic',
            'device_id': 'synthetic_device'
        }
        
        data.append(record)
    
    return data


def save_data(data: List[Dict], output_path: str):
    """Save generated data to JSON file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Synthetic Motion Detection Data')
    parser.add_argument('--num_samples', type=int, default=600, 
                       help='Number of time samples')
    parser.add_argument('--sampling_rate', type=float, default=10.0, 
                       help='Sampling rate (Hz)')
    parser.add_argument('--movement_ratio', type=float, default=0.5, 
                       help='Ratio of samples with movement (0.0 to 1.0)')
    parser.add_argument('--noise_level', type=float, default=2.0, 
                       help='Base noise level (dB)')
    parser.add_argument('--movement_variance', type=float, default=5.0, 
                       help='Additional variance during movement (dB)')
    parser.add_argument('--output', type=str, 
                       default='data/synthetic_motion_data.json',
                       help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Generating Synthetic Motion Detection Data")
    print("=" * 70)
    print(f"Number of samples: {args.num_samples}")
    print(f"Sampling rate: {args.sampling_rate} Hz")
    print(f"Duration: {args.num_samples / args.sampling_rate:.1f} seconds")
    print(f"Movement ratio: {args.movement_ratio:.1%}")
    print()
    
    # Generate data
    data = generate_synthetic_motion_data(
        num_samples=args.num_samples,
        sampling_rate=args.sampling_rate,
        movement_ratio=args.movement_ratio,
        noise_level=args.noise_level,
        movement_variance=args.movement_variance,
        seed=args.seed
    )
    
    # Display statistics
    movement_count = sum(1 for item in data if item['movement'])
    no_movement_count = len(data) - movement_count
    
    print("Data Statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Movement samples: {movement_count} ({movement_count/len(data)*100:.1f}%)")
    print(f"  No-movement samples: {no_movement_count} ({no_movement_count/len(data)*100:.1f}%)")
    print(f"  RSSI range: [{min(item['rssi'] for item in data):.2f}, {max(item['rssi'] for item in data):.2f}] dBm")
    print()
    
    # Save data
    save_data(data, args.output)
    
    print("✅ Data generation completed!")


if __name__ == "__main__":
    main()

