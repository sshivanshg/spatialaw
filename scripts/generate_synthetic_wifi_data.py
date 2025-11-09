#!/usr/bin/env python3
"""
Synthetic WiFi Data Generation for Phase-1 Baseline
Generates realistic WiFi signal data with position coordinates
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import argparse


def generate_synthetic_wifi_data(
    num_samples: int = 200,
    room_width: float = 10.0,
    room_height: float = 8.0,
    num_access_points: int = 3,
    noise_level: float = 5.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic WiFi signal data with position coordinates.
    
    Args:
        num_samples: Number of data samples to generate
        room_width: Width of the room in meters
        room_height: Height of the room in meters
        num_access_points: Number of WiFi access points
        noise_level: Noise level for signal variation (dB)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: position_x, position_y, rssi, snr, signal_strength, channel, etc.
    """
    np.random.seed(seed)
    
    # Generate random positions in the room
    positions_x = np.random.uniform(0, room_width, num_samples)
    positions_y = np.random.uniform(0, room_height, num_samples)
    
    # Generate access point positions (fixed locations)
    ap_positions = []
    for i in range(num_access_points):
        ap_x = np.random.uniform(room_width * 0.2, room_width * 0.8)
        ap_y = np.random.uniform(room_height * 0.2, room_height * 0.8)
        ap_positions.append((ap_x, ap_y))
    
    # Generate WiFi signal data for each position
    data = []
    base_time = datetime.now()
    
    for i in range(num_samples):
        x, y = positions_x[i], positions_y[i]
        
        # Calculate distance to each access point
        rssi_values = []
        snr_values = []
        signal_strengths = []
        channels = []
        
        for ap_idx, (ap_x, ap_y) in enumerate(ap_positions):
            # Calculate distance
            distance = np.sqrt((x - ap_x)**2 + (y - ap_y)**2)
            
            # Path loss model: RSSI decreases with distance
            # Free space path loss: RSSI = P0 - 20*log10(d) - noise
            # Simplified model: RSSI = -30 - 20*log10(d/1.0) + noise
            base_rssi = -30 - 20 * np.log10(max(distance, 0.1))
            noise = np.random.normal(0, noise_level)
            rssi = base_rssi + noise
            
            # Clamp RSSI to reasonable range
            rssi = np.clip(rssi, -100, -30)
            
            # Calculate SNR (signal-to-noise ratio)
            noise_floor = -95  # Typical noise floor
            snr = rssi - noise_floor
            
            # Calculate signal strength (0-100 scale)
            signal_strength = max(0, min(100, ((rssi + 100) * 100 / 70)))
            
            # Assign channel (2.4GHz or 5GHz)
            if ap_idx % 2 == 0:
                channel = np.random.choice([1, 6, 11, 36, 40, 44, 48])  # Common channels
            else:
                channel = np.random.choice([36, 40, 44, 48, 149, 153, 157, 161])  # 5GHz
            
            rssi_values.append(rssi)
            snr_values.append(snr)
            signal_strengths.append(signal_strength)
            channels.append(channel)
        
        # Use the strongest signal (closest AP)
        best_ap_idx = np.argmax(rssi_values)
        
        # Add multipath effects (signal variations)
        multipath_factor = 1.0 + 0.1 * np.sin(2 * np.pi * x / room_width) * np.cos(2 * np.pi * y / room_height)
        rssi = rssi_values[best_ap_idx] * multipath_factor
        rssi = np.clip(rssi, -100, -30)
        
        # Create data record
        record = {
            'position_x': float(x),
            'position_y': float(y),
            'rssi': float(rssi),
            'snr': float(rssi - noise_floor),
            'signal_strength': float(max(0, min(100, ((rssi + 100) * 100 / 70)))),
            'channel': int(channels[best_ap_idx]),
            'noise': float(noise_floor + np.random.normal(0, 2)),
            'num_antennas': 1,
            'phy_mode': '802.11ac',
            'timestamp': (base_time + timedelta(seconds=i)).isoformat(),
            'unix_timestamp': (base_time + timedelta(seconds=i)).timestamp(),
            'location': 'synthetic_room',
            'collection_method': 'synthetic',
            'device_id': 'synthetic_device'
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


def save_data(df: pd.DataFrame, output_path: str, format: str = 'json'):
    """
    Save generated data to file.
    
    Args:
        df: DataFrame with WiFi data
        output_path: Output file path
        format: File format ('json' or 'csv')
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if format == 'json':
        # Convert to list of dicts
        data_list = df.to_dict('records')
        with open(output_path, 'w') as f:
            json.dump(data_list, f, indent=2)
    elif format == 'csv':
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"✅ Saved {len(df)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Synthetic WiFi Data')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples to generate')
    parser.add_argument('--room_width', type=float, default=10.0, help='Room width in meters')
    parser.add_argument('--room_height', type=float, default=8.0, help='Room height in meters')
    parser.add_argument('--num_aps', type=int, default=3, help='Number of access points')
    parser.add_argument('--noise_level', type=float, default=5.0, help='Noise level (dB)')
    parser.add_argument('--output', type=str, default='data/synthetic_wifi_data.json', help='Output file path')
    parser.add_argument('--format', type=str, default='json', choices=['json', 'csv'], help='Output format')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Generating Synthetic WiFi Data")
    print("=" * 70)
    print(f"Number of samples: {args.num_samples}")
    print(f"Room size: {args.room_width}m × {args.room_height}m")
    print(f"Number of access points: {args.num_aps}")
    print(f"Noise level: {args.noise_level} dB")
    print()
    
    # Generate data
    df = generate_synthetic_wifi_data(
        num_samples=args.num_samples,
        room_width=args.room_width,
        room_height=args.room_height,
        num_access_points=args.num_aps,
        noise_level=args.noise_level,
        seed=args.seed
    )
    
    # Display statistics
    print("Data Statistics:")
    print(f"  Position X range: [{df['position_x'].min():.2f}, {df['position_x'].max():.2f}]")
    print(f"  Position Y range: [{df['position_y'].min():.2f}, {df['position_y'].max():.2f}]")
    print(f"  RSSI range: [{df['rssi'].min():.2f}, {df['rssi'].max():.2f}] dBm")
    print(f"  SNR range: [{df['snr'].min():.2f}, {df['snr'].max():.2f}] dB")
    print(f"  Signal strength range: [{df['signal_strength'].min():.2f}, {df['signal_strength'].max():.2f}]")
    print()
    
    # Save data
    save_data(df, args.output, args.format)
    
    print("\n✅ Data generation completed!")


if __name__ == "__main__":
    main()

