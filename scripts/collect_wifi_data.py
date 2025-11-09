#!/usr/bin/env python3
"""
WiFi Data Collection Script
Collects WiFi signal data using Mac system utilities
"""

import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.wifi_collector import WiFiCollector


def main():
    parser = argparse.ArgumentParser(description='Collect WiFi Data')
    parser.add_argument('--duration', type=float, default=60.0, help='Collection duration in seconds')
    parser.add_argument('--sampling_rate', type=float, default=2.0, help='Sampling rate in Hz')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--format', type=str, default='json', choices=['json', 'csv'], help='Output format')
    parser.add_argument('--interface', type=str, default='en0', help='Network interface name')
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data", exist_ok=True)
        args.output = f"data/wifi_data_{timestamp}.{args.format}"
    
    print("=" * 70)
    print("Real WiFi Data Collection")
    print("=" * 70)
    print(f"Duration: {args.duration} seconds")
    print(f"Sampling rate: {args.sampling_rate} Hz")
    print(f"Interface: {args.interface}")
    print(f"Output: {args.output}")
    print()
    print("⚠️  NOTE: Only REAL WiFi data will be collected.")
    print("   Please ensure you are connected to WiFi before starting.")
    print()
    
    # Create collector (will raise error if real data collection is not available)
    try:
        collector = WiFiCollector(
            interface=args.interface,
            sampling_rate=args.sampling_rate
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
    
    # Collect data
    try:
        collector.collect_continuous(
            duration=args.duration,
            save_path=args.output
        )
    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
        collector.save_data(args.output, format=args.format)
    
    # Print summary
    df = collector.get_dataframe()
    print(f"\nCollection complete!")
    print(f"Total samples: {len(df)}")
    if len(df) > 0:
        print(f"\nData summary:")
        print(df.describe())


if __name__ == "__main__":
    main()

