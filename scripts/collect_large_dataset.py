#!/usr/bin/env python3
"""
Collect Large Dataset for Training 50M Parameter Model
Collects WiFi data continuously to build a large training dataset
"""

import sys
import os
import argparse
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.wifi_collector import WiFiCollector


def collect_large_dataset(
    total_samples: int = 10000,
    sampling_rate: float = 2.0,
    output_dir: str = "data/large_dataset",
    use_mock: bool = False,
    save_interval: int = 1000
):
    """
    Collect a large dataset for training.
    
    Args:
        total_samples: Total number of samples to collect
        sampling_rate: Samples per second
        output_dir: Directory to save data
        use_mock: Use mock data instead of real WiFi
        save_interval: Save data every N samples
    """
    print("=" * 70)
    print("Large Dataset Collection for 50M Parameter Model")
    print("=" * 70)
    print()
    print(f"Target samples: {total_samples:,}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Estimated time: {total_samples / sampling_rate / 60:.1f} minutes")
    print(f"Output directory: {output_dir}")
    print(f"Collection method: {'Mock data' if use_mock else 'Real WiFi'}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create collector
    collector = WiFiCollector(sampling_rate=sampling_rate, use_mock=use_mock)
    
    # Calculate duration
    duration = total_samples / sampling_rate
    
    print(f"Starting collection...")
    print(f"This will take approximately {duration / 60:.1f} minutes")
    print(f"Press Ctrl+C to stop early\n")
    
    start_time = time.time()
    sample_count = 0
    file_count = 0
    
    try:
        # Collect samples
        while sample_count < total_samples:
            # Get sample
            sample = collector.get_wifi_info()
            collector.data.append(sample)
            sample_count += 1
            
            # Progress update
            if sample_count % 100 == 0:
                elapsed = time.time() - start_time
                rate = sample_count / elapsed if elapsed > 0 else 0
                remaining = (total_samples - sample_count) / rate if rate > 0 else 0
                
                print(f"Progress: {sample_count:,}/{total_samples:,} samples "
                      f"({sample_count/total_samples*100:.1f}%) | "
                      f"Rate: {rate:.1f} samples/sec | "
                      f"Elapsed: {elapsed/60:.1f} min | "
                      f"Remaining: {remaining/60:.1f} min")
            
            # Save periodically
            if sample_count % save_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"wifi_data_batch_{file_count:04d}_{timestamp}.json"
                filepath = os.path.join(output_dir, filename)
                collector.save_data(filepath, format='json')
                print(f"  💾 Saved batch {file_count} ({sample_count:,} samples) to {filename}")
                
                # Clear data to save memory (optional - keep last batch)
                if file_count > 0:  # Keep data for final save
                    collector.data = collector.data[-save_interval:]  # Keep last batch
                
                file_count += 1
            
            # Sleep to maintain sampling rate
            time.sleep(1.0 / sampling_rate)
        
        # Final save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wifi_data_final_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        collector.save_data(filepath, format='json')
        print(f"\n💾 Final save: {len(collector.data)} samples to {filename}")
        
        # Create combined dataset file
        print(f"\nCreating combined dataset...")
        combine_datasets(output_dir)
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Collection interrupted by user.")
        print(f"Collected {sample_count:,} samples so far.")
        
        # Save what we have
        if collector.data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wifi_data_interrupted_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            collector.save_data(filepath, format='json')
            print(f"💾 Saved {len(collector.data)} samples to {filename}")
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n" + "=" * 70)
    print("Collection Summary")
    print("=" * 70)
    print(f"Total samples collected: {sample_count:,}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average rate: {sample_count/elapsed:.2f} samples/second")
    print(f"Files created: {file_count + 1}")
    print(f"Output directory: {output_dir}")
    print()


def combine_datasets(data_dir: str, output_dir: str = None):
    """Combine all JSON files in data_dir into a single dataset."""
    import json
    import glob
    
    if output_dir is None:
        output_dir = data_dir
    
    # Find all JSON files
    json_files = sorted(glob.glob(os.path.join(data_dir, "wifi_data_*.json")))
    
    if not json_files:
        print("No data files found to combine.")
        return None
    
    print(f"Found {len(json_files)} data files to combine...")
    
    # Combine all data
    all_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        except Exception as e:
            print(f"Warning: Could not read {json_file}: {e}")
    
    if not all_data:
        print("No data to combine.")
        return None
    
    # Save combined dataset
    output_file = os.path.join(output_dir, "combined_dataset.json")
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"✅ Combined {len(all_data):,} samples into {output_file}")
    
    # Also save as CSV for easy analysis
    try:
        import pandas as pd
        df = pd.DataFrame(all_data)
        csv_file = os.path.join(output_dir, "combined_dataset.csv")
        df.to_csv(csv_file, index=False)
        print(f"✅ Also saved as CSV: {csv_file}")
        return output_file
    except Exception as e:
        print(f"Warning: Could not save CSV: {e}")
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Collect Large Dataset for Training')
    parser.add_argument('--total_samples', type=int, default=10000, 
                       help='Total number of samples to collect (default: 10000)')
    parser.add_argument('--sampling_rate', type=float, default=2.0,
                       help='Sampling rate in Hz (default: 2.0)')
    parser.add_argument('--output_dir', type=str, default='data/large_dataset',
                       help='Output directory (default: data/large_dataset)')
    parser.add_argument('--use_mock', action='store_true',
                       help='Use mock data instead of real WiFi')
    parser.add_argument('--save_interval', type=int, default=1000,
                       help='Save data every N samples (default: 1000)')
    parser.add_argument('--auto_continue', action='store_true',
                       help='Auto-continue without prompting')
    
    args = parser.parse_args()
    
    # Set auto_continue if not interactive
    import sys
    if not sys.stdin.isatty():
        args.auto_continue = True
    
    # Calculate estimated time
    estimated_minutes = args.total_samples / args.sampling_rate / 60
    print(f"⚠️  This will collect {args.total_samples:,} samples")
    print(f"⚠️  Estimated time: {estimated_minutes:.1f} minutes ({estimated_minutes/60:.1f} hours)")
    print(f"⚠️  Make sure you have enough disk space!")
    print()
    
    # Auto-continue if flag set or non-interactive
    import sys
    if args.auto_continue or not sys.stdin.isatty():
        print("Auto-continuing...")
    else:
        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
    
    collect_large_dataset(
        total_samples=args.total_samples,
        sampling_rate=args.sampling_rate,
        output_dir=args.output_dir,
        use_mock=args.use_mock,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()

