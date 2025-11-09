#!/usr/bin/env python3
"""
Combine Data from Multiple Devices and Locations
Merges WiFi data collected from different laptops at different locations
"""

import sys
import os
import argparse
import json
import glob
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def find_data_files(data_dir: str, recursive: bool = True) -> list:
    """Find all WiFi data JSON files."""
    if recursive:
        pattern = os.path.join(data_dir, "**", "*.json")
    else:
        pattern = os.path.join(data_dir, "*.json")
    
    json_files = glob.glob(pattern, recursive=recursive)
    
    # Filter out metadata files and session files
    data_files = [
        f for f in json_files 
        if not f.endswith('_metadata.json') 
        and 'session_' not in os.path.basename(f)
        and 'combined' not in os.path.basename(f)
    ]
    
    return sorted(data_files)


def load_and_validate_data(filepath: str) -> list:
    """Load and validate WiFi data file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return []
        
        # Validate data
        valid_data = []
        for sample in data:
            if isinstance(sample, dict) and 'rssi' in sample:
                valid_data.append(sample)
        
        return valid_data
    except Exception as e:
        print(f"⚠️  Warning: Could not load {filepath}: {e}")
        return []


def combine_data(
    data_dir: str = "data",
    output_file: str = "data/combined_multi_device.json",
    recursive: bool = True,
    group_by: str = None
):
    """
    Combine data from multiple devices and locations.
    
    Args:
        data_dir: Directory containing data files
        output_file: Output file path
        group_by: Group by 'location', 'device', or None (combine all)
    """
    print("=" * 70)
    print("Combining Multi-Device WiFi Data")
    print("=" * 70)
    print()
    print(f"Data directory: {data_dir}")
    print(f"Output file: {output_file}")
    print(f"Group by: {group_by if group_by else 'All (no grouping)'}")
    print()
    
    # Find all data files
    data_files = find_data_files(data_dir, recursive=recursive)
    
    if not data_files:
        print("❌ No data files found!")
        return
    
    print(f"Found {len(data_files)} data files")
    print()
    
    # Load and organize data
    all_data = []
    stats = defaultdict(lambda: {'files': 0, 'samples': 0, 'locations': set(), 'devices': set()})
    
    for filepath in data_files:
        data = load_and_validate_data(filepath)
        
        if not data:
            continue
        
        # Get file info
        rel_path = os.path.relpath(filepath, data_dir)
        path_parts = Path(rel_path).parts
        
        # Extract location and device from path
        location = None
        device = None
        
        if len(path_parts) >= 2:
            location = path_parts[0]
            device = path_parts[1]
        
        # Add to all data
        all_data.extend(data)
        
        # Update statistics
        key = group_by if group_by else 'all'
        if group_by == 'location' and location:
            key = location
        elif group_by == 'device' and device:
            key = device
        
        stats[key]['files'] += 1
        stats[key]['samples'] += len(data)
        if location:
            stats[key]['locations'].add(location)
        if device:
            stats[key]['devices'].add(device)
    
    if not all_data:
        print("❌ No valid data found!")
        return
    
    # Print statistics
    print("Data Statistics:")
    print(f"  Total files: {len(data_files)}")
    print(f"  Total samples: {len(all_data):,}")
    print()
    
    if group_by:
        print(f"Grouped by {group_by}:")
        for key, stat in stats.items():
            print(f"  {key}:")
            print(f"    Files: {stat['files']}")
            print(f"    Samples: {stat['samples']:,}")
            if stat['locations']:
                print(f"    Locations: {', '.join(sorted(stat['locations']))}")
            if stat['devices']:
                print(f"    Devices: {', '.join(sorted(stat['devices']))}")
        print()
    else:
        # Overall statistics
        locations = set()
        devices = set()
        for sample in all_data:
            if 'location' in sample:
                locations.add(sample['location'])
            if 'device_id' in sample:
                devices.add(sample['device_id'])
        
        print("Overall Statistics:")
        print(f"  Unique locations: {len(locations)}")
        if locations:
            print(f"    {', '.join(sorted(locations))}")
        print(f"  Unique devices: {len(devices)}")
        if devices:
            print(f"    {', '.join(sorted(devices))}")
        print()
    
    # Save combined data
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"✅ Combined data saved to: {output_file}")
    print(f"   Total samples: {len(all_data):,}")
    
    # Also save statistics
    stats_file = output_file.replace('.json', '_stats.json')
    stats_dict = {
        'total_files': len(data_files),
        'total_samples': len(all_data),
        'unique_locations': list(locations) if not group_by else None,
        'unique_devices': list(devices) if not group_by else None,
        'grouped_stats': {k: {
            'files': v['files'],
            'samples': v['samples'],
            'locations': list(v['locations']),
            'devices': list(v['devices'])
        } for k, v in stats.items()}
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"✅ Statistics saved to: {stats_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Combine WiFi Data from Multiple Devices and Locations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine all data
  python scripts/combine_multi_device_data.py

  # Combine and group by location
  python scripts/combine_multi_device_data.py --group_by location

  # Combine and group by device
  python scripts/combine_multi_device_data.py --group_by device

  # Specify custom data directory
  python scripts/combine_multi_device_data.py --data_dir data/collections
        """
    )
    
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--output', type=str, default='data/combined_multi_device.json',
                       help='Output file (default: data/combined_multi_device.json)')
    parser.add_argument('--group_by', type=str, choices=['location', 'device', None],
                       default=None,
                       help='Group data by location or device')
    parser.add_argument('--no_recursive', action='store_true',
                       help='Do not search recursively')
    
    args = parser.parse_args()
    
    combine_data(
        data_dir=args.data_dir,
        output_file=args.output,
        recursive=not args.no_recursive,
        group_by=args.group_by
    )


if __name__ == "__main__":
    main()

