#!/usr/bin/env python3
"""
Enhanced WiFi Data Collection with Location Tracking
Collects WiFi data with location, scenario, and metadata information
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection.wifi_collector import WiFiCollector


def collect_with_location(
    location_name: str,
    duration: float = 60.0,
    sampling_rate: float = 2.0,
    scenario: str = "default",
    notes: str = "",
    output_dir: str = "data/collections",
    interface: str = "en0"
):
    """
    Collect REAL WiFi data with location and metadata.
    
    Args:
        location_name: Name/label for the collection location
        duration: Collection duration in seconds
        sampling_rate: Sampling rate in Hz
        scenario: Scenario description (e.g., "static", "walking", "indoor", "outdoor")
        notes: Additional notes about the collection
        output_dir: Output directory
        interface: Network interface
        
    Raises:
        RuntimeError: If real WiFi data collection is not available
    """
    print("=" * 70)
    print("Real WiFi Data Collection with Location Tracking")
    print("=" * 70)
    print()
    print(f"Location: {location_name}")
    print(f"Scenario: {scenario}")
    print(f"Duration: {duration} seconds")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Expected samples: ~{int(duration * sampling_rate)}")
    if notes:
        print(f"Notes: {notes}")
    print()
    print("⚠️  NOTE: Only REAL WiFi data will be collected.")
    print("   Please ensure you are connected to WiFi before starting.")
    print()
    
    # Create output directory structure
    location_dir = os.path.join(output_dir, location_name.replace(" ", "_").lower())
    os.makedirs(location_dir, exist_ok=True)
    
    # Create collector (will raise error if real data collection is not available)
    try:
        collector = WiFiCollector(
            interface=interface,
            sampling_rate=sampling_rate
        )
    except RuntimeError as e:
        print(str(e))
        raise
    
    # Collect data
    print("Starting collection...")
    print("Press Ctrl+C to stop early\n")
    
    try:
        collector.collect_continuous(duration=duration)
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user.")
    
    # Create metadata
    metadata = {
        "location_name": location_name,
        "scenario": scenario,
        "notes": notes,
        "duration_seconds": duration,
        "sampling_rate": sampling_rate,
        "collection_method": collector.collection_method,
        "num_samples": len(collector.data),
        "start_time": collector.data[0]['timestamp'] if collector.data else None,
        "end_time": collector.data[-1]['timestamp'] if collector.data else None,
        "collection_timestamp": datetime.now().isoformat(),
        "interface": interface,
        "real_data_only": True
    }
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_clean = scenario.replace(" ", "_").lower()
    filename = f"{location_name.replace(' ', '_').lower()}_{scenario_clean}_{timestamp}"
    
    # Save data
    data_file = os.path.join(location_dir, f"{filename}.json")
    collector.save_data(data_file, format='json')
    
    # Save metadata
    metadata_file = os.path.join(location_dir, f"{filename}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print()
    print("=" * 70)
    print("Collection Summary")
    print("=" * 70)
    print(f"Location: {location_name}")
    print(f"Scenario: {scenario}")
    print(f"Samples collected: {len(collector.data)}")
    print(f"Data file: {data_file}")
    print(f"Metadata file: {metadata_file}")
    print()
    
    # Data statistics
    if collector.data:
        df = collector.get_dataframe()
        print("Data Statistics:")
        if 'rssi' in df.columns:
            print(f"  RSSI range: {df['rssi'].min():.0f} to {df['rssi'].max():.0f} dBm")
            print(f"  RSSI mean: {df['rssi'].mean():.1f} dBm")
        if 'signal_strength' in df.columns:
            print(f"  Signal strength range: {df['signal_strength'].min():.0f} to {df['signal_strength'].max():.0f}%")
        if 'channel' in df.columns:
            unique_channels = df['channel'].unique()
            print(f"  Channels: {', '.join(map(str, unique_channels))}")
        print()
    
    return data_file, metadata_file


def main():
    parser = argparse.ArgumentParser(
        description='Collect WiFi Data with Location Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data at a specific location
  python scripts/collect_with_location.py --location "Room 101" --duration 120

  # Collect data for different scenarios
  python scripts/collect_with_location.py --location "Library" --scenario "indoor_static" --duration 60
  python scripts/collect_with_location.py --location "Library" --scenario "indoor_walking" --duration 120

  # Collect with notes
  python scripts/collect_with_location.py --location "Lab" --scenario "experiment_1" \\
      --notes "Testing WiFi signal variations" --duration 180
        """
    )
    
    parser.add_argument('--location', type=str, required=True,
                       help='Location name (e.g., "Room 101", "Library", "Lab")')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Collection duration in seconds (default: 60)')
    parser.add_argument('--sampling_rate', type=float, default=2.0,
                       help='Sampling rate in Hz (default: 2.0)')
    parser.add_argument('--scenario', type=str, default='default',
                       help='Scenario description (e.g., "static", "walking", "indoor", "outdoor")')
    parser.add_argument('--notes', type=str, default='',
                       help='Additional notes about the collection')
    parser.add_argument('--output_dir', type=str, default='data/collections',
                       help='Output directory (default: data/collections)')
    parser.add_argument('--interface', type=str, default='en0',
                       help='Network interface (default: en0)')
    
    args = parser.parse_args()
    
    try:
        collect_with_location(
            location_name=args.location,
            duration=args.duration,
            sampling_rate=args.sampling_rate,
            scenario=args.scenario,
            notes=args.notes,
            output_dir=args.output_dir,
            interface=args.interface
        )
    except RuntimeError as e:
        print(f"\n❌ Collection failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

