#!/usr/bin/env python3
"""
Multi-Location WiFi Data Collection
Collects data from multiple locations in a guided session
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import collect_with_location function
# We need to import it from the module, not as a script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "collect_with_location", 
    os.path.join(os.path.dirname(__file__), "collect_with_location.py")
)
collect_with_location_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect_with_location_module)
collect_with_location = collect_with_location_module.collect_with_location


def interactive_multi_location_collection(
    locations: list = None,
    duration_per_location: float = 60.0,
    sampling_rate: float = 2.0,
    scenarios: list = None,
    output_dir: str = "data/collections"
):
    """
    Interactive multi-location REAL WiFi data collection.
    
    Args:
        locations: List of location names (if None, will prompt)
        duration_per_location: Duration per location in seconds
        sampling_rate: Sampling rate in Hz
        scenarios: List of scenarios to collect (if None, uses "default")
        output_dir: Output directory
        
    Raises:
        RuntimeError: If real WiFi data collection is not available
    """
    print("=" * 70)
    print("Multi-Location Real WiFi Data Collection")
    print("=" * 70)
    print()
    print("⚠️  NOTE: Only REAL WiFi data will be collected.")
    print("   Please ensure you are connected to WiFi before starting.")
    print()
    
    # Get locations if not provided
    if locations is None:
        print("Enter locations to collect data from (one per line, empty line to finish):")
        locations = []
        while True:
            location = input("Location: ").strip()
            if not location:
                break
            locations.append(location)
    
    if not locations:
        print("No locations provided. Exiting.")
        return
    
    # Get scenarios if not provided
    if scenarios is None:
        scenarios = ["default"] * len(locations)
    elif len(scenarios) == 1:
        scenarios = scenarios * len(locations)
    
    # Session metadata
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_metadata = {
        "session_id": session_id,
        "locations": locations,
        "scenarios": scenarios,
        "duration_per_location": duration_per_location,
        "sampling_rate": sampling_rate,
        "collection_timestamp": datetime.now().isoformat(),
        "real_data_only": True,
        "collections": []
    }
    
    print(f"\nSession ID: {session_id}")
    print(f"Locations: {', '.join(locations)}")
    print(f"Duration per location: {duration_per_location} seconds")
    print(f"Total estimated time: {len(locations) * duration_per_location / 60:.1f} minutes")
    print()
    
    # Collect from each location
    for i, location in enumerate(locations):
        scenario = scenarios[i] if i < len(scenarios) else "default"
        
        print(f"\n{'=' * 70}")
        print(f"Location {i+1}/{len(locations)}: {location}")
        print(f"Scenario: {scenario}")
        print(f"{'=' * 70}")
        print()
        
        if i > 0:
            response = input("Ready to collect from next location? (press Enter to continue, 'q' to quit): ")
            if response.lower() == 'q':
                print("Collection cancelled.")
                break
        
        # Collect data
        try:
            data_file, metadata_file = collect_with_location(
                location_name=location,
                duration=duration_per_location,
                sampling_rate=sampling_rate,
                scenario=scenario,
                output_dir=output_dir
            )
            
            # Record in session metadata
            session_metadata["collections"].append({
                "location": location,
                "scenario": scenario,
                "data_file": data_file,
                "metadata_file": metadata_file
            })
            
        except KeyboardInterrupt:
            print("\n\nCollection interrupted by user.")
            break
    
    # Save session metadata
    session_file = os.path.join(output_dir, f"session_{session_id}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(session_file, 'w') as f:
        json.dump(session_metadata, f, indent=2)
    
    print()
    print("=" * 70)
    print("Session Complete")
    print("=" * 70)
    print(f"Locations collected: {len(session_metadata['collections'])}")
    print(f"Session metadata: {session_file}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Location WiFi Data Collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive collection (will prompt for locations)
  python scripts/collect_multi_location.py --duration 60

  # Collect from specific locations
  python scripts/collect_multi_location.py --locations "Room 101" "Room 102" "Lab" --duration 120

  # Collect with different scenarios
  python scripts/collect_multi_location.py --locations "Library" "Lab" \\
      --scenarios "indoor_static" "outdoor_static" --duration 60
        """
    )
    
    parser.add_argument('--locations', type=str, nargs='+',
                       help='List of location names (if not provided, will prompt)')
    parser.add_argument('--duration', type=float, default=60.0,
                       help='Duration per location in seconds (default: 60)')
    parser.add_argument('--sampling_rate', type=float, default=2.0,
                       help='Sampling rate in Hz (default: 2.0)')
    parser.add_argument('--scenarios', type=str, nargs='+',
                       help='List of scenarios (default: "default" for all)')
    parser.add_argument('--output_dir', type=str, default='data/collections',
                       help='Output directory (default: data/collections)')
    
    args = parser.parse_args()
    
    try:
        interactive_multi_location_collection(
            locations=args.locations,
            duration_per_location=args.duration,
            sampling_rate=args.sampling_rate,
            scenarios=args.scenarios,
            output_dir=args.output_dir
        )
    except RuntimeError as e:
        print(f"\n❌ Collection failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

