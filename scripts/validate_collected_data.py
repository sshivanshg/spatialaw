#!/usr/bin/env python3
"""
Validate Collected WiFi Data
Checks data quality, completeness, and identifies issues
"""

import sys
import os
import argparse
import json
import glob
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def validate_data_file(filepath: str) -> dict:
    """
    Validate a single data file.
    
    Args:
        filepath: Path to data file
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "file": filepath,
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }
    
    try:
        # Load data
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            results["errors"].append("Data is not a list")
            results["valid"] = False
            return results
        
        if len(data) == 0:
            results["errors"].append("Data is empty")
            results["valid"] = False
            return results
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data)
        
        # Check required fields
        required_fields = ['rssi', 'signal_strength', 'timestamp']
        for field in required_fields:
            if field not in df.columns:
                results["errors"].append(f"Missing required field: {field}")
                results["valid"] = False
        
        # Check data quality
        if 'rssi' in df.columns:
            # RSSI should be between -100 and 0 dBm
            invalid_rssi = df[(df['rssi'] < -100) | (df['rssi'] > 0)]
            if len(invalid_rssi) > 0:
                results["warnings"].append(f"{len(invalid_rssi)} samples with invalid RSSI values")
            
            results["stats"]["rssi_mean"] = float(df['rssi'].mean())
            results["stats"]["rssi_std"] = float(df['rssi'].std())
            results["stats"]["rssi_min"] = float(df['rssi'].min())
            results["stats"]["rssi_max"] = float(df['rssi'].max())
        
        if 'signal_strength' in df.columns:
            # Signal strength should be between 0 and 100
            invalid_signal = df[(df['signal_strength'] < 0) | (df['signal_strength'] > 100)]
            if len(invalid_signal) > 0:
                results["warnings"].append(f"{len(invalid_signal)} samples with invalid signal strength")
            
            results["stats"]["signal_strength_mean"] = float(df['signal_strength'].mean())
        
        if 'channel' in df.columns:
            unique_channels = df['channel'].unique()
            results["stats"]["channels"] = [int(ch) for ch in unique_channels if pd.notna(ch)]
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                results["warnings"].append(f"{duplicate_timestamps} duplicate timestamps")
        
        # Check collection method
        if 'collection_method' in df.columns:
            collection_methods = df['collection_method'].unique()
            results["stats"]["collection_methods"] = list(collection_methods)
            if 'mock' in collection_methods:
                results["warnings"].append("Data contains mock data")
        
        # General statistics
        results["stats"]["num_samples"] = len(df)
        results["stats"]["num_fields"] = len(df.columns)
        results["stats"]["fields"] = list(df.columns)
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            missing_fields = missing_values[missing_values > 0].to_dict()
            results["warnings"].append(f"Missing values: {missing_fields}")
        
    except json.JSONDecodeError as e:
        results["errors"].append(f"Invalid JSON: {str(e)}")
        results["valid"] = False
    except Exception as e:
        results["errors"].append(f"Error processing file: {str(e)}")
        results["valid"] = False
    
    return results


def validate_directory(directory: str, recursive: bool = True) -> dict:
    """
    Validate all data files in a directory.
    
    Args:
        directory: Directory path
        recursive: Whether to search recursively
        
    Returns:
        Dictionary with validation results for all files
    """
    results = {
        "directory": directory,
        "files": [],
        "summary": {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "total_samples": 0,
            "total_errors": 0,
            "total_warnings": 0
        }
    }
    
    # Find all JSON files
    if recursive:
        pattern = os.path.join(directory, "**", "*.json")
    else:
        pattern = os.path.join(directory, "*.json")
    
    json_files = glob.glob(pattern, recursive=recursive)
    
    # Filter out metadata files
    json_files = [f for f in json_files if not f.endswith('_metadata.json') and 'session_' not in os.path.basename(f)]
    
    results["summary"]["total_files"] = len(json_files)
    
    # Validate each file
    for filepath in json_files:
        file_results = validate_data_file(filepath)
        results["files"].append(file_results)
        
        if file_results["valid"]:
            results["summary"]["valid_files"] += 1
            if "num_samples" in file_results["stats"]:
                results["summary"]["total_samples"] += file_results["stats"]["num_samples"]
        else:
            results["summary"]["invalid_files"] += 1
        
        results["summary"]["total_errors"] += len(file_results["errors"])
        results["summary"]["total_warnings"] += len(file_results["warnings"])
    
    return results


def print_validation_report(results: dict, detailed: bool = False):
    """
    Print validation report.
    
    Args:
        results: Validation results
        detailed: Whether to print detailed information
    """
    if "directory" in results:
        # Directory validation
        print("=" * 70)
        print("Data Validation Report")
        print("=" * 70)
        print()
        print(f"Directory: {results['directory']}")
        print(f"Total files: {results['summary']['total_files']}")
        print(f"Valid files: {results['summary']['valid_files']}")
        print(f"Invalid files: {results['summary']['invalid_files']}")
        print(f"Total samples: {results['summary']['total_samples']:,}")
        print(f"Total errors: {results['summary']['total_errors']}")
        print(f"Total warnings: {results['summary']['total_warnings']}")
        print()
        
        if detailed:
            print("File Details:")
            print("-" * 70)
            for file_results in results["files"]:
                status = "✓" if file_results["valid"] else "✗"
                print(f"{status} {os.path.basename(file_results['file'])}")
                
                if file_results["stats"]:
                    if "num_samples" in file_results["stats"]:
                        print(f"    Samples: {file_results['stats']['num_samples']}")
                    if "rssi_mean" in file_results["stats"]:
                        print(f"    RSSI: {file_results['stats']['rssi_mean']:.1f} dBm "
                              f"(range: {file_results['stats']['rssi_min']:.0f} to {file_results['stats']['rssi_max']:.0f})")
                
                if file_results["errors"]:
                    for error in file_results["errors"]:
                        print(f"    ERROR: {error}")
                
                if file_results["warnings"]:
                    for warning in file_results["warnings"]:
                        print(f"    WARNING: {warning}")
                
                print()
        
        # Summary of issues
        invalid_files = [f for f in results["files"] if not f["valid"]]
        if invalid_files:
            print("Invalid Files:")
            for file_results in invalid_files:
                print(f"  - {os.path.basename(file_results['file'])}")
                for error in file_results["errors"]:
                    print(f"    {error}")
            print()
    
    else:
        # Single file validation
        print("=" * 70)
        print("Data Validation Report")
        print("=" * 70)
        print()
        print(f"File: {results['file']}")
        print(f"Status: {'✓ VALID' if results['valid'] else '✗ INVALID'}")
        print()
        
        if results["stats"]:
            print("Statistics:")
            for key, value in results["stats"].items():
                print(f"  {key}: {value}")
            print()
        
        if results["errors"]:
            print("Errors:")
            for error in results["errors"]:
                print(f"  - {error}")
            print()
        
        if results["warnings"]:
            print("Warnings:")
            for warning in results["warnings"]:
                print(f"  - {warning}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Validate Collected WiFi Data')
    parser.add_argument('path', type=str,
                       help='Path to data file or directory')
    parser.add_argument('--recursive', action='store_true',
                       help='Search directories recursively')
    parser.add_argument('--detailed', action='store_true',
                       help='Print detailed validation report')
    parser.add_argument('--output', type=str, default=None,
                       help='Save validation report to JSON file')
    
    args = parser.parse_args()
    
    # Validate
    if os.path.isfile(args.path):
        results = validate_data_file(args.path)
    elif os.path.isdir(args.path):
        results = validate_directory(args.path, recursive=args.recursive)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        return
    
    # Print report
    print_validation_report(results, detailed=args.detailed)
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Validation report saved to {args.output}")


if __name__ == "__main__":
    main()

