#!/usr/bin/env python3
"""
Process Widar3.0 Dataset for Motion Detection
Converts Widar3.0 gesture recognition data to our motion detection format
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import zipfile

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("⚠️  h5py not installed. Install with: pip install h5py")

# Simplified MATLAB v7.3 loader function
def load_mat73(file_path: str) -> Dict:
    """Load MATLAB v7.3 .mat file using h5py."""
    data = {}
    with h5py.File(file_path, 'r') as f:
        top_keys = list(f.keys())
        refs_group = f.get('#refs#')
        
        for var_name in top_keys:
            if var_name == '#refs#':
                continue
            try:
                obj = f[var_name]
                if isinstance(obj, h5py.Dataset):
                    dtype = obj.dtype
                    if dtype == 'object' or 'ref' in str(dtype).lower():
                        # Reference - dereference it
                        try:
                            ref_value = obj[()]
                            if isinstance(ref_value, np.ndarray) and ref_value.size > 0:
                                ref_id = ref_value.flat[0]
                                if refs_group and ref_id in refs_group:
                                    data[var_name] = np.array(refs_group[ref_id])
                                else:
                                    data[var_name] = ref_value
                        except:
                            pass
                    else:
                        # Regular dataset
                        arr = np.array(obj)
                        # Handle compound dtype (complex numbers)
                        if arr.dtype.names is not None:
                            if 'real' in arr.dtype.names and 'imag' in arr.dtype.names:
                                arr = arr['real'] + 1j * arr['imag']
                            elif 'r' in arr.dtype.names and 'i' in arr.dtype.names:
                                arr = arr['r'] + 1j * arr['i']
                        data[var_name] = arr
            except:
                pass
        
        # If no data found, search for large arrays
        if len(data) == 0:
            def find_data(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.size > 1000:
                    arr = np.array(obj)
                    data['csi_complex_data'] = arr
            f.visititems(find_data)
    
    return data


def load_widar3_csi(csi_file: Path) -> Dict:
    """
    Load CSI data from Widar3.0 format.
    Supports .mat (MATLAB v7.3 with h5py, older versions with scipy), .json, and .npy formats.
    """
    data = {}
    
    # Try MATLAB .mat format
    if csi_file.suffix == '.mat' or 'mat' in str(csi_file).lower():
        # First, try with scipy (for older .mat files)
        if HAS_SCIPY:
            try:
                mat_data = loadmat(str(csi_file), simplify_cells=True)
                # Remove MATLAB metadata
                data = {k: v for k, v in mat_data.items() 
                       if not k.startswith('__')}
                print(f"   Loaded .mat file (scipy): {len(data)} variables")
                return data
            except Exception as e:
                error_msg = str(e)
                # Check if it's a v7.3 file that needs h5py
                if 'HDF' in error_msg or 'v7.3' in error_msg:
                    print(f"  Detected MATLAB v7.3 format, using h5py...")
                else:
                    print(f"  ⚠️  Error loading .mat file with scipy: {e}")
        
        # Try with h5py for MATLAB v7.3 files
        if HAS_H5PY:
            try:
                # Use the simplified loader
                data = load_mat73(str(csi_file))
                
                # Post-process: find CSI data
                if data:
                    # Look for CSI data by name or find largest 3D array
                    if 'csi_complex_data' not in data:
                        # Find largest 3D array (likely CSI)
                        for key, value in data.items():
                            if isinstance(value, np.ndarray) and value.ndim == 3 and value.size > 1000:
                                data['csi_complex_data'] = value
                                print(f"  Found CSI data in variable: {key}")
                                break
                        # If still not found, find any large array
                        if 'csi_complex_data' not in data:
                            largest_key = max(data.keys(), 
                                            key=lambda k: data[k].size if isinstance(data[k], np.ndarray) else 0,
                                            default=None)
                            if largest_key and isinstance(data[largest_key], np.ndarray):
                                data['csi_complex_data'] = data[largest_key]
                                print(f"  Using largest array as CSI data: {largest_key}")
                    
                    # Print summary
                    print(f"  ✅ Loaded .mat file (h5py): {len(data)} variables")
                    for key, value in list(data.items())[:3]:
                        if isinstance(value, np.ndarray):
                            print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
                            
            except Exception as e:
                print(f"  ⚠️  Error loading .mat file with h5py: {e}")
                import traceback
                traceback.print_exc()
                data = {}
        else:
            print(f"  ❌ h5py not available. Install with: pip install h5py")
            data = {}
    
    # Try JSON format
    elif csi_file.suffix == '.json':
        try:
            with open(csi_file, 'r') as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = {'samples': content}
                else:
                    data = content
        except Exception as e:
            print(f"  ⚠️  Error loading JSON: {e}")
    
    # Try NumPy format
    elif csi_file.suffix == '.npy':
        try:
            array_data = np.load(csi_file, allow_pickle=True)
            data = {'csi': array_data}
        except Exception as e:
            print(f"  ⚠️  Error loading .npy: {e}")
    
    else:
        print(f"  ⚠️  Unknown file format: {csi_file.suffix}")
    
    return data


def convert_widar3_to_motion_format(
    widar_data: Dict,
    file_name: str = "",
    gesture_label: Optional[str] = None
) -> List[Dict]:
    """
    Convert CSI localization data format to our motion detection format.
    
    This dataset structure (from readme):
    - csi_complex_data: 3D matrix (3 antennas x 30 subcarriers x number of packets)
    - File names contain location info: loc_30deg_1m.mat = 30 degrees, 1 meter
    
    For motion detection: We can treat each packet as a time sample.
    Since this is localization data (not gesture), we can use it for spatial analysis.
    """
    converted = []
    
    # Extract CSI data - look for common variable names
    csi_data = None
    for key in ['csi_complex_data', 'csi', 'CSI', 'data', 'csi_data']:
        if key in widar_data:
            csi_data = widar_data[key]
            break
    
    # If not found, look for any 3D array
    if csi_data is None:
        for key, value in widar_data.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                csi_data = value
                print(f"  Found CSI data in variable: {key}")
                break
    
    if csi_data is None:
        print("  ⚠️  No CSI data found in file")
        print(f"  Available keys: {list(widar_data.keys())[:10]}")
        return []
    
    # Parse location from filename
    angle = None
    distance = None
    if '30deg' in file_name:
        angle = 30
    elif 'minus60deg' in file_name or '-60deg' in file_name:
        angle = -60
    
    if '1m' in file_name:
        distance = 1.0
    elif '2m' in file_name:
        distance = 2.0
    elif '3m' in file_name:
        distance = 3.0
    elif '4m' in file_name:
        distance = 4.0
    elif '5m' in file_name:
        distance = 5.0
    
    # Handle CSI data structure
    # From readme: "3 x 30 x number of packets" = (antennas, subcarriers, packets) in MATLAB
    # But when loaded with h5py, MATLAB's column-major becomes row-major, so it might be transposed
    # Actual shape we're seeing: (958767, 30, 3) = (packets, subcarriers, antennas)
    if isinstance(csi_data, np.ndarray):
        if csi_data.ndim == 3:
            # Determine the correct interpretation based on size
            # If first dimension is very large, it's likely packets
            dim0, dim1, dim2 = csi_data.shape
            
            # The largest dimension is usually packets (hundreds of thousands)
            # The readme says: 3 antennas x 30 subcarriers x packets
            # So if we see (large, 30, 3), it's (packets, subcarriers, antennas)
            if dim0 > dim1 and dim0 > dim2:
                # Shape: (packets, subcarriers, antennas)
                num_packets, num_subcarriers, num_antennas = dim0, dim1, dim2
                print(f"  CSI shape: {csi_data.shape} (packets={num_packets}, subcarriers={num_subcarriers}, antennas={num_antennas})")
            elif dim2 > dim0 and dim2 > dim1:
                # Shape: (antennas, subcarriers, packets) - transpose needed
                num_antennas, num_subcarriers, num_packets = dim0, dim1, dim2
                # Transpose to (packets, subcarriers, antennas)
                csi_data = np.transpose(csi_data, (2, 1, 0))
                print(f"  CSI shape: {csi_data.shape} (transposed to: packets={num_packets}, subcarriers={num_subcarriers}, antennas={num_antennas})")
            else:
                # Default: assume (packets, subcarriers, antennas)
                num_packets, num_subcarriers, num_antennas = dim0, dim1, dim2
                print(f"  CSI shape: {csi_data.shape} (assuming: packets={num_packets}, subcarriers={num_subcarriers}, antennas={num_antennas})")
        elif csi_data.ndim == 2:
            # Shape: (subcarriers, packets) or (packets, subcarriers)
            num_packets = max(csi_data.shape)
            num_subcarriers = min(csi_data.shape)
            num_antennas = 1
            print(f"  CSI shape: {csi_data.shape} (2D array)")
        else:
            print(f"  ⚠️  Unexpected CSI shape: {csi_data.shape}")
            return []
    else:
        print(f"  ⚠️  CSI data is not a numpy array: {type(csi_data)}")
        return []
    
    # Process each packet as a time sample
    sampling_rate = 2500.0  # Hz (from readme: packet rate = 2500 Hz)
    
    # Limit number of samples to process (to avoid huge JSON files)
    # Process every Nth packet to get a manageable dataset
    max_samples = 10000  # Limit to 10k samples per file
    step = max(1, num_packets // max_samples) if num_packets > max_samples else 1
    
    samples_to_process = (num_packets + step - 1) // step  # Round up
    print(f"  Processing {num_packets} packets (every {step}th packet, ~{samples_to_process} samples)")
    
    # Process packets with sampling
    packet_indices = list(range(0, num_packets, step))
    
    for idx, i in enumerate(packet_indices):
        # Extract CSI for this packet
        if csi_data.ndim == 3:
            # Shape: (packets, subcarriers, antennas)
            # Extract all antennas and subcarriers for this packet
            packet_csi = csi_data[i, :, :]  # Shape: (30, 3) = (subcarriers, antennas)
            # Flatten to 1D: (90,) = 30 subcarriers * 3 antennas
            sample_csi = packet_csi.flatten()
        elif csi_data.ndim == 2:
            # Shape: (subcarriers, packets) or (packets, subcarriers)
            if csi_data.shape[0] < csi_data.shape[1]:
                # (subcarriers, packets)
                sample_csi = csi_data[:, i]
            else:
                # (packets, subcarriers)
                sample_csi = csi_data[i, :]
        else:
            continue
        
        # Calculate RSSI from CSI magnitude (dBm)
        # RSSI is typically calculated as: 10 * log10(sum(|CSI|^2))
        magnitude_squared = np.abs(sample_csi) ** 2
        power = np.sum(magnitude_squared)
        rssi = 10 * np.log10(power + 1e-10)  # Add small epsilon to avoid log(0)
        
        # Calculate average magnitude for signal strength estimation
        avg_magnitude = np.mean(np.abs(sample_csi))
        
        # For localization data, we don't have explicit movement labels
        # But we can use it for spatial analysis
        # For motion detection purposes, we'll mark all as having signal (not movement)
        # since this is static localization data
        movement = False  # Static localization, not motion
        movement_label = 0
        
        # Convert complex CSI to magnitude/phase for JSON serialization
        # Store magnitude and phase separately (or just magnitude)
        sample_csi_magnitude = np.abs(sample_csi).tolist()  # Magnitude only
        sample_csi_phase = np.angle(sample_csi).tolist()  # Phase in radians
        
        # Limit size for JSON storage
        if len(sample_csi_magnitude) > 30:
            sample_csi_magnitude = sample_csi_magnitude[:30]
            sample_csi_phase = sample_csi_phase[:30]
        
        # Create converted sample
        converted_sample = {
            'rssi': float(rssi),
            'snr': float(rssi) + 30,  # Estimate SNR (noise floor ~-90 dBm)
            'signal_strength': max(0, min(100, ((float(rssi) + 100) * 100 / 70))),
            'channel': 64,  # From readme: channel 64
            'timestamp': f'packet_{i}',
            'unix_timestamp': 0,
            'time_index': idx,  # Sequential index in converted samples
            'packet_index': i,  # Original packet index
            'elapsed_time': float(i / sampling_rate),  # Time in seconds
            'movement': movement,
            'movement_label': movement_label,
            'location': f'angle_{angle}_dist_{distance}m' if angle and distance else 'unknown',
            'angle': float(angle) if angle else None,
            'distance': float(distance) if distance else None,
            'device_id': 'localization_dataset',
            'device_hostname': 'localization_dataset',
            'device_platform': 'NUC_Intel5300',
            'collection_method': 'localization_dataset',
            'csi_magnitude': float(avg_magnitude),
            'num_antennas': int(num_antennas),
            'num_subcarriers': int(num_subcarriers),
            'csi_magnitude_samples': sample_csi_magnitude,  # Magnitude of CSI (real numbers)
            'csi_phase_samples': sample_csi_phase,  # Phase of CSI (real numbers)
            'file_name': file_name
        }
        converted.append(converted_sample)
    
    return converted


def process_widar3_dataset(
    data_dir: str = "data_wifii",
    extract_dir: Optional[str] = None,
    output_file: str = "data/widar3_motion_data.json"
):
    """
    Process entire Widar3.0 dataset for motion detection.
    """
    data_dir = Path(data_dir)
    
    # Use data_dir directly if extract_dir not specified (for .mat files)
    if extract_dir is None:
        extract_dir = data_dir
    else:
        extract_dir = Path(extract_dir)
    
    print("=" * 70)
    print("Processing Widar3.0 Dataset for Motion Detection")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Processing directory: {extract_dir}")
    print()
    
    # Check if directory exists
    if not extract_dir.exists():
        print(f"❌ Directory not found: {extract_dir}")
        return
    
    # Find all data files (.mat, .json, .npy)
    csi_files = (list(extract_dir.glob("*.mat")) + 
                 list(extract_dir.glob("*.json")) + 
                 list(extract_dir.glob("*.npy")))
    csi_files = [f for f in csi_files if f.is_file()]
    
    print(f"Found {len(csi_files)} data files")
    if len(csi_files) == 0:
        print(f"  Looking for: .mat, .json, or .npy files in {extract_dir}")
        print(f"  Files in directory: {list(extract_dir.glob('*'))[:10]}")
    
    all_data = []
    
    for csi_file in csi_files:  # Process all files
        print(f"\nProcessing {csi_file.name}...")
        try:
            widar_data = load_widar3_csi(csi_file)
            if widar_data:
                converted = convert_widar3_to_motion_format(widar_data, file_name=csi_file.name)
                if converted:
                    all_data.extend(converted)
                    print(f"   Converted {len(converted)} samples")
                else:
                    print(f"  ⚠️  No samples converted")
            else:
                print(f"  ⚠️  No data found in file")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Save processed data
    if all_data:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"\n Processed {len(all_data)} samples")
        print(f" Saved to: {output_path}")
        
        # Statistics
        movement_count = sum(1 for s in all_data if s.get('movement', False))
        print(f"\nStatistics:")
        print(f"  Total samples: {len(all_data)}")
        print(f"  Movement samples: {movement_count} ({movement_count/len(all_data)*100:.1f}%)")
        print(f"  No-movement samples: {len(all_data)-movement_count} ({(len(all_data)-movement_count)/len(all_data)*100:.1f}%)")
    else:
        print("\n❌ No data processed. Check file formats.")


def main():
    parser = argparse.ArgumentParser(description='Process Widar3.0 Dataset for Motion Detection')
    parser.add_argument('--data_dir', type=str, default='data_wifii',
                       help='Directory containing Widar3.0 files')
    parser.add_argument('--extract_dir', type=str, default=None,
                       help='Directory with extracted files (default: same as data_dir)')
    parser.add_argument('--output', type=str, default='data/widar3_motion_data.json',
                       help='Output file for processed data')
    
    args = parser.parse_args()
    
    process_widar3_dataset(
        data_dir=args.data_dir,
        extract_dir=args.extract_dir,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

