"""
Parser for Intel 5300 CSI binary .dat files.

Uses csiread library for accurate parsing of Intel 5300 CSI format.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import csiread
    HAS_CSIREAD = True
except ImportError:
    HAS_CSIREAD = False


def load_dat_file(path: Path) -> np.ndarray:
    """
    Load Intel 5300 CSI .dat file and return CSI amplitude matrix.
    
    Uses csiread library if available, otherwise falls back to basic parser.
    
    Parameters
    ----------
    path:
        Path to .dat file.
    
    Returns
    -------
    np.ndarray
        Array of shape (n_packets, n_subcarriers) with CSI amplitudes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    
    if HAS_CSIREAD:
        # Use csiread library for accurate parsing
        csi_data = csiread.Intel(str(path))
        csi_data.read()
        
        if len(csi_data.csi) == 0:
            raise ValueError(f"No CSI data found in {path}")
        
        # Extract CSI amplitudes
        # csi_data.csi shape: (n_packets, subcarriers, Nrx, Ntx)
        csi = csi_data.csi
        
        # Take magnitude and average over antennas
        if csi.ndim == 4:
            # Shape: (n_packets, subcarriers, Nrx, Ntx)
            # Take magnitude, average over receive and transmit antennas
            amplitudes = np.abs(csi)  # (n_packets, subcarriers, Nrx, Ntx)
            amplitudes = np.mean(amplitudes, axis=(2, 3))  # Average over Nrx and Ntx
        elif csi.ndim == 3:
            # Shape: (n_packets, subcarriers, antennas)
            amplitudes = np.abs(csi)
            amplitudes = np.mean(amplitudes, axis=2)  # Average over antennas
        else:
            amplitudes = np.abs(csi)
        
        return amplitudes.astype(np.float64)
    else:
        raise ImportError(
            "csiread library required for .dat file support. "
            "Install with: pip install csiread"
        )


def _read_bfee_legacy(bytes_data: bytes) -> dict:
    """
    Parse a single CSI entry from bytes (simplified version).
    
    This is a basic implementation. For full accuracy, refer to the
    original read_bfee.c implementation.
    """
    idx = 0
    
    # Read header fields (simplified - actual format is more complex)
    if len(bytes_data) < 20:
        raise ValueError("Insufficient data for CSI entry")
    
    # Basic structure (this is simplified - actual format needs full parsing)
    # For now, we'll try to extract what we can
    timestamp_low = struct.unpack('<I', bytes_data[idx:idx+4])[0]
    idx += 4
    
    bfee_count = struct.unpack('<H', bytes_data[idx:idx+2])[0]
    idx += 2
    
    Nrx = bytes_data[idx] & 0x03
    Ntx = (bytes_data[idx] >> 2) & 0x03
    idx += 1
    
    rssi_a = struct.unpack('b', bytes_data[idx:idx+1])[0]
    idx += 1
    rssi_b = struct.unpack('b', bytes_data[idx:idx+1])[0]
    idx += 1
    rssi_c = struct.unpack('b', bytes_data[idx:idx+1])[0]
    idx += 1
    
    noise = struct.unpack('b', bytes_data[idx:idx+1])[0]
    idx += 1
    
    # Permutation
    perm = list(bytes_data[idx:idx+3])
    idx += 3
    
    # Rate
    rate = struct.unpack('<H', bytes_data[idx:idx+2])[0]
    idx += 2
    
    # CSI matrix (complex values)
    # This is simplified - actual parsing is more complex
    # For 30 subcarriers, 3 antennas: 30 * 3 * 2 (real/imag) = 180 values
    # Each is int16
    n_subcarriers = 30
    csi_size = n_subcarriers * Nrx * Ntx * 2  # real + imag
    
    if len(bytes_data) < idx + csi_size * 2:
        # Not enough data, return what we have
        return {
            "timestamp_low": timestamp_low,
            "bfee_count": bfee_count,
            "Nrx": Nrx,
            "Ntx": Ntx,
            "rssi_a": rssi_a,
            "rssi_b": rssi_b,
            "rssi_c": rssi_c,
            "noise": noise,
            "perm": perm,
            "rate": rate,
            "csi": None,
        }
    
    csi_raw = struct.unpack(f'<{csi_size}h', bytes_data[idx:idx+csi_size*2])
    idx += csi_size * 2
    
    # Reshape CSI: (Ntx, Nrx, subcarriers)
    csi_complex = np.array(csi_raw[:csi_size:2]) + 1j * np.array(csi_raw[1:csi_size:2])
    csi = csi_complex.reshape((Ntx, Nrx, n_subcarriers))
    
    return {
        "timestamp_low": timestamp_low,
        "bfee_count": bfee_count,
        "Nrx": Nrx,
        "Ntx": Ntx,
        "rssi_a": rssi_a,
        "rssi_b": rssi_b,
        "rssi_c": rssi_c,
        "noise": noise,
        "perm": perm,
        "rate": rate,
        "csi": csi,
    }



