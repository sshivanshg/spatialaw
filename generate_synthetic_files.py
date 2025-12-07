import numpy as np
import os

def generate_empty_room_data(duration_sec=60, fs=30, n_subcarriers=30, filename="synthetic_empty_room.npy"):
    """
    Generates a synthetic CSI file mimicking an empty room (low variance).
    """
    n_samples = duration_sec * fs
    
    # Base signal: Constant amplitude (e.g., 20) with random baseline per subcarrier
    # Shape: (n_samples, n_subcarriers)
    # Empty room = static reflections -> constant amplitude over time
    baseline = np.random.uniform(15, 25, size=(1, n_subcarriers))
    
    # 1. Very low white noise (thermal noise)
    white_noise_level = 0.05  # Was 0.5, reduced significantly
    white_noise = np.random.normal(0, white_noise_level, size=(n_samples, n_subcarriers))
    
    # 2. Random Walk (Drift) - mimics slow environmental changes/phase drift from hardware
    # Generate random steps
    steps = np.random.normal(0, 0.02, size=(n_samples, n_subcarriers))
    drift = np.cumsum(steps, axis=0)
    
    csi_data = baseline + white_noise + drift
    
    # Ensure all positive
    csi_data = np.abs(csi_data)
    
    # Save as .npy
    np.save(filename, csi_data)
    print(f"Generated {filename}: Shape {csi_data.shape}, Duration {duration_sec}s")

if __name__ == "__main__":
    generate_empty_room_data()
