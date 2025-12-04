import numpy as np
from pathlib import Path

def check_shape():
    files = sorted(list(Path("data/processed/windows").glob("window_*.npy")))
    if files:
        data = np.load(files[0])
        print(f"Window shape: {data.shape}")
    else:
        print("No window files found.")

if __name__ == "__main__":
    check_shape()
