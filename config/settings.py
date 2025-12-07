"""Configuration settings for spatialaw project."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
WINDOWS_DIR = PROCESSED_DATA_DIR / "windows"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"
BINARY_DIR = PROCESSED_DATA_DIR / "binary"
WINDOWS_BINARY_DIR = PROCESSED_DATA_DIR / "windows_binary"

# Model parameters
WINDOW_SIZE = 256
STRIDE = 64
SAMPLE_RATE = 100.0  # Hz
N_SUBCARRIERS = 30

# Feature extraction
N_FEATURES = 14

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
