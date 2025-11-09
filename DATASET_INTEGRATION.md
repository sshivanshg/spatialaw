# ESP32 WiFi CSI Dataset Integration Guide

## Dataset Overview

**WiFi CSI-Based Long-Range Through-Wall Human Activity Recognition with the ESP32**

This dataset is **highly relevant** for your project! Here's why:

### What It Provides

1. **Presence Detection** (DP_LOS, DP_NLOS)
   - 6 classes: no presence, presence in room 1-5
   - 392-384 CSI amplitude spectrograms
   - Both LOS and NLOS (through-wall) scenarios

2. **Activity Recognition** (DA_LOS, DA_NLOS)
   - 3 classes: no activity, walking, walking + arm-waving
   - 392-384 CSI amplitude spectrograms
   - Both LOS and NLOS scenarios

### Why It's Useful for Your Project

✅ **Extends Your Motion Detection**
- You currently have: movement vs no movement
- This adds: specific activities (walking, arm-waving)
- Could enhance your motion detector

✅ **Adds Presence Detection**
- New capability: detect if someone is in a room
- Works through walls (NLOS) - very practical!

✅ **Real-World Scenarios**
- Through-wall detection (NLOS) is highly relevant
- Multiple rooms (5 rooms in NLOS dataset)
- Real WiFi CSI data from ESP32

✅ **Data Format Compatibility**
- CSI amplitude spectrograms (you work with CSI)
- Similar to your localization data format
- Can be processed and integrated

✅ **Benchmark Dataset**
- Published research dataset
- Can compare your models against published results
- Validates your approach

## Integration Plan

### Option 1: Extend Motion Detection

**Current**: Binary classification (movement / no movement)
**Extended**: Multi-class activity recognition

```python
# Current classes
classes = ['no_movement', 'movement']

# Extended classes (from DA_LOS/DA_NLOS)
classes = ['no_activity', 'walking', 'walking_arm_waving']
```

### Option 2: Add Presence Detection

**New capability**: Detect presence in different rooms

```python
# Presence detection (from DP_LOS/DP_NLOS)
classes = [
    'no_presence',
    'presence_room_1',
    'presence_room_2',
    'presence_room_3',
    'presence_room_4',
    'presence_room_5'
]
```

### Option 3: Use as Benchmark

- Compare your models against published results
- Validate your preprocessing pipeline
- Test generalization across datasets

## Data Processing

### Dataset Structure

```
DA_LOS/
├── trainLabels.csv      # Training labels [index, class]
├── validationLabels.csv  # Validation labels
├── testLabels.csv       # Test labels
├── meanStd.csv          # Mean and std for normalization
└── *.png                # CSI amplitude spectrograms
```

### Processing Steps

1. **Load Spectrograms**
   - Read PNG images (CSI amplitude spectrograms)
   - Convert to numpy arrays
   - Normalize using meanStd.csv

2. **Extract Features**
   - Use existing time-series feature extraction
   - Or use spectrogram directly as input
   - Extract statistical features

3. **Train Models**
   - Use existing motion detector architecture
   - Extend to multi-class classification
   - Train on presence/activity labels

## Implementation

### Step 1: Download Dataset

```bash
# Download datasets (from the repository)
# Place in data/esp32_activity/ directory
data/
└── esp32_activity/
    ├── DA_LOS/
    ├── DA_NLOS/
    ├── DP_LOS/
    └── DP_NLOS/
```

### Step 2: Create Processing Script

```python
# scripts/process_esp32_dataset.py
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def load_spectrogram(image_path):
    """Load CSI amplitude spectrogram."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32)

def process_esp32_dataset(dataset_path, labels_path):
    """Process ESP32 dataset to your format."""
    # Load labels
    labels_df = pd.read_csv(labels_path, header=None, names=['index', 'class'])
    
    # Load spectrograms
    data = []
    for idx, row in labels_df.iterrows():
        img_path = dataset_path / f"{row['index']}.png"
        spectrogram = load_spectrogram(img_path)
        
        # Extract features (or use spectrogram directly)
        features = extract_features(spectrogram)
        
        data.append({
            'features': features,
            'label': row['class'],
            'spectrogram': spectrogram
        })
    
    return data
```

### Step 3: Extend Motion Detector

```python
# src/models/motion_detector.py
class ActivityDetector(nn.Module):
    """Extended motion detector for activity recognition."""
    
    def __init__(self, num_classes=3):  # no_activity, walking, walking_arm_waving
        super().__init__()
        # Extend existing motion detector
        # Add multi-class classification head
```

### Step 4: Train on ESP32 Data

```python
# scripts/train_activity_detector.py
# Train on DA_LOS/DA_NLOS for activity recognition
# Train on DP_LOS/DP_NLOS for presence detection
```

## Use Cases

### 1. Enhanced Motion Detection
- Current: Binary (movement/no movement)
- Enhanced: Activity recognition (walking, arm-waving)

### 2. Presence Detection
- Detect if someone is in a room
- Works through walls (NLOS)
- Multi-room detection

### 3. Benchmarking
- Compare your models against published results
- Validate your approach
- Test generalization

## Advantages

✅ **Real-World Data**: Collected with ESP32 (similar to your setup)
✅ **Through-Wall**: NLOS scenarios are very practical
✅ **Multiple Activities**: More than just movement detection
✅ **Published Results**: Can compare your performance
✅ **Small Dataset**: Easy to process and experiment with

## Recommendations

1. **Start with Activity Recognition** (DA_LOS/DA_NLOS)
   - Extends your current motion detection
   - 3 classes (easier than 6)
   - More relevant for your use case

2. **Test on NLOS Data**
   - Through-wall detection is very practical
   - Tests model robustness
   - Real-world scenario

3. **Combine with Your Data**
   - Train on ESP32 data
   - Fine-tune on your collected data
   - Best of both worlds

## Next Steps

1. Download the dataset
2. Create processing script (`scripts/process_esp32_dataset.py`)
3. Extend motion detector for multi-class
4. Train and evaluate
5. Compare with published results

## References

- Paper: "WiFi CSI-Based Long-Range Through-Wall Human Activity Recognition with the ESP32"
- Dataset: Available on the repository
- Citation: See dataset page for BibTeX

