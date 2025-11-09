# WiFi Data Collection Guide

This guide explains how to collect WiFi data for your spatial awareness project.

## Overview

Data collection is a crucial step for training your spatial awareness model. This guide covers:
- Basic data collection
- Location-based collection
- Multi-location collection
- Data validation
- Best practices

## Quick Start

### Basic Collection

```bash
# Collect data for 60 seconds (2 samples per second)
python scripts/collect_wifi_data.py --duration 60

# Collect with custom sampling rate
python scripts/collect_wifi_data.py --duration 120 --sampling_rate 5.0

# Collect with mock data (for testing)
python scripts/collect_wifi_data.py --duration 60 --use_mock
```

### Location-Based Collection

```bash
# Collect data at a specific location
python scripts/collect_with_location.py --location "Room 101" --duration 120

# Collect with scenario information
python scripts/collect_with_location.py \
    --location "Library" \
    --scenario "indoor_static" \
    --duration 60 \
    --notes "Testing WiFi signal in library"

# Collect with different scenarios
python scripts/collect_with_location.py --location "Lab" --scenario "experiment_1" --duration 180
```

### Multi-Location Collection

```bash
# Interactive collection (will prompt for locations)
python scripts/collect_multi_location.py --duration 60

# Collect from specific locations
python scripts/collect_multi_location.py \
    --locations "Room 101" "Room 102" "Lab" \
    --duration 120

# Collect with different scenarios per location
python scripts/collect_multi_location.py \
    --locations "Library" "Lab" \
    --scenarios "indoor_static" "outdoor_static" \
    --duration 60
```

## Collection Methods

### 1. Basic Collection (`collect_wifi_data.py`)

Simple data collection without location tracking.

**Use when:**
- Quick data collection
- Testing collection setup
- Single location collection

**Example:**
```bash
python scripts/collect_wifi_data.py --duration 60 --sampling_rate 2.0
```

### 2. Location-Based Collection (`collect_with_location.py`)

Collect data with location and metadata tracking.

**Use when:**
- Collecting data from specific locations
- Need to track location information
- Organizing data by location

**Features:**
- Location name tracking
- Scenario labeling
- Metadata storage
- Organized file structure

**Example:**
```bash
python scripts/collect_with_location.py \
    --location "Room 101" \
    --scenario "indoor_static" \
    --duration 120 \
    --notes "Testing WiFi signal variations"
```

### 3. Multi-Location Collection (`collect_multi_location.py`)

Collect data from multiple locations in a single session.

**Use when:**
- Collecting data from multiple locations
- Need consistent collection parameters
- Organizing data by session

**Features:**
- Multiple locations in one session
- Session metadata tracking
- Guided collection process
- Consistent parameters

**Example:**
```bash
python scripts/collect_multi_location.py \
    --locations "Room 101" "Room 102" "Lab" \
    --duration 60 \
    --sampling_rate 2.0
```

### 4. Large Dataset Collection (`collect_large_dataset.py`)

Collect large amounts of data for training.

**Use when:**
- Need large training dataset
- Long-term data collection
- Batch processing

**Features:**
- Automatic batch saving
- Progress tracking
- Resume capability
- Large dataset management

**Example:**
```bash
python scripts/collect_large_dataset.py \
    --total_samples 10000 \
    --sampling_rate 2.0 \
    --save_interval 1000
```

## Data Organization

### File Structure

```
data/
├── collections/                    # Location-based collections
│   ├── room_101/
│   │   ├── room_101_indoor_static_20251109_120000.json
│   │   └── room_101_indoor_static_20251109_120000_metadata.json
│   └── library/
│       ├── library_indoor_static_20251109_130000.json
│       └── library_indoor_static_20251109_130000_metadata.json
├── large_dataset/                  # Large dataset collections
│   ├── wifi_data_batch_0000_20251109_155803.json
│   └── combined_dataset.json
└── wifi_data_20251109_150557.json  # Basic collections
```

### Metadata

Each collection includes metadata:
- Location name
- Scenario
- Collection timestamp
- Sampling rate
- Number of samples
- Collection method
- Notes

## Data Validation

### Validate Collected Data

```bash
# Validate a single file
python scripts/validate_collected_data.py data/wifi_data.json

# Validate a directory
python scripts/validate_collected_data.py data/collections --recursive

# Detailed validation report
python scripts/validate_collected_data.py data/collections --detailed

# Save validation report
python scripts/validate_collected_data.py data/collections --output validation_report.json
```

### What Gets Validated

- Data format (JSON structure)
- Required fields (RSSI, signal_strength, timestamp)
- Data quality (RSSI range, signal strength range)
- Missing values
- Duplicate timestamps
- Collection method

## Best Practices

### 1. Collection Duration

- **Minimum**: 30 seconds (for basic testing)
- **Recommended**: 60-120 seconds (for training data)
- **Long-term**: 5-10 minutes (for comprehensive data)

### 2. Sampling Rate

- **Low (0.5-1 Hz)**: For static scenarios
- **Medium (2-5 Hz)**: For general collection (recommended)
- **High (10+ Hz)**: For dynamic scenarios (walking, movement)

### 3. Location Labeling

Use descriptive location names:
- ✅ Good: "Room_101", "Library_Floor2", "Lab_BuildingA"
- ❌ Bad: "loc1", "test", "data"

### 4. Scenario Labeling

Use consistent scenario labels:
- `indoor_static`: Indoor, stationary
- `indoor_walking`: Indoor, moving
- `outdoor_static`: Outdoor, stationary
- `outdoor_walking`: Outdoor, moving
- `experiment_1`, `experiment_2`: For specific experiments

### 5. Data Collection Workflow

1. **Plan**: Decide locations and scenarios
2. **Collect**: Use appropriate collection script
3. **Validate**: Check data quality
4. **Organize**: Organize by location/scenario
5. **Document**: Add notes and metadata

### 6. Collection Tips

- **Stay Connected**: Ensure WiFi connection is stable
- **Avoid Movement**: For static scenarios, stay in one place
- **Multiple Runs**: Collect multiple runs per location
- **Time Variations**: Collect at different times of day
- **Environment Notes**: Document environment conditions

## Collection Scenarios

### Scenario 1: Single Location Testing

```bash
# Quick test collection
python scripts/collect_wifi_data.py --duration 30 --use_mock
```

### Scenario 2: Location Mapping

```bash
# Collect from multiple locations
python scripts/collect_multi_location.py \
    --locations "Room 101" "Room 102" "Room 103" \
    --duration 60 \
    --scenarios "indoor_static"
```

### Scenario 3: Scenario Comparison

```bash
# Collect same location, different scenarios
python scripts/collect_with_location.py --location "Lab" --scenario "indoor_static" --duration 60
python scripts/collect_with_location.py --location "Lab" --scenario "indoor_walking" --duration 60
```

### Scenario 4: Large Dataset

```bash
# Collect large dataset for training
python scripts/collect_large_dataset.py \
    --total_samples 10000 \
    --sampling_rate 2.0 \
    --save_interval 1000 \
    --output_dir data/large_dataset
```

## Data Analysis

### Check Collected Data

```bash
# View data summary
python scripts/check_collected_data.py data/wifi_data.json

# Analyze multiple files
python scripts/check_collected_data.py data/collections --recursive
```

### Cluster Analysis

```bash
# Cluster collected data
python scripts/run_kmeans.py \
    --data_path data/collections \
    --n_clusters 5 \
    --feature_type signal \
    --visualize
```

## Troubleshooting

### Issue: No Real WiFi Data

**Solution**: Check WiFi connection and use `--use_mock` for testing
```bash
python scripts/collect_wifi_data.py --use_mock --duration 60
```

### Issue: Low Sample Rate

**Solution**: Reduce sampling rate or increase duration
```bash
python scripts/collect_wifi_data.py --duration 120 --sampling_rate 1.0
```

### Issue: Missing Data Fields

**Solution**: Validate data and check collection method
```bash
python scripts/validate_collected_data.py data/wifi_data.json --detailed
```

### Issue: File Organization

**Solution**: Use location-based collection
```bash
python scripts/collect_with_location.py --location "Room 101" --duration 60
```

## Next Steps

1. **Collect Data**: Start with basic collection
2. **Validate**: Check data quality
3. **Organize**: Organize by location/scenario
4. **Analyze**: Use clustering and analysis tools
5. **Train**: Use collected data for model training

## Examples

### Example 1: Basic Collection Session

```bash
# Collect 60 seconds of data
python scripts/collect_wifi_data.py --duration 60

# Validate collected data
python scripts/validate_collected_data.py data/wifi_data_*.json

# Check data summary
python scripts/check_collected_data.py data/wifi_data_*.json
```

### Example 2: Location-Based Collection

```bash
# Collect from Room 101
python scripts/collect_with_location.py \
    --location "Room 101" \
    --scenario "indoor_static" \
    --duration 120

# Collect from Library
python scripts/collect_with_location.py \
    --location "Library" \
    --scenario "indoor_static" \
    --duration 120

# Validate all collections
python scripts/validate_collected_data.py data/collections --recursive
```

### Example 3: Multi-Location Session

```bash
# Collect from multiple locations
python scripts/collect_multi_location.py \
    --locations "Room 101" "Room 102" "Lab" \
    --duration 60 \
    --sampling_rate 2.0

# Cluster collected data
python scripts/run_kmeans.py \
    --data_path data/collections \
    --n_clusters 3 \
    --feature_type signal \
    --visualize
```

## Summary

- Use `collect_wifi_data.py` for basic collection
- Use `collect_with_location.py` for location-based collection
- Use `collect_multi_location.py` for multi-location sessions
- Use `collect_large_dataset.py` for large datasets
- Always validate collected data
- Organize data by location and scenario
- Document collection with notes and metadata

