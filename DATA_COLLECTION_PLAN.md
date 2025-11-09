# Data Collection Plan for 50M Parameter Model

## Model Requirements

### Model Specifications
- **Target Parameters**: ~50 million
- **Model Memory**: ~200 MB (for model weights)
- **Training Memory**: ~800 MB - 1 GB (with gradients, optimizer states, activations)

### Data Requirements

**Rule of Thumb**: For good generalization, need 10-100 samples per parameter
- **Ideal**: 500M - 5B samples (unrealistic)
- **Practical**: 10K - 100K samples (reasonable)
- **Minimum**: 5K - 10K samples (for baseline)

**Our Target**: **10,000 - 50,000 samples**

## Collection Strategy

### Option 1: Real WiFi Data (Recommended)
```bash
# Collect 10,000 samples (real WiFi data)
python scripts/collect_large_dataset.py \
  --total_samples 10000 \
  --sampling_rate 2.0 \
  --output_dir data/large_dataset \
  --save_interval 1000

# Time: ~83 minutes (1.4 hours) at 2 samples/second
```

### Option 2: Mock Data (Faster, for Testing)
```bash
# Collect 10,000 samples (mock data - instant)
python scripts/collect_large_dataset.py \
  --total_samples 10000 \
  --sampling_rate 10.0 \
  --output_dir data/large_dataset \
  --use_mock \
  --save_interval 1000

# Time: ~17 minutes at 10 samples/second
```

### Option 3: Multiple Collection Sessions
```bash
# Collect in multiple sessions
# Session 1: 5,000 samples
python scripts/collect_large_dataset.py --total_samples 5000 --output_dir data/large_dataset

# Session 2: 5,000 more samples
python scripts/collect_large_dataset.py --total_samples 5000 --output_dir data/large_dataset

# Then combine all datasets
python scripts/collect_large_dataset.py --combine_only data/large_dataset
```

## Collection Timeline

### Minimum Dataset (5,000 samples)
- **Real WiFi**: ~42 minutes
- **Mock Data**: ~8 minutes
- **Purpose**: Quick testing

### Recommended Dataset (10,000 samples)
- **Real WiFi**: ~83 minutes (1.4 hours)
- **Mock Data**: ~17 minutes
- **Purpose**: Good baseline training

### Large Dataset (50,000 samples)
- **Real WiFi**: ~417 minutes (7 hours)
- **Mock Data**: ~83 minutes (1.4 hours)
- **Purpose**: Production training

## Storage Requirements

### Data Size Estimates
- **Per sample**: ~1-2 KB (JSON format)
- **10,000 samples**: ~10-20 MB
- **50,000 samples**: ~50-100 MB
- **Storage**: Minimal (easily manageable)

## Collection Commands

### Quick Start (10K samples, Mock Data)
```bash
source venv/bin/activate
python scripts/collect_large_dataset.py \
  --total_samples 10000 \
  --use_mock \
  --output_dir data/large_dataset
```

### Real WiFi Data (10K samples)
```bash
source venv/bin/activate
python scripts/collect_large_dataset.py \
  --total_samples 10000 \
  --sampling_rate 2.0 \
  --output_dir data/large_dataset
```

### Large Dataset (50K samples, Real WiFi)
```bash
source venv/bin/activate
# Run overnight or in background
nohup python scripts/collect_large_dataset.py \
  --total_samples 50000 \
  --sampling_rate 2.0 \
  --output_dir data/large_dataset \
  --save_interval 2000 > collection.log 2>&1 &
```

## Training with Collected Data

### After Collection
```bash
# Train 50M parameter model
python scripts/train_50m_model.py \
  --data_path data/large_dataset/combined_dataset.json \
  --batch_size 4 \
  --num_epochs 50 \
  --lr 0.0001
```

## Recommendations

### For Development/Testing
- **Use mock data**: 10K samples, ~17 minutes
- **Fast iteration**: Quick to collect and test

### For Real Training
- **Use real WiFi data**: 10K-50K samples
- **Collect over time**: Multiple sessions
- **Combine datasets**: Merge all collections

### For Production
- **Large dataset**: 50K+ samples
- **Real data**: Actual WiFi signals
- **Diverse conditions**: Different times, locations

## Next Steps

1. **Start collection**: Choose real WiFi or mock data
2. **Monitor progress**: Check collection logs
3. **Combine datasets**: Merge all collections
4. **Train model**: Use collected data for training
5. **Evaluate**: Check model performance

## Notes

- Mock data is fine for development
- Real WiFi data needed for production
- Can combine multiple collection sessions
- Data collection can run in background
- Check disk space before large collections

