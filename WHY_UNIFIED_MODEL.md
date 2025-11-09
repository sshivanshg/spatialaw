# Why Use a Unified Model Instead of Separate Files?

## The Problem with Separate Files

### Current Situation
- `baseline_model.py` - Small model (~3.4M parameters)
- `medium_model.py` - Medium model (~50M parameters)  
- `large_model.py` - Large model (~100M+ parameters)

### Issues with This Approach

1. **Code Duplication**
   - Same architecture, just different sizes
   - Changes need to be made in multiple places
   - Bug fixes need to be applied to all files

2. **Maintenance Burden**
   - Three files to maintain
   - Inconsistent implementations
   - Hard to keep in sync

3. **Confusing for Users**
   - Which file to use?
   - What's the difference?
   - How to switch between sizes?

4. **Not Scalable**
   - Need new file for every size?
   - Can't easily experiment with sizes
   - Hard to fine-tune parameters

## The Solution: Unified Model

### Single File with Configuration

**One file** (`spatial_model.py`) that can create models of any size:

```python
# Small model (~3.4M params)
model = SpatialModel(model_size=ModelSize.SMALL)

# Medium model (~50M params)
model = SpatialModel(model_size=ModelSize.MEDIUM)

# Large model (~100M+ params)
model = SpatialModel(model_size=ModelSize.LARGE)

# Custom model
model = SpatialModel(
    model_size=ModelSize.CUSTOM,
    base_channels=200,
    latent_dim=600,
    num_encoder_blocks=4,
    num_decoder_blocks=5
)
```

### Benefits

1. **Single Source of Truth**
   - One file to maintain
   - One implementation
   - Consistent behavior

2. **Easy Configuration**
   - Change size with one parameter
   - Predefined configurations
   - Custom configurations supported

3. **Easy Experimentation**
   - Try different sizes easily
   - Fine-tune parameters
   - Compare models

4. **Backward Compatible**
   - Can still use old function names
   - Gradual migration
   - No breaking changes

## Migration Plan

### Step 1: Create Unified Model ✅
- Created `spatial_model.py` with configurable architecture
- Supports SMALL, MEDIUM, LARGE, and CUSTOM sizes
- Backward compatible functions

### Step 2: Update Imports (Optional)
```python
# Old way
from src.models.baseline_model import BaselineSpatialModel
from src.models.medium_model import MediumSpatialModel

# New way (recommended)
from src.models.spatial_model import SpatialModel, ModelSize

# Or use backward compatible functions
from src.models.spatial_model import BaselineSpatialModel, MediumSpatialModel
```

### Step 3: Update Training Scripts
```python
# Old way
model = BaselineSpatialModel(...)

# New way
model = SpatialModel(..., model_size=ModelSize.SMALL)

# Or keep using old way (still works)
model = BaselineSpatialModel(...)  # Still works!
```

## Recommendation

### Use Unified Model Going Forward

**Benefits:**
- ✅ Single file to maintain
- ✅ Easy to configure
- ✅ Easy to experiment
- ✅ Consistent implementation
- ✅ Backward compatible

**Migration:**
- Keep old files for now (backward compatibility)
- Use unified model for new code
- Gradually migrate old code
- Remove old files later

## Example Usage

### Create Different Sized Models

```python
from src.models.spatial_model import SpatialModel, ModelSize

# Small model (baseline)
small_model = SpatialModel(
    input_channels=2,
    num_subcarriers=64,
    num_antennas=3,
    output_channels=3,
    output_size=(64, 64),
    model_size=ModelSize.SMALL
)

# Medium model (50M params)
medium_model = SpatialModel(
    input_channels=2,
    num_subcarriers=64,
    num_antennas=3,
    output_channels=3,
    output_size=(64, 64),
    model_size=ModelSize.MEDIUM
)

# Large model (100M+ params)
large_model = SpatialModel(
    input_channels=2,
    num_subcarriers=64,
    num_antennas=3,
    output_channels=3,
    output_size=(64, 64),
    model_size=ModelSize.LARGE
)

# Custom model
custom_model = SpatialModel(
    input_channels=2,
    num_subcarriers=64,
    num_antennas=3,
    output_channels=3,
    output_size=(64, 64),
    model_size=ModelSize.CUSTOM,
    base_channels=150,
    latent_dim=500,
    num_encoder_blocks=4,
    num_decoder_blocks=5
)
```

### Get Model Information

```python
info = model.get_model_info()
print(f"Parameters: {info['total_parameters']:,}")
print(f"Memory: {info['model_memory_mb']:.2f} MB")
print(f"Config: {info}")
```

## Summary

### Why Separate Files Were Created
- Quick prototyping
- Different requirements
- Testing different architectures

### Why Unified Model is Better
- ✅ Single source of truth
- ✅ Easy configuration
- ✅ Less code duplication
- ✅ Easier maintenance
- ✅ Better for experimentation

### Recommendation
- **Use unified model** for new code
- **Keep old files** for backward compatibility
- **Gradually migrate** to unified model
- **Remove old files** once migration complete

This is a common pattern in software development: start with separate implementations for quick prototyping, then consolidate into a unified, configurable solution.

