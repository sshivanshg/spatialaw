# Training Summary

## Training Configuration Used

```bash
python scripts/train_baseline.py \
  --num_mock_samples 200 \
  --batch_size 4 \
  --num_epochs 5 \
  --output_size 32 32
```

---

## Training Details

### Model Parameters (Neural Network Weights)
- **Total Parameters**: 3,423,459 (3.4 million)
- **Trainable Parameters**: 3,423,459 (all parameters are trainable)
- **Model Memory**: ~13.06 MB (in float32 format)
- **Model Type**: CNN Encoder-Decoder

### Training Data
- **Total Data Samples**: 200 mock CSI samples
- **Training Samples**: 160 (80% of total)
- **Validation Samples**: 40 (20% of total)
- **Data Type**: Mock CSI data (amplitude + phase)

### Training Configuration
- **Batch Size**: 4 samples per batch
- **Number of Epochs**: 5 epochs
- **Batches per Epoch**: 40 batches (160 samples ÷ 4 batch size)
- **Total Training Batches**: 200 batches (40 batches × 5 epochs)
- **Total Training Iterations**: 200 iterations

### Input/Output Specifications
- **Input Size**: (batch, 2, 3, 64)
  - 2 channels: Amplitude and Phase
  - 3 antennas
  - 64 subcarriers
- **Output Size**: (batch, 3, 32, 32)
  - 3 channels: RGB
  - 32×32 pixels: Spatial dimensions

---

## Training Statistics

### What Was Trained
- **Model Parameters**: 3,423,459 weights/biases
- **Training Samples**: 160 CSI samples
- **Training Iterations**: 200 batches
- **Epochs**: 5 complete passes through training data

### Training Process
- Each epoch: 40 batches × 4 samples = 160 samples
- Total training: 5 epochs × 160 samples = 800 sample presentations
- (Note: Same 160 samples seen 5 times, not 800 unique samples)

### Computational Requirements
- **Device**: CPU (Mac)
- **Training Time**: ~26 seconds for 2 epochs (estimated ~65 seconds for 5 epochs)
- **Memory Usage**: ~13 MB for model + training overhead
- **Storage**: ~41 MB per checkpoint

---

## Key Numbers Summary

| Metric | Value |
|--------|-------|
| **Model Parameters** | 3,423,459 |
| **Training Samples** | 160 |
| **Validation Samples** | 40 |
| **Batch Size** | 4 |
| **Epochs** | 5 |
| **Total Batches** | 200 |
| **Model Memory** | ~13 MB |
| **Checkpoint Size** | ~41 MB |

---

## What This Means

### Model Complexity
- **3.4 million parameters** = Model has 3.4 million learnable weights
- These parameters are updated during training
- Larger model = more capacity to learn complex patterns

### Training Data
- **160 training samples** = Model sees 160 different CSI samples
- **5 epochs** = Each sample is seen 5 times during training
- **Total presentations** = 160 samples × 5 epochs = 800 presentations

### Training Scale
- **Small dataset** (160 samples) = Good for testing/development
- **Large model** (3.4M params) = Can learn complex patterns
- **Ratio**: ~4,600 parameters per training sample (may be overfitting)

---

## Interpretation

### Model Size vs Data Size
- **Model**: 3,423,459 parameters (large model)
- **Training Data**: 160 samples (small dataset)
- **Ratio**: ~21,396 parameters per sample

**Implications:**
- Model has high capacity (can learn complex patterns)
- Small dataset may lead to overfitting
- More data would improve generalization
- Current setup is good for baseline/development

### Training Scale
- **Small-scale training** = Development/testing phase
- **Production training** would need:
  - More data (1000+ samples)
  - More epochs (50+)
  - Larger batch size (if GPU available)

---

## Comparison

### Current Training (Development)
- Data: 160 samples
- Epochs: 5
- Purpose: Test pipeline, verify code works

### Recommended Training (Production)
- Data: 1000+ samples
- Epochs: 50+
- Purpose: Train actual model
- Device: GPU (Colab) for speed

---

## Answer to Your Question

**"How many parameters did we train on?"**

**Answer:**
- **Model has**: 3,423,459 parameters (these are the weights being trained)
- **Trained on**: 160 training samples (data)
- **Training iterations**: 200 batches over 5 epochs

**In simple terms:**
- We have a model with **3.4 million parameters** (weights)
- We trained it on **160 CSI samples**
- The model saw each sample **5 times** (5 epochs)
- Total: **200 training batches**

---

## For Your Teacher

**You can say:**

"We trained a neural network model with **3.4 million parameters** on **160 training samples** for **5 epochs**. The model is a CNN encoder-decoder that learns to reconstruct spatial information from WiFi CSI data. We used a small dataset for development/testing, and the model successfully learned (loss decreased from 0.064 to 0.012). For production, we would train on a larger dataset (1000+ samples) for more epochs (50+)."

---

## Technical Details

### Model Architecture
- **Encoder**: 3 convolutional blocks → Latent representation
- **Decoder**: Transposed convolutions → Spatial reconstruction
- **Total Layers**: ~15-20 layers
- **Activation**: ReLU, Tanh (output)

### Training Process
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Combined (MSE + SSIM + Spatial Loss)
- **Learning Rate**: 0.001 (with decay scheduler)
- **Weight Decay**: 1e-5 (regularization)

### Results
- **Initial Loss**: ~0.064
- **Final Loss**: ~0.012
- **Improvement**: ~81% reduction in loss
- **Status**: Model learning successfully

---

This summary shows exactly what was trained and the scale of the training.

