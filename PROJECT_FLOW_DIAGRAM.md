# Project Flow Diagram

## Complete System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPATIAL AWARENESS SYSTEM                      │
│              Through Ambient Wireless Signals                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        1. DATA COLLECTION                        │
└─────────────────────────────────────────────────────────────────┘

    WiFi Network (Ruckus Access Points)
            │
            ▼
    ┌───────────────┐
    │ WiFi Collector │  ← Collects RSSI, Signal Strength, Channel
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │  CSI Processor │  ← Extracts Amplitude & Phase
    └───────────────┘
            │
            ▼
    Data Files (JSON/CSV)
    - RSSI values
    - Signal strength
    - Channel info
    - Timestamps


┌─────────────────────────────────────────────────────────────────┐
│                       2. DATA PREPROCESSING                      │
└─────────────────────────────────────────────────────────────────┘

    Data Files
            │
            ▼
    ┌───────────────┐
    │  Data Loader   │  ← Loads and batches data
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │   Transforms   │  ← Normalizes, augments data
    └───────────────┘
            │
            ▼
    Preprocessed Data
    - Normalized CSI
    - Batched for training
    - Train/Val split


┌─────────────────────────────────────────────────────────────────┐
│                         3. MODEL TRAINING                        │
└─────────────────────────────────────────────────────────────────┘

    Preprocessed Data
            │
            ▼
    ┌─────────────────────────────────────┐
    │      BASELINE MODEL (CNN)            │
    │                                      │
    │  ┌──────────────┐                   │
    │  │   ENCODER     │  ← CSI → Features │
    │  │  (3 Conv      │                   │
    │  │   Blocks)     │                   │
    │  └──────────────┘                   │
    │         │                           │
    │         ▼                           │
    │  ┌──────────────┐                   │
    │  │   LATENT      │  ← Compressed     │
    │  │  REPRESENTATION│   representation │
    │  └──────────────┘                   │
    │         │                           │
    │         ▼                           │
    │  ┌──────────────┐                   │
    │  │   DECODER     │  ← Features →     │
    │  │  (Transposed  │     Spatial Info  │
    │  │   Convolutions)│                  │
    │  └──────────────┘                   │
    └─────────────────────────────────────┘
            │
            ▼
    Spatial Reconstruction
    - RGB images
    - Spatial maps
    - Feature maps


┌─────────────────────────────────────────────────────────────────┐
│                         4. TRAINING LOOP                         │
└─────────────────────────────────────────────────────────────────┘

    Model + Data
            │
            ▼
    ┌───────────────┐
    │  Forward Pass  │  ← CSI → Prediction
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │  Loss Function │  ← MSE + SSIM + Spatial
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ Backpropagation│  ← Update weights
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │   Optimizer    │  ← Adam optimizer
    └───────────────┘
            │
            ▼
    Updated Model
    (Repeat for N epochs)


┌─────────────────────────────────────────────────────────────────┐
│                         5. EVALUATION                            │
└─────────────────────────────────────────────────────────────────┘

    Trained Model
            │
            ▼
    ┌───────────────┐
    │   Evaluator    │  ← Test on validation set
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │    Metrics     │  ← MSE, PSNR, SSIM
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │  Visualizer    │  ← Generate plots, images
    └───────────────┘
            │
            ▼
    Results & Visualizations


┌─────────────────────────────────────────────────────────────────┐
│                      DATA FLOW SUMMARY                           │
└─────────────────────────────────────────────────────────────────┘

WiFi Signals
    │
    ▼
CSI Data (Amplitude + Phase)
    │
    ▼
Preprocessed Data
    │
    ▼
Model (Encoder-Decoder)
    │
    ▼
Spatial Reconstruction
    │
    ▼
Evaluation & Visualization


┌─────────────────────────────────────────────────────────────────┐
│                    COMPONENT INTERACTIONS                        │
└─────────────────────────────────────────────────────────────────┘

WiFi Collector ──→ CSI Processor ──→ Data Loader
                                            │
                                            ▼
                                    Preprocessing
                                            │
                                            ▼
                                    Model Training
                                            │
                                            ▼
                                    Evaluation
                                            │
                                            ▼
                                    Visualization


┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING WORKFLOW                           │
└─────────────────────────────────────────────────────────────────┘

1. Initialize Model
        │
        ▼
2. Load Data
        │
        ▼
3. For each epoch:
        │
        ├─→ Forward pass
        ├─→ Calculate loss
        ├─→ Backward pass
        ├─→ Update weights
        └─→ Validate
        │
        ▼
4. Save checkpoint
        │
        ▼
5. Evaluate
        │
        ▼
6. Visualize results


┌─────────────────────────────────────────────────────────────────┐
│                    INPUT/OUTPUT SPECIFICATIONS                   │
└─────────────────────────────────────────────────────────────────┘

INPUT:
- Format: (batch, 2, 3, 64)
  - 2 channels: Amplitude, Phase
  - 3 antennas
  - 64 subcarriers
- Type: Float32 tensor
- Range: Normalized [0, 1]

OUTPUT:
- Format: (batch, 3, 32, 32) or (batch, 3, 64, 64)
  - 3 channels: RGB
  - 32×32 or 64×64: Spatial dimensions
- Type: Float32 tensor
- Range: [-1, 1] (tanh output)


┌─────────────────────────────────────────────────────────────────┐
│                      SYSTEM ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │  WiFi Signals │
                    └──────┬───────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
    ┌───────▼──────┐            ┌────────▼───────┐
    │ Data Collector│            │  CSI Processor │
    └───────┬──────┘            └────────┬───────┘
            │                            │
            └────────────┬───────────────┘
                         │
                 ┌───────▼───────┐
                 │  Data Loader   │
                 └───────┬───────┘
                         │
                 ┌───────▼───────┐
                 │ Baseline Model │
                 └───────┬───────┘
                         │
            ┌────────────┴────────────┐
            │                         │
    ┌───────▼──────┐        ┌────────▼───────┐
    │   Trainer     │        │   Evaluator    │
    └───────┬──────┘        └────────┬───────┘
            │                        │
            └────────────┬───────────┘
                         │
                 ┌───────▼───────┐
                 │  Visualizer    │
                 └────────────────┘


This diagram shows the complete flow of data through the system,
from WiFi signal collection to spatial reconstruction and evaluation.

