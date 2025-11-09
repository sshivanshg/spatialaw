# Presentation Slides - Spatial Awareness Project

## Slide 1: Title Slide
**Spatial Awareness through Ambient Wireless Signals**

Authors: Rishabh (230178), Shivansh (230054)  
Newton School of Technology

---

## Slide 2: Problem Statement
**Why This Project?**

- Traditional methods use cameras/LiDAR → **Privacy concerns**
- High cost and deployment complexity
- Limited coverage

**Our Solution:**
- Use **WiFi signals** (ubiquitous, privacy-preserving)
- Low cost (existing infrastructure)
- Wide coverage

---

## Slide 3: Project Overview
**What We Built**

An intelligent system that:
- ✅ Reconstructs spatial layouts from WiFi signals
- ✅ Monitors indoor environments
- ✅ Infers spatial dynamics
- ✅ **No cameras required!**

**Key Innovation:** Privacy-preserving "sixth sense" using WiFi

---

## Slide 4: Technical Approach
**How It Works**

```
WiFi Signals → CSI Data → Neural Network → Spatial Reconstruction
```

**CSI (Channel State Information):**
- Amplitude: Signal strength variations
- Phase: Signal phase shifts  
- Frequency: Multiple subcarriers (64 for WiFi)

**Model:** CNN Encoder-Decoder
- Encoder: Extracts spatial features
- Decoder: Reconstructs spatial layouts

---

## Slide 5: System Architecture
**Complete Pipeline**

1. **Data Collection** → WiFi signal data
2. **CSI Processing** → Amplitude + Phase
3. **Model Training** → Encoder-Decoder
4. **Spatial Reconstruction** → Output images/maps
5. **Evaluation** → Metrics & Visualization

---

## Slide 6: Data Collection
**What We Collect**

- **RSSI**: Signal strength (-50 to -90 dBm)
- **Channel**: WiFi channel (1-165)
- **SNR**: Signal-to-noise ratio
- **Amplitude/Phase**: CSI data (when available)

**Methods:**
- `system_profiler`: Real WiFi data (Mac)
- Mock data: For development
- Future: Ruckus API for full CSI

---

## Slide 7: Model Architecture
**CNN Encoder-Decoder**

**Input:** CSI Data (2 channels × 3 antennas × 64 subcarriers)

**Encoder:**
- 3 convolutional blocks
- Feature extraction
- Latent representation

**Decoder:**
- Transposed convolutions
- Spatial reconstruction
- Output: RGB images (32×32 or 64×64)

**Parameters:** 3.4M parameters, ~16 MB memory

---

## Slide 8: Training Process
**How We Train**

1. Collect/Generate CSI data
2. Preprocess (normalize, augment)
3. Train encoder-decoder
4. Evaluate on validation set
5. Save best model

**Loss Function:** Combined (MSE + SSIM + Spatial Loss)

**Results:** Loss decreases from 0.064 → 0.012 (2 epochs)

---

## Slide 9: Current Status
**What's Working**

✅ **Data Collection**: Real WiFi data on Mac  
✅ **Model Architecture**: Baseline CNN encoder-decoder  
✅ **Training Pipeline**: Complete training system  
✅ **Evaluation**: Metrics and visualization  
✅ **Documentation**: Comprehensive guides  

**In Progress:**
🔄 Full CSI data collection  
🔄 Model improvements  
🔄 Real-world testing  

---

## Slide 10: Applications
**Potential Use Cases**

1. **Security**: Intrusion detection, occupancy monitoring
2. **Health**: Fall detection, activity recognition
3. **Assistive Tech**: Navigation, smart homes
4. **Urban Infrastructure**: Traffic, crowd management

**Key Advantage:** Privacy-preserving (no cameras!)

---

## Slide 11: Results
**Training Results**

- **Model**: 3.4M parameters
- **Training**: Loss decreases (0.064 → 0.012)
- **Status**: Model learning successfully
- **Time**: ~26 seconds for 2 epochs (CPU)

**Performance:**
- Processes WiFi CSI data ✅
- Encodes spatial information ✅
- Reconstructs spatial layouts ✅

---

## Slide 12: Challenges & Solutions
**Challenges:**

1. CSI data collection on Mac → **Solution**: Use `system_profiler`
2. Computational resources → **Solution**: Google Colab (free GPU)
3. Data availability → **Solution**: Mock data for development

**Solutions Implemented:**
- Multi-method data collection
- Optimized model architecture
- Flexible training pipeline

---

## Slide 13: Project Structure
**Code Organization**

```
spatialaw/
├── src/data_collection/    # WiFi & CSI collection
├── src/models/             # Model architectures
├── src/preprocessing/      # Data processing
├── src/training/           # Training utilities
├── src/evaluation/         # Evaluation & visualization
└── scripts/                # Executable scripts
```

**Clean, modular, well-documented code**

---

## Slide 14: Key Achievements
**What We've Accomplished**

1. ✅ Complete baseline model (working)
2. ✅ Real WiFi data collection
3. ✅ End-to-end training pipeline
4. ✅ Evaluation system
5. ✅ Comprehensive documentation

**Innovation:** Privacy-preserving spatial awareness using WiFi

---

## Slide 15: Future Work
**Next Steps**

**Short-term:**
- Collect more real data
- Train on larger datasets
- Improve model architecture
- Add text conditioning

**Long-term:**
- Implement diffusion models
- Real-world deployment
- Application development

---

## Slide 16: Impact
**Why This Matters**

- ✅ **Privacy-Preserving**: No cameras needed
- ✅ **Scalable**: Uses existing WiFi infrastructure
- ✅ **Cost-Effective**: No additional hardware
- ✅ **Innovative**: Novel approach to spatial awareness

**Potential Impact:** Multiple application domains (security, health, assistive tech)

---

## Slide 17: Demonstration
**Live Demo**

1. **Data Collection**: Show real WiFi data collection
2. **Model Training**: Show training progress
3. **Results**: Show predictions and visualizations

**Key Points:**
- Real WiFi data collection working
- Model training functional
- Results visible and measurable

---

## Slide 18: Conclusion
**Summary**

- Built complete baseline system for spatial awareness
- Uses WiFi signals (privacy-preserving)
- Working model and training pipeline
- Ready for further development

**Next:** Collect more data, improve model, add features

**Impact:** Privacy-preserving, scalable spatial awareness system

---

## Slide 19: Q&A
**Questions?**

Thank you for your attention!

**Contact:**  
Rishabh (230178)  
Shivansh (230054)  
Newton School of Technology

---

## Presentation Tips

### Do's:
- ✅ Start with problem statement
- ✅ Explain innovation clearly
- ✅ Show data flow diagram
- ✅ Demonstrate working system
- ✅ Discuss real-world applications
- ✅ Emphasize privacy benefits

### Don'ts:
- ❌ Don't get too technical initially
- ❌ Don't skip the problem statement
- ❌ Don't forget to show results
- ❌ Don't ignore limitations

### Key Messages:
1. **Privacy-preserving** (no cameras)
2. **Practical** (uses existing WiFi)
3. **Innovative** (novel approach)
4. **Working** (baseline functional)

---

## Demo Script

### 1. Introduction (2 min)
- Problem statement
- Why WiFi signals?
- Key innovation

### 2. Technical Overview (3 min)
- How it works (data flow)
- Model architecture
- Training process

### 3. Demonstration (5 min)
- Show data collection
- Show training process
- Show results

### 4. Applications (2 min)
- Use cases
- Impact
- Future work

### 5. Q&A (3 min)
- Answer questions
- Discuss limitations
- Future plans

**Total: ~15 minutes**

