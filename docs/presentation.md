# SpatialAw: WiFi-Based Presence Detection — Presentation (12 Slides)

> Audience: Technical + product stakeholders | Duration: 12–15 minutes
> Project: SpatialAw — Device-free human presence detection using WiFi CSI

---

## 1. Title & Motivation
- **Project**: SpatialAw — Device-Free Human Presence Detection via Ambient WiFi Signals
- **Team**: Rishabh (230158), Shivansh (230054), Newton School of Technology
- **Why it matters**:
  - Privacy-preserving alternative to cameras
  - Ubiquitous WiFi hardware, no wearables required
  - Smart buildings: HVAC, lighting, occupancy analytics
- **Outcome**: Binary presence detector achieving **99.40% accuracy** on WiAR benchmark
  - **100% precision** (zero false alarms)
  - **98.05% recall** (catches almost all activity)
  - **<10ms inference** (real-time capable)
  - ⚠️ *With important caveats about overfitting (discussed in slide 7b)*

---

## 2. Problem & Requirements
- **Goal**: Detect human presence from WiFi Channel State Information (CSI)
- **Constraints**:
  - Commodity IEEE 802.11n Intel 5300 NIC (30 subcarriers)
  - Robust to low-motion scenarios and static false alarms
  - Reproducible training pipeline, interpretable features, and live demo
- **Inputs**: `.dat` CSI recordings or preprocessed `.npy` windows
- **Outputs**: Presence probability per window; dashboard visualization and metrics

---

## 3. System Overview
- **Pipeline**:
  1. Data ingestion (`csiread`) → CSI amplitude
  2. Windowing (`T=256`, `stride=64`)
  3. Preprocessing: moving-average denoising + per-subcarrier z-score normalization
  4. Feature extraction: 14 physics-informed features
  5. Classification: Random Forest (baseline) or compact 1D-CNN
- **Artifacts**: `models/` (RF + scaler + metrics), plots, dashboard `app.py`
- **Config**: Centralized in `config/settings.py`

---

## 4. Dataset & Labeling
- **Benchmark**: WiAR (16 activities; ~2,000 windows after preprocessing)
- **Binary labels**: Presence vs No activity
  - Derived via motion-score thresholding to avoid domain leakage
- **Splitting**: Group-aware (by recording) to prevent temporal leakage
- **Balance**: Class-weighted RF; SMOTE considered during design

---

## 5. Preprocessing & Feature Engineering
- **Denoising**: Causal MA filter (K=5) preserves motion (<5 Hz)
- **Normalization**: Per-window z-score per subcarrier
- **Features (14)**:
  - Variance (mean/std/max), Envelope (mean/std), Entropy
  - Velocity (mean/max), MAD (mean/std), Motion Period (mean/std)
  - Normalized Std (mean/std)
- **Rationale**: Grounded in Fresnel diffraction and Doppler physics

---

## 6. Models & Training
- **Random Forest**:
  - `n_estimators=150`, class_weight=`balanced`, bootstrap ensemble
  - Fast training and inference; interpretable feature importance
- **1D-CNN**:
  - 2 conv blocks + pooling + FC (≈50K params)
  - Operates on normalized windows; stronger with larger datasets
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC; McNemar’s test

---

## 7. Results (Quantitative)
- **Random Forest** (WiAR test set):
  - **Accuracy: 99.40%** | **Precision: 100%** | **Recall: 98.05%** | **F1: 0.99**
  - Test set: 638 windows (281 No Activity, 357 Activity)
  - **Zero false positives** (no false alarms!)
  - 7 false negatives (missed 7 low-motion activities)
  - **Inference: <10ms per window** (real-time capable)
- **1D-CNN**: Accuracy ≈ 89%, ROC-AUC ≈ 0.93
- **⚠️ Critical Observation: Likely Overfitting**
  - Near-perfect accuracy is a red flag on single-environment data
  - Model likely memorized dataset-specific patterns
  - See next slide for detailed analysis
- **Feature importance**: variance, entropy, velocity dominate
- **Artifacts**: `models/presence_detector_classification_report.txt`, training plots

---

## 7b. Why 99.40% Accuracy? (Overfitting Analysis)
- **The Result**:
  - 99.40% accuracy, **100% precision** (0 false positives), 98.05% recall
  - Only 7 missed detections out of 638 test windows
- **Root Causes**:
  1. **Single environment**: WiAR collected in one lab room (5m × 6m)
  2. **Homogeneous hardware**: Same Intel 5300 NIC, same antenna positions
  3. **Label derivation circularity**: Labels from motion-score ← same features used for prediction
  4. **Limited diversity**: Same room geometry, furniture, participants
- **Evidence of Overfitting**:
  - RF (99.4%) vs CNN (89%) gap → RF overfits to hand-crafted features
  - 100% precision = model learned WiAR's specific "No Activity" signature
  - CNN learns from raw data, can't memorize as easily
- **Expected Real-World Performance**:
  - Same room: 90-95% (temporal drift)
  - Different room, same building: 70-85%
  - Different building: 50-70% (needs transfer learning)
- **Takeaway**: Near-perfect accuracy = warning sign, not achievement

---

## 8. Dashboard Demo (Streamlit)
- **Features**:
  - Upload `.dat` or `.npy`; auto windowing + preprocessing
  - Probability timeline, smoothing, live simulation
  - CSI heatmap visualization, threshold guides
- **Run**:
  - `streamlit run app.py`
  - Requires `models/presence_detector_rf.joblib` and `presence_detector_scaler.joblib`

---

## 9. Architecture & Codebase (Post-Restructure)
- **Structure**:
  - `src/spatialaw/`: `preprocessing/`, `models/`, `data/`, `utils/`
  - `scripts/`: `data_preparation/`, `visualization/`; `run_pipeline.py`
  - `training/`: RF + CNN training + tuning + prediction
  - `config/`: `settings.py`
  - `docs/`: Restructuring summary, recommendations, cleanup guide
- **Benefits**: Maintainable, importable package; reproducible scripts; centralized config

---

## 10. Limitations & Risks
- **🚨 Overfitting**: 99.40% accuracy (100% precision) indicates model learned dataset-specific patterns, not generalizable features
- Single environment (WiAR lab) → domain generalization untested and expected to degrade significantly
- Derived binary labels vs explicit occupancy ground truth (circularity concern)
- No multi-person cases; interactions and counting out of scope
- Potential drift over long durations; needs online adaptation
- Hardware-specific (Intel 5300); other NICs untested
- **Mitigation needed**: Cross-environment validation, transfer learning, diverse datasets

---

## 11. Roadmap & Extensions
- **Priority 1: Address Overfitting**
  - Collect data from multiple environments (different rooms, buildings)
  - Cross-environment validation (train on room A, test on room B)
  - Domain adaptation / transfer learning techniques
- **Priority 2: Model Improvements**
  - Regularization (limit tree depth, reduce n_estimators)
  - Feature selection to reduce dataset-specific features
  - Hybrid models: features + learned reps (attention fusion)
- **Future Work**
  - Multi-person presence and occupancy counting
  - Self-supervised pretraining on unlabeled CSI
  - Edge deployment (Raspberry Pi) + smart-home integration (Home Assistant)
  - Privacy-preserving training (federated, DP)

---

## 12. References & Try-It-Now
- **Key References**: WiAR dataset, Intel 5300 CSI Tool, csiread, Random Forest, CNN
- **Quick Start**:
  ```bash
  # Setup
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

  # Train RF (example)
  python training/train_random_forest.py

  # Run dashboard
  streamlit run app.py
  ```
- **Contacts**: Rishabh (230158), Shivansh (230054)

---

## Appendix: Speaker Notes (Highlights)
- Emphasize privacy vs cameras; no line-of-sight needed
- Explain CSI intuitively: multipath + motion → amplitude/phase changes
- Why 14 features: physics-informed, robust with small data
- **Key talking point on 99.40% accuracy**:
  - "Our model achieves 99.40% accuracy with zero false positives—which sounds great but is concerning"
  - "The 100% precision means it perfectly learned what 'No Activity' looks like in WiAR's lab"
  - "This indicates the model learned to distinguish WiAR recordings, not generalize to new rooms"
  - "In a new room with different furniture, we'd expect 70-85% at best"
  - "This is common in WiFi sensing research; cross-environment generalization is the hard problem"
  - "The 7 false negatives (98% recall) are low-motion activities—phone calls, subtle gestures"
- RF vs CNN trade-off: RF overfits more on limited data; CNN may generalize better with more data
- Demo: show probability timeline and CSI heatmap; mention smoothing options
- Acknowledge limitations honestly; pitch roadmap for proper validation
- **Why the project went this way**:
  - Started with WiAR (established benchmark) for reproducibility
  - Binary presence is simpler than 16-class activity recognition
  - Hand-crafted features work well on small data but risk overfitting
  - Real-time inference (<10ms) proves technical feasibility
  - Next steps: collect diverse data, validate cross-environment
