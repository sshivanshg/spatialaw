# Spatial Awareness: Human Presence Detection using WiFi CSI

> **Turn your WiFi signals into an invisible motion sensor.**
> *A Privacy-Preserving Alternative to Cameras.*

## Project Overview
This project utilizes **Channel State Information (CSI)** from standard WiFi signals to detect human presence. Unlike cameras, it preserves privacy and works in the dark. By analyzing the distortions in WiFi waves (Phase and Amplitude), our Machin Learning model can distinguish between an **Empty Room** and **Human Activity** with high precision.

### Key Features
*   **High Accuracy:** Achieved **99.40%** test accuracy using a Random Forest Classifier.
*   **Real-Time Detection:** Live Streamlit Dashboard for instant visualization.
*   **Robustness:** Optimized feature engineering (Variance, Entropy, Doppler) to filter out electrical noise.
*   **Privacy-First:** No video or audio recording; only signal physics.

---

## Quick Start

### 1. Installation
Ensure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
Launch the real-time application:
```bash
streamlit run app.py
```
*Upload a `.dat` file (e.g., `csi_a12_2.dat`) to see the detection in action.*

### 3. Training the Model (Optional)
The core model logic is located in `model_tools/`. To retrain:
```bash
python model_tools/train_random_forest.py
```

---

## Project Structure

```text
.
├── app.py                            # Main Application (Streamlit Dashboard)
├── model_tools/                      # Model Training Scripts (The "Core Logic")
│   └── train_random_forest.py        # Random Forest Trainer (99.4% Accuracy)
├── models/                           # Saved Model Artifacts (.joblib, .json)
├── data/                             # Dataset Store
│   └── raw/WiAR/                     # Original Intel 5300 .dat files
├── requirements.txt                  # Dependencies
└── _archive/                         # (LEGACY) Raw scripts & experiments.
                                      # Not used in final production.
```

> **Note:** The `_archive/` folder contains initial data processing pipelines and experimental code. It is preserved for reference but is **not** part of the final executable flow.

---

## Performance Metrics

Our final **Random Forest Model** (150 Estimators) achieved the following on independent test data:

| Metric | Score | Meaning |
| :--- | :--- | :--- |
| **Accuracy** | **99.40%** | Overall Correctness |
| **Precision** | **100.0%** | Zero False Positives (Reliable) |
| **Recall** | **98.05%** | Highly Sensitive to Motion |
| **Inference Time** | **<10ms** | Real-Time Capable |

---

## 👥 Authors
*   **Shivansh (230054)**
*   **Rishabh (230158)**

*Newton School of Technology — Computer Networks + AI/ML Capstone*
