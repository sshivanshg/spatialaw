import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys
import json
import joblib
from pathlib import Path
import tempfile
import os

# Add _archive to path to import project modules
ROOT = Path(__file__).parent
ARCHIVE_DIR = ROOT / "_archive"
if str(ARCHIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_DIR))

try:
    from src.preprocess.dat_loader import load_dat_file
    from src.preprocess.preprocess import window_csi, denoise_window, normalize_window
    from src.models.motion_detector import MotionDetector
except ImportError as e:
    st.error(f"Error importing project modules: {e}")
    st.stop()

# Page Config
st.set_page_config(
    page_title="Spatial Awareness Dashboard",
    layout="wide"
)

# Title
st.title("Spatial Awareness: WiFi Presence Detection")
st.markdown("Upload a CSI recording (`.dat` or `.npy`) to detect human presence.")

# Sidebar
st.sidebar.header("Configuration")
model_type = st.sidebar.radio("Select Model", ["Random Forest (Classic)", "1D-CNN (Deep Learning)"])
model_dir = st.sidebar.text_input("Model Directory", "models")
binary_dir = st.sidebar.text_input("Binary Data Directory", "data/processed/binary")

# Load Models (Cached)
@st.cache_resource
def load_rf_model(model_dir_path, binary_dir_path):
    try:
        model_path = Path(model_dir_path) / "presence_detector_rf.joblib"
        scaler_path = Path(model_dir_path) / "presence_detector_scaler.joblib"
        feature_names_path = Path(binary_dir_path) / "feature_names.json"
        
        if not model_path.exists():
            return None, f"Model not found at {model_path}"
        
        with open(feature_names_path) as f:
            feature_names = json.load(f)
            
        # Filter out features that were dropped during training
        features_to_drop = ['csi_variance_mean', 'csi_velocity_mean']
        feature_names = [f for f in feature_names if f not in features_to_drop]
            
        detector = MotionDetector(
            feature_names=feature_names,
            model_path=model_path,
            scaler_path=scaler_path
        )
        return detector, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def load_cnn_model(model_dir_path):
    try:
        model_path = Path(model_dir_path) / "presence_detector_cnn.pth"
        if not model_path.exists():
            return None, f"Model not found at {model_path}"
            
        import torch
        import torch.nn as nn
        
        # Define Architecture (Must match training script)
        class SimpleCNN(nn.Module):
            def __init__(self, input_channels=60, num_classes=2):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=7, padding=3)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool1d(kernel_size=2)
                
                self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
                self.relu2 = nn.ReLU()
                self.pool2 = nn.MaxPool1d(kernel_size=2)
                
                self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
                self.relu3 = nn.ReLU()
                self.pool3 = nn.MaxPool1d(kernel_size=2)
                
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(128, num_classes)

            def forward(self, x):
                x = self.pool1(self.relu1(self.conv1(x)))
                x = self.pool2(self.relu2(self.conv2(x)))
                x = self.pool3(self.relu3(self.conv3(x)))
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        input_channels = checkpoint.get('input_channels', 60)
        model = SimpleCNN(input_channels=input_channels)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

# Load Selected Model
detector = None
cnn_model = None
error_msg = None

if model_type == "Random Forest (Classic)":
    detector, error_msg = load_rf_model(model_dir, binary_dir)
else:
    cnn_model, error_msg = load_cnn_model(model_dir)

if error_msg:
    st.sidebar.error(f"Failed to load model: {error_msg}")
    st.sidebar.warning("Please ensure you have run the training script.")
else:
    st.sidebar.success(f"{model_type} Loaded! ")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSI file", type=["dat", "npy"])

if uploaded_file is not None:
    with st.spinner("Processing file..."):
        # Save to temp file because load_dat_file expects a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)
        
        try:
            # 1. Load Data
            if uploaded_file.name.endswith(".npy"):
                csi = np.load(tmp_path)
            else:
                csi = load_dat_file(tmp_path)
            
            # 2. Windowing
            # Check if input is already a window (Subcarriers, Time)
            if csi.ndim == 2 and csi.shape[1] == 256 and csi.shape[0] in [30, 60]:
                st.info(f"Detected single window file with shape {csi.shape}. Skipping windowing.")
                windows = np.expand_dims(csi, axis=0)
            else:
                # Check if we need to transpose (Subcarriers, Time) -> (Time, Subcarriers) for window_csi
                if csi.shape[0] in [30, 60] and csi.shape[1] > 256:
                    csi = csi.T
                
                windows = window_csi(csi, T=256, stride=64)
            
            if len(windows) == 0:
                st.error(f"File is too short to generate windows. Shape: {csi.shape}, Required: {256} samples.")
            else:
                # 3. Process & Predict
                processed_windows = []
                noise_levels = []
                
                for w in windows:
                    # Denoise for visualization and RF (CNN usually takes raw, but let's see)
                    # Our CNN training script loaded raw windows directly from disk.
                    # But wait, the windows on disk were generated by generate_windows.py which applies denoise+norm?
                    # Let's check generate_windows.py.
                    # It calls preprocess.window_csi -> preprocess.denoise_window -> preprocess.normalize_window.
                    # So the "raw" windows on disk ARE denoised and normalized.
                    # So we should apply the same preprocessing here for CNN too.
                    
                    w_denoised = denoise_window(w)
                    # Calculate noise metric (variance) for visualization
                    noise_levels.append(np.mean(np.var(w_denoised, axis=1)))
                    
                    w_norm = normalize_window(w_denoised)
                    processed_windows.append(w_norm)
                
                processed_windows = np.stack(processed_windows)
                
                # Predict
                if model_type == "Random Forest (Classic)":
                    probs = detector.predict_proba_from_windows(processed_windows)[:, 1]
                else:
                    # CNN Inference
                    import torch
                    import torch.nn.functional as F
                    
                    # Ensure shape is (Batch, Channels, Time)
                    # processed_windows is (Batch, Channels, Time)
                    # Check channels
                    if processed_windows.shape[1] == 30 and cnn_model.conv1.in_channels == 60:
                        # If model expects 60 but we have 30, we might need to duplicate or error?
                        # For now, let's assume data matches.
                        pass
                        
                    tensor_input = torch.FloatTensor(processed_windows)
                    with torch.no_grad():
                        outputs = cnn_model(tensor_input)
                        probs = F.softmax(outputs, dim=1)[:, 1].numpy()

                preds = (probs > 0.5).astype(int)
                
                # Time axis
                fs = 100.0
                stride_time = 64 / fs
                time_points = np.arange(len(probs)) * stride_time
                
                # Metrics
                st.divider()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Duration", f"{len(csi)/fs:.2f}s")
                col2.metric("Windows Processed", len(windows))
                col3.metric("Presence Detected", f"{np.mean(preds)*100:.1f}%")
                
                # Plots
                st.subheader("Analysis")
                
                # Stability Filter (Rolling Average)
                st.sidebar.divider()
                st.sidebar.subheader("Stability Filter")
                use_smoothing = st.sidebar.checkbox("Enable Smoothing", value=True)
                window_size = st.sidebar.slider("Rolling Window Size", min_value=1, max_value=20, value=5, help="Number of windows to average over. Higher = More stable but slower reaction.")
                
                # Real-Time Simulation Toggle
                st.sidebar.divider()
                simulate_live = st.sidebar.checkbox("Simulate Live Mode", value=False, help="Play back the file as if it were a live stream.")
                playback_speed = st.sidebar.slider("Playback Speed", 0.1, 5.0, 1.0, help="Multiplier for playback speed (1.0 = Real Time)")

                if use_smoothing and len(probs) > window_size:
                    probs_smooth = pd.Series(probs).rolling(window=window_size, min_periods=1).mean().values
                else:
                    probs_smooth = probs

                # Prepare Data
                df_plot = pd.DataFrame({
                    "Time (s)": time_points,
                    "Raw Probability": probs,
                    "Smoothed Probability": probs_smooth,
                    "Noise Variance": np.array(noise_levels) / np.max(noise_levels) # Normalize for overlay
                })
                
                plot_cols = ["Raw Probability", "Noise Variance"]
                if use_smoothing:
                    plot_cols = ["Smoothed Probability", "Raw Probability", "Noise Variance"]

                # Plotting Logic
                chart_placeholder = st.empty()
                
                if simulate_live:
                    import time
                    # Calculate delay based on stride (0.64s) and playback speed
                    delay = (0.64 / playback_speed)
                    
                    # Iterate and update
                    for i in range(1, len(df_plot) + 1):
                        current_df = df_plot.iloc[:i]
                        
                        fig_prob = px.line(current_df, x="Time (s)", y=plot_cols, 
                                           title="Presence Probability vs Time (LIVE)",
                                           color_discrete_map={
                                               "Smoothed Probability": "green", 
                                               "Raw Probability": "lightgreen", 
                                               "Noise Variance": "orange"
                                           })
                        
                        # Fix for single-point plots
                        if len(current_df) < 2:
                            fig_prob.update_traces(mode='markers+lines')
                            
                        fig_prob.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
                        fig_prob.update_layout(yaxis_range=[-0.1, 1.1], xaxis_range=[0, max(time_points[-1], 10)])
                        
                        chart_placeholder.plotly_chart(fig_prob, use_container_width=True)
                        time.sleep(delay)
                else:
                    # Static Plot
                    fig_prob = px.line(df_plot, x="Time (s)", y=plot_cols, 
                                       title="Presence Probability vs Time",
                                       color_discrete_map={
                                           "Smoothed Probability": "green", 
                                           "Raw Probability": "lightgreen", 
                                           "Noise Variance": "orange"
                                       })
                    
                    if len(df_plot) < 2:
                        fig_prob.update_traces(mode='markers+lines')
                    
                    fig_prob.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
                    fig_prob.update_layout(yaxis_range=[-0.1, 1.1])
                    chart_placeholder.plotly_chart(fig_prob, use_container_width=True)
                
                # 2. CSI Heatmap
                st.subheader("CSI Signal Heatmap")
                # Downsample for display if too large
                heatmap_data = csi
                if heatmap_data.shape[1] > 10000:
                    heatmap_data = heatmap_data[:, ::10]
                
                fig_heat, ax = plt.subplots(figsize=(12, 4))
                # imshow expects (rows, cols). We want Subcarriers (rows) x Time (cols).
                # csi is (Subcarriers, Time), so we pass it directly.
                im = ax.imshow(heatmap_data, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
                ax.set_xlabel("Time (samples)")
                ax.set_ylabel("Subcarriers")
                plt.colorbar(im, label="Amplitude")
                st.pyplot(fig_heat)
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
        finally:
            # Cleanup temp file
            os.unlink(tmp_path)
