#!/usr/bin/env python3
"""
Visualize Motion Detection Results
Creates time-series plots with motion regions highlighted
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.time_series_features import extract_simple_features


def load_time_series_data(data_path: str) -> list:
    """Load time series WiFi data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    return data


def predict_motion(model, scaler, time_series_data: list, window_size: int = 20):
    """Predict motion for time series data."""
    # Extract features
    features, labels = extract_simple_features(time_series_data)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    predictions = model.predict(features_scaled)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[:, 1]
    else:
        probabilities = predictions.astype(float)
    
    return predictions, probabilities, labels


def plot_time_series_with_motion(
    time_series_data: list,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    true_labels: np.ndarray,
    save_path: str = None
):
    """Plot time series with motion regions highlighted."""
    # Extract time series
    timestamps = [item.get('elapsed_time', i) for i, item in enumerate(time_series_data)]
    rssi_values = [item['rssi'] for item in time_series_data]
    snr_values = [item['snr'] for item in time_series_data]
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot 1: RSSI over time
    axes[0].plot(timestamps, rssi_values, 'b-', alpha=0.7, linewidth=1)
    axes[0].set_ylabel('RSSI (dBm)')
    axes[0].set_title('WiFi Signal Strength Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Highlight motion regions (true labels)
    for i in range(len(timestamps) - 1):
        if i < len(true_labels) and true_labels[min(i, len(true_labels)-1)] == 1:
            axes[0].axvspan(timestamps[i], timestamps[min(i+1, len(timestamps)-1)], 
                           alpha=0.3, color='green', label='True Motion' if i == 0 else '')
    
    # Plot 2: SNR over time
    axes[1].plot(timestamps, snr_values, 'g-', alpha=0.7, linewidth=1)
    axes[1].set_ylabel('SNR (dB)')
    axes[1].set_title('Signal-to-Noise Ratio Over Time')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: True labels
    true_labels_full = [item.get('movement_label', 0) for item in time_series_data]
    axes[2].plot(timestamps, true_labels_full, 'r-', linewidth=2, label='True Label')
    axes[2].set_ylabel('Movement')
    axes[2].set_title('True Movement Labels')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Plot 4: Predictions and probabilities
    # Map predictions to full time series (simple expansion)
    pred_full = []
    prob_full = []
    window_size_actual = len(time_series_data) // len(predictions) if len(predictions) > 0 else 1
    for pred, prob in zip(predictions, probabilities):
        pred_full.extend([pred] * window_size_actual)
        prob_full.extend([prob] * window_size_actual)
    
    # Pad to match time series length
    while len(pred_full) < len(timestamps):
        pred_full.append(pred_full[-1] if pred_full else 0)
        prob_full.append(prob_full[-1] if prob_full else 0)
    pred_full = pred_full[:len(timestamps)]
    prob_full = prob_full[:len(timestamps)]
    
    axes[3].plot(timestamps, pred_full, 'b-', linewidth=2, label='Predicted', alpha=0.7)
    axes[3].plot(timestamps, prob_full, 'orange', linewidth=1, label='Probability', alpha=0.5)
    axes[3].set_ylabel('Movement Prediction')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_title('Motion Detection Predictions')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Motion Detection')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to time series data file')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model')
    parser.add_argument('--scaler_path', type=str, required=True, 
                       help='Path to scaler')
    parser.add_argument('--output', type=str, 
                       default='visualizations/motion_detection_timeseries.png',
                       help='Output image path')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Motion Detection Visualization")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Model path: {args.model_path}")
    print()
    
    # Load data
    print("Loading data...")
    time_series_data = load_time_series_data(args.data_path)
    print(f"Loaded {len(time_series_data)} samples")
    print()
    
    # Load model
    print("Loading model...")
    model = joblib.load(args.model_path)
    scaler = joblib.load(args.scaler_path)
    print("Model loaded")
    print()
    
    # Predict
    print("Predicting motion...")
    predictions, probabilities, true_labels = predict_motion(
        model, scaler, time_series_data
    )
    print(f"Predictions: {len(predictions)} windows")
    print(f"  Movement: {np.sum(predictions == 1)}")
    print(f"  No-movement: {np.sum(predictions == 0)}")
    print()
    
    # Visualize
    print("Creating visualization...")
    plot_time_series_with_motion(
        time_series_data, predictions, probabilities, true_labels, args.output
    )
    
    print("✅ Visualization completed!")


if __name__ == "__main__":
    main()

