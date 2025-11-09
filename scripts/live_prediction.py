#!/usr/bin/env python3
"""
Live Human Detection from WiFi Signals
Uses existing trained models to predict human movement in real-time
"""

import sys
import os
import joblib
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.collect_time_series_data import WiFiCollector
from src.preprocessing.time_series_features import extract_simple_features

def load_model(model_dir='checkpoints'):
    """Load trained motion detector model."""
    model_path = Path(model_dir) / 'motion_detector_random_forest.pkl'
    scaler_path = Path(model_dir) / 'motion_detector_random_forest_scaler.pkl'
    
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("   Please train the model first:")
        print("   python scripts/train_motion_detector.py")
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"✅ Loaded model from {model_path}")
    return model, scaler

def predict_motion(model, scaler, wifi_data, window_size=20):
    """Predict motion from WiFi data."""
    if len(wifi_data) < window_size:
        return None, None
    
    # Extract features
    features, _ = extract_simple_features(wifi_data)
    
    if len(features) == 0:
        return None, None
    
    # Use last feature window
    features = features[-1:].reshape(1, -1)
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability

def main():
    print("=" * 70)
    print("Live Human Detection from WiFi Signals")
    print("=" * 70)
    print()
    
    # Load model
    print("Loading model...")
    model, scaler = load_model()
    if model is None:
        return
    print()
    
    # Initialize WiFi collector
    print("Initializing WiFi collector...")
    try:
        collector = WiFiCollector(interface="en0", sampling_rate=10.0)
        print("✅ WiFi collector ready")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    print()
    
    # Configuration
    window_size = 20  # Number of samples for prediction (2 seconds at 10Hz)
    sampling_interval = 0.1  # 10Hz sampling rate
    buffer = []
    
    print("Starting live detection...")
    print(f"Window size: {window_size} samples ({window_size * sampling_interval:.1f} seconds)")
    print(f"Sampling rate: {1/sampling_interval:.1f} Hz")
    print()
    print("Press Ctrl+C to stop")
    print("-" * 70)
    print()
    
    try:
        sample_count = 0
        while True:
            # Collect WiFi sample
            wifi_info = collector.get_wifi_info()
            
            if wifi_info:
                # Add to buffer
                buffer.append(wifi_info)
                sample_count += 1
                
                # Keep only last window_size samples
                if len(buffer) > window_size:
                    buffer = buffer[-window_size:]
                
                # Predict when we have enough samples
                if len(buffer) >= window_size:
                    prediction, probability = predict_motion(model, scaler, buffer, window_size)
                    
                    if prediction is not None:
                        # Get current signal info
                        current_rssi = wifi_info.get('rssi', 0)
                        current_snr = wifi_info.get('snr', 0)
                        
                        # Display result
                        if prediction == 1:
                            confidence = probability[1] * 100
                            status = "🟢 MOVEMENT"
                        else:
                            confidence = probability[0] * 100
                            status = "🔴 NO MOVEMENT"
                        
                        # Print result
                        print(f"\r[{sample_count:4d}] {status} | "
                              f"Confidence: {confidence:5.1f}% | "
                              f"RSSI: {current_rssi:4.0f} dBm | "
                              f"SNR: {current_snr:3.0f} dB", end='', flush=True)
            
            # Wait before next sample
            time.sleep(sampling_interval)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Live detection stopped")
        print(f"Total samples collected: {sample_count}")
        print("=" * 70)

if __name__ == "__main__":
    main()

