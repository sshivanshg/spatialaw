#!/usr/bin/env python3
"""
Test Localization Model
Evaluates trained models and makes predictions on new data
"""

import json
import numpy as np
import argparse
from pathlib import Path
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_localization_data(data_path: str):
    """Load processed localization data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def prepare_features(data):
    """Extract features from localization data."""
    features = []
    
    for sample in data:
        # Features: RSSI, SNR, signal_strength, CSI magnitude
        feature_vector = [
            sample.get('rssi', 0),
            sample.get('snr', 0),
            sample.get('signal_strength', 0),
            sample.get('csi_magnitude', 0),
        ]
        
        # Add CSI magnitude samples (first few)
        csi_mag_samples = sample.get('csi_magnitude_samples', [])
        if len(csi_mag_samples) > 0:
            # Take first 10 CSI magnitude values
            feature_vector.extend(csi_mag_samples[:10])
        else:
            # Pad with zeros if no CSI samples
            feature_vector.extend([0] * 10)
        
        features.append(feature_vector)
    
    return np.array(features)

def load_model(model_path: str, scaler_path: str):
    """Load trained model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def evaluate_model(model, scaler, X, y, task_name: str):
    """Evaluate model performance."""
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"Model Performance: {task_name}")
    print(f"{'='*70}")
    print(f"  R² Score:  {r2:.4f}")
    print(f"  RMSE:      {rmse:.4f}")
    print(f"  MAE:       {mae:.4f}")
    print(f"  MSE:       {mse:.4f}")
    print()
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'y_true': y,
        'y_pred': y_pred
    }

def plot_predictions(y_true, y_pred, task_name: str, save_path: str = None):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R² for title
    r2 = r2_score(y_true, y_pred)
    
    plt.xlabel(f'Actual {task_name}', fontsize=12)
    plt.ylabel(f'Predicted {task_name}', fontsize=12)
    plt.title(f'{task_name} Prediction: R² = {r2:.4f}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved prediction plot to {save_path}")
    else:
        plt.show()
    plt.close()

def predict_position(model_angle, model_distance, scaler_angle, scaler_distance, features):
    """Predict angle and distance from features."""
    # Scale features
    features_scaled_angle = scaler_angle.transform(features)
    features_scaled_distance = scaler_distance.transform(features)
    
    # Make predictions
    angle_pred = model_angle.predict(features_scaled_angle)
    distance_pred = model_distance.predict(features_scaled_distance)
    
    return angle_pred, distance_pred

def test_on_sample_data(data_path: str, model_dir: str = 'checkpoints', num_samples: int = 100):
    """Test model on sample data from the dataset."""
    print("="*70)
    print("Testing Localization Models")
    print("="*70)
    print()
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = load_localization_data(data_path)
    print(f"✅ Loaded {len(data)} samples")
    
    # Prepare features and labels
    print("Preparing features and labels...")
    X = prepare_features(data)
    
    # Filter data to only samples with both angle and distance
    valid_indices = []
    valid_angles = []
    valid_distances = []
    for i, sample in enumerate(data):
        angle = sample.get('angle')
        distance = sample.get('distance')
        if angle is not None and distance is not None:
            valid_indices.append(i)
            valid_angles.append(angle)
            valid_distances.append(distance)
    
    X_valid = X[valid_indices]
    angles_valid = np.array(valid_angles)
    distances_valid = np.array(valid_distances)
    
    print(f"✅ Valid samples: {len(X_valid)}")
    print()
    
    # Load models
    print("Loading trained models...")
    angle_model_path = Path(model_dir) / 'localization_angle_random_forest.pkl'
    angle_scaler_path = Path(model_dir) / 'localization_angle_random_forest_scaler.pkl'
    distance_model_path = Path(model_dir) / 'localization_distance_random_forest.pkl'
    distance_scaler_path = Path(model_dir) / 'localization_distance_random_forest_scaler.pkl'
    
    if not angle_model_path.exists():
        print(f"❌ Error: Model not found at {angle_model_path}")
        print("   Please train the model first using:")
        print("   python scripts/train_localization_model.py --predict both")
        return
    
    model_angle, scaler_angle = load_model(str(angle_model_path), str(angle_scaler_path))
    model_distance, scaler_distance = load_model(str(distance_model_path), str(distance_scaler_path))
    print("✅ Models loaded")
    print()
    
    # Sample data for testing (use subset if too large)
    if len(X_valid) > num_samples:
        indices = np.random.choice(len(X_valid), num_samples, replace=False)
        X_test = X_valid[indices]
        angles_test = angles_valid[indices]
        distances_test = distances_valid[indices]
        print(f"Testing on {num_samples} randomly sampled data points...")
    else:
        X_test = X_valid
        angles_test = angles_valid
        distances_test = distances_valid
        print(f"Testing on all {len(X_test)} data points...")
    print()
    
    # Evaluate angle prediction
    angle_results = evaluate_model(model_angle, scaler_angle, X_test, angles_test, "Angle Prediction")
    
    # Evaluate distance prediction
    distance_results = evaluate_model(model_distance, scaler_distance, X_test, distances_test, "Distance Prediction")
    
    # Plot predictions
    print("Generating prediction plots...")
    plot_predictions(
        angle_results['y_true'],
        angle_results['y_pred'],
        "Angle (degrees)",
        save_path="visualizations/test_angle_prediction.png"
    )
    
    plot_predictions(
        distance_results['y_true'],
        distance_results['y_pred'],
        "Distance (meters)",
        save_path="visualizations/test_distance_prediction.png"
    )
    
    # Show some example predictions
    print("\n" + "="*70)
    print("Example Predictions")
    print("="*70)
    print(f"{'Index':<8} {'Actual Angle':<15} {'Pred Angle':<15} {'Actual Dist':<15} {'Pred Dist':<15} {'Error Angle':<15} {'Error Dist':<15}")
    print("-"*70)
    
    num_examples = min(10, len(X_test))
    for i in range(num_examples):
        actual_angle = angles_test[i]
        pred_angle = angle_results['y_pred'][i]
        actual_dist = distances_test[i]
        pred_dist = distance_results['y_pred'][i]
        error_angle = abs(actual_angle - pred_angle)
        error_dist = abs(actual_dist - pred_dist)
        
        print(f"{i:<8} {actual_angle:<15.2f} {pred_angle:<15.2f} {actual_dist:<15.2f} {pred_dist:<15.2f} {error_angle:<15.2f} {error_dist:<15.2f}")
    
    print("\n" + "="*70)
    print("✅ Testing completed!")
    print("="*70)
    print()
    print("Results saved to:")
    print("  - visualizations/test_angle_prediction.png")
    print("  - visualizations/test_distance_prediction.png")
    print()

def predict_single_sample(rssi: float, snr: float, signal_strength: float, 
                         csi_magnitude: float, csi_samples: list = None,
                         model_dir: str = 'checkpoints'):
    """Predict position for a single sample."""
    # Load models
    angle_model_path = Path(model_dir) / 'localization_angle_random_forest.pkl'
    angle_scaler_path = Path(model_dir) / 'localization_angle_random_forest_scaler.pkl'
    distance_model_path = Path(model_dir) / 'localization_distance_random_forest.pkl'
    distance_scaler_path = Path(model_dir) / 'localization_distance_random_forest_scaler.pkl'
    
    if not angle_model_path.exists():
        print("❌ Error: Models not found. Please train the models first.")
        return None, None
    
    model_angle, scaler_angle = load_model(str(angle_model_path), str(angle_scaler_path))
    model_distance, scaler_distance = load_model(str(distance_model_path), str(distance_scaler_path))
    
    # Prepare features
    feature_vector = [rssi, snr, signal_strength, csi_magnitude]
    if csi_samples:
        feature_vector.extend(csi_samples[:10])
    else:
        feature_vector.extend([0] * 10)
    
    features = np.array([feature_vector])
    
    # Predict
    angle_pred, distance_pred = predict_position(
        model_angle, model_distance, scaler_angle, scaler_distance, features
    )
    
    return angle_pred[0], distance_pred[0]

def main():
    parser = argparse.ArgumentParser(description='Test Localization Models')
    parser.add_argument('--data_path', type=str, default='data/localization_data.json',
                       help='Path to localization data')
    parser.add_argument('--model_dir', type=str, default='checkpoints',
                       help='Directory containing trained models')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to test on')
    parser.add_argument('--predict', type=str, default=None,
                       help='Predict for single sample: "rssi,snr,signal_strength,csi_magnitude"')
    
    args = parser.parse_args()
    
    if args.predict:
        # Single prediction mode
        values = [float(x.strip()) for x in args.predict.split(',')]
        if len(values) < 4:
            print("❌ Error: Need at least 4 values: rssi, snr, signal_strength, csi_magnitude")
            return
        
        rssi, snr, signal_strength, csi_magnitude = values[:4]
        csi_samples = values[4:] if len(values) > 4 else None
        
        angle, distance = predict_single_sample(
            rssi, snr, signal_strength, csi_magnitude, csi_samples, args.model_dir
        )
        
        if angle is not None and distance is not None:
            print(f"\nPredicted Position:")
            print(f"  Angle:    {angle:.2f} degrees")
            print(f"  Distance: {distance:.2f} meters")
    else:
        # Test on dataset
        test_on_sample_data(args.data_path, args.model_dir, args.num_samples)

if __name__ == "__main__":
    main()

