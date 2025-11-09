#!/usr/bin/env python3
"""
Train Localization Model from CSI Data
Predicts position (angle, distance) from CSI/RSSI features
"""

import json
import numpy as np
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_localization_data(data_path: str):
    """Load processed localization data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def prepare_features_labels(data):
    """Extract features and labels from localization data."""
    features = []
    labels_angle = []
    labels_distance = []
    
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
        
        features.append(feature_vector)
        
        # Labels: angle and distance
        angle = sample.get('angle')
        distance = sample.get('distance')
        
        if angle is not None:
            labels_angle.append(angle)
        else:
            labels_angle.append(0)
        
        if distance is not None:
            labels_distance.append(distance)
        else:
            labels_distance.append(0)
    
    return np.array(features), np.array(labels_angle), np.array(labels_distance)

def train_model(X, y, model_type='random_forest', test_size=0.2, random_state=42):
    """Train a regression model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'linear':
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    print(f"Training {model_type} model...")
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    metrics = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    return model, scaler, metrics, X_test, y_test, y_test_pred

def plot_predictions(y_true, y_pred, label_name, save_path=None):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel(f'Actual {label_name}')
    plt.ylabel(f'Predicted {label_name}')
    plt.title(f'{label_name} Prediction: Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # Add R² score
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train Localization Model')
    parser.add_argument('--data_path', type=str, default='data/localization_data.json',
                       help='Path to processed localization data')
    parser.add_argument('--model_type', type=str, default='random_forest',
                       choices=['random_forest', 'linear'],
                       help='Model type')
    parser.add_argument('--predict', type=str, default='both',
                       choices=['angle', 'distance', 'both'],
                       help='What to predict')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Training Localization Model")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Model type: {args.model_type}")
    print(f"Predict: {args.predict}")
    print()
    
    # Load data
    print("Loading data...")
    data = load_localization_data(args.data_path)
    print(f"Loaded {len(data)} samples")
    
    # Prepare features and labels
    print("Preparing features and labels...")
    X, y_angle, y_distance = prepare_features_labels(data)
    print(f"Features shape: {X.shape}")
    print(f"Angle range: [{y_angle.min():.1f}, {y_angle.max():.1f}]")
    print(f"Distance range: [{y_distance.min():.1f}, {y_distance.max():.1f}]")
    print()
    
    # Train models
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.predict in ['angle', 'both']:
        print("Training angle prediction model...")
        model_angle, scaler_angle, metrics_angle, X_test, y_test_angle, y_pred_angle = train_model(
            X, y_angle, model_type=args.model_type
        )
        
        print(f"\nAngle Prediction Metrics:")
        print(f"  Train MSE: {metrics_angle['train_mse']:.4f}")
        print(f"  Test MSE: {metrics_angle['test_mse']:.4f}")
        print(f"  Train MAE: {metrics_angle['train_mae']:.4f}")
        print(f"  Test MAE: {metrics_angle['test_mae']:.4f}")
        print(f"  Train R²: {metrics_angle['train_r2']:.4f}")
        print(f"  Test R²: {metrics_angle['test_r2']:.4f}")
        
        # Save model
        model_path = Path(args.output_dir) / f'localization_angle_{args.model_type}.pkl'
        scaler_path = Path(args.output_dir) / f'localization_angle_{args.model_type}_scaler.pkl'
        joblib.dump(model_angle, model_path)
        joblib.dump(scaler_angle, scaler_path)
        print(f"✅ Saved angle model to {model_path}")
        
        # Plot
        plot_path = f'visualizations/localization_angle_prediction.png'
        plot_predictions(y_test_angle, y_pred_angle, 'Angle (degrees)', plot_path)
        print()
    
    if args.predict in ['distance', 'both']:
        print("Training distance prediction model...")
        model_distance, scaler_distance, metrics_distance, X_test, y_test_distance, y_pred_distance = train_model(
            X, y_distance, model_type=args.model_type
        )
        
        print(f"\nDistance Prediction Metrics:")
        print(f"  Train MSE: {metrics_distance['train_mse']:.4f}")
        print(f"  Test MSE: {metrics_distance['test_mse']:.4f}")
        print(f"  Train MAE: {metrics_distance['train_mae']:.4f}")
        print(f"  Test MAE: {metrics_distance['test_mae']:.4f}")
        print(f"  Train R²: {metrics_distance['train_r2']:.4f}")
        print(f"  Test R²: {metrics_distance['test_r2']:.4f}")
        
        # Save model
        model_path = Path(args.output_dir) / f'localization_distance_{args.model_type}.pkl'
        scaler_path = Path(args.output_dir) / f'localization_distance_{args.model_type}_scaler.pkl'
        joblib.dump(model_distance, model_path)
        joblib.dump(scaler_distance, scaler_path)
        print(f"✅ Saved distance model to {model_path}")
        
        # Plot
        plot_path = f'visualizations/localization_distance_prediction.png'
        plot_predictions(y_test_distance, y_pred_distance, 'Distance (meters)', plot_path)
        print()
    
    print("✅ Training completed!")

if __name__ == "__main__":
    main()

