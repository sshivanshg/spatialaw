#!/usr/bin/env python3
"""
Baseline Model Training for Phase-1
Trains simple scikit-learn models (RandomForest, Linear Regression) for WiFi signal prediction
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_data(data_path: str) -> pd.DataFrame:
    """Load WiFi data from JSON or CSV file."""
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    return df


def prepare_features(df: pd.DataFrame, predict_signal: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and targets for training.
    
    Args:
        df: DataFrame with WiFi data
        predict_signal: If True, predict signal from position; else predict position from signal
    
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    if predict_signal:
        # Predict signal from position
        # Features: position_x, position_y
        # Target: rssi (or signal_strength)
        X = df[['position_x', 'position_y']].values
        y = df['rssi'].values
        feature_names = ['position_x', 'position_y']
        target_name = 'rssi'
    else:
        # Predict position from signal
        # Features: rssi, snr, signal_strength, channel
        # Target: position_x, position_y (multitarget)
        X = df[['rssi', 'snr', 'signal_strength', 'channel']].values
        y = df[['position_x', 'position_y']].values
        feature_names = ['rssi', 'snr', 'signal_strength', 'channel']
        target_name = 'position'
    
    return X, y, feature_names, target_name


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'random_forest',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train a baseline model.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: Model type ('random_forest' or 'linear')
        test_size: Test set size
        random_state: Random seed
    
    Returns:
        Trained model, scaler, and evaluation metrics
    """
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
    
    # Train model
    print(f"Training {model_type} model...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    if y.ndim == 1:  # Single target
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
    else:  # Multiple targets
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
    
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2
    }
    
    return model, scaler, metrics, X_test, y_test, y_test_pred


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, target_name: str, save_path: str = None):
    """Plot predictions vs actual values."""
    if y_true.ndim == 1:  # Single target
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel(f'Actual {target_name}')
        plt.ylabel(f'Predicted {target_name}')
        plt.title(f'Predictions vs Actual: {target_name}')
        plt.grid(True, alpha=0.3)
    else:  # Multiple targets
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        for i, name in enumerate(['position_x', 'position_y']):
            axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
            axes[i].plot([y_true[:, i].min(), y_true[:, i].max()], 
                        [y_true[:, i].min(), y_true[:, i].max()], 'r--', lw=2)
            axes[i].set_xlabel(f'Actual {name}')
            axes[i].set_ylabel(f'Predicted {name}')
            axes[i].set_title(f'Predictions vs Actual: {name}')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved prediction plot to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Baseline Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to WiFi data file')
    parser.add_argument('--model_type', type=str, default='random_forest', choices=['random_forest', 'linear'],
                       help='Model type')
    parser.add_argument('--predict_signal', action='store_true', default=True,
                       help='Predict signal from position (default: True)')
    parser.add_argument('--predict_position', action='store_true',
                       help='Predict position from signal (overrides --predict_signal)')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory for model')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Determine prediction mode
    predict_signal = not args.predict_position
    
    print("=" * 70)
    print("Baseline Model Training - Phase-1")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Model type: {args.model_type}")
    print(f"Prediction mode: {'Signal from Position' if predict_signal else 'Position from Signal'}")
    print()
    
    # Load data
    print("Loading data...")
    df = load_data(args.data_path)
    print(f"Loaded {len(df)} samples")
    print()
    
    # Prepare features
    X, y, feature_names, target_name = prepare_features(df, predict_signal=predict_signal)
    print(f"Features: {feature_names}")
    print(f"Target: {target_name}")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print()
    
    # Train model
    model, scaler, metrics, X_test, y_test, y_test_pred = train_model(
        X, y, model_type=args.model_type, test_size=args.test_size, random_state=args.random_state
    )
    
    # Print metrics
    print("\nModel Metrics:")
    print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"  Train MAE: {metrics['train_mae']:.4f}")
    print(f"  Test MAE: {metrics['test_mae']:.4f}")
    print(f"  Train R²: {metrics['train_r2']:.4f}")
    print(f"  Test R²: {metrics['test_r2']:.4f}")
    print()
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f'baseline_{args.model_type}_model.pkl')
    scaler_path = os.path.join(args.output_dir, f'baseline_{args.model_type}_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved model to {model_path}")
    print(f"✅ Saved scaler to {scaler_path}")
    
    # Plot predictions
    plot_path = os.path.join('visualizations', f'baseline_{args.model_type}_predictions.png')
    plot_predictions(y_test, y_test_pred, target_name, save_path=plot_path)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'baseline_{args.model_type}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics to {metrics_path}")
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()

