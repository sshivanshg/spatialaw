#!/usr/bin/env python3
"""
Training Script for Motion Detection
Trains a classifier to detect movement from WiFi time series
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.time_series_features import extract_simple_features, extract_time_series_features


def load_time_series_data(data_path: str) -> list:
    """Load time series WiFi data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    return data


def prepare_dataset(data_paths: list, window_size: int = 20, overlap: float = 0.5):
    """
    Prepare dataset from time series data files.
    
    Args:
        data_paths: List of paths to time series data files
        window_size: Size of sliding window
        overlap: Overlap ratio between windows
    
    Returns:
        Tuple of (features, labels)
    """
    all_features = []
    all_labels = []
    
    for data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"⚠️  File not found: {data_path}")
            continue
        
        data = load_time_series_data(data_path)
        
        # Extract features
        features, labels = extract_simple_features(data)
        
        all_features.append(features)
        all_labels.append(labels)
        
        print(f"Loaded {len(data)} samples from {data_path}")
        print(f"  Extracted {len(features)} feature windows")
        print(f"  Movement: {np.sum(labels == 1)}, No-movement: {np.sum(labels == 0)}")
    
    # Combine all data
    if all_features:
        features = np.vstack(all_features)
        labels = np.hstack(all_labels)
    else:
        raise ValueError("No data loaded!")
    
    return features, labels


def train_sklearn_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'random_forest',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train a scikit-learn classifier.
    
    Args:
        X: Feature matrix
        y: Labels
        model_type: Model type ('random_forest', 'logistic', 'svm')
        test_size: Test set size
        random_state: Random seed
    
    Returns:
        Trained model, scaler, metrics, test data
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=random_state, max_iter=1000)
    elif model_type == 'svm':
        model = SVC(kernel='rbf', probability=True, random_state=random_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    print(f"Training {model_type} model...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    # ROC AUC
    if y_test_proba is not None:
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
    else:
        test_roc_auc = 0.0
    
    metrics = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc
    }
    
    return model, scaler, metrics, X_test, y_test, y_test_pred, y_test_proba


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Movement', 'Movement'],
                yticklabels=['No Movement', 'Movement'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Motion Detection')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved confusion matrix to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: str = None):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Motion Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved ROC curve to {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Motion Detection Model')
    parser.add_argument('--data_paths', type=str, nargs='+', required=True, 
                       help='Paths to time series data files')
    parser.add_argument('--model_type', type=str, default='random_forest', 
                       choices=['random_forest', 'logistic', 'svm'],
                       help='Model type')
    parser.add_argument('--window_size', type=int, default=20, 
                       help='Window size for feature extraction')
    parser.add_argument('--overlap', type=float, default=0.5, 
                       help='Overlap ratio between windows')
    parser.add_argument('--output_dir', type=str, default='checkpoints', 
                       help='Output directory for model')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Motion Detection Model Training")
    print("=" * 70)
    print(f"Data paths: {args.data_paths}")
    print(f"Model type: {args.model_type}")
    print(f"Window size: {args.window_size}")
    print()
    
    # Prepare dataset
    print("Loading and preparing data...")
    features, labels = prepare_dataset(args.data_paths, args.window_size, args.overlap)
    
    print(f"\nDataset summary:")
    print(f"  Total samples: {len(features)}")
    print(f"  Features: {features.shape[1]}")
    print(f"  Movement samples: {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.1f}%)")
    print(f"  No-movement samples: {np.sum(labels == 0)} ({np.sum(labels == 0)/len(labels)*100:.1f}%)")
    print()
    
    # Train model
    model, scaler, metrics, X_test, y_test, y_test_pred, y_test_proba = train_sklearn_model(
        features, labels, model_type=args.model_type, 
        test_size=args.test_size, random_state=args.random_state
    )
    
    # Print metrics
    print("\nModel Metrics:")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  Train Precision: {metrics['train_precision']:.4f}")
    print(f"  Test Precision: {metrics['test_precision']:.4f}")
    print(f"  Train Recall: {metrics['train_recall']:.4f}")
    print(f"  Test Recall: {metrics['test_recall']:.4f}")
    print(f"  Train F1: {metrics['train_f1']:.4f}")
    print(f"  Test F1: {metrics['test_f1']:.4f}")
    print(f"  Test ROC AUC: {metrics['test_roc_auc']:.4f}")
    print()
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['No Movement', 'Movement']))
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f'motion_detector_{args.model_type}.pkl')
    scaler_path = os.path.join(args.output_dir, f'motion_detector_{args.model_type}_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Saved model to {model_path}")
    print(f"✅ Saved scaler to {scaler_path}")
    
    # Plot confusion matrix
    cm_path = os.path.join('visualizations', f'motion_detection_confusion_matrix.png')
    plot_confusion_matrix(y_test, y_test_pred, save_path=cm_path)
    
    # Plot ROC curve
    if y_test_proba is not None:
        roc_path = os.path.join('visualizations', f'motion_detection_roc_curve.png')
        plot_roc_curve(y_test, y_test_proba, save_path=roc_path)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'motion_detector_{args.model_type}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print(f"✅ Saved metrics to {metrics_path}")
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()

