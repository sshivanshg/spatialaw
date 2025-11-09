#!/usr/bin/env python3
"""
Check Which Models Are Trained
Shows status of all trained models in the checkpoints directory
"""

import os
import json
from pathlib import Path
import joblib

def check_model_files(checkpoints_dir='checkpoints'):
    """Check which model files exist."""
    checkpoints_path = Path(checkpoints_dir)
    
    print("=" * 70)
    print("Trained Models Status")
    print("=" * 70)
    print()
    
    # Check Motion Detector
    print("1. MOTION DETECTOR:")
    print("-" * 70)
    motion_model = checkpoints_path / 'motion_detector_random_forest.pkl'
    motion_scaler = checkpoints_path / 'motion_detector_random_forest_scaler.pkl'
    motion_metrics = checkpoints_path / 'motion_detector_random_forest_metrics.json'
    
    if motion_model.exists() and motion_scaler.exists():
        print("   ✅ Model: EXISTS")
        print(f"      Path: {motion_model}")
        print(f"      Size: {motion_model.stat().st_size / 1024:.1f} KB")
        
        # Load and check model
        try:
            model = joblib.load(motion_model)
            print(f"      Type: {type(model).__name__}")
            if hasattr(model, 'n_estimators'):
                print(f"      Trees: {model.n_estimators}")
        except:
            print("      ⚠️  Could not load model")
        
        if motion_metrics.exists():
            try:
                with open(motion_metrics, 'r') as f:
                    metrics = json.load(f)
                print("   ✅ Metrics: EXISTS")
                print(f"      Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.2%}")
                print(f"      Test ROC AUC: {metrics.get('test_roc_auc', 'N/A'):.3f}")
                print(f"      Test F1: {metrics.get('test_f1', 'N/A'):.3f}")
            except:
                print("   ⚠️  Metrics file exists but could not be read")
        else:
            print("   ⚠️  Metrics: NOT FOUND")
    else:
        print("   ❌ Model: NOT TRAINED")
        print("      Train with: python scripts/train_motion_detector.py")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    trained_models = []
    if motion_model.exists():
        trained_models.append("Motion Detector")
    
    if trained_models:
        print("✅ Trained Models:")
        for model in trained_models:
            print(f"   - {model}")
    else:
        print("❌ No models trained yet!")
        print()
        print("Train models with:")
        print("   - Motion Detector: python scripts/train_motion_detector.py")
    print()

if __name__ == "__main__":
    check_model_files()

