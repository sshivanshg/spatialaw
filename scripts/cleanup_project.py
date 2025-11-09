#!/usr/bin/env python3
"""Cleanup script to remove unnecessary files from the project"""

import os
from pathlib import Path
import shutil

def cleanup_checkpoints():
    """Remove baseline and localization model files."""
    checkpoints_dir = Path('checkpoints')
    
    files_to_remove = [
        'baseline_linear_model.pkl',
        'baseline_linear_scaler.pkl',
        'baseline_linear_metrics.json',
        'baseline_random_forest_model.pkl',
        'baseline_random_forest_scaler.pkl',
        'baseline_random_forest_metrics.json',
        'localization_angle_random_forest.pkl',
        'localization_angle_random_forest_scaler.pkl',
        'localization_distance_random_forest.pkl',
        'localization_distance_random_forest_scaler.pkl',
        'best_model.pth',
        'checkpoint_epoch_1.pth',
        'checkpoint_epoch_2.pth',
        'checkpoint_epoch_3.pth',
        'checkpoint_epoch_4.pth',
        'checkpoint_epoch_5.pth',
    ]
    
    removed = 0
    for filename in files_to_remove:
        filepath = checkpoints_dir / filename
        if filepath.exists():
            filepath.unlink()
            removed += 1
            print(f"✅ Removed: {filename}")
    
    print(f"\n✅ Removed {removed} model files from checkpoints/")

def cleanup_visualizations():
    """Remove old localization visualization files."""
    vis_dir = Path('visualizations')
    
    files_to_remove = [
        'localization_angle_prediction.png',
        'localization_distance_prediction.png',
        'localization_position_scatter.png',
        'localization_rssi_heatmap.png',
        'localization_signal_distribution.png',
        'test_angle_prediction.png',
        'test_distance_prediction.png',
    ]
    
    removed = 0
    for filename in files_to_remove:
        filepath = vis_dir / filename
        if filepath.exists():
            filepath.unlink()
            removed += 1
            print(f"✅ Removed: {filename}")
    
    print(f"\n✅ Removed {removed} visualization files")

def cleanup_empty_directories():
    """Remove empty directories."""
    directories = [
        'src/evaluation',
        'src/training',
        'configs',
        'notebooks',
    ]
    
    removed = 0
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            try:
                # Try to remove if empty
                path.rmdir()
                removed += 1
                print(f"✅ Removed empty directory: {dir_path}")
            except OSError:
                # Directory not empty or other error
                print(f"⚠️  Could not remove {dir_path} (may contain files)")
    
    print(f"\n✅ Removed {removed} empty directories")

def main():
    print("=" * 70)
    print("Project Cleanup")
    print("=" * 70)
    print()
    
    cleanup_checkpoints()
    cleanup_visualizations()
    cleanup_empty_directories()
    
    print()
    print("=" * 70)
    print("Cleanup Complete!")
    print("=" * 70)
    print()
    print("Remaining files in checkpoints/:")
    for f in sorted(Path('checkpoints').glob('*')):
        if f.is_file():
            print(f"  - {f.name}")
    print()
    print("Remaining scripts:")
    for f in sorted(Path('scripts').glob('*.py')):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()

