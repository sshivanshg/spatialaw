#!/bin/bash
# Cleanup script to remove old/unnecessary files

echo "Cleaning up old files..."

# Remove old PyTorch checkpoints
rm -f checkpoints/checkpoint_epoch_*.pth

# Remove old TensorBoard logs
rm -rf logs/*

# Remove old test data directories
rm -rf data/collections
rm -rf data/Room_101
rm -rf data/Test_Location
rm -rf data/Test_Room
rm -rf data/large_dataset

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "✅ Cleanup complete!"

