#!/bin/bash
# Quick script to test the localization model

echo "=========================================="
echo "Testing Localization Models"
echo "=========================================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Please run: python3 -m venv venv"
    exit 1
fi

# Check if models exist
if [ ! -f "checkpoints/localization_angle_random_forest.pkl" ]; then
    echo "⚠️  Models not found. Training models first..."
    echo ""
    python scripts/train_localization_model.py --predict both
    echo ""
fi

# Test the models
echo "Running model tests..."
python scripts/test_localization_model.py --num_samples 1000

echo ""
echo "=========================================="
echo "✅ Testing completed!"
echo "=========================================="
echo ""
echo "Check visualizations/ for prediction plots:"
echo "  - test_angle_prediction.png"
echo "  - test_distance_prediction.png"
echo ""

