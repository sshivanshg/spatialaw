#!/bin/bash
# Quick Start Script for Phase-1 Baseline

echo "=========================================="
echo "Phase-1 Baseline - Quick Start"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Step 1: Generate synthetic data
echo "Step 1: Generating synthetic WiFi data..."
python scripts/generate_synthetic_wifi_data.py \
    --num_samples 200 \
    --room_width 10.0 \
    --room_height 8.0 \
    --output data/synthetic_wifi_data.json

if [ $? -ne 0 ]; then
    echo "❌ Data generation failed"
    exit 1
fi

echo "✅ Data generated successfully"
echo ""

# Step 2: Train baseline model
echo "Step 2: Training baseline model..."
python scripts/train_baseline_model.py \
    --data_path data/synthetic_wifi_data.json \
    --model_type random_forest \
    --predict_signal

if [ $? -ne 0 ]; then
    echo "❌ Model training failed"
    exit 1
fi

echo "✅ Model trained successfully"
echo ""

# Step 3: Create visualizations
echo "Step 3: Creating visualizations..."
python scripts/visualize_baseline.py \
    --data_path data/synthetic_wifi_data.json

if [ $? -ne 0 ]; then
    echo "❌ Visualization failed"
    exit 1
fi

echo "✅ Visualizations created successfully"
echo ""

# Step 4: Display results
echo "=========================================="
echo "Results Summary"
echo "=========================================="
echo ""
echo "📊 Data: data/synthetic_wifi_data.json"
echo "🤖 Model: checkpoints/baseline_random_forest_model.pkl"
echo "📈 Visualizations: visualizations/baseline_*.png"
echo ""
echo "📓 Next step: Open Jupyter notebook for detailed analysis:"
echo "   jupyter notebook notebooks/baseline_analysis.ipynb"
echo ""
echo "✅ Phase-1 baseline pipeline completed!"

