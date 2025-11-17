#!/bin/bash
# Setup script for Spatial Awareness Project

echo "=========================================="
echo "Spatial Awareness Project - Setup"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed"
    echo "Please install Python 3 first: brew install python3"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can run:"
echo "  ./venv/bin/python scripts/train_motion_detector.py --dataset_root \"WiFi CSI HAR Dataset\" --model_type random_forest"
echo "  ./venv/bin/python scripts/check_trained_models.py"
echo ""

