#!/bin/bash

set -euo pipefail

echo "=========================================="
echo "Downloading Small WiAR Test Dataset"
echo "=========================================="

# Create data directory
mkdir -p data/test_samples

# WiAR dataset GitHub raw URLs
BASE_URL="https://raw.githubusercontent.com/linteresa/WiAR/master"

# Download 2 small sample files from different activities
echo "Downloading sample 1: Walking activity..."
curl -L "${BASE_URL}/user1/walking/walking_1.dat" -o data/test_samples/walking_1.dat || echo "Failed to download walking_1.dat"

echo "Downloading sample 2: Empty room..."
curl -L "${BASE_URL}/user1/empty/empty_1.dat" -o data/test_samples/empty_1.dat || echo "Failed to download empty_1.dat"

echo "Downloading sample 3: Sitting activity..."
curl -L "${BASE_URL}/user1/sitting/sitting_1.dat" -o data/test_samples/sitting_1.dat || echo "Failed to download sitting_1.dat"

# Check if files were downloaded
if [ -f "data/test_samples/walking_1.dat" ] || [ -f "data/test_samples/empty_1.dat" ] || [ -f "data/test_samples/sitting_1.dat" ]; then
    echo "✓ Test samples downloaded successfully to data/test_samples/"
    echo ""
    echo "Files downloaded:"
    ls -lh data/test_samples/
    echo ""
    echo "You can now test the app with: streamlit run app.py"
else
    echo "⚠️  Failed to download samples. The WiAR repository structure may have changed."
    echo "Alternative: Clone the entire WiAR repo with: bash scripts/fetch_wiar.sh"
fi
