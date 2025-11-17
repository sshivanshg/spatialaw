#!/bin/bash

set -euo pipefail

REPO_URL="https://github.com/linteresa/WiAR.git"
ZIP_URL="https://github.com/linteresa/WiAR/archive/refs/heads/master.zip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$ROOT_DIR/data/raw/WiAR"
TMP_ZIP="$ROOT_DIR/data/raw/WiAR-master.zip"

echo "=========================================="
echo "Fetching WiAR dataset"
echo "Target directory: $TARGET_DIR"
echo "=========================================="

mkdir -p "$TARGET_DIR"

# Try cloning first
if command -v git >/dev/null 2>&1; then
    if [ -d "$TARGET_DIR/.git" ]; then
        echo "WiAR already cloned at $TARGET_DIR"
        exit 0
    fi

    echo "Attempting git clone..."
    if git clone --depth 1 "$REPO_URL" "$TARGET_DIR"; then
        echo "✓ Cloned WiAR repository to $TARGET_DIR"
        exit 0
    else
        echo "⚠️  Git clone failed, falling back to zip download."
        rm -rf "$TARGET_DIR"
    fi
else
    echo "⚠️  Git not available, falling back to zip download."
fi

# Fallback to zip download
echo "Downloading zip from $ZIP_URL"
curl -L "$ZIP_URL" -o "$TMP_ZIP"
echo "Extracting zip..."
unzip -q "$TMP_ZIP" -d "$ROOT_DIR/data/raw"
rm -rf "$TARGET_DIR"
mv "$ROOT_DIR/data/raw/WiAR-master" "$TARGET_DIR"
rm -f "$TMP_ZIP"

echo "✓ Downloaded and extracted WiAR dataset to $TARGET_DIR"

