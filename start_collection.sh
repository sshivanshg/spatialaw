#!/bin/bash
# Quick script to start data collection

echo "=========================================="
echo "WiFi Data Collection with Position"
echo "=========================================="
echo ""

# Get room name
read -p "Enter room/location name (e.g., Room_101): " ROOM_NAME

if [ -z "$ROOM_NAME" ]; then
    echo "Error: Room name is required"
    exit 1
fi

echo ""
echo "Starting data collection for: $ROOM_NAME"
echo ""
echo "Instructions:"
echo "  - Enter positions as: x, y (e.g., 0, 0)"
echo "  - Use meters as units"
echo "  - Enter 'q' to quit"
echo ""

# Activate venv and run collection
source venv/bin/activate
python collect_with_position.py --location "$ROOM_NAME" --interactive --duration 30

echo ""
echo "Data collection completed!"
echo "Check data in: data/$ROOM_NAME/"

