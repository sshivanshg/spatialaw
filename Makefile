.PHONY: help setup install clean train check

help:
	@echo "Available commands:"
	@echo "  make setup     - Create virtual environment and install dependencies"
	@echo "  make install   - Install dependencies only"
	@echo "  make clean     - Remove __pycache__ and .pyc files"
	@echo "  make train     - Train offline presence detector on WiFi CSI HAR dataset"
	@echo "  make check     - Show trained model status"

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

install:
	./venv/bin/pip install -r requirements.txt

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

train:
	./venv/bin/python scripts/train_motion_detector.py --dataset_root "WiFi CSI HAR Dataset" --model_type random_forest

check:
	./venv/bin/python scripts/check_trained_models.py

