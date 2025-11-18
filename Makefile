.PHONY: help setup install clean

help:
	@echo "Available commands:"
	@echo "  make setup     - Create virtual environment and install dependencies"
	@echo "  make install   - Install dependencies only"
	@echo "  make clean     - Remove __pycache__ and .pyc files"
	@echo ""
	@echo "Note: Model training is done via Jupyter notebooks:"
	@echo "  - notebooks/train_presence_detector.ipynb"

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

install:
	./venv/bin/pip install -r requirements.txt

clean:
	find . -type d -name __pycache__ -not -path "./venv/*" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./venv/*" -delete
	find . -type f -name "*.pyo" -not -path "./venv/*" -delete

