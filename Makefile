.PHONY: help setup install clean train-rf train-cnn

help:
	@echo "Available commands:"
	@echo "  make setup       - Create virtual environment (.venv) and install dependencies"
	@echo "  make install     - Install dependencies into existing .venv"
	@echo "  make clean       - Remove __pycache__ and .pyc files"
	@echo "  make train-rf    - Train Random Forest presence detector"
	@echo "  make train-cnn   - Train CNN model (if configured)"

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

install:
	./.venv/bin/pip install -r requirements.txt

clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
	find . -type f -name "*.pyo" -not -path "./.venv/*" -delete

train-rf:
	. .venv/bin/activate && python model_tools/train_random_forest.py

train-cnn:
	. .venv/bin/activate && python model_tools/train_cnn.py

