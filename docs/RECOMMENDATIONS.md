# Additional Recommendations for Project Improvement

## Branch Workflow

**Current Status**: Restructuring is complete in the `restructured` branch.

### Before Implementing Recommendations:

1. **Test the restructured branch**:
   ```bash
   git checkout restructured
   streamlit run app.py
   python scripts/data_preparation/generate_windows.py --help
   python training/train_random_forest.py --help
   ```

2. **Merge to main** (after testing):
   ```bash
   git checkout main
   git merge restructured
   git push origin main
   ```

3. **Then proceed with recommendations below**

---

After merging the restructured branch, here are recommended next steps to further improve the project:

## 1. Package Installation & Development Setup

### Make the package installable
```bash
# First, ensure you're on the restructured branch (or have merged to main)
git checkout restructured  # or main after merge

# Install in editable mode for development
pip install -e .
```

This allows importing `spatialaw` from anywhere without path manipulation.

**Note**: This only works with the new structure in the `restructured` branch.

**Update pyproject.toml** to include all dependencies and proper package discovery:
```toml
[tool.setuptools.packages.find]
where = ["src"]
```

## 2. Update .gitignore

Add to `.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Project specific
data/
models/*.joblib
models/*.pth
models/*.json
models/*.png
*.log

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# Documentation builds
docs/_build/
```

## 3. Testing Infrastructure

### Create test structure
```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── test_preprocessing.py    # Test preprocessing functions
├── test_models.py           # Test model implementations
├── test_data.py             # Test data loading
└── integration/             # Integration tests
    └── test_pipeline.py
```

### Add pytest configuration
Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --cov=src/spatialaw
    --cov-report=html
    --cov-report=term
```

## 4. Code Quality Tools

### Add pre-commit hooks
Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.11
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=88', '--extend-ignore=E203']
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
```

### Install and run:
```bash
pip install pre-commit
pre-commit install
```

## 5. Logging Configuration

Create `src/spatialaw/utils/logging_config.py`:
```python
import logging
from pathlib import Path

def setup_logging(log_file=None, level=logging.INFO):
    """Configure logging for the project."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
```

## 6. Command-Line Interface (CLI)

Create `src/spatialaw/cli.py` for unified CLI:
```python
import click

@click.group()
def cli():
    """Spatialaw: WiFi-based presence detection."""
    pass

@cli.command()
@click.option('--input-dir', required=True)
@click.option('--output-dir', required=True)
def prepare_data(input_dir, output_dir):
    """Prepare data pipeline."""
    # Import and run pipeline
    pass

@cli.command()
@click.option('--model-type', type=click.Choice(['rf', 'cnn']))
def train(model_type):
    """Train a model."""
    pass

if __name__ == '__main__':
    cli()
```

## 7. Environment Management

Create `environment.yml` for conda users:
```yaml
name: spatialaw
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    - -r requirements.txt
```

## 8. Documentation Improvements

### Add docstrings using Google style:
```python
def window_csi(csi_data: np.ndarray, T: int = 256, stride: int = 64) -> np.ndarray:
    """
    Create sliding windows from CSI data.
    
    Args:
        csi_data: CSI amplitude data of shape (time_steps, subcarriers)
        T: Window size in samples
        stride: Stride between windows in samples
        
    Returns:
        Array of windows with shape (n_windows, subcarriers, T)
        
    Raises:
        ValueError: If csi_data is too short for windowing
        
    Example:
        >>> csi = np.random.rand(1000, 30)
        >>> windows = window_csi(csi, T=256, stride=64)
        >>> windows.shape
        (12, 30, 256)
    """
    pass
```

### Generate documentation with Sphinx:
```bash
pip install sphinx sphinx-rtd-theme
cd docs
sphinx-quickstart
```

## 9. CI/CD Pipeline

Create `.github/workflows/test.yml`:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e .[dev]
    - name: Run tests
      run: pytest
    - name: Lint
      run: |
        flake8 src/
        black --check src/
```

## 10. Data Versioning (DVC)

For tracking large data files:
```bash
pip install dvc dvc-s3
dvc init
dvc add data/raw/WiAR
git add data/raw/WiAR.dvc .dvc/
```

## 11. Experiment Tracking

Integrate MLflow or Weights & Biases:
```python
import mlflow

mlflow.set_experiment("presence_detection")

with mlflow.start_run():
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
```

## 12. Docker Support

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

And `docker-compose.yml`:
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
```

## 13. Performance Optimization

- Add caching decorators for expensive functions
- Profile code with `cProfile` or `line_profiler`
- Parallelize data processing with `multiprocessing` or `joblib`
- Use `numba` JIT compilation for numeric functions

## 14. Security

- Add `safety` check for vulnerable dependencies:
  ```bash
  pip install safety
  safety check
  ```
- Never commit credentials (use `.env` files)
- Add security scanning to CI/CD

## 15. Monitoring & Observability

For production deployments:
- Add health check endpoints
- Implement request/response logging
- Set up error tracking (Sentry)
- Monitor model performance drift

## 16. API Development

Consider creating a REST API using FastAPI:
```python
from fastapi import FastAPI, UploadFile
from spatialaw.models.motion_detector import MotionDetector

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    # Process file and return prediction
    pass
```

## Priority Order

**Immediate** (Before anything else):
0. Test and merge `restructured` branch to `main`

**High Priority** (Do these first):
1. Update .gitignore ✅ (Already done in restructured branch)
2. Make package installable
3. Add basic tests
4. Setup logging

**Medium Priority** (Nice to have):
5. Code quality tools (black, flake8)
6. CLI interface
7. Better documentation

**Low Priority** (Future enhancements):
8. CI/CD pipeline
9. Docker support
10. API development

## Implementation Timeline

**Day 1**: Test and merge `restructured` branch
**Week 1**: High priority items
**Week 2**: Medium priority items  
**Week 3**: Documentation and testing
**Week 4**: Low priority enhancements as needed

---

These recommendations will transform your project from a research prototype into a production-ready, maintainable software package.
