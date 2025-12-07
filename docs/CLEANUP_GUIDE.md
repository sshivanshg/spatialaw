# Cleanup Guide

**Status**: ✅ Cleanup already completed in the `restructured` branch!

All duplicate and unnecessary files have been removed:
- `_archive/` directory (all duplicates)
- `model_tools/` directory (merged into `training/`)
- `debug_features.py` (debug script)
- `download_test_data.sh` (redundant)

This guide now serves as reference for the cleanup that was performed.

## Branch Workflow

### 1. Review & Test on restructured Branch
```bash
# Make sure you're on the restructured branch
git checkout restructured

# Activate virtual environment
source .venv/bin/activate

# Run tests (see verification steps below)
```

### 2. Merge to Main (After Testing)

**Option A: Direct Merge** (if you have write access):
```bash
# Switch to main
git checkout main

# Merge restructured branch
git merge restructured

# Push to remote
git push origin main
```

**Option B: Pull Request** (recommended for team review):
1. Go to: https://github.com/sshivanshg/spatialaw/pull/new/restructured
2. Create PR from `restructured` to `main`
3. Review changes with team
4. Merge via GitHub interface

### 3. Clean Up Branch (Optional)
```bash
# Delete local branch after successful merge
git branch -d restructured

# Delete remote branch
git push origin --delete restructured
```

## Files & Directories Removed

### ✅ 1. Archive Directory
Removed entire `_archive/` directory containing:
- Duplicate source files from `_archive/src/`
- Duplicate scripts from `_archive/scripts/`
- Duplicate model tools from `_archive/model_tools/`
- Old HTML visualization files

**Total files removed**: ~30+ duplicate Python files

### ✅ 2. Old model_tools Directory
Removed `model_tools/` directory:
- `train_random_forest.py` → moved to `training/`
- `train_cnn.py` → moved to `training/`

### ✅ 3. Debug & Temporary Files
Removed:
- `debug_features.py` - One-off debug script
- `download_test_data.sh` - Redundant with `scripts/data_preparation/fetch_wiar.sh`

### Result
Project size reduced by ~3-8 MB of duplicate code.

## Files to Keep

### Root Level
- ✓ `app.py` - Main Streamlit dashboard
- ✓ `README.md` - Project documentation
- ✓ `pyproject.toml` - Package configuration
- ✓ `requirements.txt` - Dependencies
- ✓ `Makefile` - Build automation
- ✓ `setup.sh` - Environment setup

### New Directories
- ✓ `src/spatialaw/` - Main package
- ✓ `scripts/` - All scripts
- ✓ `training/` - Training scripts
- ✓ `config/` - Configuration
- ✓ `docs/` - Documentation
- ✓ `notebooks/` - Jupyter notebooks
- ✓ `tests/` - Unit tests
- ✓ `models/` - Model artifacts (git-ignored)
- ✓ `data/` - Datasets (git-ignored)

## Verification Steps Before Cleanup

**Important**: Perform these tests on the `restructured` branch BEFORE merging to main.

```bash
# Ensure you're on restructured branch
git checkout restructured
source .venv/bin/activate
```

### 1. Test Data Pipeline
```bash
# Run a complete pipeline test
python scripts/run_pipeline.py --test
```

### 2. Test Training
```bash
# Test RF training
python training/train_random_forest.py --dry-run

# Test CNN training  
python training/train_cnn.py --dry-run
```

### 3. Test Dashboard
```bash
# Launch dashboard
streamlit run app.py
```

### 4. Test Imports
```bash
# Test Python imports
python -c "from spatialaw.preprocessing import load_dat_file; print('OK')"
python -c "from spatialaw.models import MotionDetector; print('OK')"
```

### 5. Run Tests
```bash
# If you've added tests
pytest tests/
```

## Cleanup Commands Used

**For reference**, these are the commands that were used to clean up the project:

```bash
# Remove duplicate directories and files
git rm -rf _archive/ model_tools/
git rm debug_features.py download_test_data.sh

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Commit the cleanup
git add -A
git commit -m "chore: remove duplicate files and directories"
```

These commands have already been executed in the `restructured` branch.

## Post-Cleanup Verification

The cleanup has been completed. To verify the new structure:

```bash
# Switch to restructured branch
git checkout restructured

# Check main directories exist
ls -ld src/ scripts/ training/ config/ docs/ notebooks/ tests/

# Verify package is importable
python -c "import sys; sys.path.insert(0, 'src'); from spatialaw.preprocessing import load_dat_file; print('✓ Imports working')"

# Check git status
git status
```

## Clean Directory Structure

After cleanup, the project has this clean structure:

## What if Something Breaks?

If you encounter issues after merging the cleaned-up branch:

1. **Check import paths**:
   ```bash
   # Search for any remaining references to old paths
   grep -r "from src\." . --exclude-dir=.git --exclude-dir=.venv
   grep -r "model_tools" . --exclude-dir=.git --exclude-dir=.venv
   ```

2. **Verify new paths**:
   - Old: `from src.preprocess.X import Y`
   - New: `from spatialaw.preprocessing.X import Y`
   
3. **Revert if needed**:
   ```bash
   # Go back to main branch
   git checkout main
   ```

All functionality has been preserved - just in better organized locations!

## Optional: Git Commit Strategy

Consider committing in stages:

```bash
# Commit restructure
git add src/ scripts/ training/ config/ docs/ notebooks/
git add app.py pyproject.toml README.md .gitignore
git commit -m "refactor: restructure project into clean package layout"

# Commit cleanup separately
git add -u  # Stage deletions
git commit -m "chore: remove old directory structure"
```

## Size Reduction Estimate

After cleanup, you'll remove approximately:
- `_archive/` - ~2-5 MB of code
- `model_tools/` - ~500 KB
- `__pycache__/` directories - ~1-2 MB

Total: ~3-8 MB of redundant files

---

**Remember**: Always test thoroughly before deleting. When in doubt, keep the files until you're 100% confident.
