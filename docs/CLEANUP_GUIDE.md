# Cleanup Guide

**Note**: These changes are currently in the `restructured` branch. After merging to `main` and verifying everything works correctly, you can safely clean up old directories and files.

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

## Safe to Remove (After Merging & Testing)

### 1. Archive Directory
```bash
# The _archive/ directory contains old structure - no longer needed
rm -rf _archive/
```

**Before removing, verify:**
- ✓ All scripts in `scripts/` work correctly
- ✓ Training scripts in `training/` execute properly
- ✓ Imports in app.py and other files are updated
- ✓ No custom scripts depend on _archive/ paths

### 2. Old model_tools Directory
```bash
# Contents merged into training/
rm -rf model_tools/
```

**Verify first:**
- ✓ All training scripts copied to `training/`
- ✓ No references to `model_tools/` in code or docs

### 3. Empty paper Directory
```bash
# Contents moved to docs/
rmdir paper/  # Will only remove if empty
```

### 4. Temporary/Debug Files
```bash
# Clean up debug and temporary files
rm -f debug_features.py
rm -f download_test_data.sh  # Or move to scripts/data_preparation/
```

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

## Cleanup Commands (All at Once)

**Only run after:**
1. ✅ Testing on `restructured` branch
2. ✅ Merging to `main` branch
3. ✅ Verifying everything works on `main`

```bash
#!/bin/bash
# cleanup.sh - Run this after thorough testing

cd /Users/rishabh/pgming/spatialaw

echo "🧹 Cleaning up old structure..."

# Remove archive
if [ -d "_archive" ]; then
    echo "Removing _archive/..."
    rm -rf _archive/
fi

# Remove old model_tools
if [ -d "model_tools" ]; then
    echo "Removing model_tools/..."
    rm -rf model_tools/
fi

# Remove empty paper dir
if [ -d "paper" ]; then
    echo "Removing empty paper/..."
    rmdir paper/ 2>/dev/null || echo "paper/ not empty or doesn't exist"
fi

# Remove debug files (optional)
echo "Removing debug files..."
rm -f debug_features.py

# Clean Python cache
echo "Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

echo "✅ Cleanup complete!"
echo "📊 Current structure:"
ls -l | grep '^d' | awk '{print $9}'
```

## Post-Cleanup Verification

After cleanup, verify structure:

```bash
# Check main directories exist
ls -ld src/ scripts/ training/ config/ docs/ notebooks/ tests/

# Verify package is importable
python -c "import spatialaw; print(f'Package version: {spatialaw.__version__}')"

# Check git status
git status
```

## What if Something Breaks?

If you encounter issues after cleanup:

1. **Check git history**:
   ```bash
   git log --oneline
   git show <commit-hash>
   ```

2. **Revert if needed**:
   ```bash
   git checkout HEAD~1 -- _archive/
   ```

3. **Compare paths**:
   - Old: `_archive/src/preprocess/...`
   - New: `src/spatialaw/preprocessing/...`

4. **Check import errors**:
   - Look for any remaining references to old paths
   - Search: `grep -r "_archive" . --exclude-dir=.git`
   - Search: `grep -r "from src\." . --exclude-dir=.git`

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
