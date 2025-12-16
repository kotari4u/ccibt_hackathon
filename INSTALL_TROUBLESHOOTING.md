# Installation Troubleshooting Guide

## Issue: Failed building wheel for llvmlite

This error occurs when installing the `arch` package, which requires `numba` and `llvmlite`. These packages need to compile C code and require LLVM.

### Solution 1: Install Minimal Requirements (Recommended for Quick Start)

```bash
# Install core dependencies without arch/prophet
python3.12 -m pip install -r requirements-minimal.txt

# The application will run without GARCH models
# You can add arch/prophet later if needed
```

### Solution 2: Install LLVM and Build llvmlite

**On macOS:**
```bash
# Install LLVM via Homebrew
brew install llvm

# Set environment variables
export LLVM_CONFIG=/opt/homebrew/opt/llvm/bin/llvm-config
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"

# Then install arch
python3.12 -m pip install arch
```

**On Ubuntu/Debian:**
```bash
# Install LLVM development packages
sudo apt-get update
sudo apt-get install llvm-dev

# Then install arch
python3.12 -m pip install arch
```

### Solution 3: Use Pre-built Wheels

```bash
# Try installing numba first (which includes llvmlite)
python3.12 -m pip install numba

# Then install arch
python3.12 -m pip install arch
```

### Solution 4: Install Packages Separately

```bash
# Install core requirements first
python3.12 -m pip install -r requirements-minimal.txt

# Then try optional packages one by one
python3.12 -m pip install arch  # May fail, that's OK
python3.12 -m pip install prophet  # May fail, that's OK
python3.12 -m pip install pmdarima  # May fail, that's OK
python3.12 -m pip install shap  # May fail, that's OK
```

### Solution 5: Use Conda (Alternative)

If pip continues to fail, consider using conda:

```bash
# Install conda if not available
# Then:
conda install -c conda-forge arch numba llvmlite
pip install -r requirements.txt
```

## What Works Without arch/prophet?

The application will run fine without these packages:
- ✅ API endpoints
- ✅ Data processing
- ✅ Feature engineering
- ✅ Anomaly detection
- ✅ Event impact modeling
- ✅ Scenario simulation
- ✅ Ensemble models (XGBoost, Random Forest)
- ⚠️ GARCH models (will show warning, use alternative)
- ⚠️ Prophet forecasts (will show warning, use alternative)

## Verify Installation

```bash
# Check what's installed
python3.12 -c "import fastapi, uvicorn, pandas, numpy, xgboost; print('✓ Core packages OK')"

# Check optional packages
python3.12 -c "
try:
    import arch
    print('✓ arch available')
except ImportError:
    print('✗ arch not available (optional)')

try:
    import prophet
    print('✓ prophet available')
except ImportError:
    print('✗ prophet not available (optional)')
"
```

## Recommended Installation Order

1. **First**: Install minimal requirements
   ```bash
   python3.12 -m pip install -r requirements-minimal.txt
   ```

2. **Test**: Verify core functionality works
   ```bash
   python3.12 run.py
   ```

3. **Then**: Try optional packages if needed
   ```bash
   # Only if you need GARCH models
   python3.12 -m pip install arch
   
   # Only if you need Prophet forecasts
   python3.12 -m pip install prophet
   ```

## Alternative: Skip Problematic Packages

If you don't need GARCH or Prophet models, you can use the application with:
- XGBoost for volatility forecasting
- Random Forest for ensemble predictions
- Statistical methods for volatility calculation

These work well for most use cases!

