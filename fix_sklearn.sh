#!/bin/bash
# Script to fix scikit-learn import error

echo "Fixing scikit-learn import error..."
echo "===================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Virtual environment not detected."
    echo "Please activate your virtual environment first:"
    echo "  source venv/bin/activate"
    exit 1
fi

echo "Current Python: $(which python)"
echo "Virtual env: $VIRTUAL_ENV"
echo ""

# Uninstall existing scikit-learn
echo "Step 1: Uninstalling existing scikit-learn..."
pip uninstall -y scikit-learn

# Clear pip cache
echo "Step 2: Clearing pip cache..."
pip cache purge

# Reinstall scikit-learn with updated version
echo "Step 3: Installing scikit-learn>=1.4.0..."
pip install --upgrade "scikit-learn>=1.4.0,<1.6.0"

# Verify installation
echo ""
echo "Step 4: Verifying installation..."
python -c "from sklearn.decomposition import PCA; from sklearn.neighbors import NeighborhoodComponentsAnalysis; print('✓ scikit-learn import successful!')"

echo ""
echo "Done! scikit-learn has been fixed."

