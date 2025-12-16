#!/usr/bin/env python3
"""
Verify deployment readiness for Vertex AI Agent Engine.
Checks for common issues that cause build failures.
"""

import sys
import os
from pathlib import Path

def check_imports():
    """Check if all critical imports work."""
    print("Checking imports...")
    errors = []
    
    try:
        from src.api.main import app
        print("✓ src.api.main import successful")
    except Exception as e:
        errors.append(f"✗ Failed to import src.api.main: {e}")
    
    try:
        from src.utils.config import settings
        print("✓ src.utils.config import successful")
    except Exception as e:
        errors.append(f"✗ Failed to import src.utils.config: {e}")
    
    try:
        from src.data.bigquery_client import BigQueryClient
        print("✓ src.data.bigquery_client import successful")
    except Exception as e:
        errors.append(f"✗ Failed to import src.data.bigquery_client: {e}")
    
    try:
        from src.api.routes import chatbot
        print("✓ src.api.routes.chatbot import successful")
    except Exception as e:
        errors.append(f"✗ Failed to import chatbot: {e}")
    
    return errors

def check_syntax():
    """Check Python syntax of all files."""
    print("\nChecking syntax...")
    import py_compile
    errors = []
    
    src_path = Path("src")
    if not src_path.exists():
        errors.append("✗ src/ directory not found")
        return errors
    
    for py_file in src_path.rglob("*.py"):
        try:
            py_compile.compile(str(py_file), doraise=True)
            print(f"✓ {py_file}")
        except py_compile.PyCompileError as e:
            errors.append(f"✗ Syntax error in {py_file}: {e}")
    
    return errors

def check_requirements():
    """Check if requirements file exists."""
    print("\nChecking requirements...")
    errors = []
    
    if not Path("requirements.txt").exists():
        errors.append("✗ requirements.txt not found")
    else:
        print("✓ requirements.txt exists")
        
        # Check for problematic packages
        with open("requirements.txt") as f:
            content = f.read()
            problematic = ['arch==', 'prophet==', 'shap==', 'pmdarima==']
            found = [pkg for pkg in problematic if pkg in content]
            if found:
                print(f"⚠ Warning: Found potentially problematic packages: {', '.join(found)}")
                print("  Consider using requirements-deploy.txt instead")
    
    return errors

def check_structure():
    """Check project structure."""
    print("\nChecking project structure...")
    errors = []
    
    required = [
        "src/api/main.py",
        "src/utils/config.py",
        "requirements.txt",
    ]
    
    for path in required:
        if Path(path).exists():
            print(f"✓ {path} exists")
        else:
            errors.append(f"✗ {path} not found")
    
    return errors

def main():
    """Run all checks."""
    print("=" * 60)
    print("Deployment Verification for Vertex AI Agent Engine")
    print("=" * 60)
    
    all_errors = []
    
    # Run checks
    all_errors.extend(check_structure())
    all_errors.extend(check_syntax())
    all_errors.extend(check_requirements())
    all_errors.extend(check_imports())
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if all_errors:
        print(f"\n✗ Found {len(all_errors)} issue(s):")
        for error in all_errors:
            print(f"  {error}")
        print("\n⚠ Please fix these issues before deploying.")
        return 1
    else:
        print("\n✓ All checks passed! Ready for deployment.")
        print("\nRecommendations:")
        print("1. Use requirements-deploy.txt for deployment")
        print("2. Set environment variables in Vertex AI console")
        print("3. Entry point: src.api.main:app")
        return 0

if __name__ == "__main__":
    sys.exit(main())

