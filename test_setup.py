"""
Quick Test Script to Validate Project Setup
This script performs a quick validation to ensure all modules can be imported
and basic functionality works before running the full pipeline.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing module imports...")

    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False

    try:
        import pandas as pd
        print("  ✓ pandas")
    except ImportError as e:
        print(f"  ✗ pandas: {e}")
        return False

    try:
        import matplotlib.pyplot as plt
        print("  ✓ matplotlib")
    except ImportError as e:
        print(f"  ✗ matplotlib: {e}")
        return False

    try:
        import seaborn as sns
        print("  ✓ seaborn")
    except ImportError as e:
        print(f"  ✗ seaborn: {e}")
        return False

    try:
        import sklearn
        print("  ✓ scikit-learn")
    except ImportError as e:
        print(f"  ✗ scikit-learn: {e}")
        return False

    try:
        import xgboost
        print("  ✓ xgboost")
    except ImportError as e:
        print(f"  ✗ xgboost: {e}")
        return False

    try:
        import lightgbm
        print("  ✓ lightgbm")
    except ImportError as e:
        print(f"  ✗ lightgbm: {e}")
        return False

    try:
        import catboost
        print("  ✓ catboost")
    except ImportError as e:
        print(f"  ✗ catboost: {e}")
        return False

    try:
        import imblearn
        print("  ✓ imbalanced-learn")
    except ImportError as e:
        print(f"  ✗ imbalanced-learn: {e}")
        return False

    return True

def test_custom_modules():
    """Test if custom modules can be imported"""
    print("\nTesting custom modules...")

    try:
        import utils
        print("  ✓ utils.py")
    except ImportError as e:
        print(f"  ✗ utils.py: {e}")
        return False

    try:
        import data_preprocess
        print("  ✓ data_preprocess.py")
    except ImportError as e:
        print(f"  ✗ data_preprocess.py: {e}")
        return False

    try:
        import models
        print("  ✓ models.py")
    except ImportError as e:
        print(f"  ✗ models.py: {e}")
        return False

    try:
        import clustering
        print("  ✓ clustering.py")
    except ImportError as e:
        print(f"  ✗ clustering.py: {e}")
        return False

    return True

def test_data_file():
    """Test if data file exists"""
    print("\nTesting data file...")

    if os.path.exists('diabetic_data.csv'):
        file_size = os.path.getsize('diabetic_data.csv') / (1024 * 1024)  # MB
        print(f"  ✓ diabetic_data.csv found ({file_size:.2f} MB)")
        return True
    else:
        print("  ✗ diabetic_data.csv not found")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("PROJECT SETUP VALIDATION")
    print("="*60)

    tests_passed = True

    # Test imports
    if not test_imports():
        tests_passed = False

    # Test custom modules
    if not test_custom_modules():
        tests_passed = False

    # Test data file
    if not test_data_file():
        tests_passed = False

    print("\n" + "="*60)
    if tests_passed:
        print("✅ ALL TESTS PASSED - Ready to run main.py")
        print("="*60)
        print("\nTo run the full project, execute:")
        print("  python main.py")
    else:
        print("❌ SOME TESTS FAILED - Please fix the issues above")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()

