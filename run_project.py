#!/usr/bin/env python3
"""
Simple runner script to execute main.py with proper logging
"""

import subprocess
import sys
import time
from datetime import datetime

print("="*80)
print("DIABETES READMISSION PREDICTION - EXECUTION RUNNER")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()

try:
    # Run main.py
    print("Executing main.py...")
    print("-"*80)

    result = subprocess.run(
        [sys.executable, 'main.py'],
        capture_output=False,
        text=True,
        check=True
    )

    print()
    print("="*80)
    print("✅ EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    print("Check the outputs/ folder for all results!")

except subprocess.CalledProcessError as e:
    print()
    print("="*80)
    print("❌ EXECUTION FAILED!")
    print(f"Error code: {e.returncode}")
    print("="*80)
    sys.exit(1)

except KeyboardInterrupt:
    print()
    print("="*80)
    print("⚠️  EXECUTION INTERRUPTED BY USER")
    print("="*80)
    sys.exit(1)

except Exception as e:
    print()
    print("="*80)
    print(f"❌ UNEXPECTED ERROR: {e}")
    print("="*80)
    sys.exit(1)

