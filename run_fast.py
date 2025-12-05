#!/usr/bin/env python3
"""
Fast Runner - Executes main.py with progress monitoring and prevents sleep
"""

import subprocess
import sys
import os
from datetime import datetime

print("="*80)
print("DIABETES READMISSION PREDICTION - FAST EXECUTION")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print()
print("⚠️  IMPORTANT: Keep this terminal window in focus to prevent sleep!")
print("   Estimated time: 8-12 minutes")
print()

# Prevent display sleep on macOS
try:
    caffeinate_process = subprocess.Popen(['caffeinate', '-d'])
    print("✓ Sleep prevention activated")
except:
    print("⚠️  Could not activate sleep prevention - please keep laptop awake!")

print("-"*80)
print()

try:
    # Run main.py
    result = subprocess.run(
        [sys.executable, 'main.py'],
        check=True
    )

    print()
    print("="*80)
    print("✅ EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    print("📁 Check the outputs/ folder for all results!")
    print("📄 Read outputs/final_report.txt for metrics")

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

finally:
    # Stop caffeinate
    try:
        caffeinate_process.terminate()
    except:
        pass

print()
print("✅ All done! You're ready for tomorrow's review!")

