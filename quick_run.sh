#!/bin/bash
# Simple execution script with progress monitoring

cd "/Users/ojasbayas/vac cp"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "                 DIABETES READMISSION PREDICTION - QUICK RUN"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Optimized for FAST execution: ~8-10 minutes"
echo "  Hyperparameter tuning reduced: 30 → 5 iterations"
echo "  Sleep prevention: ACTIVE"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Prevent sleep and run
echo "Starting execution with sleep prevention..."
echo ""

caffeinate -d python3 main.py

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ EXECUTION COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Check your results:"
echo "  cd outputs/"
echo "  cat final_report.txt"
echo ""

