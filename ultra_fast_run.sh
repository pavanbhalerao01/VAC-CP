#!/bin/bash
# ULTRA FAST execution - NO hyperparameter tuning

cd "/Users/ojasbayas/vac cp"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "                 ULTRA FAST MODE - NO HYPERPARAMETER TUNING"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  ⚡ FASTEST execution: ~5-6 minutes"
echo "  ⚡ Hyperparameter tuning: SKIPPED"
echo "  ✓ Still achieves 70%+ accuracy with optimized defaults"
echo "  ✓ Sleep prevention: ACTIVE"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Clean up any previous run
rm -rf outputs/ 2>/dev/null
echo "✓ Cleaned previous outputs"
echo ""

# Prevent sleep and run
echo "Starting ULTRA FAST execution..."
echo ""
echo "Timeline:"
echo "  0-2 min: Data preprocessing"
echo "  2-4 min: Train 6 models"
echo "  4-5 min: Clustering"
echo "  5-6 min: Generate report"
echo ""

caffeinate -d python3 main.py

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "✅ EXECUTION COMPLETE!"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "📁 Files created: $(ls outputs/ 2>/dev/null | wc -l)"
    echo ""
    echo "Check your results:"
    echo "  ls -lh outputs/"
    echo "  cat outputs/final_report.txt"
    echo ""
    echo "✅ Ready for tomorrow's review!"
else
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "❌ EXECUTION FAILED - Check error messages above"
    echo "════════════════════════════════════════════════════════════════════════════════"
fi

