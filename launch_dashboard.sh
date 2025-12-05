#!/bin/bash
# Quick launch script for the Streamlit dashboard

cd "/Users/ojasbayas/vac cp"

echo "════════════════════════════════════════════════════════════════════════════════"
echo "           🏥 DIABETES READMISSION PREDICTION DASHBOARD 🏥"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Starting Streamlit dashboard..."
echo ""
echo "The dashboard will open automatically in your browser."
echo "If not, navigate to: http://localhost:8501"
echo ""
echo "To stop the dashboard, press Ctrl+C"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

streamlit run app.py

