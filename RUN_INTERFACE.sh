#!/bin/bash

##############################################################################
# Hybrid RAG System - Interface Launcher
# Quick script to start the Streamlit web interface
##############################################################################

echo "=================================="
echo "Hybrid RAG System - Web Interface"
echo "=================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Current directory: $SCRIPT_DIR"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "  Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "  Python found: $(python3 --version)"

# Check if Streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo ""
    echo "   Streamlit not installed. Installing dependencies..."
    pip install streamlit transformers faiss-cpu sentence-transformers
    echo "  Dependencies installed"
fi

echo ""
echo "=================================="
echo "Starting Streamlit Application"
echo "=================================="
echo ""
echo "  Launching app at http://localhost:8501"
echo ""
echo "  To stop: Press Ctrl+C"
echo ""

# Run Streamlit app
streamlit run interface/app.py

##############################################################################
# Exit
##############################################################################
