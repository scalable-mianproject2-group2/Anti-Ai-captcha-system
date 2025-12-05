#!/bin/bash

# ================================================
# Anti-AI-Captcha System - One-Click Launcher
# For Linux/Mac (runme.sh)
# ================================================

clear
echo "================================================"
echo "   ANTI-AI-CAPTCHA SYSTEM - STARTUP"
echo "================================================"
echo ""

# 1. Check Python installation
echo "[1/5] Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found!"
    echo "Please install Python 3.8+:"
    echo "  Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "  macOS: brew install python"
    echo "  Or download from: https://python.org"
    exit 1
fi
python3 --version

# 2. Check/Create virtual environment
echo ""
echo "[2/5] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment!"
        echo "Try: pip3 install virtualenv"
        exit 1
    fi
    echo "Virtual environment created successfully."
else
    echo "Virtual environment already exists."
fi

# 3. Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment!"
    exit 1
fi
echo "Virtual environment activated."

# 4. Install/Upgrade dependencies
echo ""
echo "[4/5] Installing dependencies..."
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[WARNING] Some dependencies failed to install!"
        echo "Trying to install common dependencies individually..."
        pip install Flask opencv-python numpy selenium requests
    fi
else
    echo "No requirements.txt found!"
    echo "Installing basic dependencies..."
    pip install Flask opencv-python numpy selenium requests
fi

# 5. Run the application
echo ""
echo "[5/5] Starting Anti-AI-Captcha System..."
echo "================================================"
echo "Server will start at: http://localhost:5000"
echo "Press Ctrl+C to stop the server."
echo "================================================"
echo ""
sleep 2

python3 combined_app.py

# If the app exits
echo ""
echo "================================================"
echo "Application has stopped."
echo "================================================"