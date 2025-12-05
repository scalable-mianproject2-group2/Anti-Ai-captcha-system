@echo off
REM ================================================
REM Anti-AI-Captcha System - One-Click Launcher
REM For Windows (runme.bat)
REM ================================================

echo.
echo ================================================
echo    ANTI-AI-CAPTCHA SYSTEM - STARTUP
echo ================================================
echo.

REM 1. Check Python installation
echo [1/5] Checking Python environment...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.8+ from: https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
python --version

REM 2. Check/Create virtual environment
echo.
echo [2/5] Setting up virtual environment...
if not exist "venv\" (
    echo Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        echo Try: pip install virtualenv
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

REM 3. Activate virtual environment
echo.
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo Virtual environment activated.

REM 4. Install/Upgrade dependencies
echo.
echo [4/5] Installing dependencies...
if exist "requirements.txt" (
    echo Installing packages from requirements.txt...
    pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [WARNING] Some dependencies failed to install!
        echo Trying to install common dependencies individually...
        pip install Flask opencv-python numpy selenium requests
    )
) else (
    echo No requirements.txt found!
    echo Installing basic dependencies...
    pip install Flask opencv-python numpy selenium requests
)

REM 5. Run the application
echo.
echo [5/5] Starting Anti-AI-Captcha System...
echo ================================================
echo Server will start at: http://localhost:5000
echo Press Ctrl+C to stop the server.
echo ================================================
echo.
timeout /t 2 /nobreak >nul

python combined_app.py

REM If the app exits, pause to show any error messages
echo.
echo ================================================
echo Application has stopped.
echo ================================================
pause