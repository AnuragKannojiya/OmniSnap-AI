@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo OmniSnap Setup for Snapdragon PCs (Windows ARM64)
echo ========================================================

:: 1. Check Python installation
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python for ARM64 and try again.
    exit /b 1
)

:: 2. Create Virtual Environment
echo [INFO] Creating Python virtual environment...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)

:: 3. Activate Virtual Environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: 4. Install Dependencies
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing standard dependencies from requirements.txt...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install standard dependencies.
    exit /b 1
)

:: 5. Install Snapdragon NPU Acceleration (ONNX Runtime QNN)
echo [INFO] Installing ONNX Runtime QNN for Snapdragon NPU...
:: Attempting to install the QNN package. Ensure that pip uses the right wheels for ARM64 if available.
pip install onnxruntime-qnn>=1.16.0
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to install onnxruntime-qnn. Execution will fallback to CPU/DirectML.
) else (
    echo [INFO] Successfully installed onnxruntime-qnn.
)

:: 6. Run System Check Report
echo.
echo ========================================================
echo Running Snapdragon Readiness Report...
echo ========================================================
python deploy_snapdragon.py

:: 7. Launch App (Placeholder for actual app launch script)
echo.
echo ========================================================
echo Setup Complete.
echo You can now launch OmniSnap.
echo Example: python main.py or uvicorn app:app --reload
echo ========================================================

endlocal
