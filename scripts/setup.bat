@echo off
REM =============================================================================
REM Autism Physio-AI Pipeline — Local Environment Setup (Windows)
REM =============================================================================
setlocal EnableDelayedExpansion

set REPO_ROOT=%~dp0..
set MODULE_DIR=%REPO_ROOT%
set VENV_DIR=%REPO_ROOT%\.venv

echo.
echo =====================================================
echo   Autism Physio-AI Pipeline -- Environment Setup
echo =====================================================
echo   Repository : %REPO_ROOT%
echo.

REM Step 1: Check Python
echo [Step 1] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    echo         Make sure to check "Add Python to PATH" during installation.
    pause & exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VER=%%v
echo [INFO]  Found Python %PYTHON_VER%

REM Step 2: Create virtual environment
echo.
echo [Step 2] Creating virtual environment...
if exist "%VENV_DIR%" (
    echo [WARN]  .venv already exists -- skipping creation
) else (
    python -m venv "%VENV_DIR%"
    echo [INFO]  .venv created
)

REM Step 3: Upgrade pip
echo.
echo [Step 3] Upgrading pip...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet
echo [INFO]  pip upgraded

REM Step 4: Install ALL dependencies from root requirements.txt
echo.
echo [Step 4] Installing all dependencies...
"%VENV_DIR%\Scripts\pip.exe" install -r "%REPO_ROOT%\requirements.txt"
if errorlevel 1 (
    echo [ERROR] Dependency installation failed
    pause & exit /b 1
)
echo [INFO]  All dependencies installed

REM Step 5: Verify key packages
echo.
echo [Step 5] Verifying packages...
"%VENV_DIR%\Scripts\python.exe" -c "import numpy, scipy, pandas, matplotlib, sklearn, antropy; print('[INFO]  All packages verified OK')"
if errorlevel 1 (
    echo [ERROR] Package verification failed
    pause & exit /b 1
)

REM Step 6: Smoke test
echo.
echo [Step 6] Running smoke test...
cd "%REPO_ROOT%\module_1a_data_simulation"
"%VENV_DIR%\Scripts\python.exe" main.py --duration 60 --n_events 2 --noise low --seed 42 --no_plots --out output\smoke_test
if errorlevel 1 (
    echo [ERROR] Smoke test failed
    pause & exit /b 1
)
echo [INFO]  Smoke test passed

echo.
echo =====================================================
echo   Setup Complete!
echo =====================================================
echo.
echo   To activate:   .venv\Scripts\activate
echo   Full pipeline:  run_full_pipeline
echo   Preprocessing:  run_data_preprocessing
echo   Acquisition:    run_data_acquisition_module
echo.
pause
