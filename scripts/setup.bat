@echo off
REM =============================================================================
REM Autism Physio-AI Pipeline — Local Environment Setup (Windows)
REM =============================================================================
REM Run from the repository root in Command Prompt or PowerShell:
REM   scripts\setup.bat
REM =============================================================================

setlocal EnableDelayedExpansion

set REPO_ROOT=%~dp0..
set MODULE_DIR=%REPO_ROOT%\module_1a_data_simulation
set VENV_DIR=%REPO_ROOT%\.venv

echo.
echo =====================================================
echo   Autism Physio-AI Pipeline -- Environment Setup
echo =====================================================
echo   Repository : %REPO_ROOT%
echo   Module     : %MODULE_DIR%
echo.

REM ── Step 1: Check Python ──────────────────────────────────────────────────
echo [Step 1] Checking Python installation...

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VER=%%v
echo [INFO]  Found Python %PYTHON_VER%

REM ── Step 2: Create virtual environment ───────────────────────────────────
echo.
echo [Step 2] Creating virtual environment...

if exist "%VENV_DIR%" (
    echo [WARN]  .venv already exists -- skipping creation
    echo         Delete .venv manually to recreate it
) else (
    python -m venv "%VENV_DIR%"
    echo [INFO]  .venv created at %VENV_DIR%
)

REM ── Step 3: Upgrade pip ───────────────────────────────────────────────────
echo.
echo [Step 3] Upgrading pip...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet
echo [INFO]  pip upgraded

REM ── Step 4: Install dependencies ─────────────────────────────────────────
echo.
echo [Step 4] Installing dependencies...
"%VENV_DIR%\Scripts\pip.exe" install -r "%MODULE_DIR%\requirements.txt"
if errorlevel 1 (
    echo [ERROR] Dependency installation failed
    pause
    exit /b 1
)
echo [INFO]  Dependencies installed

REM ── Step 5: Verify packages ───────────────────────────────────────────────
echo.
echo [Step 5] Verifying packages...
"%VENV_DIR%\Scripts\python.exe" -c "import numpy, scipy, pandas, matplotlib, seaborn; print('[INFO]  All packages verified OK')"
if errorlevel 1 (
    echo [ERROR] Package verification failed
    pause
    exit /b 1
)

REM ── Step 6: Smoke test ────────────────────────────────────────────────────
echo.
echo [Step 6] Running smoke test (60-second simulation)...

cd "%MODULE_DIR%"
"%VENV_DIR%\Scripts\python.exe" main.py ^
    --duration 60 ^
    --n_events 2 ^
    --event_dur 15 ^
    --noise medium ^
    --seed 42 ^
    --out output\smoke_test

if errorlevel 1 (
    echo [ERROR] Smoke test failed
    pause
    exit /b 1
)
echo [INFO]  Smoke test passed

REM ── Done ──────────────────────────────────────────────────────────────────
echo.
echo =====================================================
echo   Setup Complete!
echo =====================================================
echo.
echo   To activate the environment:
echo     .venv\Scripts\activate
echo.
echo   To run the simulator:
echo     cd module_1a_data_simulation
echo     python main.py --help
echo.
echo   To open in VS Code:
echo     code .
echo.
echo   VS Code: Press Ctrl+Shift+P ^> "Python: Select Interpreter"
echo            Choose:  .venv\Scripts\python.exe
echo.
pause
