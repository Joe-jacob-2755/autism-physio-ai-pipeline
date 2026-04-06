@echo off
REM =============================================================================
REM  Autism Physio-AI Pipeline
REM  Data Acquisition Module — Launcher
REM =============================================================================
REM  Run this file from the repo root to start Module 2.
REM  Double-click it, or type:  run_data_acquisition_module
REM =============================================================================

setlocal

REM ── Locate the repo root (same folder as this script) ──────────────────────
set "REPO_ROOT=%~dp0"
set "VENV_PYTHON=%REPO_ROOT%.venv\Scripts\python.exe"
set "MODULE_DIR=%REPO_ROOT%module_2_data_acquisition"
set "MAIN_SCRIPT=%MODULE_DIR%\main.py"

REM ── Check virtual environment exists ────────────────────────────────────────
if not exist "%VENV_PYTHON%" (
    echo.
    echo  [ERROR] Virtual environment not found.
    echo.
    echo  Please run the setup script first:
    echo    scripts\setup.bat
    echo.
    pause
    exit /b 1
)

REM ── Check module exists ──────────────────────────────────────────────────────
if not exist "%MAIN_SCRIPT%" (
    echo.
    echo  [ERROR] module_2_data_acquisition\main.py not found.
    echo  Please ensure you have pulled the latest code:
    echo    git pull
    echo.
    pause
    exit /b 1
)

REM ── Launch ───────────────────────────────────────────────────────────────────
cls
"%VENV_PYTHON%" "%MAIN_SCRIPT%"

REM ── On exit ──────────────────────────────────────────────────────────────────
if errorlevel 1 (
    echo.
    echo  [ERROR] Module exited with an error (code %errorlevel%).
    echo  Check the output above for details.
    pause
)

endlocal
