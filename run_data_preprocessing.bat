@echo off
REM =============================================================================
REM  Autism Physio-AI Pipeline
REM  Data Preprocessing Launcher  (Module 3 standalone)
REM =============================================================================
REM  Run from the repo root:
REM    run_data_preprocessing
REM =============================================================================
setlocal

set "REPO_ROOT=%~dp0"
set "VENV_PYTHON=%REPO_ROOT%.venv\Scripts\python.exe"
set "MAIN=%REPO_ROOT%pipeline_main.py"

if not exist "%VENV_PYTHON%" (
    echo.
    echo  [ERROR] Virtual environment not found.
    echo  Please run:  scripts\setup.bat
    echo.
    pause & exit /b 1
)

cls
"%VENV_PYTHON%" "%MAIN%" --mode preprocess %*

if errorlevel 1 (
    echo.
    echo  [ERROR] Preprocessing exited with an error. Check the output above.
    pause
)
endlocal
