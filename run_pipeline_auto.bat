@echo off
REM Automated Full Pipeline Runner (all modules, no prompts)
REM Usage: run_pipeline_auto
REM        run_pipeline_auto --n_users 10 --seed 42
REM        run_pipeline_auto --n_users 5 --no_plots
setlocal
set "REPO_ROOT=%~dp0"
set "VENV_PYTHON=%REPO_ROOT%.venv\Scripts\python.exe"
set "MAIN=%REPO_ROOT%run_pipeline_auto.py"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment not found.
    echo Please run: scripts\setup.bat
    pause & exit /b 1
)

cls
"%VENV_PYTHON%" "%MAIN%" %*

if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline exited with an error.
    pause
)
endlocal
