@echo off
set "REPO_ROOT=%~dp0"
set "VENV_PYTHON=%REPO_ROOT%.venv\Scripts\python.exe"
set "MAIN=%REPO_ROOT%pipeline_main.py"
if not exist "%VENV_PYTHON%" (echo [ERROR] Run scripts\setup.bat first && pause && exit /b 1)
cls
"%VENV_PYTHON%" "%MAIN%" --mode analyse %*
if errorlevel 1 (echo. && echo [ERROR] Analyser exited with an error. && pause)
