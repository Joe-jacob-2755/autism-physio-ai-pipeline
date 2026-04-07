@echo off
set "REPO_ROOT=%~dp0"
set "VENV_PYTHON=%REPO_ROOT%.venv\Scripts\python.exe"
set "MAIN=%REPO_ROOT%module_1_data_acquisition\main.py"
if not exist "%VENV_PYTHON%" (echo [ERROR] Run scripts\setup.bat first && pause && exit /b 1)
cls
"%VENV_PYTHON%" "%MAIN%" %*
if errorlevel 1 (echo. && echo [ERROR] Module exited with an error. && pause)
