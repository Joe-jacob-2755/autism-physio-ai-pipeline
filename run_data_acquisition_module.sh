#!/usr/bin/env bash
# =============================================================================
#  Autism Physio-AI Pipeline
#  Data Acquisition Module — Launcher
# =============================================================================
#  Run from the repo root:
#    ./run_data_acquisition_module.sh
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
MODULE_DIR="$REPO_ROOT/module_2_data_acquisition"
MAIN_SCRIPT="$MODULE_DIR/main.py"

# Check virtual environment
if [ ! -f "$VENV_PYTHON" ]; then
    echo ""
    echo " [ERROR] Virtual environment not found."
    echo " Please run:  ./scripts/setup.sh"
    echo ""
    exit 1
fi

# Check module exists
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo ""
    echo " [ERROR] module_2_data_acquisition/main.py not found."
    echo " Please run:  git pull"
    echo ""
    exit 1
fi

# Launch
clear
"$VENV_PYTHON" "$MAIN_SCRIPT"
