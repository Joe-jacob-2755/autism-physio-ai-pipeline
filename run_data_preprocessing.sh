#!/usr/bin/env bash
# =============================================================================
#  Autism Physio-AI Pipeline
#  Data Preprocessing Launcher  (Module 3 standalone)
# =============================================================================
#  Run from the repo root:
#    ./run_data_preprocessing.sh
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
MAIN="$REPO_ROOT/pipeline_main.py"

if [ ! -f "$VENV_PYTHON" ]; then
    echo ""
    echo " [ERROR] Virtual environment not found."
    echo " Please run:  ./scripts/setup.sh"
    echo ""
    exit 1
fi

clear
"$VENV_PYTHON" "$MAIN" --mode preprocess "$@"
