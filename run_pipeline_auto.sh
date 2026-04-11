#!/usr/bin/env bash
# Automated Full Pipeline Runner (all modules, no prompts)
# Usage: ./run_pipeline_auto.sh
#        ./run_pipeline_auto.sh --n_users 10 --seed 42
#        ./run_pipeline_auto.sh --n_users 5 --no_plots
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
MAIN="$REPO_ROOT/run_pipeline_auto.py"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[ERROR] Virtual environment not found."
    echo "Please run: ./scripts/setup.sh"
    exit 1
fi

clear
"$VENV_PYTHON" "$MAIN" "$@"
