#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
[ ! -f "$VENV_PYTHON" ] && echo "[ERROR] Run ./scripts/setup.sh first" && exit 1
clear
"$VENV_PYTHON" "$REPO_ROOT/module_3_preprocessing/main.py" "$@"
