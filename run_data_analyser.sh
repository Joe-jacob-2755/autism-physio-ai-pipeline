#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ ! -f "$REPO_ROOT/.venv/bin/python" ] && echo "[ERROR] Run ./scripts/setup.sh first" && exit 1
clear
"$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/pipeline_main.py" --mode analyse "$@"
