#!/usr/bin/env bash
# =============================================================================
# Autism Physio-AI Pipeline — Local Environment Setup (Mac / Linux)
# =============================================================================
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
GREEN='\033[0;32m'; NC='\033[0m'
info() { echo -e "${GREEN}[INFO]${NC}  $*"; }

echo "======================================================"
echo "  Autism Physio-AI Pipeline — Environment Setup"
echo "======================================================"

# Check Python
PYTHON_CMD=""
for cmd in python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    MAJOR=$(echo "$VER" | cut -d. -f1); MINOR=$(echo "$VER" | cut -d. -f2)
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
      PYTHON_CMD="$cmd"; info "Found $cmd $VER"; break
    fi
  fi
done
[ -z "$PYTHON_CMD" ] && echo "ERROR: Python 3.10+ not found" && exit 1

# Create venv
[ -d "$VENV_DIR" ] && info ".venv already exists — skipping" || \
  ("$PYTHON_CMD" -m venv "$VENV_DIR" && info ".venv created")

# Upgrade pip
"$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet
info "pip upgraded"

# Install ALL dependencies from root requirements.txt
info "Installing all dependencies from requirements.txt ..."
"$VENV_DIR/bin/pip" install -r "$REPO_ROOT/requirements.txt"
info "All dependencies installed"

# Verify
"$VENV_DIR/bin/python" -c \
  "import numpy, scipy, pandas, matplotlib, sklearn, antropy; print('All packages verified OK')"

# Smoke test
info "Running smoke test ..."
cd "$REPO_ROOT/module_2a_data_simulation"
"$VENV_DIR/bin/python" main.py --duration 60 --n_events 2 --noise low \
  --seed 42 --no_plots --out output/smoke_test 2>&1
info "Smoke test passed"

echo ""
echo "======================================================"
echo "  Setup Complete!"
echo "======================================================"
echo "  Activate:      source .venv/bin/activate"
echo "  Full pipeline: ./run_full_pipeline.sh"
echo "  Preprocessing: ./run_data_preprocessing.sh"
echo "======================================================"
