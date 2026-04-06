#!/usr/bin/env bash
# =============================================================================
# Autism Physio-AI Pipeline — Local Environment Setup (Mac / Linux)
# =============================================================================
# Run from the repository root:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
# =============================================================================

set -e  # Exit immediately on any error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
MODULE_DIR="$REPO_ROOT/module_1a_data_simulation"
PYTHON_MIN="3.10"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${GREEN}══════════════════════════════════════${NC}"; echo -e "${GREEN}  $*${NC}"; echo -e "${GREEN}══════════════════════════════════════${NC}"; }

section "Autism Physio-AI Pipeline — Environment Setup"
echo "  Repository : $REPO_ROOT"
echo "  Module     : $MODULE_DIR"

# ── 1. Check Python version ───────────────────────────────────────────────────
section "Step 1 — Checking Python"

PYTHON_CMD=""
for cmd in python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    MAJOR=$(echo "$VER" | cut -d. -f1)
    MINOR=$(echo "$VER" | cut -d. -f2)
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
      PYTHON_CMD="$cmd"
      info "Found $cmd $VER ✓"
      break
    else
      warn "$cmd $VER is below minimum $PYTHON_MIN — skipping"
    fi
  fi
done

[ -z "$PYTHON_CMD" ] && error "Python $PYTHON_MIN+ not found. Install from https://python.org"

# ── 2. Create virtual environment ─────────────────────────────────────────────
section "Step 2 — Creating Virtual Environment"

if [ -d "$VENV_DIR" ]; then
  warn ".venv already exists — skipping creation (delete it manually to recreate)"
else
  info "Creating .venv with $PYTHON_CMD ..."
  "$PYTHON_CMD" -m venv "$VENV_DIR"
  info ".venv created at $VENV_DIR ✓"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
section "Step 3 — Upgrading pip"
"$VENV_PYTHON" -m pip install --upgrade pip --quiet
info "pip upgraded ✓"

# ── 4. Install dependencies ───────────────────────────────────────────────────
section "Step 4 — Installing Dependencies"
info "Installing from $MODULE_DIR/requirements.txt ..."
"$VENV_PIP" install -r "$MODULE_DIR/requirements.txt"
info "All dependencies installed ✓"

# ── 5. Verify installation ────────────────────────────────────────────────────
section "Step 5 — Verifying Installation"

PACKAGES=("numpy" "scipy" "pandas" "matplotlib" "seaborn")
ALL_OK=true
for pkg in "${PACKAGES[@]}"; do
  if "$VENV_PYTHON" -c "import $pkg; print(f'  ✓ {\"$pkg\":12s} {$pkg.__version__}')" 2>/dev/null; then
    :
  else
    warn "  ✗ $pkg — import failed"
    ALL_OK=false
  fi
done

$ALL_OK || error "Some packages failed to import. Re-run this script."

# ── 6. Quick smoke test ───────────────────────────────────────────────────────
section "Step 6 — Smoke Test"
info "Running a 60-second simulation to verify everything works..."

cd "$MODULE_DIR"
"$VENV_PYTHON" main.py \
  --duration 60 \
  --n_events 2 \
  --event_dur 15 \
  --noise medium \
  --seed 42 \
  --out output/smoke_test 2>&1

if [ $? -eq 0 ]; then
  info "Smoke test passed ✓"
  info "Output written to: $MODULE_DIR/output/smoke_test/"
else
  error "Smoke test failed — check the error above"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
section "Setup Complete ✅"
echo ""
echo "  To activate the environment:"
echo "    source .venv/bin/activate"
echo ""
echo "  To run the simulator:"
echo "    cd module_1a_data_simulation"
echo "    python main.py --help"
echo ""
echo "  To open in VS Code:"
echo "    code ."
echo ""
echo "  VS Code: Press Ctrl+Shift+P → 'Python: Select Interpreter'"
echo "           Choose:  .venv/bin/python"
echo ""
