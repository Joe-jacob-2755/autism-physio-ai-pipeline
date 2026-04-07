# Local Development Setup Guide

Step-by-step instructions to get a fully working VS Code environment on your machine.

---

## Prerequisites

| Tool | Minimum Version | Download |
|------|----------------|---------|
| Python | 3.10+ | https://python.org/downloads |
| Git | Any recent | https://git-scm.com |
| VS Code | Any recent | https://code.visualstudio.com |

---

## Step 1 — Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/Joe-jacob-2755/autism-physio-ai-pipeline.git
cd autism-physio-ai-pipeline
```

---

## Step 2 — Run the Setup Script

**Mac / Linux:**
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

**Windows (Command Prompt):**
```bat
scripts\setup.bat
```

The script will:
- ✅ Check your Python version
- ✅ Create a `.venv` virtual environment
- ✅ Install all dependencies
- ✅ Verify all packages imported correctly
- ✅ Run a 60-second smoke test to confirm everything works

---

## Step 3 — Open in VS Code

```bash
code .
```

---

## Step 4 — Select the Python Interpreter

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type `Python: Select Interpreter`
3. Choose the one that shows `.venv` in the path:
   - **Mac/Linux:** `.venv/bin/python`
   - **Windows:** `.venv\Scripts\python.exe`

VS Code will now use the correct environment for linting, debugging, and the terminal.

---

## Step 5 — Install Recommended Extensions

VS Code will show a popup: **"Do you want to install the recommended extensions?"**

Click **Install All**. This installs:
- Python + Debugpy (core Python support)
- Black formatter + Flake8 linter
- Jupyter notebooks
- CSV viewer
- GitLens
- Markdown preview

If the popup doesn't appear: `Ctrl+Shift+P` → `Extensions: Show Recommended Extensions`

---

## Step 6 — Run Your First Simulation

### Option A — Using the Run & Debug panel (F5)

1. Press `F5` or click the **Run & Debug** icon in the left sidebar
2. Select a configuration from the dropdown at the top, e.g.:
   - `▶ Run: Default Simulation`
   - `▶ Run: Anger (single emotion)`
   - `▶ Run: High Arousal (Fear, Anger, Surprise)`
3. Press the green play button

### Option B — Using Tasks (Ctrl+Shift+P)

1. Press `Ctrl+Shift+P`
2. Type `Tasks: Run Task`
3. Choose any task, e.g. `▶ Simulate: Default (5 min, random)`

### Option C — Using the integrated terminal

```bash
# Activate environment (if not already active)
source .venv/bin/activate       # Mac/Linux
.venv\Scripts\activate          # Windows

cd module_2a_data_simulation
python main.py --duration 300 --n_events 5 --noise medium --seed 42
```

---

## Step 7 — View the Outputs

After a simulation runs, output files appear in:

```
module_2a_data_simulation/output/<run_name>/
├── combined_signals.png    ← Open this to see all signals
├── signal_EDA.png
├── signal_BVP.png
├── signal_IBI.png
├── signal_ST.png
├── signal_ACC_combined.png
├── combined_signals.csv
├── EDA.csv
├── annotations_events.csv
└── metadata.json
```

Click any `.png` file in the Explorer panel to view it directly inside VS Code.
Click any `.csv` to view it as a spreadsheet (with the Excel Viewer extension).

---

## Step 8 — Run the Tests

```bash
# In the integrated terminal
python -m pytest tests/ -v --tb=short
```

Or use the **Testing** panel (beaker icon in the left sidebar) to run and debug individual tests.

---

## Available Debug Configurations (F5 dropdown)

| Configuration | What it does |
|---|---|
| `▶ Run: Default Simulation` | 5 min, 5 random events, medium noise |
| `▶ Run: Anger (single emotion)` | 4 min, 4 Anger events, high noise |
| `▶ Run: High Arousal (Fear, Anger, Surprise)` | 6 min, mixed high-arousal events |
| `▶ Run: Physiological Needs` | 8 min, needs states with random durations |
| `▶ Run: Fully Random` | Everything random, low noise |
| `▶ Run: CSV Only (no plots)` | 10 min, no plots generated (fast) |
| `▶ List All Emotion Labels` | Print all 10 available states |
| `🐛 Debug: Current File` | Debug whichever file is open |
| `🧪 Run Tests` | Run the full test suite |

---

## Troubleshooting

**`python: command not found`**
- Install Python 3.10+ from https://python.org
- On Windows, ensure "Add Python to PATH" is checked during installation

**VS Code shows wrong Python interpreter**
- Press `Ctrl+Shift+P` → `Python: Select Interpreter` → choose `.venv`

**`ModuleNotFoundError`**
- Make sure the virtual environment is active and the setup script completed successfully
- Re-run: `./scripts/setup.sh`

**Plots not showing / blank figures**
- This is expected when running non-interactively — plots are saved as PNG files, not displayed in a window. Check the `output/` directory.

**Tests failing**
- Make sure `pytest` is installed: `.venv/bin/pip install pytest`
- Run from the repo root, not from inside the module directory

---

## Project Structure Reference

```
autism-physio-ai-pipeline/
├── .vscode/
│   ├── settings.json       ← Python interpreter, formatter, linting
│   ├── launch.json         ← F5 debug configurations
│   ├── tasks.json          ← Ctrl+Shift+P task runner
│   └── extensions.json     ← Recommended extensions
├── scripts/
│   ├── setup.sh            ← Mac/Linux one-shot setup
│   └── setup.bat           ← Windows one-shot setup
├── tests/
│   └── test_module_1a.py   ← Full test suite
├── module_2a_data_simulation/
│   ├── main.py             ← Entry point
│   ├── config.py           ← All configuration
│   ├── simulator.py        ← Master pipeline
│   ├── signal_models.py    ← Signal generation
│   ├── event_scheduler.py  ← Event timing
│   ├── noise_injector.py   ← Noise injection
│   ├── annotator.py        ← Auto-annotation
│   ├── visualizer.py       ← PNG generation
│   ├── exporter.py         ← CSV/JSON export
│   └── requirements.txt    ← Dependencies
└── SETUP_GUIDE.md          ← This file
```
