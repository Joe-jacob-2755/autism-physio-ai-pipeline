"""
=============================================================================
MODULE 5 – DATA PREPROCESSING  |  main.py  (v2.0.0)
=============================================================================
Interactive CLI entry point for signal cleaning and filtering.

Feature extraction, normalisation, and encoding are now in Module 4.

Launch from repo root:
  Windows:    run_preprocessing_module.bat
  Mac/Linux:  ./run_preprocessing_module.sh

Or directly:
  cd module_5_preprocessing
  python main.py
  python main.py --source ../module_2a_data_simulation/outputs/M2A_v1.1.0_run_001
  python main.py --source <path> --filter butterworth --no_plots
=============================================================================
"""
from __future__ import annotations
from config import MODULE_VERSION
from preprocessor import DataPreprocessor
import argparse
import sys
import os
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
M2_DIR = REPO_ROOT / "module_1_data_acquisition"
M2A_DIR = REPO_ROOT / "module_2a_data_simulation"

sys.path.insert(0, str(MODULE_DIR))

WIDTH = 64


def _line():
    print("─" * WIDTH)


def _header(t):
    print()
    print("═" * WIDTH)
    print(f"  {t}")
    print("═" * WIDTH)


def _section(t):
    print(f"\n  {'─' * (WIDTH - 4)}\n  {t}\n  {'─' * (WIDTH - 4)}")


def _ask(prompt, default=""):
    return input(f"\n  {prompt} [{default}]: ").strip() or str(default)


def _ask_yn(prompt, default=True):
    raw = input(f"\n  {prompt} ({'Y/n' if default else 'y/N'}): ").strip().lower()
    return default if raw == "" else raw in ("y", "yes")


def _browse_source() -> Path:
    """Browse for a Module 2 / 2A output folder to preprocess."""
    _section("Select data source")
    candidates = []

    # Module 2 outputs
    m2_out = M2_DIR / "outputs"
    if m2_out.exists():
        for d in sorted(m2_out.iterdir()):
            if d.is_dir():
                csv_count = len(list(d.rglob("*.csv")))
                if csv_count > 0:
                    candidates.append((d, f"Module 2: {d.name}  ({csv_count} CSVs)"))

    # Module 2A outputs
    m2a_out = M2A_DIR / "outputs"
    if m2a_out.exists():
        for d in sorted(m2a_out.iterdir()):
            if d.is_dir() and any((d / f).exists()
                                  for f in ("EDA.csv", "combined_signals.csv")):
                candidates.append((d, f"Module 2A: {d.name}"))

    if not candidates:
        print("  No output folders found. Enter path manually.")
    else:
        print()
        for i, (p, lbl) in enumerate(candidates, 1):
            print(f"    {i}.  {lbl}")
        print(f"    {len(candidates) + 1}.  Enter path manually")

    while True:
        raw = input("\n  Select [1]: ").strip() or "1"
        try:
            idx = int(raw) - 1
            if idx == len(candidates):
                p = Path(input("  Enter path: ").strip())
                return p if p.exists() else None
            if 0 <= idx < len(candidates):
                return candidates[idx][0]
        except ValueError:
            p = Path(raw)
            if p.exists():
                return p
        print("  Invalid selection.")


def _interactive_mode():
    """Full interactive guided session."""
    os.system("cls" if os.name == "nt" else "clear")
    print()
    print("  " + "=" * (WIDTH - 4))
    print(f"  |    AUTISM PHYSIO-AI PIPELINE" + " " * (WIDTH - 38) + "|")
    print(f"  |    Module 5 - Data Preprocessing" + " " * (WIDTH - 42) + "|")
    print(f"  |    v{MODULE_VERSION}" + " " * (WIDTH - 14) + "|")
    print("  " + "=" * (WIDTH - 4))

    # Source
    source = _browse_source()
    if source is None:
        print("  No valid source selected. Exiting.")
        return

    # Filter
    _section("Signal filtering")
    print("""
  Butterworth: Zero-phase frequency filter (low-pass / band-pass).
               Best general choice for most signals.
  Kalman:      Optimal Gaussian noise smoother.
               Best for EDA tonic and ST (slow-varying signals).
  Hampel only: Outlier removal only, no frequency filtering.
  None:        Skip all filtering (Hampel still applied by default).
""")
    filt_opts = ["butterworth", "kalman", "hampel_only", "none"]
    for i, f in enumerate(filt_opts, 1):
        print(f"    {i}.  {f}")
    f_idx = int(_ask("Select filter", "1")) - 1
    filter_type = filt_opts[max(0, min(f_idx, 3))]
    apply_hampel = _ask_yn("Apply Hampel outlier pre-filter?", default=True)

    # Plots
    _section("Visualisation")
    gen_plots = _ask_yn("Generate signal plots?", default=True)

    # Summary
    _section("Preprocessing Parameters Summary")
    print(f"""
  Source         : {source}
  Filter         : {filter_type}{' + Hampel' if apply_hampel else ''}
  Generate plots : {gen_plots}

  Note: Feature extraction and normalisation are now in Module 4.
  Run Module 4 on this output to extract features.
""")
    if not _ask_yn("Start preprocessing?", default=True):
        print("  Cancelled.")
        return

    # Run
    preprocessor = DataPreprocessor(
        filter_type=filter_type,
        apply_hampel=apply_hampel,
        generate_plots=gen_plots,
        verbose=True,
    )
    result = preprocessor.run(
        signals_input=source,
        session_id=source.name,
        user_id="unknown",
    )

    print(f"\n  Output ready for Module 4: {result['run_folder']}")


def _cli_mode():
    """Argument-driven mode."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=f"Module 5 - Data Preprocessing (Clean + Filter)  v{MODULE_VERSION}"
    )
    parser.add_argument("--source", default=None,
                        help="Path to Module 2/2A output folder or signal CSV folder")
    parser.add_argument("--filter", default="butterworth",
                        choices=["butterworth", "kalman", "hampel_only", "none"])
    parser.add_argument("--no_hampel", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--user_id", default="unknown")
    parser.add_argument("--session_id", default="cli_session")

    args = parser.parse_args()

    if not args.source:
        parser.error("--source is required in CLI mode")

    preprocessor = DataPreprocessor(
        filter_type=args.filter,
        apply_hampel=not args.no_hampel,
        generate_plots=not args.no_plots,
        verbose=True,
    )
    preprocessor.run(
        signals_input=Path(args.source),
        session_id=args.session_id,
        user_id=args.user_id,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _interactive_mode()
    else:
        _cli_mode()
