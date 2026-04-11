"""
=============================================================================
MODULE 6 – FEATURE ENGINEERING  |  main.py  (v1.0.0)
=============================================================================
Interactive CLI entry point.

Launch from repo root:
  Windows:    run_feature_engineering.bat
  Mac/Linux:  ./run_feature_engineering.sh

Or directly:
  cd module_6_feature_engineering
  python main.py
  python main.py --source ../module_5_preprocessing/outputs/M5_v1.0.0_run_001
  python main.py --source <path> --window 60 --top_n 20
=============================================================================
"""
from __future__ import annotations
from config import MODULE_VERSION, DEFAULT_TOP_N_FEATURES
from feature_engineer import FeatureEngineer
import argparse
import sys
import os
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
M5_DIR = REPO_ROOT / "module_5_preprocessing"
M2A_DIR = REPO_ROOT / "module_2a_data_simulation"

sys.path.insert(0, str(MODULE_DIR))

WIDTH = 64


def _section(t):
    print(f"\n  {'─' * (WIDTH - 4)}\n  {t}\n  {'─' * (WIDTH - 4)}")


def _ask(prompt, default=""):
    return input(f"\n  {prompt} [{default}]: ").strip() or str(default)


def _ask_yn(prompt, default=True):
    raw = input(f"\n  {prompt} ({'Y/n' if default else 'y/N'}): ").strip().lower()
    return default if raw == "" else raw in ("y", "yes")


def _browse_source() -> Path:
    """Browse for Module 5 or Module 2A output folder."""
    _section("Select data source")
    candidates = []

    # Module 5 outputs
    m5_out = M5_DIR / "outputs"
    if m5_out.exists():
        for d in sorted(m5_out.iterdir()):
            if d.is_dir():
                has_signals = ((d / "cleaned_signals").exists()
                               or (d / "EDA.csv").exists())
                if has_signals:
                    candidates.append(
                        (d, f"Module 5: {d.name}"))

    # Module 2A outputs (can also be used directly)
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
    print(f"  |    Module 6 - Feature Engineering" + " " * (WIDTH - 43) + "|")
    print(f"  |    v{MODULE_VERSION}" + " " * (WIDTH - 14) + "|")
    print("  " + "=" * (WIDTH - 4))

    # Source
    source = _browse_source()
    if source is None:
        print("  No valid source selected. Exiting.")
        return

    # Window
    _section("Feature extraction window")
    print("""
  Recommended: 60 s windows with 50% overlap for emotion classification.
  Shorter windows (30 s) reduce latency but may miss slow HRV features.
""")
    window_s = float(_ask("Window size (seconds)", "60"))
    overlap = float(_ask("Overlap fraction (0.0-0.9)", "0.5"))

    # Feature selection
    _section("Feature selection")
    print(f"""
  Select the number of most important features to keep.
  All features will still be saved; this creates an additional
  reduced set for model training.
""")
    top_n = int(_ask("Number of top features", str(DEFAULT_TOP_N_FEATURES)))

    methods = ["ensemble", "mutual_information", "random_forest",
               "f_statistic", "variance_threshold"]
    print("\n  Selection methods:")
    for i, m in enumerate(methods, 1):
        print(f"    {i}.  {m}")
    m_idx = int(_ask("Select method", "1")) - 1
    sel_method = methods[max(0, min(m_idx, len(methods) - 1))]

    # Summary
    _section("Feature Engineering Parameters Summary")
    print(f"""
  Source           : {source}
  Window           : {window_s:.0f}s  overlap={overlap:.0%}
  Scaler           : RobustScaler
  Top features     : {top_n}  (method={sel_method})
  Dim. reduction   : PCA + PLS-VIP + SVD (auto-compared)
  One-hot encoding : Yes (demographics)
""")
    if not _ask_yn("Start feature engineering?", default=True):
        print("  Cancelled.")
        return

    # Run
    engineer = FeatureEngineer(
        window_s=window_s,
        overlap=overlap,
        top_n_features=top_n,
        selection_method=sel_method,
        verbose=True,
    )
    result = engineer.run(
        signals_input=source,
        session_id=source.name,
        user_id="unknown",
    )

    print(f"\n  Output ready for Module 5: {result['run_folder']}")


def _cli_mode():
    """Argument-driven mode."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=f"Module 6 - Feature Engineering  v{MODULE_VERSION}"
    )
    parser.add_argument("--source", default=None,
                        help="Path to Module 5 output folder or signal CSVs")
    parser.add_argument("--window", type=float, default=60.0,
                        help="Feature window size in seconds (default: 60)")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Window overlap fraction (default: 0.5)")
    parser.add_argument("--scaler", default="robust",
                        choices=["robust", "standard", "minmax"])
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N_FEATURES,
                        help=f"Top N features to select "
                             f"(default: {DEFAULT_TOP_N_FEATURES})")
    parser.add_argument("--method", default="ensemble",
                        choices=["ensemble", "mutual_information",
                                 "random_forest", "f_statistic",
                                 "variance_threshold"])
    parser.add_argument("--pca_var", type=float, default=0.95,
                        help="PCA variance retention (default: 0.95)")
    parser.add_argument("--user_id", default="unknown")
    parser.add_argument("--session_id", default="cli_session")

    args = parser.parse_args()

    if not args.source:
        parser.error("--source is required in CLI mode")

    engineer = FeatureEngineer(
        window_s=args.window,
        overlap=args.overlap,
        scaler_type=args.scaler,
        top_n_features=args.top_n,
        selection_method=args.method,
        pca_variance=args.pca_var,
        verbose=True,
    )
    engineer.run(
        signals_input=Path(args.source),
        session_id=args.session_id,
        user_id=args.user_id,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _interactive_mode()
    else:
        _cli_mode()
