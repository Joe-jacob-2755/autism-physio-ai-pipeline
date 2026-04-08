"""
=============================================================================
MODULE 3 – DATA PREPROCESSING  |  main.py
=============================================================================
Interactive CLI entry point.

Launch from repo root:
  Windows:    run_preprocessing_module.bat
  Mac/Linux:  ./run_preprocessing_module.sh

Or directly:
  cd module_3_preprocessing
  python main.py
  python main.py --source ../module_2_data_acquisition/outputs/M2_v1.0.0_mode2_2_run_001
  python main.py --source <path> --filter butterworth --window 60 --no_plots
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
M2_DIR = REPO_ROOT / "module_2_data_acquisition"
M1A_DIR = REPO_ROOT / "module_1a_data_simulation"

sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(M2_DIR))
sys.path.insert(0, str(M1A_DIR))


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
    """Browse for a Module 2 output folder to preprocess."""
    _section("📂  Select data source")
    candidates = []

    # Module 2 outputs
    m2_out = M2_DIR / "outputs"
    if m2_out.exists():
        for d in sorted(m2_out.iterdir()):
            if d.is_dir():
                csv_count = len(list(d.rglob("*.csv")))
                if csv_count > 0:
                    candidates.append((d, f"Module 2: {d.name}  ({csv_count} CSVs)"))

    # Module 1A outputs (direct path use)
    m1a_out = M1A_DIR / "outputs"
    if m1a_out.exists():
        for d in sorted(m1a_out.iterdir()):
            if d.is_dir() and any((d / f).exists() for f in ("EDA.csv", "combined_signals.csv")):
                candidates.append((d, f"Module 1A: {d.name}"))

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


def _random_demographics(seed=None) -> dict:
    """Draw demographics from ASD population distributions via Module 2A."""
    m2a_dir = MODULE_DIR.parent / "module_2a_data_simulation"
    saved_path = list(__import__('sys').path)
    saved_modules = {k: v for k, v in __import__('sys').modules.items()
                     if k.split('.')[0] == 'config'}
    for k in saved_modules:
        __import__('sys').modules.pop(k, None)
    __import__('sys').path.insert(0, str(m2a_dir))
    try:
        from user_profiles import random_demographics as _rd
        return _rd(seed=seed)
    except Exception:
        import random
        return {
            "age": random.randint(5, 15),
            "gender": random.choice(["Male", "Female", "Non-binary"]),
            "ethnicity": "White British",
            "autism_severity": random.choice(["Low", "Medium", "Severe"]),
            "verbal_status": random.choice(["Verbal", "Minimally verbal", "Non-verbal"]),
            "comorbidity": random.choice(["Yes", "No"]),
        }
    finally:
        __import__('sys').path[:] = saved_path
        for k in list(__import__('sys').modules):
            if k.split('.')[0] == 'config':
                __import__('sys').modules.pop(k, None)
        __import__('sys').modules.update(saved_modules)


def _choose_dem(prompt, options, labels):
    """Numbered choice menu returning selected index."""
    print()
    for i, lbl in enumerate(labels, 1):
        print(f"    {i}.  {lbl}")
    while True:
        raw = _ask(prompt, "1").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(f"  ✗  Enter 1–{len(options)}.")


def _collect_demographics() -> dict:
    """
    Collect participant demographics — consistent with pipeline_main.py interface.

    Three modes:
      1. Random  — draw from ASD population distributions (default)
                   Each participant gets independently drawn demographics,
                   so ages and profiles differ across a multi-user session.
      2. Manual  — enter each field individually
      3. Skip    — no demographics appended to feature matrix
    """
    _section("👤  Participant demographics")
    print("""
  Demographics (age, gender, severity, verbal status, comorbidity) are fused
  with signal features in the combined CSV for multimodal models.
""")
    dem_idx = _choose_dem("Demographics mode", range(3), [
        "Random  — auto-generate from ASD population distributions  (default)",
        "Manual  — enter each field individually",
        "Skip    — omit demographics from feature matrix",
    ])

    if dem_idx == 2:
        return {}

    if dem_idx == 0:
        dem = _random_demographics()
        print(f"""
  Generated demographics:
    Age             : {dem["age"]}
    Gender          : {dem["gender"]}
    Ethnicity       : {dem["ethnicity"]}
    Autism severity : {dem["autism_severity"]}
    Verbal status   : {dem["verbal_status"]}
    Comorbidity     : {dem["comorbidity"]}

  ℹ  In multi-user sessions each participant gets independently drawn
     demographics (different ages, genders, severity levels etc.)
""")
        if not _ask_yn("Use these demographics?", default=True):
            dem = _random_demographics()
            print(f"  Re-generated: Age={dem['age']}  "
                  f"Gender={dem['gender']}  Severity={dem['autism_severity']}")
        return dem

    # Manual entry
    try:
        age = max(5, min(15, int(_ask("Age (5–15)", "10"))))
    except ValueError:
        age = 10

    gender = ["Male", "Female", "Non-binary"][_choose_dem(
        "Gender", range(3), ["Male", "Female", "Non-binary"])]

    ethnicity = [
        "White British", "Asian / Asian British",
        "Black / African / Caribbean", "Mixed / Multiple", "Other",
    ][_choose_dem("Ethnicity", range(5), [
        "White British", "Asian / Asian British",
        "Black / African / Caribbean", "Mixed / Multiple", "Other",
    ])]

    severity = ["Low", "Medium", "Severe"][_choose_dem(
        "Autism severity  (DSM-5 Level)", range(3),
        ["Low (Level 1)", "Medium (Level 2)", "Severe (Level 3)"])]

    verbal = ["Verbal", "Minimally verbal", "Non-verbal"][_choose_dem(
        "Verbal status", range(3),
        ["Verbal", "Minimally verbal", "Non-verbal"])]

    comorbidity = "Yes" if _ask_yn(
        "Any co-occurring conditions? (ADHD, anxiety, etc.)", default=True) else "No"

    return {
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "autism_severity": severity,
        "verbal_status": verbal,
        "comorbidity": comorbidity,
    }


def _interactive_mode():
    """Full interactive guided session."""
    os.system("cls" if os.name == "nt" else "clear")
    print()
    print("  " + "═" * (WIDTH - 4))
    print(f"  ║    🧠  AUTISM PHYSIO-AI PIPELINE" + " " * (WIDTH - 42) + "║")
    print(f"  ║        Module 3 — Data Preprocessing" + " " * (WIDTH - 45) + "║")
    print(f"  ║        v{MODULE_VERSION}" + " " * (WIDTH - 14) + "║")
    print("  " + "═" * (WIDTH - 4))

    # Source
    source = _browse_source()
    if source is None:
        print("  No valid source selected. Exiting.")
        return

    # Demographics
    demographics = _collect_demographics()

    # Filter
    _section("🔧  Signal filtering")
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

    # Window
    _section("⏱  Feature extraction window")
    print("""
  Recommended: 60 s windows with 50% overlap for emotion classification.
  Shorter windows (30 s) reduce latency but may miss slow HRV features.
""")
    window_s = float(_ask("Window size (seconds)", "60"))
    overlap = float(_ask("Overlap fraction (0.0–0.9)", "0.5"))

    # Plots
    _section("📊  Visualisation")
    gen_plots = _ask_yn("Generate signal plots?", default=True)

    # Summary
    _section("✅  Preprocessing Parameters Summary")
    print(f"""
  Source         : {source}
  Demographics   : {demographics if demographics else 'Not provided'}
  Filter         : {filter_type}{' + Hampel' if apply_hampel else ''}
  Window         : {window_s:.0f}s  overlap={overlap:.0%}
  Scaler         : RobustScaler
  Generate plots : {gen_plots}
""")
    if not _ask_yn("Start preprocessing?", default=True):
        print("  Cancelled.")
        return

    # Run
    preprocessor = DataPreprocessor(
        filter_type=filter_type,
        apply_hampel=apply_hampel,
        window_s=window_s,
        overlap=overlap,
        generate_plots=gen_plots,
        verbose=True,
    )
    preprocessor.run(
        signals_input=source,
        demographics=demographics if demographics else None,
        session_id=source.name,
        user_id=demographics.get("gender", "unknown") if demographics else "unknown",
    )


def _cli_mode():
    """Argument-driven mode."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=f"Module 3 — Data Preprocessing  v{MODULE_VERSION}"
    )
    parser.add_argument("--source", default=None,
                        help="Path to Module 2 output folder or signal CSV folder")
    parser.add_argument("--filter", default="butterworth",
                        choices=["butterworth", "kalman", "hampel_only", "none"])
    parser.add_argument("--no_hampel", action="store_true")
    parser.add_argument("--window", type=float, default=60.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--user_id", default="unknown")
    parser.add_argument("--session_id", default="cli_session")
    # Demographics
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--gender", default=None)
    parser.add_argument("--severity", default=None,
                        choices=["Low", "Medium", "Severe"])
    parser.add_argument("--verbal", default=None,
                        choices=["Verbal", "Minimally verbal", "Non-verbal"])
    parser.add_argument("--comorbidity", default=None, choices=["Yes", "No"])

    args = parser.parse_args()

    if not args.source:
        parser.error("--source is required in CLI mode")

    demographics = {}
    if args.age:
        demographics["age"] = args.age
        demographics["gender"] = args.gender or "Male"
        demographics["ethnicity"] = "White British"
        demographics["autism_severity"] = args.severity or "Medium"
        demographics["verbal_status"] = args.verbal or "Verbal"
        demographics["comorbidity"] = args.comorbidity or "Yes"

    preprocessor = DataPreprocessor(
        filter_type=args.filter,
        apply_hampel=not args.no_hampel,
        window_s=args.window,
        overlap=args.overlap,
        generate_plots=not args.no_plots,
        verbose=True,
    )
    preprocessor.run(
        signals_input=Path(args.source),
        demographics=demographics if demographics else None,
        session_id=args.session_id,
        user_id=args.user_id,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _interactive_mode()
    else:
        _cli_mode()
