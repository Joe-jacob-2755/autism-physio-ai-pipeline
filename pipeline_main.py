"""
=============================================================================
AUTISM PHYSIO-AI PIPELINE  |  pipeline_main.py
=============================================================================
Master pipeline orchestrator.

Runs the full sequence:   Module 2  →  Module 3
Or Module 3 independently on existing data.

Launch commands (from repo root)
---------------------------------
  run_full_pipeline              Windows
  ./run_full_pipeline.sh         Mac / Linux

  run_data_preprocessing         Windows   (Module 3 only)
  ./run_data_preprocessing.sh    Mac / Linux

HOW PATH ISOLATION WORKS
------------------------
Every module has a file called config.py.  Adding all three module
directories to sys.path at once causes Python's module cache to serve
the wrong config.py to the wrong importer.

The solution: use _isolated_import() — a context manager that:
  1. Inserts ONLY the target module's directory at sys.path[0]
  2. Clears all cached 'config' entries from sys.modules before entry
  3. Restores both sys.path and sys.modules on exit

This gives each module a clean, isolated import environment without
needing subprocesses or renaming any files.
=============================================================================
"""
from __future__ import annotations
from output_manager import (
    next_pipeline_run_folder, standalone_run_folder,
    module_subdir, user_subdir, RunManifest, count_files,
)
import os
import sys
import re
import json
import time
import importlib
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent

# Module root directories
M2A_DIR = REPO_ROOT / "module_2a_data_simulation"
M1_DIR = REPO_ROOT / "module_1_data_acquisition"
M3_DIR = REPO_ROOT / "module_3_preprocessing"
M4_DIR = REPO_ROOT / "module_4_data_analyser"

PIPELINE_VERSION = "1.0.0"

# Central output manager — all pipeline runs write here
sys.path.insert(0, str(REPO_ROOT))
CENTRAL_OUTPUTS = REPO_ROOT / "outputs"
WIDTH = 66


# ─────────────────────────────────────────────────────────────────────────────
# PATH ISOLATION — prevents config.py collisions between modules
# ─────────────────────────────────────────────────────────────────────────────

# Names of modules that conflict across the three pipeline modules
_CONFLICTING_MODULES = {
    "config", "signal_filters", "signal_cleaner",
    "annotator", "exporter", "visualiser", "visualizer",
    "simulator", "main",
    "analyser", "reporter", "signal_analyser", "statistical_analyser",
}


@contextmanager
def _isolated_import(*module_dirs: Path):
    """
    Context manager: make module_dirs the exclusive front of sys.path,
    and clear any cached modules that share names across pipeline modules.

    Usage:
        with _isolated_import(M1_DIR):
            from acquisition_module import DataAcquisitionModule
    """
    saved_path = list(sys.path)
    saved_modules = {k: v for k, v in sys.modules.items()
                     if _is_pipeline_module(k)}

    # Clear conflicting cached modules
    for k in list(sys.modules.keys()):
        if _is_pipeline_module(k):
            del sys.modules[k]

    # Put the target module dirs at the very front of sys.path
    new_path = [str(d) for d in reversed(module_dirs)]
    sys.path[:] = new_path + [p for p in saved_path
                              if p not in new_path]
    try:
        yield
    finally:
        # Restore everything
        for k in list(sys.modules.keys()):
            if _is_pipeline_module(k):
                del sys.modules[k]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def _is_pipeline_module(name: str) -> bool:
    """Return True if this module name could conflict across pipeline modules."""
    top = name.split(".")[0]
    return top in _CONFLICTING_MODULES


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _banner(subtitle=""):
    _clear()
    print()
    print("  " + "═" * (WIDTH - 2))
    print("  ║" + " " * (WIDTH - 4) + "║")
    print("  ║    🧠  AUTISM PHYSIO-AI PIPELINE" + " " * (WIDTH - 38) + "║")
    if subtitle:
        pad = WIDTH - 6 - len(subtitle)
        print(f"  ║    {subtitle}" + " " * max(pad, 0) + "║")
    print(f"  ║    v{PIPELINE_VERSION}" + " " * (WIDTH - 12) + "║")
    print("  ║" + " " * (WIDTH - 4) + "║")
    print("  " + "═" * (WIDTH - 2))
    print()


def _section(title):
    print()
    print("  " + "─" * (WIDTH - 4))
    print(f"  {title}")
    print("  " + "─" * (WIDTH - 4))


def _step_header(n, total, title):
    print()
    print("  " + "═" * (WIDTH - 4))
    print(f"  ▶  Step {n}/{total}  —  {title}")
    print("  " + "═" * (WIDTH - 4))


def _ok(msg):
    print(f"  ✓  {msg}")


def _info(msg):
    print(f"     {msg}")


def _ask(prompt, default=""):
    suffix = f" [{default}]" if default != "" else ""
    raw = input(f"\n  {prompt}{suffix}: ").strip()
    return raw if raw else str(default)


def _ask_int(prompt, default, lo=1, hi=9999):
    while True:
        raw = _ask(prompt, str(default))
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
            print(f"  ✗  Enter {lo}–{hi}.")
        except ValueError:
            print("  ✗  Please enter a whole number.")


def _ask_float(prompt, default, lo=0.0):
    while True:
        raw = _ask(prompt, str(default))
        try:
            v = float(raw)
            if v >= lo:
                return v
            print(f"  ✗  Must be ≥ {lo}.")
        except ValueError:
            print("  ✗  Please enter a number.")


def _ask_int_or_random(prompt, default):
    raw = _ask(f"{prompt}  (number or 'random')", str(default))
    return "random" if raw.lower() == "random" else int(raw)


def _ask_float_or_random(prompt, default):
    raw = _ask(f"{prompt}  (number or 'random')", str(default))
    return "random" if raw.lower() == "random" else float(raw)


def _ask_yn(prompt, default=True):
    hint = "Y/n" if default else "y/N"
    raw = input(f"\n  {prompt}  ({hint}): ").strip().lower()
    return default if raw == "" else raw in ("y", "yes")


def _choose(prompt, options, labels=None):
    display = labels or [str(o) for o in options]
    print()
    for i, lbl in enumerate(display, 1):
        print(f"    {i}.  {lbl}")
    while True:
        raw = _ask(prompt, "1")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
            print(f"  ✗  Enter 1–{len(options)}.")
        except ValueError:
            print("  ✗  Please enter a number.")


def _pause(msg="  Press Enter to continue ..."):
    input(msg)


# ─────────────────────────────────────────────────────────────────────────────
# FOLDER BROWSER
# ─────────────────────────────────────────────────────────────────────────────

def _browse_data_folder(purpose="existing data") -> Optional[Path]:
    _section(f"📂  Select {purpose} folder")

    candidates, labels = [], []
    for out_root, tag in [
        (M2A_DIR / "outputs", "M2A"),
        (M1_DIR / "outputs", "M1"),
        (M3_DIR / "outputs", "M3"),
    ]:
        if out_root.exists():
            for d in sorted(out_root.iterdir()):
                if not d.is_dir():
                    continue
                n_csv = len(list(d.rglob("*.csv")))
                if n_csv == 0:
                    continue
                try:
                    rel = d.relative_to(REPO_ROOT)
                except ValueError:
                    rel = d
                n_users = len([x for x in d.iterdir()
                               if x.is_dir() and x.name.startswith("user_")])
                suffix = f"  ({n_users} users)" if n_users > 0 \
                    else f"  ({n_csv} CSVs)"
                candidates.append(d)
                labels.append(f"[{tag}]  {rel.name}{suffix}")

    labels.append("📁  Type a custom path")
    print()
    for i, lbl in enumerate(labels[:-1], 1):
        print(f"    {i}.  {lbl}")
    print(f"    {len(candidates) + 1}.  {labels[-1]}")

    while True:
        raw = input(f"\n  Select [1]: ").strip() or "1"
        if raw.startswith(".") or raw.startswith("/") or \
           (len(raw) > 2 and raw[1] == ":"):
            p = Path(raw)
            return p if p.exists() else None
        try:
            idx = int(raw) - 1
        except ValueError:
            p = Path(raw)
            return p if p.exists() else None

        if idx == len(candidates):
            p = Path(_ask("Enter the full folder path"))
            return p if p.exists() else None

        if 0 <= idx < len(candidates):
            chosen = candidates[idx]
            user_dirs = sorted([x for x in chosen.iterdir()
                                if x.is_dir() and x.name.startswith("user_")])
            if user_dirs:
                _section("Multiple users found in this folder")
                sub_labels = ["📦  Use entire folder (all users)"] + \
                             [u.name for u in user_dirs]
                sub_idx = _choose("Which data to use?",
                                  range(len(sub_labels)), sub_labels)
                if sub_idx == 0:
                    return chosen
                return user_dirs[sub_idx - 1]
            return chosen
        print(f"  ✗  Enter 1–{len(candidates) + 1}.")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def _collect_m2_params() -> dict:
    _section("📡  Module 2 — Data Acquisition")
    print("""
  Choose how data enters the pipeline.
""")
    mode_idx = _choose(
        "Data source",
        ["import", "simulate", "live_file", "live_e4", "deployment"],
        [
            "Use existing data      — import CSVs from a folder",
            "Create new simulation  — generate via Module 1A",
            "Live device (test)     — replay a CSV as a live stream",
            "Live device (E4)       — Empatica E4 via BLE server",
            "Deployment data        — non-annotated inference data",
        ],
    )
    modes = ["import", "simulate", "live_file", "live_e4", "deployment"]
    mode = modes[mode_idx]
    params = {"mode": mode}

    if mode == "import":
        params["source"] = _browse_data_folder("existing signal data")
        params["user_id"] = _ask("Participant ID", "imported_user")

    elif mode == "simulate":
        _section("⚙️  Simulation settings")
        params["n_users"] = _ask_int("Number of users", 1, 1, 500)
        params["duration_s"] = _ask_float("Duration per user (seconds)", 300, 30)
        params["n_events"] = _ask_int_or_random("Events per user", 5)
        params["event_dur"] = _ask_float_or_random("Event duration (seconds)", 30)

        # Get emotion list safely — M1A emotions don't need import
        EMOTIONS_ALL = [
            "Happy", "Anger", "Fear", "Disgust", "Sad", "Surprise",
            "Hunger", "Thirst", "Toilet", "Tired",
        ]
        print(f"\n  Available emotions: {', '.join(EMOTIONS_ALL)}")
        em_idx = _choose("Emotion mode", range(3),
                         ["Random  — all 10 states randomly",
                          "Single  — one emotion repeated",
                          "Subset  — choose a list"])
        if em_idx == 0:
            params["emotions"] = None
        elif em_idx == 1:
            while True:
                em = _ask("Emotion name")
                if em in EMOTIONS_ALL:
                    params["emotions"] = em
                    break
                print(f"  ✗  Not recognised. Valid: {EMOTIONS_ALL}")
        else:
            while True:
                raw = _ask("Comma-separated list  e.g. Fear,Anger,Tired")
                lst = [e.strip() for e in raw.split(",") if e.strip()]
                bad = [e for e in lst if e not in EMOTIONS_ALL]
                if bad:
                    print(f"  ✗  Unknown: {bad}")
                elif lst:
                    params["emotions"] = lst
                    break

        params["noise"] = ["low", "medium", "high"][_choose(
            "Noise level", range(3),
            ["Low    – clean signal",
             "Medium – realistic (recommended)",
             "High   – poor contact / movement"])]

        params["shared_events"] = (
            _ask_yn("Share event schedule across all users?", False)
            if params["n_users"] > 1 else False
        )
        seed_raw = _ask("Random seed  (0 = random each run)", "42")
        try:
            params["seed"] = int(seed_raw) or \
                __import__("random").randint(1, 999_999)
        except ValueError:
            params["seed"] = 42

        params["save_m1a"] = _ask_yn("Also save Module 1A CSV/plots?", False)

    elif mode in ("live_file", "live_e4"):
        params["user_id"] = _ask("Participant ID", "participant_001")
        if mode == "live_file":
            folder = _browse_data_folder("replay CSV")
            if folder and folder.is_dir():
                combined = folder / "combined_signals.csv"
                params["source"] = combined if combined.exists() else folder
            else:
                params["source"] = folder
            params["speed"] = _ask_float("Replay speed multiplier", 10.0, 0.1)
        else:
            params["e4_host"] = _ask("E4 server host", "127.0.0.1")
            params["e4_port"] = _ask_int("E4 server port", 28000, 1, 65535)
        max_dur = _ask("Auto-stop after N seconds  (Enter = manual stop)", "")
        params["max_dur"] = float(max_dur) if max_dur else None

    elif mode == "deployment":
        params["source"] = _browse_data_folder("deployment signal data")
        params["user_id"] = _ask("Participant ID  (can be anonymised)", "anon_001")
        params["strip_labels"] = _ask_yn("Strip annotation columns?", True)

    return params


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def _random_demographics(seed: int = None) -> dict:
    """
    Draw one participant's demographics from ASD population distributions.
    Uses module_2a user_profiles.random_demographics() in an isolated import
    so the 2A config is found instead of M3 config.
    """
    with _isolated_import(M2A_DIR):
        from user_profiles import random_demographics as _rd
        return _rd(seed=seed)


def _collect_m3_params(skip_source=False) -> dict:
    params = {}

    if not skip_source:
        _section("📂  Module 3 — Select data to preprocess")
        params["source"] = _browse_data_folder("data to preprocess")

    _section("👤  Participant demographics")
    print("""
  Demographics (age, gender, severity, verbal status, comorbidity) are fused
  with signal features in the combined CSV.
""")
    dem_idx = _choose("Demographics mode", range(3),
                      ["Random  — auto-generate from ASD population distributions",
                       "Manual  — enter each field individually",
                       "Skip    — omit demographics from feature matrix"])

    if dem_idx == 0:
        # Random — generate a sample draw to preview, but re-draw per user at runtime
        dem = _random_demographics()
        print(f"""
  Sample demographics (each user will get their own random draw):
    Age             : {dem["age"]}
    Gender          : {dem["gender"]}
    Ethnicity       : {dem["ethnicity"]}
    Autism severity : {dem["autism_severity"]}
    Verbal status   : {dem["verbal_status"]}
    Comorbidity     : {dem["comorbidity"]}

  ℹ  Each participant will receive independently drawn demographics.
     Ages are uniform 5–15. Gender/severity drawn from ASD population distributions.
""")
        params["demographics"] = dem
        params["_dem_mode"] = "random"

    elif dem_idx == 1:
        # Manual entry
        try:
            age = max(5, min(15, int(_ask("Age (5–15)", "10"))))
        except ValueError:
            age = 10
        params["demographics"] = {
            "age": age,
            "gender": ["Male", "Female", "Non-binary"][_choose(
                "Gender", range(3), ["Male", "Female", "Non-binary"])],
            "ethnicity": [
                "White British", "Asian / Asian British",
                "Black / African / Caribbean", "Mixed / Multiple", "Other",
            ][_choose("Ethnicity", range(5),
                      ["White British", "Asian / Asian British",
                       "Black / African / Caribbean", "Mixed / Multiple", "Other"])],
            "autism_severity": ["Low", "Medium", "Severe"][_choose(
                "Autism severity  (DSM-5 Level)", range(3),
                ["Low (Level 1)", "Medium (Level 2)", "Severe (Level 3)"])],
            "verbal_status": ["Verbal", "Minimally verbal", "Non-verbal"][_choose(
                "Verbal status", range(3),
                ["Verbal", "Minimally verbal", "Non-verbal"])],
            "comorbidity": "Yes" if _ask_yn(
                "Any co-occurring conditions?", True) else "No",
        }
    else:
        params["demographics"] = None

    _section("🔧  Signal filtering")
    print("""
  Butterworth: Zero-phase LP/BP filter  (recommended for all signals)
  Kalman:      Gaussian noise smoother  (best for ST and EDA tonic)
  Hampel only: Outlier removal only, no frequency filtering
  None:        Skip frequency filtering
""")
    filt_names = ["butterworth", "kalman", "hampel_only", "none"]
    filt_idx = _choose("Filter type", range(4),
                       ["Butterworth  (recommended)",
                        "Kalman smoother",
                        "Hampel outlier removal only",
                        "No frequency filter"])
    params["filter_type"] = filt_names[filt_idx]
    params["apply_hampel"] = _ask_yn("Apply Hampel outlier pre-filter?", True)

    _section("⏱  Feature extraction window")
    print("""
  Recommended: 60 s with 50% overlap.
  Shorter windows (30 s) suit real-time use but weaken HRV features.
""")
    params["window_s"] = _ask_float("Window size (seconds)", 60.0, 10.0)
    params["overlap"] = _ask_float("Overlap fraction  (0.0–0.9)", 0.5, 0.0)
    params["gen_plots"] = _ask_yn("Generate visualisation plots?", True)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# RUN MODULE 2  (isolated imports)
# ─────────────────────────────────────────────────────────────────────────────

def run_module_2(params: dict) -> list:
    """Execute Module 2 with isolated sys.path so M2's config.py is found."""
    mode = params["mode"]

    with _isolated_import(M1_DIR, M2A_DIR):
        from acquisition_module import DataAcquisitionModule

        acq = DataAcquisitionModule(
            mode={
                "import": "2.1",
                "simulate": "2.2",
                "live_file": "2.3",
                "live_e4": "2.3",
                "deployment": "2.4",
            }[mode],
            save_packets=True,
            verbose=True,
        )

        if mode == "import":
            packet = acq.run_import(
                source_path=params["source"],
                user_id=params.get("user_id", "imported_user"),
            )
            return [packet]

        elif mode == "simulate":
            packets = acq.run_simulate(
                duration_s=params["duration_s"],
                n_events=params["n_events"],
                event_duration_s=params["event_dur"],
                emotions=params.get("emotions"),
                noise_level=params["noise"],
                seed=params["seed"],
                n_users=params["n_users"],
                shared_events=params.get("shared_events", False),
                save_m1a_output=params.get("save_m1a", False),
            )
            return packets

        elif mode == "live_file":
            packet = acq.run_live_file(
                csv_path=params["source"],
                user_id=params.get("user_id", "participant_001"),
                speed_factor=params.get("speed", 10.0),
                max_duration_s=params.get("max_dur"),
            )
            return [packet]

        elif mode == "live_e4":
            packet = acq.run_live_e4(
                user_id=params.get("user_id", "participant_001"),
                host=params.get("e4_host", "127.0.0.1"),
                port=params.get("e4_port", 28000),
                max_duration_s=params.get("max_dur"),
            )
            return [packet]

        elif mode == "deployment":
            packet = acq.run_deployment(
                source_path=params["source"],
                user_id=params.get("user_id", "anon_001"),
                strip_labels=params.get("strip_labels", True),
            )
            return [packet]

    return []


# ─────────────────────────────────────────────────────────────────────────────
# RUN MODULE 3  (isolated imports)
# ─────────────────────────────────────────────────────────────────────────────

def run_module_3(
    source,
    params: dict,
    session_id: str = "pipeline_session",
    user_id: str = "unknown",
    demographics: dict = None,
) -> dict:
    """Execute Module 3 with isolated sys.path so M3's config.py is found."""
    with _isolated_import(M3_DIR):
        from preprocessor import DataPreprocessor

        pp = DataPreprocessor(
            filter_type=params.get("filter_type", "butterworth"),
            apply_hampel=params.get("apply_hampel", True),
            window_s=params.get("window_s", 60.0),
            overlap=params.get("overlap", 0.5),
            generate_plots=params.get("gen_plots", True),
            verbose=True,
        )
        return pp.run(
            signals_input=source,
            demographics=demographics or params.get("demographics"),
            session_id=session_id,
            user_id=user_id,
        )


def run_module_4(
    source,
    threshold_window_s: float = 300.0,
    threshold_mad_factor: float = 3.0,
    threshold_sustain_s: float = 30.0,
    session_id: str = "pipeline_session",
    user_id: str = "unknown",
) -> dict:
    """Execute Module 4 data analysis with isolated sys.path."""
    with _isolated_import(M4_DIR):
        from analyser import DataAnalyser
        da = DataAnalyser(
            threshold_window_s=threshold_window_s,
            threshold_mad_factor=threshold_mad_factor,
            threshold_sustain_s=threshold_sustain_s,
            verbose=True,
        )
        return da.run(source=source, session_id=session_id, user_id=user_id)


def _collect_m4_params() -> dict:
    """Collect Module 4 threshold parameters."""
    _section("⚙️  Module 4 — Threshold settings")
    print("""
  Adaptive threshold flags sustained signal deviations from running median.
  Recommended: window=300s (5 min), N=3×MAD, sustain=30s.
""")
    win_s = _ask_float("Threshold window (seconds)", 300.0, 30.0)
    mad_n = _ask_float("Deviation multiplier (N × MAD)", 3.0, 1.0)
    sustain = _ask_float("Minimum sustained duration (seconds)", 30.0, 1.0)
    return {"window_s": win_s, "mad_factor": mad_n, "sustain_s": sustain}


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE  (M2 → M3)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline():
    """
    Interactive full pipeline with CENTRAL output directory.

    All module outputs go to:  outputs/run_NNN/
        module_1_acquisition/user_NNN/   <- M1 signal CSVs + metadata
        module_2a_simulation/user_NNN/   <- M2A CSVs + plots (if simulated)
        module_3_preprocessing/user_NNN/ <- feature CSVs + plots
        module_4_analysis/user_NNN/      <- analysis report + tables + plots
    """
    _banner("Full Pipeline  —  M1 → M2A → M3 → M4")
    print("""
  Runs the complete pipeline. All outputs go to a single numbered
  folder under  outputs/run_NNN/  at the repo root.

  Flow:
    ① Data Acquisition   (Module 1)
    ② Preprocessing      (Module 3)
    ③ Analysis           (Module 4, optional)
""")
    if not _ask_yn("Continue?", True):
        return

    t_start = time.time()

    # ── Create central run folder ─────────────────────────────────────────
    run_folder = next_pipeline_run_folder()
    manifest = RunManifest(run_folder)
    rel_run = run_folder.relative_to(REPO_ROOT)

    print(f"\n  📁  Central output folder: {rel_run}\n")

    # ── Step 1: Module 1 — Data Acquisition ──────────────────────────────
    _step_header(1, 3, "Data Acquisition  (Module 1)")
    m1_params = _collect_m2_params()

    # Route M1 outputs to central folder
    m1_out = module_subdir(run_folder, "module_1_acquisition")
    m1_params["_output_dir"] = str(m1_out)

    print("\n  Starting Module 1 ...")
    packets = run_module_2(m1_params)
    _ok(f"Module 1 complete — {len(packets)} packet(s) acquired")

    manifest.record_module("M1", m1_out, {"mode": m1_params.get("mode"), "n_packets": len(packets)})
    manifest.save()

    if not packets:
        print("\n  ✗  No data packets returned. Aborting.")
        return

    # ── Step 2: Module 3 — Preprocessing ─────────────────────────────────
    _step_header(2, 3, "Data Preprocessing  (Module 3)")
    m3_params = _collect_m3_params(skip_source=True)
    m3_out = module_subdir(run_folder, "module_3_preprocessing")

    all_m3_results = []
    for i, packet in enumerate(packets, 1):
        uid = getattr(packet, "user_id", f"user_{i:03d}")
        sid = getattr(packet, "session_id", f"session_{i:03d}")
        print(f"\n  Preprocessing packet {i}/{len(packets)}  [{uid}]")

        # Per-user subdirectory inside module_3_preprocessing/
        user_m3_dir = user_subdir(m3_out, uid)

        # Auto-extract demographics from packet metadata
        dem = m3_params.get("demographics")
        if dem is None and hasattr(packet, "metadata"):
            meta_user = packet.metadata.get("user", {})
            if meta_user:
                dem = {k: meta_user[k]
                       for k in ("age", "gender", "ethnicity",
                                 "autism_severity", "verbal_status", "comorbidity")
                       if k in meta_user and meta_user[k] is not None}

        result = run_module_3(
            source=packet, params=m3_params,
            session_id=sid, user_id=uid,
            demographics=dem,
            output_dir=str(user_m3_dir),
        )
        all_m3_results.append(result)
        manifest.record_user(uid, dem)

    manifest.record_module("M3", m3_out, {"n_users": len(all_m3_results)})
    manifest.save()

    # ── Step 3: Module 4 — Analysis (optional) ────────────────────────────
    if all_m3_results:
        _section("📊  Optional — Data Analysis  (Module 4)")
        print("""
  Module 4 analyses preprocessed signals and extracted features:
    • Descriptive, statistical and correlation analysis
    • Temporal dynamics (time to peak, subside, return to median)
    • Adaptive threshold detection
    • 3D time-synchronised signal projection
    • Comprehensive HTML report + CSV tables
""")
        run_m4 = _ask_yn("Run data analysis (Module 4)?", True)
        if run_m4:
            m4_params = _collect_m4_params()
            m4_out = module_subdir(run_folder, "module_4_analysis")

            for result in all_m3_results:
                m4_uid = result["metadata"].get("user_id", "user_001")
                m4_sid = result["metadata"].get("session_id", "session_001")
                user_m4_dir = user_subdir(m4_out, m4_uid)

                print(f"\n  Analysing [{m4_uid}] ...")
                run_module_4(
                    source=result["run_folder"],
                    threshold_window_s=m4_params["window_s"],
                    threshold_mad_factor=m4_params["mad_factor"],
                    threshold_sustain_s=m4_params["sustain_s"],
                    session_id=m4_sid,
                    user_id=m4_uid,
                    output_dir=str(user_m4_dir),
                )

            manifest.record_module("M4", m4_out)

    # ── Finalise ──────────────────────────────────────────────────────────
    manifest_path = manifest.complete()
    elapsed = time.time() - t_start

    _section(f"✅  Pipeline Complete  ({elapsed:.1f}s)")
    total = count_files(run_folder)
    print(f"""
  📁  {rel_run}
      {total['csv']} CSV files  |  {total['png']} PNG plots  |  {total['html']} HTML reports
      {total['json']} JSON metadata files  |  {total['total']} total files

  Folder structure:
    {rel_run}/
      module_1_acquisition/  ← signal CSVs from acquisition
      module_3_preprocessing/ ← feature CSVs + signal plots
      module_4_analysis/      ← analysis report + tables + 3D plot
      run_manifest.json       ← full run index
""")
    _pause()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 STANDALONE
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing_standalone():
    """Run Module 3 independently on any existing data folder."""
    _banner("Data Preprocessing  —  Module 3")
    print("""
  Preprocess existing physiological signal data.

  Accepted sources:
    • Module 1A output folder  (EDA.csv, BVP.csv …)
    • Module 2 output folder   (PipelinePacket CSVs)
    • Any folder with signal CSV files

  Pipeline: Clean → Filter → Feature extraction → Normalise → Export
""")
    if not _ask_yn("Continue?", True):
        return

    t_start = time.time()
    m3_params = _collect_m3_params(skip_source=False)
    source = m3_params.pop("source", None)

    if source is None or not Path(source).exists():
        print("  ✗  No valid source selected. Aborting.")
        return

    source = Path(source)

    # Detect multi-user folders
    user_dirs = sorted([d for d in source.iterdir()
                        if d.is_dir() and d.name.startswith("user_")])
    if user_dirs:
        print(f"\n  Multi-user folder: {len(user_dirs)} users detected.")
        all_or_one = _ask_yn(
            f"Preprocess all {len(user_dirs)} users?  (No = select one)", True
        )
        targets = user_dirs if all_or_one else \
            [_browse_data_folder("user subfolder")]
    else:
        targets = [source]

    all_results = []
    for i, target in enumerate(targets, 1):
        # Try to load demographics from metadata.json
        dem = m3_params.get("demographics")
        for meta_candidate in (target / "metadata.json",
                               source / "metadata.json"):
            if dem is None and meta_candidate.exists():
                try:
                    with open(meta_candidate) as f:
                        meta = json.load(f)
                    user_meta = meta.get("user", {})
                    if user_meta:
                        dem = {k: user_meta[k]
                               for k in ("age", "gender", "ethnicity",
                                         "autism_severity", "verbal_status",
                                         "comorbidity")
                               if k in user_meta and user_meta[k] is not None}
                except Exception:
                    pass

        uid = target.name if target != source else "user_001"
        print(f"\n  [{i}/{len(targets)}]  Preprocessing: {uid}")

        user_out = run_folder / uid
        result = run_module_3(
            source=target, params=m3_params,
            session_id=source.name, user_id=uid,
            demographics=dem,
            output_dir=str(user_out),
        )
        all_results.append(result)

    elapsed = time.time() - t_start
    total = count_files(run_folder)
    _section(f"✅  Preprocessing Complete  ({elapsed:.1f}s)")
    print(f"""
  📁  {rel_run}
      {total['csv']} CSV files  |  {total['png']} PNG plots
      {total['total']} total files
""")
    _pause()


def run_analysis_standalone():
    """Run Module 4 independently on any existing Module 3 output."""
    _banner("Data Analysis  —  Module 4")
    print("""
  Analyse preprocessed physiological signals and extracted features.

  Accepted sources:
    • Module 3 output folder (features_raw/, cleaned signals)
    • Any folder with signal CSVs and feature CSVs
""")
    if not _ask_yn("Continue?", True):
        return

    source = _browse_data_folder("Module 3 output to analyse")
    if source is None or not Path(source).exists():
        print("  ✗  No valid source. Aborting.")
        return

    m4_params = _collect_m4_params()
    session_id = _ask("Session ID", Path(source).name)
    user_id = _ask("User ID", "user_001")

    t0 = time.time()
    run_module_4(
        source=source,
        threshold_window_s=m4_params["window_s"],
        threshold_mad_factor=m4_params["mad_factor"],
        threshold_sustain_s=m4_params["sustain_s"],
        session_id=session_id,
        user_id=user_id,
    )
    elapsed = time.time() - t0
    _section(f"✅  Analysis Complete  ({elapsed:.1f}s)")
    _pause()


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL MENU
# ─────────────────────────────────────────────────────────────────────────────

def main_menu():
    _banner()
    print("""
  Choose what to run.
""")
    idx = _choose("Select", range(4), [
        "Full pipeline        —  M2 (acquire) → M3 (preprocess) → M4 (analyse)",
        "Data preprocessing   —  M3 only on existing data",
        "Data analysis        —  M4 only on existing M3 output",
        "Exit",
    ])
    if idx == 0:
        run_full_pipeline()
    elif idx == 1:
        run_preprocessing_standalone()
    elif idx == 2:
        run_analysis_standalone()
    else:
        print("\n  Goodbye.\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", default=None,
                        choices=["full", "preprocess", "analyse", "menu"])
    args, _ = parser.parse_known_args()

    if args.mode == "full":
        run_full_pipeline()
    elif args.mode == "preprocess":
        run_preprocessing_standalone()
    elif args.mode == "analyse":
        run_analysis_standalone()
    else:
        main_menu()
