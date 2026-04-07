"""
=============================================================================
AUTISM PHYSIO-AI PIPELINE  |  pipeline_main.py
=============================================================================
Master pipeline orchestrator.

Runs the full sequence:
  Module 2  →  Module 1A  →  Module 3

Or Module 3 independently on existing data.

Launch commands (from repo root)
---------------------------------
  run_full_pipeline          Windows batch file
  ./run_full_pipeline.sh     Mac / Linux shell script

  run_data_preprocessing     Windows batch file  (Module 3 only)
  ./run_data_preprocessing.sh Mac / Linux shell script
=============================================================================
"""
from __future__ import annotations
import os
import sys
import re
import json
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "module_1a_data_simulation"))
sys.path.insert(0, str(REPO_ROOT / "module_2_data_acquisition"))
sys.path.insert(0, str(REPO_ROOT / "module_3_preprocessing"))

PIPELINE_VERSION = "1.0.0"
WIDTH = 66

# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _clear():
    os.system("cls" if os.name == "nt" else "clear")

def _hline(char="─"):
    print(char * WIDTH)

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
            print(f"  ✗  Enter a number between {lo} and {hi}.")
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
    raw  = input(f"\n  {prompt}  ({hint}): ").strip().lower()
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
# FOLDER BROWSER (shared between modules)
# ─────────────────────────────────────────────────────────────────────────────

def _browse_data_folder(purpose="existing data") -> Optional[Path]:
    """
    Numbered list of candidate data folders from M1A and M2 outputs.
    User can also type a custom path.
    """
    _section(f"📂  Select {purpose} folder")

    candidates = []
    labels     = []

    for out_root, tag in [
        (REPO_ROOT / "module_1a_data_simulation" / "outputs", "M1A"),
        (REPO_ROOT / "module_2_data_acquisition"  / "outputs", "M2"),
        (REPO_ROOT / "module_3_preprocessing"     / "outputs", "M3"),
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
                         else f"  ({n_csv} CSV files)"
                candidates.append(d)
                labels.append(f"[{tag}]  {rel.name}{suffix}")

    labels.append("📁  Type a custom path")

    print()
    for i, lbl in enumerate(candidates, 1):
        print(f"    {i}.  {labels[i-1]}")
    print(f"    {len(candidates)+1}.  {labels[-1]}")

    while True:
        raw = input(f"\n  Select [1]: ").strip() or "1"

        # Allow typing a path directly
        if raw.startswith(".") or raw.startswith("/") or \
           (len(raw) > 2 and raw[1] == ":"):
            p = Path(raw)
            if p.exists():
                return p
            print(f"  ✗  Path not found: {raw}")
            continue

        try:
            idx = int(raw) - 1
        except ValueError:
            p = Path(raw)
            return p if p.exists() else None

        if idx == len(candidates):
            raw_path = _ask("Enter the full folder path")
            p = Path(raw_path)
            return p if p.exists() else None

        if 0 <= idx < len(candidates):
            chosen = candidates[idx]
            # Handle multi-user subfolders
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

        print(f"  ✗  Enter 1–{len(candidates)+1}.")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def _collect_m2_params() -> dict:
    """
    Ask user what data acquisition mode to use.
    Returns a dict the pipeline uses to call Module 2.
    """
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
    mode  = modes[mode_idx]

    params = {"mode": mode}

    if mode == "import":
        folder = _browse_data_folder("existing signal data")
        params["source"] = folder
        params["user_id"] = _ask("Participant ID", "imported_user")

    elif mode == "simulate":
        # These params are passed to Module 1A via M2's simulate runner
        _section("⚙️  Simulation settings")
        params["n_users"]    = _ask_int("Number of users", 1, 1, 500)
        params["duration_s"] = _ask_float("Duration per user (seconds)", 300, 30)
        params["n_events"]   = _ask_int_or_random("Events per user", 5)
        params["event_dur"]  = _ask_float_or_random("Event duration (seconds)", 30)

        try:
            from config import EMOTIONS_ALL
        except ImportError:
            EMOTIONS_ALL = ["Happy","Anger","Fear","Disgust","Sad","Surprise",
                            "Hunger","Thirst","Toilet","Tired"]

        print(f"\n  Available emotions: {', '.join(EMOTIONS_ALL)}")
        em_idx = _choose("Emotion mode",
                          ["random", "single", "subset"],
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

        params["noise"]  = ["low","medium","high"][_choose(
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
            params["seed"] = int(seed_raw) or __import__("random").randint(1,999999)
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
            params["speed"]  = _ask_float("Replay speed multiplier", 10.0, 0.1)
        else:
            params["e4_host"] = _ask("E4 server host", "127.0.0.1")
            params["e4_port"] = _ask_int("E4 server port", 28000, 1, 65535)
        max_dur = _ask("Auto-stop after N seconds  (Enter = manual stop)", "")
        params["max_dur"] = float(max_dur) if max_dur else None

    elif mode == "deployment":
        folder = _browse_data_folder("deployment signal data")
        params["source"]       = folder
        params["user_id"]      = _ask("Participant ID  (can be anonymised)", "anon_001")
        params["strip_labels"] = _ask_yn("Strip annotation columns?", True)

    return params


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def _collect_m3_params(skip_source=False) -> dict:
    """Collect Module 3 preprocessing parameters."""
    params = {}

    if not skip_source:
        _section("📂  Module 3 — Select data to preprocess")
        folder = _browse_data_folder("data to preprocess")
        params["source"] = folder

    _section("👤  Participant demographics")
    print("""
  Demographics are fused with signal features in the combined CSV.
  Required for multimodal models. Press Enter to skip.
""")
    if _ask_yn("Add participant demographics?", True):
        try:
            age = max(5, min(15, int(_ask("Age (5–15)", "10"))))
        except ValueError:
            age = 10
        params["demographics"] = {
            "age": age,
            "gender": ["Male","Female","Non-binary"][_choose(
                "Gender", range(3), ["Male","Female","Non-binary"])],
            "ethnicity": [
                "White British","Asian / Asian British",
                "Black / African / Caribbean","Mixed / Multiple","Other"
            ][_choose("Ethnicity", range(5),
                ["White British","Asian / Asian British",
                 "Black / African / Caribbean","Mixed / Multiple","Other"])],
            "autism_severity": ["Low","Medium","Severe"][_choose(
                "Autism severity  (DSM-5 Level)", range(3),
                ["Low (Level 1)","Medium (Level 2)","Severe (Level 3)"])],
            "verbal_status": ["Verbal","Minimally verbal","Non-verbal"][_choose(
                "Verbal status", range(3),
                ["Verbal","Minimally verbal","Non-verbal"])],
            "comorbidity": "Yes" if _ask_yn(
                "Any co-occurring conditions?", True) else "No",
        }
    else:
        params["demographics"] = None

    _section("🔧  Signal filtering")
    print("""
  Butterworth: Zero-phase LP/BP filter  (recommended for all signals)
  Kalman:      Gaussian noise smoother  (best for ST, EDA tonic)
  Hampel only: Outlier removal only
  None:        Skip frequency filtering  (Hampel still applied)
""")
    filt_names = ["butterworth","kalman","hampel_only","none"]
    filt_idx   = _choose("Filter type", range(4),
                         ["Butterworth  (recommended)",
                          "Kalman smoother",
                          "Hampel outlier removal only",
                          "No frequency filter"])
    params["filter_type"]  = filt_names[filt_idx]
    params["apply_hampel"] = _ask_yn("Apply Hampel outlier pre-filter?", True)

    _section("⏱  Feature extraction window")
    print("""
  Recommended: 60 s with 50% overlap.
  Shorter windows (30 s) suit real-time use but weaken HRV features.
""")
    params["window_s"] = _ask_float("Window size (seconds)", 60.0, 10.0)
    params["overlap"]  = _ask_float("Overlap fraction  (0.0–0.9)", 0.5, 0.0)

    params["gen_plots"] = _ask_yn("Generate visualisation plots?", True)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# RUN MODULE 2
# ─────────────────────────────────────────────────────────────────────────────

def run_module_2(params: dict) -> list:
    """Execute Module 2 with the given parameters. Returns list of PipelinePackets."""
    from acquisition_module import DataAcquisitionModule

    mode = params["mode"]
    acq  = DataAcquisitionModule(
        mode         = {
            "import":     "2.1",
            "simulate":   "2.2",
            "live_file":  "2.3",
            "live_e4":    "2.3",
            "deployment": "2.4",
        }[mode],
        save_packets = True,
        verbose      = True,
    )

    if mode == "import":
        packet = acq.run_import(
            source_path = params["source"],
            user_id     = params.get("user_id", "imported_user"),
        )
        return [packet]

    elif mode == "simulate":
        packets = acq.run_simulate(
            duration_s       = params["duration_s"],
            n_events         = params["n_events"],
            event_duration_s = params["event_dur"],
            emotions         = params.get("emotions"),
            noise_level      = params["noise"],
            seed             = params["seed"],
            n_users          = params["n_users"],
            shared_events    = params.get("shared_events", False),
            save_m1a_output  = params.get("save_m1a", False),
        )
        return packets

    elif mode == "live_file":
        packet = acq.run_live_file(
            csv_path       = params["source"],
            user_id        = params.get("user_id", "participant_001"),
            speed_factor   = params.get("speed", 10.0),
            max_duration_s = params.get("max_dur"),
        )
        return [packet]

    elif mode == "live_e4":
        packet = acq.run_live_e4(
            user_id        = params.get("user_id", "participant_001"),
            host           = params.get("e4_host", "127.0.0.1"),
            port           = params.get("e4_port", 28000),
            max_duration_s = params.get("max_dur"),
        )
        return [packet]

    elif mode == "deployment":
        packet = acq.run_deployment(
            source_path  = params["source"],
            user_id      = params.get("user_id", "anon_001"),
            strip_labels = params.get("strip_labels", True),
        )
        return [packet]

    return []


# ─────────────────────────────────────────────────────────────────────────────
# RUN MODULE 3
# ─────────────────────────────────────────────────────────────────────────────

def run_module_3(
    source,
    params:      dict,
    session_id:  str = "pipeline_session",
    user_id:     str = "unknown",
    demographics: dict = None,
) -> dict:
    """Execute Module 3 preprocessing on a source (packet, path, or dict)."""
    from preprocessor import DataPreprocessor

    pp = DataPreprocessor(
        filter_type    = params.get("filter_type",  "butterworth"),
        apply_hampel   = params.get("apply_hampel", True),
        window_s       = params.get("window_s",     60.0),
        overlap        = params.get("overlap",      0.5),
        generate_plots = params.get("gen_plots",    True),
        verbose        = True,
    )
    return pp.run(
        signals_input = source,
        demographics  = demographics or params.get("demographics"),
        session_id    = session_id,
        user_id       = user_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE  (M2 → M1A → M3)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline():
    """
    Interactive full pipeline:
      Step 1 — Module 2: Data Acquisition
      Step 2 — Module 3: Preprocessing (per packet / per user)
    """
    _banner("Full Pipeline  —  M2 → M3")
    print("""
  This runs the complete data ingestion and preprocessing pipeline.

  Flow:
    ① Data Acquisition  (Module 2)
         Import / Simulate / Live / Deployment
    ② Preprocessing     (Module 3)
         Clean → Filter → Extract → Normalise → Export
""")

    if not _ask_yn("Continue?", True):
        return

    t_start = time.time()

    # ── Step 1: Module 2 ──────────────────────────────────────────────
    _step_header(1, 2, "Data Acquisition  (Module 2)")
    m2_params = _collect_m2_params()

    print("\n  Starting Module 2 ...")
    packets = run_module_2(m2_params)
    _ok(f"Module 2 complete — {len(packets)} packet(s) acquired")

    if not packets:
        print("\n  ✗  No data packets returned from Module 2. Aborting.")
        return

    # ── Step 2: Module 3 — one run per packet ─────────────────────────
    _step_header(2, 2, "Data Preprocessing  (Module 3)")

    # Collect M3 params once (shared for all packets)
    m3_params = _collect_m3_params(skip_source=True)

    all_m3_results = []
    for i, packet in enumerate(packets, 1):
        uid = getattr(packet, "user_id", f"user_{i:03d}")
        sid = getattr(packet, "session_id", f"session_{i:03d}")
        print(f"\n  Processing packet {i}/{len(packets)}  [{uid}]")

        # Extract demographics from packet metadata if available
        dem = m3_params.get("demographics")
        if dem is None and hasattr(packet, "metadata"):
            meta_user = packet.metadata.get("user", {})
            if meta_user:
                dem = {
                    k: meta_user.get(k)
                    for k in ("age","gender","ethnicity",
                              "autism_severity","verbal_status","comorbidity")
                    if meta_user.get(k) is not None
                }

        result = run_module_3(
            source       = packet,
            params       = m3_params,
            session_id   = sid,
            user_id      = uid,
            demographics = dem,
        )
        all_m3_results.append(result)

    elapsed = time.time() - t_start
    _section(f"✅  Full Pipeline Complete  ({elapsed:.1f}s)")
    print(f"""
  Packets processed : {len(packets)}
  M3 output folders :""")
    for r in all_m3_results:
        try:
            rel = r["run_folder"].relative_to(REPO_ROOT)
        except Exception:
            rel = r["run_folder"]
        print(f"    {rel}")
    print()
    _pause()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 STANDALONE
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing_standalone():
    """
    Run Module 3 independently on any existing data folder.
    Used via run_data_preprocessing launcher.
    """
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

    # Collect params (includes source selection)
    m3_params = _collect_m3_params(skip_source=False)
    source    = m3_params.pop("source", None)

    if source is None or not Path(source).exists():
        print("  ✗  No valid source selected. Aborting.")
        return

    # Check for multi-user folders (run M3 per user subfolder)
    source = Path(source)
    user_dirs = sorted([d for d in source.iterdir()
                        if d.is_dir() and d.name.startswith("user_")])

    if user_dirs:
        print(f"\n  Multi-user folder detected: {len(user_dirs)} users.")
        all_or_one = _ask_yn(
            f"Preprocess all {len(user_dirs)} users? "
            f"(No = select one user)", True
        )
        targets = user_dirs if all_or_one else [_browse_data_folder("user subfolder")]
    else:
        targets = [source]

    all_results = []
    for i, target in enumerate(targets, 1):
        # Attempt to read demographics from metadata.json
        dem = m3_params.get("demographics")
        meta_path = target / "metadata.json"
        if not meta_path.exists():
            meta_path = source / "metadata.json"
        if dem is None and meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                user_meta = meta.get("user", {})
                if user_meta:
                    dem = {
                        k: user_meta.get(k)
                        for k in ("age","gender","ethnicity",
                                  "autism_severity","verbal_status","comorbidity")
                        if user_meta.get(k) is not None
                    }
            except Exception:
                pass

        uid = target.name if target != source else "user_001"
        sid = source.name
        print(f"\n  [{i}/{len(targets)}]  Processing: {uid}")

        result = run_module_3(
            source       = target,
            params       = m3_params,
            session_id   = sid,
            user_id      = uid,
            demographics = dem,
        )
        all_results.append(result)

    elapsed = time.time() - t_start
    _section(f"✅  Preprocessing Complete  ({elapsed:.1f}s)")
    for r in all_results:
        try:
            rel = r["run_folder"].relative_to(REPO_ROOT)
        except Exception:
            rel = r["run_folder"]
        print(f"    {rel}")
    print()
    _pause()


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL MENU  (when run with no arguments)
# ─────────────────────────────────────────────────────────────────────────────

def main_menu():
    """Show top-level menu when pipeline_main.py is run directly."""
    _banner()
    print("""
  Choose what to run.
""")
    idx = _choose(
        "Select",
        range(3),
        [
            "Full pipeline        —  M2 (acquire)  →  M3 (preprocess)",
            "Data preprocessing   —  M3 only on existing data",
            "Exit",
        ],
    )
    if idx == 0:
        run_full_pipeline()
    elif idx == 1:
        run_preprocessing_standalone()
    else:
        print("\n  Goodbye.\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", default=None,
                        choices=["full", "preprocess", "menu"])
    args, _ = parser.parse_known_args()

    if args.mode == "full":
        run_full_pipeline()
    elif args.mode == "preprocess":
        run_preprocessing_standalone()
    else:
        main_menu()
