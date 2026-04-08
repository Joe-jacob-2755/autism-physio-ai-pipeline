"""
=============================================================================
MODULE 2 – DATA ACQUISITION  |  main.py
=============================================================================
Interactive menu-driven entry point.

HOW TO LAUNCH (from the repo root)
-----------------------------------
  Windows:    run_data_acquisition_module.bat
  Mac/Linux:  ./run_data_acquisition_module.sh

Or directly:
  cd module_2_data_acquisition
  python main.py
=============================================================================
"""

from __future__ import annotations
from acquisition_module import (
    DataAcquisitionModule, MODULE_VERSION,
    MODE_IMPORT, MODE_SIMULATE, MODE_LIVE, MODE_DEPLOYMENT,
)
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Optional

# ── Ensure module directory is on sys.path ────────────────────────────────────
MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
M1A_DIR = REPO_ROOT / "module_2a_data_simulation"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(M1A_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

WIDTH = 64


def _line(char="─"):
    print(char * WIDTH)


def _header(title: str):
    print()
    print("═" * WIDTH)
    print(f"  {title}")
    print("═" * WIDTH)


def _section(title: str):
    print()
    print("  " + "─" * (WIDTH - 4))
    print(f"  {title}")
    print("  " + "─" * (WIDTH - 4))


def _ask(prompt: str, default: str = "") -> str:
    """Prompt user and return input, falling back to default on empty."""
    suffix = f" [{default}]" if default != "" else ""
    raw = input(f"\n  {prompt}{suffix}: ").strip()
    return raw if raw else str(default)


def _ask_int(prompt: str, default: int, min_val: int = 1,
             max_val: int = 9999) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  ✗  Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  ✗  Please enter a whole number.")


def _ask_float(prompt: str, default: float, min_val: float = 0.0) -> float:
    while True:
        raw = _ask(prompt, str(default))
        try:
            val = float(raw)
            if val >= min_val:
                return val
            print(f"  ✗  Must be ≥ {min_val}.")
        except ValueError:
            print("  ✗  Please enter a number.")


def _ask_int_or_random(prompt: str, default) -> object:
    raw = _ask(f"{prompt}  (number or 'random')", str(default))
    return "random" if raw.lower() == "random" else int(raw)


def _ask_float_or_random(prompt: str, default) -> object:
    raw = _ask(f"{prompt}  (number or 'random')", str(default))
    return "random" if raw.lower() == "random" else float(raw)


def _ask_yn(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = _ask(f"{prompt}  ({hint})", "")
    if raw == "":
        return default
    return raw.lower() in ("y", "yes")


def _choose(prompt: str, options: list, labels: list = None) -> int:
    """
    Present a numbered menu and return the 0-based index of the choice.
    options : list of option values (returned)
    labels  : optional display strings (defaults to str(options))
    """
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
            print(f"  ✗  Enter a number between 1 and {len(options)}.")
        except ValueError:
            print("  ✗  Please enter a number.")


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _pause():
    input("\n  Press Enter to continue ...")


# ─────────────────────────────────────────────────────────────────────────────
# WELCOME BANNER
# ─────────────────────────────────────────────────────────────────────────────

def _show_banner():
    _clear()
    print()
    print("  " + "═" * (WIDTH - 4))
    print("  ║" + " " * (WIDTH - 6) + "║")
    print("  ║    🧠  AUTISM PHYSIO-AI PIPELINE" + " " * (WIDTH - 41) + "║")
    print("  ║        Module 2 — Data Acquisition" + " " * (WIDTH - 43) + "║")
    print(f"  ║        v{MODULE_VERSION}" + " " * (WIDTH - 14) + "║")
    print("  ║" + " " * (WIDTH - 6) + "║")
    print("  " + "═" * (WIDTH - 4))


# ─────────────────────────────────────────────────────────────────────────────
# FOLDER BROWSER
# ─────────────────────────────────────────────────────────────────────────────

def _browse_for_folder(purpose: str = "data") -> Optional[Path]:
    """
    Present a numbered list of candidate data folders for the user to choose,
    or allow them to type a custom path.

    Searches (in order):
      1. Module 1A outputs
      2. Module 2 outputs
      3. Repo root
    """
    _section(f"📂  Select {purpose} folder")

    candidates = []

    # Collect Module 1A outputs
    m1a_out = M1A_DIR / "outputs"
    if m1a_out.exists():
        for d in sorted(m1a_out.iterdir()):
            if d.is_dir() and any(
                (d / f).exists()
                for f in ("combined_signals.csv", "EDA.csv", "BVP.csv")
            ):
                candidates.append(d)

    # Collect Module 2 outputs (for re-import)
    m2_out = MODULE_DIR / "outputs"
    if m2_out.exists():
        for d in sorted(m2_out.iterdir()):
            if d.is_dir() and (d / "combined_signals.csv").exists():
                candidates.append(d)

    # Build display labels
    labels = []
    for c in candidates:
        try:
            rel = c.relative_to(REPO_ROOT)
        except ValueError:
            rel = c
        # Count signal files as a hint
        n_csv = len(list(c.glob("*.csv")))
        # Check for multi-user (subfolders with user_*)
        n_users = len([x for x in c.iterdir()
                       if x.is_dir() and x.name.startswith("user_")])
        tag = f"  ({n_users} users)" if n_users > 0 else f"  ({n_csv} CSV files)"
        labels.append(f"{rel}{tag}")

    labels.append("📁  Type a custom folder path")

    print(f"\n  Available {purpose} folders:\n")
    for i, lbl in enumerate(labels, 1):
        print(f"    {i}.  {lbl}")

    while True:
        raw = _ask("Select folder number (or type a path directly)", "1")

        # If user typed a path directly
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
            print("  ✗  Enter a number or a folder path.")
            continue

        if idx == len(labels) - 1:
            # Custom path option
            raw_path = _ask("Enter the full folder path")
            p = Path(raw_path)
            if p.exists():
                return p
            print(f"  ✗  Folder not found: {raw_path}")
            continue

        if 0 <= idx < len(candidates):
            chosen = candidates[idx]
            # If multi-user folder, let user pick a sub-user or whole folder
            user_dirs = sorted([
                x for x in chosen.iterdir()
                if x.is_dir() and x.name.startswith("user_")
            ])
            if user_dirs:
                _section("Multiple users found in this folder")
                sub_labels = [f"{u.name}" for u in user_dirs]
                sub_labels.insert(0, "📦  Use entire multi-user folder")
                sub_idx = _choose("Which data to use?", range(len(sub_labels)),
                                  sub_labels)
                if sub_idx == 0:
                    return chosen
                return user_dirs[sub_idx - 1]
            return chosen

        print(f"  ✗  Enter a number between 1 and {len(labels)}.")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2.1 — USE EXISTING DATA
# ─────────────────────────────────────────────────────────────────────────────

def _run_import_flow(acq: DataAcquisitionModule):
    _header("Mode 2.1  —  Use Existing Data")
    print("""
  This mode loads pre-existing physiological signal data into the pipeline.
  It accepts any folder produced by Module 1A, or a custom CSV file.
""")

    folder = _browse_for_folder("existing signal data")
    if folder is None:
        print("  No folder selected. Returning to menu.")
        return

    user_id = _ask("Participant ID (label for this data)", "imported_user")

    print(f"\n  ✔  Loading from:  {folder}")
    print(f"  ✔  User ID:       {user_id}\n")

    acq.run_import(source_path=folder, user_id=user_id)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2.2 — CREATE NEW SIMULATED DATA
# ─────────────────────────────────────────────────────────────────────────────

def _run_simulate_flow(acq: DataAcquisitionModule):
    _header("Mode 2.2  —  Create New Simulated Data")
    print("""
  This mode generates new synthetic physiological signals using Module 1A.
  You will be guided through all parameters step by step.
""")

    # ── Number of users ───────────────────────────────────────────────────
    _section("👥  How many users to simulate?")
    print("""
  Single user  → one set of physiological signals (for user-specific models)
  Multiple users → multiple sets with realistic inter-subject variability
                   (for global / population-level models)
""")
    n_users = _ask_int("Number of users", default=1, min_val=1, max_val=500)

    # ── Recording duration ────────────────────────────────────────────────
    _section("⏱  How long should each recording be?")
    print("""
  Recommended minimum: 120 s (2 min)
  Typical research session: 300–600 s (5–10 min)
""")
    duration_s = _ask_float("Duration per user (seconds)", default=300.0,
                            min_val=30.0)

    # ── Number of events ──────────────────────────────────────────────────
    _section("📍  How many emotion / behaviour events?")
    print("""
  Each event is a labelled window where a specific emotion or behaviour
  is simulated. Enter a number, or 'random' to let the system decide.
""")
    n_events = _ask_int_or_random("Number of events per user", default=5)

    # ── Event duration ────────────────────────────────────────────────────
    _section("⏳  How long should each event last?")
    print("""
  Typical event duration: 15–45 seconds.
  Enter a number in seconds, or 'random' to vary per event (10–60 s).
""")
    event_dur = _ask_float_or_random("Event duration (seconds)", default=30.0)

    # ── Target emotions ───────────────────────────────────────────────────
    _section("🎭  Which emotions / behaviours to simulate?")

    EMOTION_OPTIONS = [
        ("Random (all 10 states)",
         "Each event picks randomly from all available states"),
        ("Specific single emotion",
         "All events use one emotion you choose"),
        ("Select a subset",
         "Events randomly chosen from a list you specify"),
    ]
    print()
    for i, (name, desc) in enumerate(EMOTION_OPTIONS, 1):
        print(f"    {i}.  {name}")
        print(f"         {desc}")

    emotion_choice = _choose("Choose emotion mode", range(3),
                             [e[0] for e in EMOTION_OPTIONS])
    emotions = None

    try:
        from config import EMOTIONS_ALL, EMOTIONS_AFFECTIVE, EMOTIONS_NEEDS
    except ImportError:
        EMOTIONS_ALL = [
            "Happy", "Anger", "Fear", "Disgust", "Sad", "Surprise",
            "Hunger", "Thirst", "Toilet", "Tired",
        ]
        EMOTIONS_AFFECTIVE = ["Happy", "Anger", "Fear", "Disgust", "Sad", "Surprise"]
        EMOTIONS_NEEDS = ["Hunger", "Thirst", "Toilet", "Tired"]

    if emotion_choice == 1:
        # Single emotion
        print(f"\n  Affective emotions : {', '.join(EMOTIONS_AFFECTIVE)}")
        print(f"  Physiological needs: {', '.join(EMOTIONS_NEEDS)}")
        while True:
            raw = _ask("Enter emotion name (exact, case-sensitive)")
            if raw in EMOTIONS_ALL:
                emotions = raw
                break
            print(f"  ✗  '{raw}' not recognised. Valid: {EMOTIONS_ALL}")

    elif emotion_choice == 2:
        # Subset
        print(f"\n  Available emotions: {', '.join(EMOTIONS_ALL)}")
        while True:
            raw = _ask("Enter comma-separated list  e.g. Fear,Anger,Tired")
            lst = [e.strip() for e in raw.split(",") if e.strip()]
            bad = [e for e in lst if e not in EMOTIONS_ALL]
            if bad:
                print(f"  ✗  Unknown: {bad}.  Valid: {EMOTIONS_ALL}")
            elif len(lst) == 0:
                print("  ✗  Please enter at least one emotion.")
            else:
                emotions = lst
                break

    # ── Noise level ───────────────────────────────────────────────────────
    _section("📡  Signal noise level")
    print("""
  Low    — clean signal, ideal conditions (lab)
  Medium — realistic wearable noise (recommended)
  High   — poor electrode contact, movement artefacts
""")
    noise_idx = _choose("Noise level",
                        ["low", "medium", "high"],
                        ["Low    – clean signal",
                         "Medium – realistic (recommended)",
                         "High   – noisy, poor contact"])
    noise_level = ["low", "medium", "high"][noise_idx]

    # ── Shared events (multi-user only) ───────────────────────────────────
    shared_events = False
    if n_users > 1:
        _section("🔗  Event schedule across users")
        print("""
  Independent (default): each user gets a different randomly timed
                          event schedule — more realistic variability.

  Shared:                 all users see the exact same events at the
                          same times — useful for controlled comparison
                          of individual physiological responses.
""")
        shared_events = _ask_yn("Share the same event schedule for all users?",
                                default=False)

    # ── Random seed ───────────────────────────────────────────────────────
    _section("🎲  Reproducibility seed")
    print("""
  Using the same seed will reproduce identical data in future runs.
  Use 0 for a random seed each time.
""")
    seed_raw = _ask("Random seed (integer, or 0 for random)", "42")
    try:
        seed = int(seed_raw)
        if seed == 0:
            import random
            seed = random.randint(1, 999_999)
            print(f"  Using random seed: {seed}")
    except ValueError:
        seed = 42

    # ── Save Module 1A outputs ────────────────────────────────────────────
    _section("💾  Save Module 1A signal files?")
    print("""
  The simulation always produces a PipelinePacket for downstream use.
  Optionally also save the full Module 1A outputs (individual signal
  CSVs + PNG plots) to disk for inspection.
""")
    save_m1a = _ask_yn("Save Module 1A CSV files and plots?", default=False)

    # ── Summary + confirm ─────────────────────────────────────────────────
    _section("✅  Simulation Parameters Summary")
    print(f"""
  Users           : {n_users}
  Duration/user   : {duration_s:.0f} s  ({duration_s / 60:.1f} min)
  Events/user     : {n_events}
  Event duration  : {event_dur} s
  Emotions        : {emotions if emotions else 'random (all 10 states)'}
  Noise level     : {noise_level}
  Shared events   : {shared_events if n_users > 1 else 'N/A (single user)'}
  Seed            : {seed}
  Save M1A output : {save_m1a}
""")

    if not _ask_yn("Start simulation with these settings?", default=True):
        print("  Simulation cancelled.")
        return

    print()
    acq.run_simulate(
        duration_s=duration_s,
        n_events=n_events,
        event_duration_s=event_dur,
        emotions=emotions,
        noise_level=noise_level,
        seed=seed,
        n_users=n_users,
        shared_events=shared_events,
        save_m1a_output=save_m1a,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2.3 — LIVE DATA FROM DEVICE
# ─────────────────────────────────────────────────────────────────────────────

def _run_live_flow(acq: DataAcquisitionModule):
    _header("Mode 2.3  —  Live Data from Wearable Device")
    print("""
  Stream physiological signals from a wearable device in real time.
  A researcher annotates emotion / behaviour events during recording.
""")

    _section("📟  Select data source")
    device_idx = _choose(
        "Device or source",
        ["file", "e4"],
        [
            "Test replay  — replay an existing CSV file at speed "
            "(no device needed)",
            "Empatica E4  — stream from a connected E4 device "
            "(E4 Streaming Server must be running)",
        ],
    )

    user_id = _ask("Participant ID", "participant_001")
    max_dur_raw = _ask(
        "Auto-stop after N seconds  (press Enter for manual stop)", "")
    max_dur = float(max_dur_raw) if max_dur_raw else None

    if device_idx == 0:
        # File replay
        csv_path = _browse_for_folder("test replay CSV")
        if csv_path is None:
            print("  No file selected.")
            return
        # If a folder was selected, look for combined_signals.csv inside
        if csv_path.is_dir():
            combined = csv_path / "combined_signals.csv"
            if not combined.exists():
                print(f"  ✗  combined_signals.csv not found in {csv_path}")
                return
            csv_path = combined

        speed = _ask_float("Replay speed multiplier  (1=realtime, 10=10× faster)",
                           default=10.0, min_val=0.1)
        acq.run_live_file(csv_path=csv_path, user_id=user_id,
                          speed_factor=speed, max_duration_s=max_dur)

    else:
        # Empatica E4
        host = _ask("E4 Streaming Server host", "127.0.0.1")
        port = _ask_int("E4 Streaming Server port", default=28000,
                        min_val=1, max_val=65535)
        acq.run_live_e4(user_id=user_id, host=host,
                        port=port, max_duration_s=max_dur)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2.4 — DEPLOYMENT (NON-ANNOTATED)
# ─────────────────────────────────────────────────────────────────────────────

def _run_deployment_flow(acq: DataAcquisitionModule):
    _header("Mode 2.4  —  Deployment Data (No Annotation)")
    print("""
  Load raw, unannotated physiological data for inference in the
  deployed model.  No ground-truth labels are required.

  The data is routed directly to Module 9 (Deployment Inference)
  and will NOT enter the training/testing pipeline.
""")

    folder = _browse_for_folder("deployment signal data")
    if folder is None:
        print("  No folder selected.")
        return

    user_id = _ask("Participant ID  (can be anonymised)", "anon_001")

    strip = _ask_yn(
        "Remove any existing annotation columns?  (recommended for privacy)",
        default=True)

    print(f"\n  ✔  Source:  {folder}")
    print(f"  ✔  User:    {user_id}")
    print(f"  ✔  Strip labels: {strip}\n")

    acq.run_deployment(source_path=folder, user_id=user_id,
                       strip_labels=strip)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MENU
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _show_banner()

    print("""
  This module is the gateway to the Autism Physio-AI Pipeline.
  All data — simulated, imported, live, or for deployment — enters
  the pipeline through here as a standardised PipelinePacket.
""")

    while True:
        _section("What would you like to do?")

        MENU_OPTIONS = [
            ("Use existing data",
             "Import pre-existing signal data from a folder into the pipeline"),
            ("Create new simulated data",
             "Generate synthetic physiological data using Module 1A"),
            ("Stream live data from device",
             "Acquire real-time signals from a wearable with annotation"),
            ("Ingest data for deployment",
             "Load non-annotated data for the deployed inference model"),
            ("Exit",
             "Quit the Data Acquisition Module"),
        ]

        print()
        for i, (name, desc) in enumerate(MENU_OPTIONS, 1):
            print(f"    {i}.  {name}")
            print(f"         {desc}")
            if i < len(MENU_OPTIONS):
                print()

        choice = _choose("Enter your choice", range(len(MENU_OPTIONS)),
                         [o[0] for o in MENU_OPTIONS])

        if choice == 4:
            print("\n  Goodbye.\n")
            break

        acq = DataAcquisitionModule(
            mode=[MODE_IMPORT, MODE_SIMULATE, MODE_LIVE,
                  MODE_DEPLOYMENT][choice],
            save_packets=True,
            verbose=True,
        )

        try:
            if choice == 0:
                _run_import_flow(acq)
            elif choice == 1:
                _run_simulate_flow(acq)
            elif choice == 2:
                _run_live_flow(acq)
            elif choice == 3:
                _run_deployment_flow(acq)
        except KeyboardInterrupt:
            print("\n\n  Interrupted. Returning to main menu.")

        _pause()
        _show_banner()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
