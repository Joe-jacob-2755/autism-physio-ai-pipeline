"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  main.py
=============================================================================
Entry point for the physiological signal simulator.

FOLDER STRUCTURE
----------------
autism-physio-ai-pipeline/              <- repo root (git clone lands here)
└── module_1a_data_simulation/          <- run ALL commands from HERE
        main.py                         <- this file
        outputs/                        <- all run outputs saved here
            M1A_v1.0.0_run_001/         <- auto-created, auto-numbered
            M1A_v1.0.0_run_002/

HOW TO RUN
----------
  # 1. From the repo root, enter the module folder:
  cd module_1a_data_simulation

  # 2. Activate virtual environment:
  ../.venv/Scripts/activate            (Windows)
  source ../.venv/bin/activate         (Mac/Linux)

  # 3. Run:
  python main.py                                   # auto-numbered output
  python main.py --duration 600                    # 10 minutes
  python main.py --emotion Anger --n_events 4
  python main.py --emotions "Anger,Fear,Surprise"
  python main.py --n_events random --event_dur random
  python main.py --out my_experiment               # custom folder name

OUTPUT FOLDERS
--------------
  Every run auto-creates a new numbered subfolder:
    outputs/M1A_v1.0.0_run_001/
    outputs/M1A_v1.0.0_run_002/
    ...

  Use --out <name> to override with a custom folder name:
    python main.py --out anger_test
    -> saved to:  outputs/anger_test/
=============================================================================
"""

from __future__ import annotations
import argparse
import sys
import os
import re
from pathlib import Path

# Ensure module directory is on sys.path regardless of where Python is invoked
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from config import (
    EMOTIONS_ALL, DEFAULT_DURATION_S, DEFAULT_NOISE_LEVEL,
    DEFAULT_SEED, DEFAULT_N_EVENTS, MODULE_VERSION, MODULE_LABEL,
    OUTPUT_ROOT,
)
from simulator  import DataSimulator
from visualizer import SignalVisualizer
from exporter   import DataExporter


# -----------------------------------------------------------------------------
# AUTO-VERSIONED OUTPUT FOLDER
# -----------------------------------------------------------------------------

def next_run_folder(custom_name: str | None = None) -> Path:
    """
    Return a unique, auto-numbered output folder path and create it.

    Auto-naming  ->  outputs/M1A_v1.0.0_run_001/
                     outputs/M1A_v1.0.0_run_002/  (next available number)

    Custom name  ->  outputs/<custom_name>/
                     outputs/<custom_name>_002/   (if already exists)
    """
    output_root = MODULE_DIR / OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    version_tag = f"{MODULE_LABEL}_v{MODULE_VERSION}"

    if custom_name:
        base = output_root / custom_name
        if not base.exists():
            base.mkdir(parents=True)
            return base
        prefix  = custom_name
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    else:
        prefix  = f"{version_tag}_run"
        pattern = re.compile(rf"^{re.escape(version_tag)}_run_(\d+)$")

    existing = [
        int(m.group(1))
        for d in output_root.iterdir()
        if d.is_dir() and (m := pattern.match(d.name))
    ]
    next_num = (max(existing) + 1) if existing else 1
    folder   = output_root / f"{prefix}_{next_num:03d}"
    folder.mkdir(parents=True)
    return folder


# -----------------------------------------------------------------------------
# CLI ARGUMENT PARSER
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "main.py",
        description = (
            "Module 1A – Physiological Signal Data Simulator\n"
            "Outputs saved to:  module_1a_data_simulation/outputs/M1A_v<ver>_run_NNN/"
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--duration",   type=float, default=DEFAULT_DURATION_S,
                   help=f"Recording duration in seconds (default: {DEFAULT_DURATION_S})")
    p.add_argument("--n_events",   default=str(DEFAULT_N_EVENTS),
                   help="Number of events: integer or 'random' (default: 5)")
    p.add_argument("--event_dur",  default="30",
                   help="Event duration in seconds: float or 'random' (default: 30)")
    p.add_argument("--emotion",    default=None,
                   help="Single target emotion e.g. 'Anger'. Overrides --emotions.")
    p.add_argument("--emotions",   default=None,
                   help="Comma-separated emotion subset e.g. 'Anger,Fear,Sad'.")
    p.add_argument("--noise",      default=DEFAULT_NOISE_LEVEL,
                   choices=["low", "medium", "high"],
                   help=f"Noise level (default: {DEFAULT_NOISE_LEVEL})")
    p.add_argument("--seed",       type=int, default=DEFAULT_SEED,
                   help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    p.add_argument("--out",        default=None,
                   help="Custom output folder name inside outputs/. Default: auto-numbered.")
    p.add_argument("--no_plots",   action="store_true",
                   help="Skip PNG generation (CSV only — faster).")
    p.add_argument("--list_emotions", action="store_true",
                   help="Print all available emotion labels and exit.")
    return p


# -----------------------------------------------------------------------------
# ARGUMENT HELPERS
# -----------------------------------------------------------------------------

def resolve_n_events(val: str):
    return "random" if val.strip().lower() == "random" else int(val)

def resolve_event_dur(val: str):
    return "random" if val.strip().lower() == "random" else float(val)

def resolve_emotions(emotion_single, emotions_csv):
    if emotion_single is not None:
        if emotion_single not in EMOTIONS_ALL:
            raise ValueError(f"Unknown emotion '{emotion_single}'.\nValid: {EMOTIONS_ALL}")
        return emotion_single
    if emotions_csv is not None:
        lst = [e.strip() for e in emotions_csv.split(",")]
        for e in lst:
            if e not in EMOTIONS_ALL:
                raise ValueError(f"Unknown emotion '{e}'.\nValid: {EMOTIONS_ALL}")
        return lst
    return None


# -----------------------------------------------------------------------------
# MAIN SIMULATION RUNNER
# -----------------------------------------------------------------------------

def run_simulation(
    duration_s,
    n_events,
    event_duration_s,
    emotions,
    noise_level:    str,
    seed:           int,
    out_name:       str | None = None,
    generate_plots: bool       = True,
):
    """Full pipeline: simulate -> visualise -> export."""

    run_folder = next_run_folder(out_name)
    rel_path   = run_folder.relative_to(MODULE_DIR)

    print("=" * 62)
    print(" MODULE 1A  –  Physiological Signal Data Simulator")
    print(" Autism Emotion/Behaviour AI Pipeline")
    print("=" * 62)
    print(f"  Module version  :  {MODULE_LABEL} v{MODULE_VERSION}")
    print(f"  Duration        :  {duration_s:.0f} s  ({duration_s/60:.1f} min)")
    print(f"  Events          :  {n_events}")
    print(f"  Event duration  :  {event_duration_s} s")
    print(f"  Emotions        :  {emotions if emotions else 'random'}")
    print(f"  Noise level     :  {noise_level}")
    print(f"  Seed            :  {seed}")
    print(f"  Output folder   :  {rel_path}")
    print("=" * 62)

    # 1. Simulate
    result = DataSimulator(
        duration_s       = duration_s,
        n_events         = n_events,
        event_duration_s = event_duration_s,
        emotions         = emotions,
        noise_level      = noise_level,
        seed             = seed,
    ).simulate()

    print()
    print(result.summary())

    # 2. Visualise
    if generate_plots:
        print("\n[Main] Generating visualisations ...")
        plots = SignalVisualizer(result, output_dir=str(run_folder)).save_all()
        print(f"  -> {len(plots)} figure(s) saved")

    # 3. Export
    print("\n[Main] Exporting data files ...")
    files = DataExporter(result, output_dir=str(run_folder)).export_all()
    print(f"  -> {len(files)} file(s) saved")

    print(f"\n[Main] Run complete.  Output folder:")
    print(f"       {run_folder}\n")
    return result


# -----------------------------------------------------------------------------
# CLI ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.list_emotions:
        from config import EMOTION_PROFILES
        print("\nAvailable emotion / behaviour labels:")
        for cat in ("affective", "physiological_need"):
            print(f"\n  [{cat}]")
            for name, prof in EMOTION_PROFILES.items():
                if prof["category"] == cat:
                    print(f"    {name:<12}  – {prof['description']}")
        return

    try:
        n_ev   = resolve_n_events(args.n_events)
        ev_dur = resolve_event_dur(args.event_dur)
        emot   = resolve_emotions(args.emotion, args.emotions)
    except ValueError as e:
        print(f"\n[Error] {e}")
        sys.exit(1)

    run_simulation(
        duration_s       = args.duration,
        n_events         = n_ev,
        event_duration_s = ev_dur,
        emotions         = emot,
        noise_level      = args.noise,
        seed             = args.seed,
        out_name         = args.out,
        generate_plots   = not args.no_plots,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments -> run with defaults
        run_simulation(
            duration_s       = DEFAULT_DURATION_S,
            n_events         = DEFAULT_N_EVENTS,
            event_duration_s = 30.0,
            emotions         = None,
            noise_level      = DEFAULT_NOISE_LEVEL,
            seed             = DEFAULT_SEED,
            out_name         = None,
            generate_plots   = True,
        )
    else:
        main()
