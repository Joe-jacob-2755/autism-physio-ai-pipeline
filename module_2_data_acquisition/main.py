"""
=============================================================================
MODULE 2 – DATA ACQUISITION  |  main.py
=============================================================================
CLI entry point for Module 2.

FOLDER STRUCTURE
----------------
autism-physio-ai-pipeline/
└── module_2_data_acquisition/     <- run ALL commands from HERE
        main.py
        outputs/
            M2_v1.0.0_mode2_1_run_001/    <- import run
                EDA.csv, BVP.csv, ...
                combined_signals.csv
                packet_metadata.json
                module2_run_summary.json

            M2_v1.0.0_mode2_2_run_001/    <- simulate run (single user)
            M2_v1.0.0_mode2_2_run_002/    <- simulate run (multi-user)
                user_001/
                user_002/
                module2_run_summary.json

            M2_v1.0.0_mode2_3_run_001/    <- live run
            M2_v1.0.0_mode2_4_run_001/    <- deployment run

HOW TO RUN
----------
  cd module_2_data_acquisition
  ..\.venv\Scripts\activate          (Windows)
  source ../.venv/bin/activate       (Mac/Linux)

  # Mode 2.1 — Import existing data
  python main.py --mode 2.1 --source "../module_1a_data_simulation/outputs/M1A_v1.1.0_run_001"

  # Mode 2.2 — Simulate new data
  python main.py --mode 2.2 --duration 300 --n_events 5 --emotions "Fear,Anger"

  # Mode 2.2 — Interactive simulation setup
  python main.py --mode 2.2 --interactive

  # Mode 2.2 — Multi-user simulation
  python main.py --mode 2.2 --n_users 10 --duration 300 --shared_events

  # Mode 2.3 — Live stream test (replay a CSV)
  python main.py --mode 2.3 --source "../module_1a_data_simulation/outputs/M1A_v1.1.0_run_001/combined_signals.csv" --speed 10

  # Mode 2.3 — Empatica E4 device
  python main.py --mode 2.3 --device e4 --user_id participant_001

  # Mode 2.4 — Deployment (non-annotated)
  python main.py --mode 2.4 --source "../module_1a_data_simulation/outputs/M1A_v1.1.0_run_001"
=============================================================================
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from acquisition_module import (
    DataAcquisitionModule, MODULE_VERSION,
    MODE_IMPORT, MODE_SIMULATE, MODE_LIVE, MODE_DEPLOYMENT,
    MODE_DESCRIPTIONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI PARSER
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "main.py",
        description = (
            f"Module 2 – Data Acquisition  |  v{MODULE_VERSION}\n"
            "Routes data from any source into the pipeline as a PipelinePacket.\n\n"
            "Modes:\n"
            "  2.1  Import existing data from a folder or CSV\n"
            "  2.2  Create new simulated data via Module 1A\n"
            "  2.3  Stream live data from a wearable device\n"
            "  2.4  Ingest non-annotated data for deployment inference"
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # ── Universal args ────────────────────────────────────────────────────
    p.add_argument("--mode", required=True, choices=["2.1","2.2","2.3","2.4"],
                   help="Acquisition mode (required)")
    p.add_argument("--user_id",  default=None,
                   help="Participant identifier")
    p.add_argument("--out",      default=None,
                   help="Custom output folder name (auto-numbered if omitted)")
    p.add_argument("--no_save",  action="store_true",
                   help="Do not save packets to disk (return only)")

    # ── Mode 2.1 / 2.3 / 2.4: source path ───────────────────────────────
    p.add_argument("--source",   default=None,
                   help="Path to folder or CSV file (modes 2.1, 2.3 file, 2.4)")

    # ── Mode 2.2: simulation params ───────────────────────────────────────
    p.add_argument("--interactive", action="store_true",
                   help="[2.2] Launch interactive simulation setup prompt")
    p.add_argument("--duration",    type=float, default=300,
                   help="[2.2] Recording duration per user in seconds (default: 300)")
    p.add_argument("--n_events",    default="5",
                   help="[2.2] Events per user: integer or 'random' (default: 5)")
    p.add_argument("--event_dur",   default="30",
                   help="[2.2] Event duration: float or 'random' (default: 30)")
    p.add_argument("--emotion",     default=None,
                   help="[2.2] Single target emotion e.g. 'Anger'")
    p.add_argument("--emotions",    default=None,
                   help="[2.2] Comma-separated emotion subset e.g. 'Anger,Fear'")
    p.add_argument("--noise",       default="medium",
                   choices=["low","medium","high"],
                   help="[2.2] Noise level (default: medium)")
    p.add_argument("--seed",        type=int, default=42,
                   help="[2.2] Master random seed (default: 42)")
    p.add_argument("--n_users",     type=int, default=1,
                   help="[2.2] Number of users to simulate (default: 1)")
    p.add_argument("--shared_events", action="store_true",
                   help="[2.2] All users share the same event schedule")
    p.add_argument("--save_m1a",    action="store_true",
                   help="[2.2] Also save Module 1A CSV/PNG outputs")

    # ── Mode 2.3: live params ─────────────────────────────────────────────
    p.add_argument("--device",      default="file",
                   choices=["file", "e4"],
                   help="[2.3] Device type: 'file' (replay CSV) or 'e4' (Empatica E4)")
    p.add_argument("--speed",       type=float, default=10.0,
                   help="[2.3] Playback speed for file stream (default: 10.0 = 10x)")
    p.add_argument("--max_dur",     type=float, default=None,
                   help="[2.3] Auto-stop after N seconds (default: manual stop)")
    p.add_argument("--e4_host",     default="127.0.0.1",
                   help="[2.3] E4 streaming server host (default: 127.0.0.1)")
    p.add_argument("--e4_port",     type=int, default=28000,
                   help="[2.3] E4 streaming server port (default: 28000)")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def resolve_n_events(val):
    return "random" if str(val).strip().lower() == "random" else int(val)

def resolve_event_dur(val):
    return "random" if str(val).strip().lower() == "random" else float(val)

def resolve_emotions(single, csv_list):
    if single:
        return single
    if csv_list:
        return [e.strip() for e in csv_list.split(",")]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    acq = DataAcquisitionModule(
        mode         = args.mode,
        save_packets = not args.no_save,
        out_name     = args.out,
        verbose      = True,
    )

    # ── Mode 2.1: Import ──────────────────────────────────────────────────
    if args.mode == MODE_IMPORT:
        if not args.source:
            parser.error("--source is required for mode 2.1")
        uid = args.user_id or "imported_user"
        acq.run_import(source_path=args.source, user_id=uid)

    # ── Mode 2.2: Simulate ────────────────────────────────────────────────
    elif args.mode == MODE_SIMULATE:
        if args.interactive:
            acq.run_simulate_interactive()
        else:
            try:
                n_ev   = resolve_n_events(args.n_events)
                ev_dur = resolve_event_dur(args.event_dur)
                emot   = resolve_emotions(args.emotion, args.emotions)
            except (ValueError, TypeError) as e:
                parser.error(str(e))

            acq.run_simulate(
                duration_s       = args.duration,
                n_events         = n_ev,
                event_duration_s = ev_dur,
                emotions         = emot,
                noise_level      = args.noise,
                seed             = args.seed,
                n_users          = args.n_users,
                shared_events    = args.shared_events,
                save_m1a_output  = args.save_m1a,
            )

    # ── Mode 2.3: Live ────────────────────────────────────────────────────
    elif args.mode == MODE_LIVE:
        uid = args.user_id or "participant_001"

        if args.device == "file":
            if not args.source:
                parser.error(
                    "--source <combined_signals.csv> is required for "
                    "--mode 2.3 --device file"
                )
            acq.run_live_file(
                csv_path       = args.source,
                user_id        = uid,
                speed_factor   = args.speed,
                max_duration_s = args.max_dur,
            )
        elif args.device == "e4":
            acq.run_live_e4(
                user_id        = uid,
                host           = args.e4_host,
                port           = args.e4_port,
                max_duration_s = args.max_dur,
            )

    # ── Mode 2.4: Deployment ──────────────────────────────────────────────
    elif args.mode == MODE_DEPLOYMENT:
        if not args.source:
            parser.error("--source is required for mode 2.4")
        uid = args.user_id or "deployment_participant"
        acq.run_deployment(source_path=args.source, user_id=uid)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
    else:
        main()
