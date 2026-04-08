"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  main.py  (v1.1.0)
=============================================================================
Entry point — single-user and multi-user simulation.

FOLDER STRUCTURE
----------------
autism-physio-ai-pipeline/
└── module_1a_data_simulation/     <- run ALL commands from HERE
        main.py
        outputs/
            M1A_v1.1.0_run_001/    <- single user
                EDA.csv, BVP.csv, IBI.csv, ST.csv, ACC.csv
                combined_signals.csv
                signal_EDA.png ... combined_signals.png
                annotations_*.csv
                metadata.json

            M1A_v1.1.0_run_002/    <- multi-user (--n_users 5)
                user_001/          <- per-user subfolder
                    EDA.csv ...
                    signal_EDA.png ...
                user_002/
                    ...
                user_005/
                    ...
                all_users_combined.csv   <- all users stacked, user_id column
                run_summary.csv          <- one row per user, key physio params
                metadata.json

HOW TO RUN
----------
  cd module_1a_data_simulation
  ..\\.venv\\Scripts\activate          (Windows)
  source ../.venv/bin/activate       (Mac/Linux)

  python main.py                                        # 1 user, defaults
  python main.py --n_users 10                           # 10 users
  python main.py --n_users 5 --emotions "Fear,Anger"    # 5 users, subset
  python main.py --n_users 3 --shared_events            # same events per user
  python main.py --emotion Anger --n_users 8            # 8 users, single emotion
=============================================================================
"""

from __future__ import annotations
from user_profiles import UserProfileGenerator
from exporter import DataExporter, export_all_users_combined
from visualizer import SignalVisualizer
from simulator import DataSimulator
from config import (
    EMOTIONS_ALL, DEFAULT_DURATION_S, DEFAULT_NOISE_LEVEL,
    DEFAULT_SEED, DEFAULT_N_EVENTS, DEFAULT_N_USERS,
    MODULE_VERSION, MODULE_LABEL, OUTPUT_ROOT,
)
import argparse
import sys
import re
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-VERSIONED OUTPUT FOLDER
# ─────────────────────────────────────────────────────────────────────────────

def next_run_folder(custom_name=None) -> Path:
    output_root = MODULE_DIR / OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    version_tag = f"{MODULE_LABEL}_v{MODULE_VERSION}"

    if custom_name:
        base = output_root / custom_name
        if not base.exists():
            base.mkdir(parents=True)
            return base
        prefix = custom_name
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    else:
        prefix = f"{version_tag}_run"
        pattern = re.compile(rf"^{re.escape(version_tag)}_run_(\d+)$")

    existing = [
        int(m.group(1))
        for d in output_root.iterdir()
        if d.is_dir() and (m := pattern.match(d.name))
    ]
    next_num = (max(existing) + 1) if existing else 1
    folder = output_root / f"{prefix}_{next_num:03d}"
    folder.mkdir(parents=True)
    return folder


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Module 1A – Physiological Signal Data Simulator (v1.1.0)\n"
            "Single-user and multi-user simulation with per-user physiology.\n"
            "Output: module_1a_data_simulation/outputs/M1A_v1.1.0_run_NNN/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                   help=f"Recording duration in seconds (default: {DEFAULT_DURATION_S})")
    p.add_argument("--n_events", default=str(DEFAULT_N_EVENTS),
                   help="Events per user: integer or 'random' (default: 5)")
    p.add_argument("--event_dur", default="30",
                   help="Event duration in seconds: float or 'random' (default: 30)")
    p.add_argument("--emotion", default=None,
                   help="Single target emotion e.g. 'Anger'. Overrides --emotions.")
    p.add_argument("--emotions", default=None,
                   help="Comma-separated emotion subset e.g. 'Anger,Fear,Sad'.")
    p.add_argument("--noise", default=DEFAULT_NOISE_LEVEL,
                   choices=["low", "medium", "high"],
                   help=f"Noise level (default: {DEFAULT_NOISE_LEVEL})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Master random seed (default: {DEFAULT_SEED})")
    p.add_argument("--n_users", type=int, default=DEFAULT_N_USERS,
                   help=f"Number of users to simulate (default: {DEFAULT_N_USERS})")
    p.add_argument("--shared_events", action="store_true",
                   help="All users share the same event schedule. "
                        "Default: each user gets an independent schedule.")
    p.add_argument("--out", default=None,
                   help="Custom output folder name inside outputs/. "
                        "Default: auto-numbered.")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip PNG generation (CSV only — faster).")
    p.add_argument("--list_emotions", action="store_true",
                   help="Print all available emotion labels and exit.")
    return p


def resolve_n_events(val):
    return "random" if str(val).strip().lower() == "random" else int(val)


def resolve_event_dur(val):
    return "random" if str(val).strip().lower() == "random" else float(val)


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


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE USER SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_one_user(
    user_profile,
    duration_s, n_events, event_duration_s, emotions,
    noise_level, seed, output_dir, generate_plots,
    shared_events=None,
):
    """
    Simulate, visualise, and export data for a single user.

    Parameters
    ----------
    shared_events : pre-built list of EventConfig to reuse (shared_events mode),
                    or None to generate a fresh schedule.
    """
    # Override event schedule if shared
    from event_scheduler import EventScheduler
    import numpy as np

    sim = DataSimulator(
        duration_s=duration_s,
        n_events=n_events,
        event_duration_s=event_duration_s,
        emotions=emotions,
        noise_level=noise_level,
        seed=seed,
        user_profile=user_profile,
    )

    # If shared events provided, monkey-patch the scheduler to return them
    if shared_events is not None:
        sim._schedule_events = lambda: shared_events

    result = sim.simulate()

    if generate_plots:
        SignalVisualizer(result, output_dir=str(output_dir)).save_all()

    DataExporter(result, output_dir=str(output_dir)).export_all()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-USER RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    duration_s, n_events, event_duration_s, emotions,
    noise_level, seed, n_users=1, shared_events_flag=False,
    out_name=None, generate_plots=True,
):
    run_folder = next_run_folder(out_name)
    rel_path = run_folder.relative_to(MODULE_DIR)

    print("=" * 64)
    print(f"  MODULE 1A  –  Physiological Signal Data Simulator")
    print(f"  Autism Emotion/Behaviour AI Pipeline  |  v{MODULE_VERSION}")
    print("=" * 64)
    print(f"  Users           :  {n_users}")
    print(f"  Duration/user   :  {duration_s:.0f} s  ({duration_s / 60:.1f} min)")
    print(f"  Events/user     :  {n_events}")
    print(f"  Event duration  :  {event_duration_s} s")
    print(f"  Emotions        :  {emotions if emotions else 'random'}")
    print(f"  Noise level     :  {noise_level}")
    print(f"  Shared events   :  {shared_events_flag}")
    print(f"  Master seed     :  {seed}")
    print(f"  Output folder   :  {rel_path}")
    print("=" * 64)

    # ── Generate user physiological profiles ─────────────────────────────────
    generator = UserProfileGenerator(n_users=n_users, master_seed=seed)
    profiles = generator.generate()

    print(f"\n  User physiological profiles:")
    for p in profiles:
        print(f"   {p.summary()}")
    print()

    # ── Pre-build shared event schedule if requested ──────────────────────────
    shared_ev = None
    if shared_events_flag and n_users > 1:
        from event_scheduler import EventScheduler
        import numpy as np
        shared_ev = EventScheduler(
            duration_s=duration_s,
            n_events=n_events,
            event_duration_s=event_duration_s,
            emotions=emotions,
            rng=np.random.default_rng(seed),
        ).schedule()
        print(f"  Shared event schedule ({len(shared_ev)} events):")
        for ev in shared_ev:
            print(f"    {ev}")
        print()

    # ── Simulate each user ────────────────────────────────────────────────────
    all_results = []

    for profile in profiles:
        print(f"\n{'─' * 60}")
        print(f"  Simulating {profile.user_id}  "
              f"(EDA={profile.eda_tonic_mean:.1f}µS  "
              f"HR={profile.hr_resting:.0f}bpm  "
              f"react={profile.eda_reactivity:.2f}x  "
              f"[{profile.reactivity_class}])")
        print(f"{'─' * 60}")

        if n_users == 1:
            # Single user — save directly to run_folder
            user_dir = run_folder
        else:
            # Multi-user — save to user subfolder
            user_dir = run_folder / profile.user_id
            user_dir.mkdir(parents=True, exist_ok=True)

        result = simulate_one_user(
            user_profile=profile,
            duration_s=duration_s,
            n_events=n_events,
            event_duration_s=event_duration_s,
            emotions=emotions,
            noise_level=noise_level,
            seed=seed,
            output_dir=user_dir,
            generate_plots=generate_plots,
            shared_events=shared_ev,
        )
        all_results.append(result)

    # ── Multi-user cross exports ──────────────────────────────────────────────
    if n_users > 1:
        print(f"\n{'─' * 60}")
        print("  Exporting cross-user files ...")
        print(f"{'─' * 60}")
        export_all_users_combined(all_results, run_folder)

        # Write metadata for the whole run
        import json
        import numpy as np
        meta = {
            "module_version": MODULE_VERSION,
            "n_users": n_users,
            "duration_s": duration_s,
            "noise_level": noise_level,
            "master_seed": seed,
            "shared_events": shared_events_flag,
            "emotion_mode": str(emotions),
            "users": [r.metadata.get("user", {}) for r in all_results],
        }
        with open(run_folder / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [Exporter] Saved metadata.json")

    print(f"\n  Run complete.  Output folder:")
    print(f"  {run_folder}\n")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

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
        n_ev = resolve_n_events(args.n_events)
        ev_dur = resolve_event_dur(args.event_dur)
        emot = resolve_emotions(args.emotion, args.emotions)
    except ValueError as e:
        print(f"\n[Error] {e}")
        sys.exit(1)

    run_simulation(
        duration_s=args.duration,
        n_events=n_ev,
        event_duration_s=ev_dur,
        emotions=emot,
        noise_level=args.noise,
        seed=args.seed,
        n_users=args.n_users,
        shared_events_flag=args.shared_events,
        out_name=args.out,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_simulation(
            duration_s=DEFAULT_DURATION_S,
            n_events=DEFAULT_N_EVENTS,
            event_duration_s=30.0,
            emotions=None,
            noise_level=DEFAULT_NOISE_LEVEL,
            seed=DEFAULT_SEED,
            n_users=DEFAULT_N_USERS,
        )
    else:
        main()
