"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  main.py
=============================================================================
Entry point and demonstration script.

Usage (command line)
--------------------
  python main.py                          # default: 5-min, 5 random events
  python main.py --duration 600           # 10 minutes
  python main.py --n_events 8 --noise high
  python main.py --emotion Anger          # single emotion, 5 repetitions
  python main.py --emotions "Anger,Fear,Surprise"  # subset
  python main.py --n_events random --event_dur random
  python main.py --seed 123 --out results/run1

API usage
---------
  from simulator import DataSimulator
  from visualizer import SignalVisualizer
  from exporter import DataExporter

  sim    = DataSimulator(duration_s=300, n_events=5, noise_level='medium')
  result = sim.simulate()
  SignalVisualizer(result, output_dir='out').save_all()
  DataExporter(result, output_dir='out').export_all()
=============================================================================
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path

# ── Make sure the module directory is on the path ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    EMOTIONS_ALL, DEFAULT_DURATION_S, DEFAULT_NOISE_LEVEL,
    DEFAULT_SEED, DEFAULT_N_EVENTS,
)
from simulator  import DataSimulator
from visualizer import SignalVisualizer
from exporter   import DataExporter


# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "module_1a",
        description = "Module 1A – Physiological Signal Data Simulator for Autism AI Research",
    )
    p.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_S,
        help=f"Recording duration in seconds (default: {DEFAULT_DURATION_S})",
    )
    p.add_argument(
        "--n_events", default=str(DEFAULT_N_EVENTS),
        help="Number of events: integer or 'random' (default: 5)",
    )
    p.add_argument(
        "--event_dur", default="30",
        help="Event duration in seconds: float or 'random' (default: 30)",
    )
    p.add_argument(
        "--emotion", default=None,
        help="Single target emotion (e.g. 'Anger'). Overrides --emotions.",
    )
    p.add_argument(
        "--emotions", default=None,
        help="Comma-separated list of target emotions (e.g. 'Anger,Fear,Sad'). "
             "None = all random.",
    )
    p.add_argument(
        "--noise", default=DEFAULT_NOISE_LEVEL, choices=["low", "medium", "high"],
        help=f"Noise level (default: {DEFAULT_NOISE_LEVEL})",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    p.add_argument(
        "--out", default="output",
        help="Output directory for CSV and PNG files (default: output/)",
    )
    p.add_argument(
        "--no_plots", action="store_true",
        help="Skip plot generation (CSV output only).",
    )
    p.add_argument(
        "--list_emotions", action="store_true",
        help="List all available emotion / behaviour labels and exit.",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT RESOLUTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def resolve_n_events(val: str):
    if val.strip().lower() == "random":
        return "random"
    return int(val)


def resolve_event_dur(val: str):
    if val.strip().lower() == "random":
        return "random"
    return float(val)


def resolve_emotions(emotion_single, emotions_csv):
    """Resolve mutually-exclusive emotion args."""
    if emotion_single is not None:
        if emotion_single not in EMOTIONS_ALL:
            raise ValueError(
                f"Unknown emotion '{emotion_single}'. Valid: {EMOTIONS_ALL}"
            )
        return emotion_single
    if emotions_csv is not None:
        lst = [e.strip() for e in emotions_csv.split(",")]
        for e in lst:
            if e not in EMOTIONS_ALL:
                raise ValueError(f"Unknown emotion '{e}'. Valid: {EMOTIONS_ALL}")
        return lst
    return None   # fully random


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    duration_s:       float,
    n_events,
    event_duration_s,
    emotions,
    noise_level:      str,
    seed:             int,
    output_dir:       str,
    generate_plots:   bool = True,
):
    """End-to-end simulation, visualisation, and export."""

    print("=" * 60)
    print(" MODULE 1A – Physiological Signal Data Simulator")
    print(" Autism Emotion/Behaviour AI Pipeline")
    print("=" * 60)
    print(f" Duration     : {duration_s:.0f} s ({duration_s/60:.1f} min)")
    print(f" Events       : {n_events}")
    print(f" Event dur    : {event_duration_s} s")
    print(f" Emotions     : {emotions if emotions is not None else 'random'}")
    print(f" Noise level  : {noise_level}")
    print(f" Seed         : {seed}")
    print(f" Output dir   : {output_dir}")
    print("=" * 60)

    # ── 1. Simulate ─────────────────────────────────────────────────────────
    simulator = DataSimulator(
        duration_s       = duration_s,
        n_events         = n_events,
        event_duration_s = event_duration_s,
        emotions         = emotions,
        noise_level      = noise_level,
        seed             = seed,
    )
    result = simulator.simulate()
    print()
    print(result.summary())

    # ── 2. Visualise ─────────────────────────────────────────────────────────
    if generate_plots:
        print("\n[Main] Generating visualisations …")
        viz   = SignalVisualizer(result, output_dir=output_dir)
        plots = viz.save_all()
        print(f"  → {len(plots)} figure(s) saved to: {output_dir}/")

    # ── 3. Export CSV ────────────────────────────────────────────────────────
    print("\n[Main] Exporting data files …")
    exporter = DataExporter(result, output_dir=output_dir)
    files    = exporter.export_all()
    print(f"  → {len(files)} file(s) saved to: {output_dir}/")

    print("\n[Main] ✓ Module 1A complete.\n")
    return result


def run_demo_scenarios(output_dir: str = "output_demo"):
    """
    Run four illustrative scenarios to showcase Module 1A capabilities.

    Scenario A – Single emotion (Anger) repeated 4 times, high noise
    Scenario B – Physiological needs subset, random duration, medium noise
    Scenario C – Fully random (count, duration, emotion), low noise
    Scenario D – Short burst (60 s) of Fear + Surprise, medium noise
    """
    scenarios = [
        {
            "name":        "A_Anger_repeated",
            "duration_s":  240,
            "n_events":    4,
            "event_dur":   25.0,
            "emotions":    "Anger",
            "noise":       "high",
            "seed":        11,
        },
        {
            "name":        "B_Needs_random_dur",
            "duration_s":  360,
            "n_events":    5,
            "event_dur":   "random",
            "emotions":    ["Hunger", "Thirst", "Toilet", "Tired"],
            "noise":       "medium",
            "seed":        22,
        },
        {
            "name":        "C_FullyRandom",
            "duration_s":  300,
            "n_events":    "random",
            "event_dur":   "random",
            "emotions":    None,
            "noise":       "low",
            "seed":        33,
        },
        {
            "name":        "D_FearSurprise_short",
            "duration_s":  120,
            "n_events":    3,
            "event_dur":   15.0,
            "emotions":    ["Fear", "Surprise"],
            "noise":       "medium",
            "seed":        44,
        },
    ]

    print("\n" + "=" * 60)
    print("  MODULE 1A – DEMO SCENARIOS")
    print("=" * 60)

    for sc in scenarios:
        print(f"\n▶ Scenario {sc['name']}")
        run_simulation(
            duration_s       = sc["duration_s"],
            n_events         = sc["n_events"],
            event_duration_s = sc["event_dur"],
            emotions         = sc["emotions"],
            noise_level      = sc["noise"],
            seed             = sc["seed"],
            output_dir       = f"{output_dir}/{sc['name']}",
            generate_plots   = True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

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
        print(f"[Error] {e}")
        sys.exit(1)

    run_simulation(
        duration_s       = args.duration,
        n_events         = n_ev,
        event_duration_s = ev_dur,
        emotions         = emot,
        noise_level      = args.noise,
        seed             = args.seed,
        output_dir       = args.out,
        generate_plots   = not args.no_plots,
    )


if __name__ == "__main__":
    # If run without arguments, execute demo scenarios
    if len(sys.argv) == 1:
        run_simulation(
            duration_s       = 300,
            n_events         = 5,
            event_duration_s = 30.0,
            emotions         = None,
            noise_level      = "medium",
            seed             = 42,
            output_dir       = "output",
            generate_plots   = True,
        )
    else:
        main()
