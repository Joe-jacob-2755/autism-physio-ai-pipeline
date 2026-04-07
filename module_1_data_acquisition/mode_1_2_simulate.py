"""
=============================================================================
MODULE 2 – DATA ACQUISITION  |  mode_2_2_simulate.py
=============================================================================
Mode 2.2 — Create New Simulated Data

Connects to Module 1A (Data Simulation), collects user inputs interactively
or programmatically, triggers a simulation run, and wraps the result as a
PipelinePacket for immediate ingestion into the preprocessing pipeline.

Two usage modes
---------------
Interactive (CLI):
    The user is guided through prompts to set all simulation parameters.

Programmatic (API):
    Parameters are passed directly, no user prompts.

The simulation runs in-process — no subprocess call needed — by importing
Module 1A's DataSimulator directly.

=============================================================================
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

# ── Locate Module 1A and add to path ─────────────────────────────────────────
MODULE_DIR    = Path(__file__).resolve().parent
MODULE_1A_DIR = MODULE_DIR.parent / "module_2a_data_simulation"
if str(MODULE_1A_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_1A_DIR))

from pipeline_packet import (
    PipelinePacket, build_packet_from_dataframes,
    SOURCE_SIMULATED, _new_session_id,
)

# Module 1A imports
try:
    from simulator      import DataSimulator
    from user_profiles  import UserProfileGenerator
    from config         import (
        EMOTIONS_ALL, DEFAULT_DURATION_S, DEFAULT_N_EVENTS,
        DEFAULT_NOISE_LEVEL, DEFAULT_SEED, DEFAULT_N_USERS,
        MODULE_VERSION as M1A_VERSION,
    )
    M2A_AVAILABLE = True
except ImportError as e:
    M2A_AVAILABLE = False
    M2A_IMPORT_ERROR = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

SIM_DEFAULTS = {
    "duration_s":       300,
    "n_events":         5,
    "event_duration_s": 30.0,
    "emotions":         None,
    "noise_level":      "medium",
    "seed":             42,
    "n_users":          1,
}


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION CONNECTOR
# ─────────────────────────────────────────────────────────────────────────────

class SimulationConnector:
    """
    Connect to Module 1A and run a simulation, returning a PipelinePacket.

    Parameters
    ----------
    duration_s       : recording duration in seconds
    n_events         : number of emotion events (int or 'random')
    event_duration_s : per-event duration in seconds (float or 'random')
    emotions         : str | list | None  — target emotion(s)
    noise_level      : 'low' | 'medium' | 'high'
    seed             : master random seed
    n_users          : number of users to simulate
    shared_events    : all users share the same event schedule
    save_m1a_output  : also save Module 1A CSV/PNG outputs to disk
    output_dir       : where to save Module 1A outputs (if save_m1a_output)
    verbose          : print progress
    """

    def __init__(
        self,
        duration_s:       float                       = SIM_DEFAULTS["duration_s"],
        n_events:         Union[int, str]             = SIM_DEFAULTS["n_events"],
        event_duration_s: Union[float, str]           = SIM_DEFAULTS["event_duration_s"],
        emotions:         Optional[Union[str, list]]  = SIM_DEFAULTS["emotions"],
        noise_level:      str                         = SIM_DEFAULTS["noise_level"],
        seed:             int                         = SIM_DEFAULTS["seed"],
        n_users:          int                         = SIM_DEFAULTS["n_users"],
        shared_events:    bool                        = False,
        save_m1a_output:  bool                        = False,
        output_dir:       Optional[Path]              = None,
        verbose:          bool                        = True,
    ):
        if not M2A_AVAILABLE:
            raise ImportError(
                f"Module 2A not found at {MODULE_1A_DIR}.\n"
                f"Error: {M2A_IMPORT_ERROR}\n"
                f"Ensure module_2a_data_simulation is in the repo root."
            )

        self.duration_s       = float(duration_s)
        self.n_events         = n_events
        self.event_duration_s = event_duration_s
        self.emotions         = emotions
        self.noise_level      = noise_level
        self.seed             = seed
        self.n_users          = n_users
        self.shared_events    = shared_events
        self.save_m1a_output  = save_m1a_output
        self.output_dir       = Path(output_dir) if output_dir else None
        self.verbose          = verbose

    # ── Public API ────────────────────────────────────────────────────────

    def run(self) -> list[PipelinePacket]:
        """
        Run simulation for all users and return a list of PipelinePackets.

        Single user → list of length 1
        Multi-user  → list of length n_users
        """
        if not M2A_AVAILABLE:
            raise ImportError("Module 1A is not available.")

        self._log(f"\n[Simulator 2.2] Module 1A v{M1A_VERSION} connected.")
        self._log(f"[Simulator 2.2] Starting simulation ...")
        self._log(f"  Users       : {self.n_users}")
        self._log(f"  Duration    : {self.duration_s:.0f}s")
        self._log(f"  Events      : {self.n_events}")
        self._log(f"  Emotions    : {self.emotions or 'random'}")
        self._log(f"  Noise       : {self.noise_level}")
        self._log(f"  Seed        : {self.seed}")

        t0 = time.time()

        # Generate user profiles
        generator = UserProfileGenerator(
            n_users=self.n_users, master_seed=self.seed
        )
        profiles = generator.generate()

        # Pre-build shared events if requested
        shared_ev = None
        if self.shared_events and self.n_users > 1:
            from event_scheduler import EventScheduler
            shared_ev = EventScheduler(
                duration_s       = self.duration_s,
                n_events         = self.n_events,
                event_duration_s = self.event_duration_s,
                emotions         = self.emotions,
                rng              = np.random.default_rng(self.seed),
            ).schedule()
            self._log(f"  Shared events: {len(shared_ev)} events scheduled")

        packets = []
        for profile in profiles:
            self._log(f"\n  Simulating {profile.user_id} ...")

            sim = DataSimulator(
                duration_s       = self.duration_s,
                n_events         = self.n_events,
                event_duration_s = self.event_duration_s,
                emotions         = self.emotions,
                noise_level      = self.noise_level,
                seed             = self.seed,
                user_profile     = profile,
            )

            if shared_ev is not None:
                sim._schedule_events = lambda: shared_ev

            result = sim.simulate()

            # Optionally save Module 1A outputs to disk
            if self.save_m1a_output:
                from visualizer import SignalVisualizer
                from exporter   import DataExporter
                out_dir = self.output_dir or (
                    MODULE_1A_DIR / "outputs" / f"m2_sim_{profile.user_id}"
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                SignalVisualizer(result, output_dir=str(out_dir)).save_all()
                DataExporter(result, output_dir=str(out_dir)).export_all()

            packet = self._result_to_packet(result)
            packets.append(packet)
            self._log(f"  {profile.user_id} → PipelinePacket [{packet.session_id}]")

        elapsed = time.time() - t0
        self._log(
            f"\n[Simulator 2.2] {self.n_users} user(s) simulated in {elapsed:.2f}s. "
            f"PipelinePackets ready."
        )
        return packets

    # ── Conversion: SimulationResult → PipelinePacket ─────────────────────

    def _result_to_packet(self, result) -> PipelinePacket:
        """Convert a Module 1A SimulationResult into a PipelinePacket."""

        # ── Build per-signal DataFrames ─────────────────────────────────
        up  = result.user_profile
        uid = up.user_id if up else "simulated_user"

        def _labels(timestamps):
            n   = len(timestamps)
            lbl = np.full(n, "baseline", dtype=object)
            eid = np.zeros(n, dtype=int)
            cat = np.full(n, "baseline", dtype=object)
            for i, ev in enumerate(result.events):
                mask = (timestamps >= ev.start_s) & (timestamps < ev.end_s)
                lbl[mask] = ev.emotion
                eid[mask] = i + 1
                cat[mask] = ev.profile["category"]
            return lbl, eid, cat

        from config import SAMPLING_RATES
        signals = {}

        ts = result.time_vectors["EDA"]
        l, e, c = _labels(ts)
        signals["EDA"] = pd.DataFrame({
            "timestamp_s": np.round(ts, 4), "EDA_uS": result.signals["EDA"],
            "target_label": l, "event_id": e, "category": c})

        ts = result.time_vectors["BVP"]
        l, e, c = _labels(ts)
        signals["BVP"] = pd.DataFrame({
            "timestamp_s": np.round(ts, 5), "BVP_nT": result.signals["BVP"],
            "target_label": l, "event_id": e, "category": c})

        l, e, c = _labels(result.ibi_times_s)
        signals["IBI"] = pd.DataFrame({
            "timestamp_s": np.round(result.ibi_times_s, 5),
            "IBI_ms": result.ibi_values_ms,
            "target_label": l, "event_id": e, "category": c})

        ts = result.time_vectors["ST"]
        l, e, c = _labels(ts)
        signals["ST"] = pd.DataFrame({
            "timestamp_s": np.round(ts, 4), "ST_degC": result.signals["ST"],
            "target_label": l, "event_id": e, "category": c})

        ts = result.time_vectors["ACC_X"]
        l, e, c = _labels(ts)
        signals["ACC"] = pd.DataFrame({
            "timestamp_s": np.round(ts, 5),
            "ACC_X_g": result.signals["ACC_X"],
            "ACC_Y_g": result.signals["ACC_Y"],
            "ACC_Z_g": result.signals["ACC_Z"],
            "target_label": l, "event_id": e, "category": c})

        # ── Build combined DataFrame ──────────────────────────────────
        from scipy.interpolate import interp1d
        fs_ref = SAMPLING_RATES["BVP"]
        t_ref  = result.time_vectors["BVP"]
        n_ref  = len(result.signals["BVP"])
        l, e, c = _labels(t_ref)

        combined = pd.DataFrame({
            "timestamp_s":  np.round(t_ref, 5),
            "target_label": l, "event_id": e, "category": c,
            "BVP_nT":       result.signals["BVP"],
        })

        for sn, col in [("EDA","EDA_uS"),("ST","ST_degC"),
                        ("ACC_X","ACC_X_g"),("ACC_Y","ACC_Y_g"),("ACC_Z","ACC_Z_g")]:
            f = interp1d(result.time_vectors[sn], result.signals[sn],
                         kind="linear", bounds_error=False,
                         fill_value="extrapolate")
            combined[col] = f(t_ref)

        ibi_col = np.full(n_ref, np.nan)
        for bt, iv in zip(result.ibi_times_s, result.ibi_values_ms):
            idx = int(round(bt * fs_ref))
            if 0 <= idx < n_ref:
                ibi_col[idx] = iv
        combined["IBI_ms"] = ibi_col

        # ── Extra metadata ────────────────────────────────────────────
        extra = {
            "m1a_version":   M1A_VERSION,
            "n_events":      len(result.events),
            "events": [
                {"event_id": i+1, "emotion": ev.emotion,
                 "start_s": ev.start_s, "end_s": ev.end_s}
                for i, ev in enumerate(result.events)
            ],
        }
        if up:
            extra["user_profile"] = up.to_dict()

        return build_packet_from_dataframes(
            signals      = signals,
            combined     = combined,
            source_type  = SOURCE_SIMULATED,
            is_annotated = True,
            user_id      = uid,
            session_id   = _new_session_id("SIM"),
            extra_meta   = extra,
        )

    def _log(self, msg):
        if self.verbose:
            print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE PARAMETER COLLECTOR
# ─────────────────────────────────────────────────────────────────────────────

class InteractiveSimSetup:
    """
    Walk the user through simulation parameter prompts interactively.

    Returns a dict of parameters ready to pass to SimulationConnector.
    """

    EMOTIONS_ALL_STR = ", ".join(EMOTIONS_ALL) if M2A_AVAILABLE else ""

    def collect(self) -> dict:
        print("\n" + "=" * 60)
        print("  MODULE 2.2 — Simulation Setup")
        print("  Configure new simulated data (press Enter for defaults)")
        print("=" * 60)

        params = dict(SIM_DEFAULTS)

        params["n_users"] = self._ask_int(
            "Number of users to simulate", SIM_DEFAULTS["n_users"], min_val=1)

        params["duration_s"] = self._ask_float(
            "Recording duration per user (seconds)", SIM_DEFAULTS["duration_s"],
            min_val=30.0)

        params["n_events"] = self._ask_int_or_random(
            "Number of emotion events per user", SIM_DEFAULTS["n_events"])

        params["event_duration_s"] = self._ask_float_or_random(
            "Duration of each event (seconds)", SIM_DEFAULTS["event_duration_s"],
            min_val=5.0)

        print(f"\n  Available emotions: {self.EMOTIONS_ALL_STR}")
        emotion_input = input(
            "  Target emotion(s) — single name, comma-separated list, "
            "or press Enter for random: "
        ).strip()

        if emotion_input == "":
            params["emotions"] = None
        elif "," in emotion_input:
            params["emotions"] = [e.strip() for e in emotion_input.split(",")]
        else:
            params["emotions"] = emotion_input

        params["noise_level"] = self._ask_choice(
            "Noise level", ["low", "medium", "high"],
            SIM_DEFAULTS["noise_level"])

        params["seed"] = self._ask_int(
            "Random seed (for reproducibility)", SIM_DEFAULTS["seed"])

        if params["n_users"] > 1:
            shared = input(
                "  Share the same event schedule across all users? "
                "(y/n, default n): "
            ).strip().lower()
            params["shared_events"] = shared == "y"

        save = input(
            "\n  Also save Module 1A CSV and PNG outputs to disk? "
            "(y/n, default n): "
        ).strip().lower()
        params["save_m1a_output"] = save == "y"

        print("\n  Summary of simulation parameters:")
        for k, v in params.items():
            print(f"    {k:<22} : {v}")

        confirm = input("\n  Proceed with simulation? (y/n, default y): ").strip().lower()
        if confirm == "n":
            print("  Simulation cancelled.")
            sys.exit(0)

        return params

    # ── Input helpers ─────────────────────────────────────────────────

    def _ask_int(self, prompt, default, min_val=None):
        while True:
            raw = input(f"  {prompt} [{default}]: ").strip()
            if raw == "":
                return default
            try:
                val = int(raw)
                if min_val is not None and val < min_val:
                    print(f"  Must be >= {min_val}")
                    continue
                return val
            except ValueError:
                print("  Please enter an integer.")

    def _ask_float(self, prompt, default, min_val=None):
        while True:
            raw = input(f"  {prompt} [{default}]: ").strip()
            if raw == "":
                return default
            try:
                val = float(raw)
                if min_val is not None and val < min_val:
                    print(f"  Must be >= {min_val}")
                    continue
                return val
            except ValueError:
                print("  Please enter a number.")

    def _ask_int_or_random(self, prompt, default):
        raw = input(f"  {prompt} [{default}] (or 'random'): ").strip()
        if raw == "":
            return default
        if raw.lower() == "random":
            return "random"
        try:
            return int(raw)
        except ValueError:
            return default

    def _ask_float_or_random(self, prompt, default, min_val=None):
        raw = input(f"  {prompt} [{default}] (or 'random'): ").strip()
        if raw == "":
            return default
        if raw.lower() == "random":
            return "random"
        try:
            return float(raw)
        except ValueError:
            return default

    def _ask_choice(self, prompt, choices, default):
        opts = "/".join(choices)
        raw  = input(f"  {prompt} ({opts}) [{default}]: ").strip()
        if raw == "":
            return default
        if raw in choices:
            return raw
        print(f"  Invalid choice — using default '{default}'")
        return default


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def simulate_data(
    duration_s:       float                      = SIM_DEFAULTS["duration_s"],
    n_events:         Union[int, str]            = SIM_DEFAULTS["n_events"],
    event_duration_s: Union[float, str]          = SIM_DEFAULTS["event_duration_s"],
    emotions:         Optional[Union[str, list]] = SIM_DEFAULTS["emotions"],
    noise_level:      str                        = SIM_DEFAULTS["noise_level"],
    seed:             int                        = SIM_DEFAULTS["seed"],
    n_users:          int                        = SIM_DEFAULTS["n_users"],
    shared_events:    bool                       = False,
    save_m1a_output:  bool                       = False,
    verbose:          bool                       = True,
) -> list[PipelinePacket]:
    """
    Shorthand: simulate data and return PipelinePacket(s).

    Returns a list — one packet per user (length 1 for single user).
    """
    return SimulationConnector(
        duration_s       = duration_s,
        n_events         = n_events,
        event_duration_s = event_duration_s,
        emotions         = emotions,
        noise_level      = noise_level,
        seed             = seed,
        n_users          = n_users,
        shared_events    = shared_events,
        save_m1a_output  = save_m1a_output,
        verbose          = verbose,
    ).run()


def simulate_interactive() -> list[PipelinePacket]:
    """Launch interactive parameter setup and run simulation."""
    params = InteractiveSimSetup().collect()
    return SimulationConnector(**params).run()
