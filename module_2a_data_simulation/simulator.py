"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  simulator.py
=============================================================================
Master simulation orchestrator.

Now supports per-user physiological profiles via UserProfile. When a
UserProfile is supplied, the user's personal baseline parameters and
reactivity factors modulate every signal channel.

=============================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import copy

from config import (
    SAMPLING_RATES, SIGNAL_RANGES, BASELINE, EMOTION_PROFILES,
    DEFAULT_DURATION_S, DEFAULT_NOISE_LEVEL, DEFAULT_SEED,
)
from signal_models import (
    generate_eda_baseline, generate_eda_event_signal,
    generate_bvp_from_beats, generate_ibi_sequence,
    generate_st_baseline, generate_st_event_signal,
    generate_acc_baseline, generate_acc_event_signal,
    clip_to_range,
)
from event_scheduler import EventConfig, EventScheduler
from noise_injector import NoiseInjector
from annotator import AutoAnnotator
from user_profiles import UserProfile


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Complete output of one simulation run for one user."""
    signals:       Dict[str, np.ndarray]
    time_vectors:  Dict[str, np.ndarray]
    ibi_times_s:   np.ndarray
    ibi_values_ms: np.ndarray
    events:        List[EventConfig]
    annotations:   Dict[str, pd.DataFrame]
    metadata:      dict
    duration_s:    float
    user_profile:  Optional[UserProfile] = None

    def summary(self) -> str:
        uid = (f"  User          : {self.user_profile.user_id} "
               f"[{self.user_profile.reactivity_class}]\n"
               if self.user_profile else "")
        lines = [
            "=" * 60,
            " MODULE 1A – SIMULATION RESULT SUMMARY",
            "=" * 60,
            uid +
            f"  Duration       : {self.duration_s:.1f} s  "
            f"({self.duration_s/60:.2f} min)",
            f"  Events         : {len(self.events)}",
            f"  Noise level    : {self.metadata.get('noise_level', 'N/A')}",
            f"  Random seed    : {self.metadata.get('seed', 'N/A')}",
            "",
            "  Signals",
        ]
        for name, arr in self.signals.items():
            fs = SAMPLING_RATES.get(name, 4)
            lines.append(
                f"    {name:<8}  n={len(arr):>7,d}  "
                f"fs={fs!r:<4}  "
                f"µ={np.mean(arr):.3f}  σ={np.std(arr):.3f}"
            )
        lines += ["", "  Events"]
        for ev in self.events:
            lines.append(f"    {ev}")
        lines += ["", "  Annotations"]
        for k, df in self.annotations.items():
            lines.append(f"    {k:<20} → {len(df)} rows")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# DATA SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class DataSimulator:
    """
    Physiological signal simulator — single user.

    Parameters
    ----------
    duration_s        : total recording length (s)
    n_events          : int or 'random'
    event_duration_s  : float or 'random'
    emotions          : str | list[str] | None
    noise_level       : 'low' | 'medium' | 'high'
    seed              : master random seed
    user_profile      : optional UserProfile — if supplied, overrides baseline
                        physiology with user-specific parameters
    min_gap_s         : minimum inter-event gap (s)
    min_lead_s        : quiet baseline before first event (s)
    """

    def __init__(
        self,
        duration_s:       float                         = DEFAULT_DURATION_S,
        n_events:         Union[int, str]               = 5,
        event_duration_s: Union[float, str]             = 30.0,
        emotions:         Optional[Union[str, list]]    = None,
        noise_level:      str                           = DEFAULT_NOISE_LEVEL,
        seed:             Optional[int]                 = DEFAULT_SEED,
        user_profile:     Optional[UserProfile]         = None,
        min_gap_s:        float                         = 15.0,
        min_lead_s:       float                         = 10.0,
    ):
        self.duration_s       = float(duration_s)
        self.n_events         = n_events
        self.event_duration_s = event_duration_s
        self.emotions         = emotions
        self.noise_level      = noise_level
        self.seed             = seed
        self.user_profile     = user_profile
        self.min_gap_s        = min_gap_s
        self.min_lead_s       = min_lead_s

        # User seed overrides run seed for baseline generation,
        # keeping baselines stable across runs for the same user
        effective_seed = (user_profile.user_seed if user_profile else seed)
        self._rng = np.random.default_rng(effective_seed)

        # Build user-adjusted baseline params once
        self._baseline = self._build_user_baseline()

    # ── Public API ────────────────────────────────────────────────────────────

    def simulate(self) -> SimulationResult:
        t0 = time.time()

        uid = f"[{self.user_profile.user_id}] " if self.user_profile else ""

        print(f"[Simulator] {uid}Step 1/6 – Scheduling events ...")
        events = self._schedule_events()

        print(f"[Simulator] {uid}Step 2/6 – Generating baseline signals ...")
        raw = self._generate_baselines()

        print(f"[Simulator] {uid}Step 3/6 – Applying event modulations ...")
        raw = self._apply_events(raw, events)

        print(f"[Simulator] {uid}Step 4/6 – Injecting noise ...")
        noisy = self._inject_noise(raw)

        print(f"[Simulator] {uid}Step 5/6 – Clipping to physiological ranges ...")
        final = self._clip_signals(noisy)

        print(f"[Simulator] {uid}Step 6/6 – Auto-annotating ...")
        annotations = self._annotate(final, events)

        elapsed = time.time() - t0
        print(f"[Simulator] {uid}Done in {elapsed:.2f} s.")

        time_vectors = self._build_time_vectors(final)

        metadata = {
            "duration_s":       self.duration_s,
            "n_events":         len(events),
            "noise_level":      self.noise_level,
            "seed":             self.seed,
            "emotion_mode":     self._describe_emotion_mode(),
            "event_duration_s": self.event_duration_s,
            "elapsed_s":        round(elapsed, 3),
        }
        if self.user_profile:
            metadata["user"] = self.user_profile.to_dict()

        return SimulationResult(
            signals       = {k: v for k, v in final.items()
                             if k not in ("IBI_TIMES", "IBI_VALUES")},
            time_vectors  = time_vectors,
            ibi_times_s   = final["IBI_TIMES"],
            ibi_values_ms = final["IBI_VALUES"],
            events        = events,
            annotations   = annotations,
            metadata      = metadata,
            duration_s    = self.duration_s,
            user_profile  = self.user_profile,
        )

    # ── User-adjusted baseline builder ───────────────────────────────────────

    def _build_user_baseline(self) -> dict:
        """
        Deep-copy the global BASELINE and apply user-specific offsets.
        Returns a modified baseline dict used throughout this simulation.
        """
        bp = copy.deepcopy(BASELINE)
        up = self.user_profile
        if up is None:
            return bp

        # EDA — shift tonic mean to user's personal level
        bp["EDA"]["tonic_mean"] = up.eda_tonic_mean

        # BVP — user's resting HR and BVP amplitude
        bp["BVP"]["hr_bpm"]    = up.hr_resting
        bp["BVP"]["amplitude"] = up.bvp_amplitude

        # IBI — user's HRV
        bp["IBI"]["mean_ms"]    = round(60_000 / up.hr_resting, 1)
        bp["IBI"]["hrv_std_ms"] = up.hrv_sdnn

        # ST — user's wrist temperature
        bp["ST"]["mean_celsius"] = up.st_baseline

        return bp

    # ── Step 1: Schedule events ───────────────────────────────────────────────

    def _schedule_events(self) -> List[EventConfig]:
        scheduler = EventScheduler(
            duration_s       = self.duration_s,
            n_events         = self.n_events,
            event_duration_s = self.event_duration_s,
            emotions         = self.emotions,
            rng              = self._rng,
            min_gap_s        = self.min_gap_s,
            min_lead_s       = self.min_lead_s,
        )
        events = scheduler.schedule()
        for ev in events:
            print(f"  Scheduled: {ev}")
        return events

    # ── Step 2: Generate baselines ────────────────────────────────────────────

    def _generate_baselines(self) -> dict:
        dur = self.duration_s
        bp  = self._baseline
        rng = self._rng
        signals = {}

        signals["EDA"] = generate_eda_baseline(
            dur, SAMPLING_RATES["EDA"], bp["EDA"], rng)

        ibi_times, ibi_vals = generate_ibi_sequence(
            dur, bp["BVP"]["hr_bpm"], bp["IBI"]["hrv_std_ms"], rng)
        signals["BVP"]       = generate_bvp_from_beats(
            dur, SAMPLING_RATES["BVP"], ibi_times, ibi_vals, bp["BVP"]["amplitude"])
        signals["IBI_TIMES"] = ibi_times
        signals["IBI_VALUES"]= ibi_vals

        signals["ST"] = generate_st_baseline(
            dur, SAMPLING_RATES["ST"], bp["ST"], rng)

        ax, ay, az = generate_acc_baseline(
            dur, SAMPLING_RATES["ACC_X"], bp["ACC"], rng)
        signals["ACC_X"] = ax
        signals["ACC_Y"] = ay
        signals["ACC_Z"] = az

        return signals

    # ── Step 3: Apply event modulations ──────────────────────────────────────

    def _apply_events(self, signals: dict, events: List[EventConfig]) -> dict:
        result = {k: v.copy() for k, v in signals.items()}
        up     = self.user_profile

        # Reactivity factors (1.0 if no user profile)
        eda_rx  = up.eda_reactivity  if up else 1.0
        hr_rx   = up.hr_reactivity   if up else 1.0
        move_rx = up.movement_scale  if up else 1.0

        for ev in events:
            prof = ev.profile

            # ── EDA ───────────────────────────────────────────────────────
            fs  = SAMPLING_RATES["EDA"]
            i0  = int(ev.start_s * fs)
            i1  = min(int(ev.end_s * fs), len(result["EDA"]))
            dur = (i1 - i0) / fs
            if dur >= 1.0:
                # Scale EDA profile by user reactivity
                scaled_prof = self._scale_eda_profile(prof["EDA"], eda_rx)
                delta = generate_eda_event_signal(dur, fs, scaled_prof, self._rng)
                result["EDA"][i0:i1] += delta[:i1 - i0]

            # ── BVP / IBI ─────────────────────────────────────────────────
            fs  = SAMPLING_RATES["BVP"]
            i0  = int(ev.start_s * fs)
            i1  = min(int(ev.end_s * fs), len(result["BVP"]))
            dur = (i1 - i0) / fs
            if dur >= 1.0:
                base_hr   = self._baseline["BVP"]["hr_bpm"]
                base_hrv  = self._baseline["IBI"]["hrv_std_ms"]
                hr_delta  = prof["BVP"]["hr_delta_bpm"] * hr_rx
                event_hr  = float(np.clip(base_hr + hr_delta, 40, 200))
                event_hrv = base_hrv * prof["BVP"]["hrv_factor"]
                ev_amp    = self._baseline["BVP"]["amplitude"] * prof["BVP"]["amplitude_factor"]

                ev_t, ev_v = generate_ibi_sequence(dur, event_hr, event_hrv, self._rng)
                result["BVP"][i0:i1] = generate_bvp_from_beats(
                    dur, fs, ev_t, ev_v, ev_amp)[:i1 - i0]

                # Patch IBI arrays
                keep_before = result["IBI_TIMES"] < ev.start_s
                keep_after  = result["IBI_TIMES"] >= ev.end_s
                ev_t_global = ev_t + ev.start_s
                ev_v_patched = ev_v + self._rng.normal(
                    prof["IBI"]["delta_mean_ms"] * 0.3,
                    prof["IBI"]["delta_std_ms"],
                    len(ev_v),
                )
                result["IBI_TIMES"]  = np.concatenate([
                    result["IBI_TIMES"][keep_before],
                    ev_t_global,
                    result["IBI_TIMES"][keep_after],
                ])
                result["IBI_VALUES"] = np.concatenate([
                    result["IBI_VALUES"][keep_before],
                    ev_v_patched,
                    result["IBI_VALUES"][keep_after],
                ])

            # ── ST ────────────────────────────────────────────────────────
            fs  = SAMPLING_RATES["ST"]
            i0  = int(ev.start_s * fs)
            i1  = min(int(ev.end_s * fs), len(result["ST"]))
            dur = (i1 - i0) / fs
            if dur >= 1.0:
                delta = generate_st_event_signal(dur, fs, prof["ST"])
                result["ST"][i0:i1] += delta[:i1 - i0]

            # ── ACC ───────────────────────────────────────────────────────
            fs  = SAMPLING_RATES["ACC_X"]
            i0  = int(ev.start_s * fs)
            i1  = min(int(ev.end_s * fs), len(result["ACC_X"]))
            dur = (i1 - i0) / fs
            if dur >= 1.0:
                scaled_acc = self._scale_acc_profile(prof["ACC"], move_rx)
                dx, dy, dz = generate_acc_event_signal(
                    dur, fs, scaled_acc, self._rng)
                result["ACC_X"][i0:i1] += dx[:i1 - i0]
                result["ACC_Y"][i0:i1] += dy[:i1 - i0]
                result["ACC_Z"][i0:i1] += dz[:i1 - i0]

        return result

    # ── Step 4: Noise ─────────────────────────────────────────────────────────

    def _inject_noise(self, signals: dict) -> dict:
        noise_seed = int(self._rng.integers(0, 99_999))
        injector   = NoiseInjector(level=self.noise_level, seed=noise_seed)
        noisy      = injector.inject(signals)
        # Apply electrode contact quality to EDA noise
        if self.user_profile and "EDA" in noisy:
            contact = self.user_profile.noise_contact
            # contact < 1 = poor contact → noisier; > 1 = unusually clean
            noise_added = noisy["EDA"] - signals["EDA"]
            noisy["EDA"] = signals["EDA"] + noise_added * (1.0 / contact)
        return noisy

    # ── Step 5: Clip ──────────────────────────────────────────────────────────

    def _clip_signals(self, signals: dict) -> dict:
        clipped = {}
        for name, arr in signals.items():
            if name in ("IBI_TIMES", "IBI_VALUES"):
                clipped[name] = arr
                continue
            lo, hi = SIGNAL_RANGES.get(name, (-1e9, 1e9))
            clipped[name] = clip_to_range(arr, lo, hi)
        return clipped

    # ── Step 6: Annotate ──────────────────────────────────────────────────────

    def _annotate(self, signals: dict, events: List[EventConfig]) -> dict:
        sig_for_annot = {k: v for k, v in signals.items()
                         if k not in ("IBI_TIMES", "IBI_VALUES")}
        return AutoAnnotator(self.duration_s, events, sig_for_annot).annotate()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_time_vectors(self, signals: dict) -> Dict[str, np.ndarray]:
        tvecs = {}
        for name, arr in signals.items():
            if name == "IBI_TIMES":
                tvecs["IBI"] = arr
            elif name != "IBI_VALUES":
                fs = SAMPLING_RATES.get(name)
                if fs:
                    tvecs[name] = np.arange(len(arr)) / fs
        return tvecs

    def _describe_emotion_mode(self) -> str:
        if self.emotions is None:
            return "random_all"
        if isinstance(self.emotions, str):
            return f"specific:{self.emotions}"
        return f"subset:{','.join(self.emotions)}"

    @staticmethod
    def _scale_eda_profile(prof: dict, factor: float) -> dict:
        """Return a copy of the EDA event profile scaled by reactivity factor."""
        s = dict(prof)
        s["tonic_delta"]        = prof["tonic_delta"]        * factor
        s["scr_amplitude_mean"] = prof["scr_amplitude_mean"] * factor
        s["scr_amplitude_std"]  = prof["scr_amplitude_std"]  * factor
        return s

    @staticmethod
    def _scale_acc_profile(prof: dict, factor: float) -> dict:
        """Return a copy of the ACC event profile scaled by movement factor."""
        s = dict(prof)
        s["activity_amp"] = prof["activity_amp"] * factor
        s["jitter"]       = prof["jitter"]       * factor
        return s
