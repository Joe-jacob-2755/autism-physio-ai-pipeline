"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  simulator.py
=============================================================================
Master simulation orchestrator.

DataSimulator ties together:
  SignalModels  → physiological signal generation
  EventScheduler → event timing
  NoiseInjector → realistic noise
  AutoAnnotator  → labelling

Output: SimulationResult dataclass containing every signal array,
        timestamps, annotations, and run metadata.

=============================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from config import (
    SAMPLING_RATES, SIGNAL_RANGES, BASELINE, EMOTION_PROFILES,
    DEFAULT_DURATION_S, DEFAULT_NOISE_LEVEL, DEFAULT_SEED,
)
from signal_models import (
    generate_eda_baseline, generate_eda_event_signal,
    generate_bvp_from_beats, generate_ibi_sequence, build_hr_track,
    generate_st_baseline, generate_st_event_signal,
    generate_acc_baseline, generate_acc_event_signal,
    lowpass_filter, clip_to_range,
)
from event_scheduler import EventConfig, EventScheduler
from noise_injector import NoiseInjector
from annotator import AutoAnnotator


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION RESULT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """
    Complete output of one simulation run.

    Signals
    -------
    Each signal is stored as a 1D numpy array at its native sampling rate.
    Time vectors (in seconds) are provided for each.

    IBI is stored as (times, values) because it is event-based.

    Attributes
    ----------
    signals      : dict of {name: ndarray}  (post-noise)
    time_vectors : dict of {name: ndarray}  (in seconds)
    ibi_times_s  : beat onset times (s)
    ibi_values_ms: IBI values (ms)
    events       : list of EventConfig
    annotations  : dict returned by AutoAnnotator
    metadata     : run parameters dict
    duration_s   : recording duration
    """
    signals:       Dict[str, np.ndarray]
    time_vectors:  Dict[str, np.ndarray]
    ibi_times_s:   np.ndarray
    ibi_values_ms: np.ndarray
    events:        List[EventConfig]
    annotations:   Dict[str, pd.DataFrame]
    metadata:      dict
    duration_s:    float

    def summary(self) -> str:
        lines = [
            "=" * 60,
            " MODULE 1A – SIMULATION RESULT SUMMARY",
            "=" * 60,
            f"  Duration       : {self.duration_s:.1f} s  "
            f"({self.duration_s/60:.2f} min)",
            f"  Events         : {len(self.events)}",
            f"  Noise level    : {self.metadata.get('noise_level', 'N/A')}",
            f"  Random seed    : {self.metadata.get('seed', 'N/A')}",
            "",
            "  Signals",
        ]
        for name, arr in self.signals.items():
            fs  = SAMPLING_RATES.get(name, 4)
            lines.append(
                f"    {name:<8}  n={len(arr):>7,d}  "
                f"fs={fs or 'event'!r:<4}  "
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
# MAIN SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class DataSimulator:
    """
    Physiological signal data simulator for autism emotion/behaviour research.

    Parameters
    ----------
    duration_s        : total recording duration (seconds)
    n_events          : int or 'random'
    event_duration_s  : float or 'random'
    emotions          : str | list[str] | None (None = fully random)
    noise_level       : 'low' | 'medium' | 'high'
    seed              : integer seed for reproducibility (None = non-deterministic)
    min_gap_s         : minimum inter-event gap (s)
    min_lead_s        : quiet baseline at recording start (s)
    """

    def __init__(
        self,
        duration_s:       float                        = DEFAULT_DURATION_S,
        n_events:         Union[int, str]              = 5,
        event_duration_s: Union[float, str]            = 30.0,
        emotions:         Optional[Union[str, list]]   = None,
        noise_level:      str                          = DEFAULT_NOISE_LEVEL,
        seed:             Optional[int]                = DEFAULT_SEED,
        min_gap_s:        float                        = 15.0,
        min_lead_s:       float                        = 10.0,
    ):
        self.duration_s       = float(duration_s)
        self.n_events         = n_events
        self.event_duration_s = event_duration_s
        self.emotions         = emotions
        self.noise_level      = noise_level
        self.seed             = seed
        self.min_gap_s        = min_gap_s
        self.min_lead_s       = min_lead_s

        # ── RNG (seeded) ────────────────────────────────────────────────────
        self._rng = np.random.default_rng(seed)

    # ── Public API ──────────────────────────────────────────────────────────

    def simulate(self) -> SimulationResult:
        """
        Run the full simulation pipeline.

        Steps
        ─────
        1. Schedule events
        2. Generate baseline signals
        3. Apply event-specific modulations
        4. Inject noise
        5. Clip to physiological ranges
        6. Auto-annotate
        7. Package SimulationResult

        Returns
        -------
        SimulationResult
        """
        t_start = time.time()
        print("[Simulator] Step 1/6 – Scheduling events …")
        events = self._schedule_events()

        print("[Simulator] Step 2/6 – Generating baseline signals …")
        raw = self._generate_baselines()

        print("[Simulator] Step 3/6 – Applying event modulations …")
        raw = self._apply_events(raw, events)

        print("[Simulator] Step 4/6 – Injecting noise …")
        noisy = self._inject_noise(raw, events)

        print("[Simulator] Step 5/6 – Clipping to physiological ranges …")
        final = self._clip_signals(noisy)

        print("[Simulator] Step 6/6 – Auto-annotating …")
        annotations = self._annotate(final, events)

        elapsed = time.time() - t_start
        print(f"[Simulator] Done in {elapsed:.2f} s.")

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

        result = SimulationResult(
            signals       = {k: v for k, v in final.items()
                             if k not in ("IBI_TIMES", "IBI_VALUES")},
            time_vectors  = time_vectors,
            ibi_times_s   = final["IBI_TIMES"],
            ibi_values_ms = final["IBI_VALUES"],
            events        = events,
            annotations   = annotations,
            metadata      = metadata,
            duration_s    = self.duration_s,
        )
        return result

    # ── Step 1: Schedule events ─────────────────────────────────────────────

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

    # ── Step 2: Generate baselines ─────────────────────────────────────────

    def _generate_baselines(self) -> dict:
        dur = self.duration_s
        bp  = BASELINE
        rng = self._rng

        signals = {}

        # EDA
        signals["EDA"] = generate_eda_baseline(dur, SAMPLING_RATES["EDA"], bp["EDA"], rng)

        # BVP / IBI (using baseline HR)
        hr_track_ibi = np.full(int(dur), bp["BVP"]["hr_bpm"])  # flat baseline for IBI gen
        ibi_times, ibi_vals = generate_ibi_sequence(
            dur,
            bp["BVP"]["hr_bpm"],
            bp["IBI"]["hrv_std_ms"],
            rng,
        )
        bvp = generate_bvp_from_beats(
            dur, SAMPLING_RATES["BVP"], ibi_times, ibi_vals, bp["BVP"]["amplitude"]
        )
        signals["BVP"]        = bvp
        signals["IBI_TIMES"]  = ibi_times
        signals["IBI_VALUES"] = ibi_vals

        # ST
        signals["ST"] = generate_st_baseline(dur, SAMPLING_RATES["ST"], bp["ST"], rng)

        # ACC
        ax, ay, az = generate_acc_baseline(dur, SAMPLING_RATES["ACC_X"], bp["ACC"], rng)
        signals["ACC_X"] = ax
        signals["ACC_Y"] = ay
        signals["ACC_Z"] = az

        return signals

    # ── Step 3: Apply event modulations ────────────────────────────────────

    def _apply_events(self, signals: dict, events: List[EventConfig]) -> dict:
        result = {k: v.copy() for k, v in signals.items()}

        for ev in events:
            prof = ev.profile

            # ── EDA ─────────────────────────────────────────────────────
            fs_eda = SAMPLING_RATES["EDA"]
            i0_eda = int(ev.start_s    * fs_eda)
            i1_eda = int(ev.end_s      * fs_eda)
            i1_eda = min(i1_eda, len(result["EDA"]))
            seg_dur = (i1_eda - i0_eda) / fs_eda
            if seg_dur >= 1.0:
                eda_delta = generate_eda_event_signal(seg_dur, fs_eda, prof["EDA"], self._rng)
                result["EDA"][i0_eda:i1_eda] += eda_delta[:i1_eda - i0_eda]

            # ── BVP / IBI  ────────────────────────────────────────────────
            # Re-generate BVP for event window with elevated HR
            fs_bvp    = SAMPLING_RATES["BVP"]
            i0_bvp    = int(ev.start_s * fs_bvp)
            i1_bvp    = min(int(ev.end_s * fs_bvp), len(result["BVP"]))
            seg_dur_b = (i1_bvp - i0_bvp) / fs_bvp
            if seg_dur_b >= 1.0:
                event_hr  = BASELINE["BVP"]["hr_bpm"] + prof["BVP"]["hr_delta_bpm"]
                event_hr  = float(np.clip(event_hr, 40, 200))
                event_hrv = BASELINE["IBI"]["hrv_std_ms"] * prof["BVP"]["hrv_factor"]
                ev_amp    = BASELINE["BVP"]["amplitude"] * prof["BVP"]["amplitude_factor"]

                ev_ibi_t, ev_ibi_v = generate_ibi_sequence(
                    seg_dur_b, event_hr, event_hrv, self._rng
                )
                ev_bvp = generate_bvp_from_beats(
                    seg_dur_b, fs_bvp, ev_ibi_t, ev_ibi_v, ev_amp
                )
                result["BVP"][i0_bvp:i1_bvp] = ev_bvp[:i1_bvp - i0_bvp]

                # Patch IBI arrays for the event window
                mask_out = (result["IBI_TIMES"] >= ev.start_s) & \
                           (result["IBI_TIMES"] <  ev.end_s)
                ev_ibi_global_t = ev_ibi_t + ev.start_s
                keep_before = result["IBI_TIMES"] < ev.start_s
                keep_after  = result["IBI_TIMES"] >= ev.end_s
                new_times  = np.concatenate([
                    result["IBI_TIMES"][keep_before],
                    ev_ibi_global_t,
                    result["IBI_TIMES"][keep_after],
                ])
                delta_mean = prof["IBI"]["delta_mean_ms"]
                delta_std  = prof["IBI"]["delta_std_ms"]
                ev_ibi_patched = ev_ibi_v + self._rng.normal(
                    delta_mean * 0.3, delta_std, len(ev_ibi_v)
                )
                new_vals = np.concatenate([
                    result["IBI_VALUES"][keep_before],
                    ev_ibi_patched,
                    result["IBI_VALUES"][keep_after],
                ])
                result["IBI_TIMES"]  = new_times
                result["IBI_VALUES"] = new_vals

            # ── ST ───────────────────────────────────────────────────────
            fs_st  = SAMPLING_RATES["ST"]
            i0_st  = int(ev.start_s * fs_st)
            i1_st  = min(int(ev.end_s * fs_st), len(result["ST"]))
            seg_dur_st = (i1_st - i0_st) / fs_st
            if seg_dur_st >= 1.0:
                st_delta = generate_st_event_signal(seg_dur_st, fs_st, prof["ST"])
                result["ST"][i0_st:i1_st] += st_delta[:i1_st - i0_st]

            # ── ACC ──────────────────────────────────────────────────────
            fs_acc = SAMPLING_RATES["ACC_X"]
            i0_acc = int(ev.start_s * fs_acc)
            i1_acc = min(int(ev.end_s * fs_acc), len(result["ACC_X"]))
            seg_dur_a = (i1_acc - i0_acc) / fs_acc
            if seg_dur_a >= 1.0:
                dx, dy, dz = generate_acc_event_signal(seg_dur_a, fs_acc, prof["ACC"], self._rng)
                result["ACC_X"][i0_acc:i1_acc] += dx[:i1_acc - i0_acc]
                result["ACC_Y"][i0_acc:i1_acc] += dy[:i1_acc - i0_acc]
                result["ACC_Z"][i0_acc:i1_acc] += dz[:i1_acc - i0_acc]

        return result

    # ── Step 4: Inject noise ────────────────────────────────────────────────

    def _inject_noise(self, signals: dict, events: List[EventConfig]) -> dict:
        injector = NoiseInjector(level=self.noise_level, seed=int(self._rng.integers(0, 9999)))
        return injector.inject(signals)

    # ── Step 5: Clip ────────────────────────────────────────────────────────

    def _clip_signals(self, signals: dict) -> dict:
        clipped = {}
        for name, arr in signals.items():
            if name in ("IBI_TIMES", "IBI_VALUES"):
                clipped[name] = arr
                continue
            lo, hi = SIGNAL_RANGES.get(name, (-1e9, 1e9))
            clipped[name] = clip_to_range(arr, lo, hi)
        return clipped

    # ── Step 6: Annotate ────────────────────────────────────────────────────

    def _annotate(self, signals: dict, events: List[EventConfig]) -> dict:
        # Build signal dict without IBI timing arrays for SQI computation
        sig_for_annot = {k: v for k, v in signals.items()
                         if k not in ("IBI_TIMES", "IBI_VALUES")}
        annotator = AutoAnnotator(self.duration_s, events, sig_for_annot)
        return annotator.annotate()

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _build_time_vectors(self, signals: dict) -> Dict[str, np.ndarray]:
        tvecs = {}
        for name, arr in signals.items():
            if name == "IBI_TIMES":
                tvecs["IBI"] = arr
                continue
            if name == "IBI_VALUES":
                continue
            fs = SAMPLING_RATES.get(name)
            if fs is not None:
                tvecs[name] = np.arange(len(arr)) / fs
        return tvecs

    def _describe_emotion_mode(self) -> str:
        if self.emotions is None:
            return "random_all"
        if isinstance(self.emotions, str):
            return f"specific:{self.emotions}"
        return f"subset:{','.join(self.emotions)}"
