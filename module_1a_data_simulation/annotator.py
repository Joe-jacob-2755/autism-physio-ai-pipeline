"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  annotator.py
=============================================================================
Automatic annotation engine.

Produces a structured annotation DataFrame containing:
  • Event labels (emotion / behaviour, start, end, duration)
  • Signal quality indicators per channel per window
  • Baseline windows (periods with no active event)
  • A sample-level annotation array for each signal

=============================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from config import SAMPLING_RATES, SIGNAL_RANGES, EMOTIONS_ALL
from event_scheduler import EventConfig


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL QUALITY INDEX  (SQI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sqi(
    segment: np.ndarray,
    signal_name: str,
    fs: int,
) -> float:
    """
    Compute a simple [0, 1] Signal Quality Index for a segment.

    Heuristics used:
      1. Range check  – flag if values exceed physiological limits
      2. Flatline     – flag near-zero variance (dead sensor / saturation)
      3. Clipping     – flag if > 2 % samples at hard rails
      4. High noise   – flag if short-term variance >> long-term variance

    Returns 1.0 = perfect, 0.0 = unusable.
    """
    lo, hi = SIGNAL_RANGES.get(signal_name, (-1e9, 1e9))
    n = len(segment)
    if n < 4:
        return 0.5

    score = 1.0

    # (1) physiological range
    out_of_range = np.mean((segment < lo) | (segment > hi))
    score -= min(out_of_range * 2.0, 0.4)

    # (2) flatline
    if np.std(segment) < 1e-4:
        score -= 0.5

    # (3) clipping
    clip_frac = np.mean((segment <= lo * 1.01) | (segment >= hi * 0.99))
    score -= min(clip_frac * 3.0, 0.3)

    # (4) short-vs-long variance ratio (for BVP / ACC only)
    if signal_name in ("BVP", "ACC_X", "ACC_Y", "ACC_Z") and n >= 32:
        short_var = np.mean([
            np.var(segment[i: i + max(1, fs // 4)])
            for i in range(0, n - fs // 4, fs // 4)
        ])
        long_var = np.var(segment) + 1e-12
        ratio = short_var / long_var
        if ratio > 5.0:
            score -= 0.2

    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# AUTO ANNOTATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AutoAnnotator:
    """
    Generate structured annotations for a simulated recording.

    Parameters
    ----------
    duration_s : total recording length (s)
    events     : list of EventConfig objects from the scheduler
    signals    : dict of signal arrays {name: ndarray}
    """

    def __init__(
        self,
        duration_s: float,
        events:     List[EventConfig],
        signals:    Dict[str, np.ndarray],
    ):
        self.duration_s = duration_s
        self.events     = events
        self.signals    = signals

    # ── Master method ───────────────────────────────────────────────────────

    def annotate(self) -> Dict[str, pd.DataFrame]:
        """
        Run full annotation pipeline.

        Returns
        -------
        {
          'events'        : per-event annotation DataFrame
          'baseline_wins' : identified quiet/baseline window DataFrame
          'signal_quality': per-signal SQI DataFrame (evaluated in 10 s windows)
          'sample_labels' : per-sample label DataFrame (one column per signal)
        }
        """
        event_df   = self._build_event_annotations()
        baseline_df = self._build_baseline_windows(event_df)
        sqi_df     = self._build_sqi_table()
        sample_df  = self._build_sample_labels()

        return {
            "events":         event_df,
            "baseline_wins":  baseline_df,
            "signal_quality": sqi_df,
            "sample_labels":  sample_df,
        }

    # ── Event annotation DataFrame ─────────────────────────────────────────

    def _build_event_annotations(self) -> pd.DataFrame:
        rows = []
        for i, ev in enumerate(self.events):
            row = {
                "event_id":       i + 1,
                "emotion":        ev.emotion,
                "category":       ev.profile["category"],
                "valence":        ev.profile.get("valence", "unknown"),
                "arousal":        ev.profile.get("arousal", "unknown"),
                "start_s":        round(ev.start_s, 3),
                "end_s":          round(ev.end_s,   3),
                "duration_s":     round(ev.duration_s, 3),
                "color_hex":      ev.color,
                "description":    ev.profile["description"],
                # Expected signal directions (for validation reference)
                "expect_EDA_dir":  "up"   if ev.profile["EDA"]["tonic_delta"] > 0 else "down",
                "expect_HR_dir":   "up"   if ev.profile["BVP"]["hr_delta_bpm"] > 0 else "down",
                "expect_ST_dir":   "up"   if ev.profile["ST"]["delta_celsius"] > 0 else "down",
                "expect_ACC_dir":  "high" if ev.profile["ACC"]["activity_amp"] > 0.5 else "low",
            }
            rows.append(row)

        return pd.DataFrame(rows)

    # ── Baseline windows DataFrame ─────────────────────────────────────────

    def _build_baseline_windows(self, event_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify continuous quiet periods (no events).
        A window must be ≥ 5 s to be reported.
        """
        # Build a sorted list of (start, end) tuples for occupied time
        occupied = sorted(
            [(ev.start_s, ev.end_s) for ev in self.events],
            key=lambda x: x[0],
        )

        gaps    = []
        cursor  = 0.0
        win_id  = 1

        for (ev_start, ev_end) in occupied:
            if ev_start - cursor >= 5.0:
                gaps.append({
                    "baseline_id":  win_id,
                    "start_s":      round(cursor, 3),
                    "end_s":        round(ev_start, 3),
                    "duration_s":   round(ev_start - cursor, 3),
                    "label":        "baseline",
                })
                win_id += 1
            cursor = ev_end

        # Tail gap
        if self.duration_s - cursor >= 5.0:
            gaps.append({
                "baseline_id":  win_id,
                "start_s":      round(cursor, 3),
                "end_s":        round(self.duration_s, 3),
                "duration_s":   round(self.duration_s - cursor, 3),
                "label":        "baseline",
            })

        return pd.DataFrame(gaps)

    # ── SQI table ─────────────────────────────────────────────────────────

    def _build_sqi_table(self, window_s: float = 10.0) -> pd.DataFrame:
        """Evaluate SQI in non-overlapping 10 s windows for each signal."""
        rows = []
        for sig_name, sig_arr in self.signals.items():
            if sig_name in ("IBI_TIMES", "IBI_VALUES"):
                continue
            fs_key = sig_name if sig_name in SAMPLING_RATES else "EDA"
            fs     = SAMPLING_RATES.get(fs_key, 4)
            win_n  = int(window_s * fs)
            n      = len(sig_arr)

            for start_idx in range(0, n - win_n + 1, win_n):
                seg      = sig_arr[start_idx: start_idx + win_n]
                t_start  = start_idx / fs
                sqi      = compute_sqi(seg, sig_name, fs)

                # Find which event (if any) this window falls in
                label = "baseline"
                for ev in self.events:
                    if t_start >= ev.start_s and t_start < ev.end_s:
                        label = ev.emotion
                        break

                rows.append({
                    "signal":       sig_name,
                    "window_start_s": round(t_start, 2),
                    "window_end_s":   round(t_start + window_s, 2),
                    "sqi":           round(sqi, 4),
                    "state_label":   label,
                })

        return pd.DataFrame(rows)

    # ── Sample-level label array ─────────────────────────────────────────

    def _build_sample_labels(self) -> pd.DataFrame:
        """
        Build a per-sample label DataFrame for each signal channel.

        Each row = one sample for the highest-rate signal (BVP, 64 Hz).
        Columns: time_s, label, event_id, category
        """
        fs_ref = SAMPLING_RATES["BVP"]
        n_ref  = int(self.duration_s * fs_ref)
        t_ref  = np.arange(n_ref) / fs_ref

        labels    = np.full(n_ref, "baseline", dtype=object)
        event_ids = np.full(n_ref, 0, dtype=int)
        cats      = np.full(n_ref, "baseline", dtype=object)

        for ev in self.events:
            i0 = int(ev.start_s   * fs_ref)
            i1 = int(ev.end_s     * fs_ref)
            labels[i0:i1]    = ev.emotion
            event_ids[i0:i1] = self.events.index(ev) + 1
            cats[i0:i1]      = ev.profile["category"]

        return pd.DataFrame({
            "time_s":   np.round(t_ref, 4),
            "label":    labels,
            "event_id": event_ids,
            "category": cats,
        })
