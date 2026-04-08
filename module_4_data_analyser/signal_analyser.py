"""
=============================================================================
MODULE 4 – DATA ANALYSER  |  signal_analyser.py
=============================================================================
Temporal dynamics analysis of raw physiological signals per event.

Analyses for each event × signal:
  1. Change % from pre-event baseline to peak
  2. Time to peak (onset → peak)
  3. Whether signal peak aligns with the observed event label
  4. Time to subside (peak → 50% return)
  5. Return to median — does it return? how many events return?
  6. Median drift — does the running median shift after events?
  7. Adaptive threshold — flag sustained deviations from running median
=============================================================================
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import (
    BASELINE_WINDOW_S, RETURN_TOLERANCE, SUBSIDE_THRESHOLD,
    DEFAULT_THRESHOLD_WINDOW_S, DEFAULT_THRESHOLD_MAD_FACTOR,
    DEFAULT_THRESHOLD_SUSTAIN_S, VALUE_COLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EventDynamics:
    """Temporal dynamics metrics for one event × one signal."""
    event_id:          int
    target_label:      str
    signal_name:       str
    # Baseline
    baseline_median:   float
    baseline_std:      float
    # Peak
    peak_value:        float
    peak_time_s:       float       # seconds from event onset to peak
    change_pct:        float       # (peak - baseline) / |baseline| × 100
    # Peak alignment
    peak_in_event:     bool        # True if peak occurs within event window
    peak_delay_s:      float       # seconds from event start to peak
    # Subside
    subside_time_s:    float       # seconds from peak to 50% return (nan if no return)
    # Return to median
    returned_to_median:bool
    time_to_return_s:  float       # nan if did not return
    # Post-event median
    post_event_median: float
    median_drift:      float       # post-event median − pre-event median

    def to_dict(self) -> dict:
        return {
            "event_id":           self.event_id,
            "target_label":       self.target_label,
            "signal":             self.signal_name,
            "baseline_median":    round(self.baseline_median, 4),
            "baseline_std":       round(self.baseline_std,    4),
            "peak_value":         round(self.peak_value,      4),
            "peak_time_s":        round(self.peak_time_s,     2),
            "change_pct":         round(self.change_pct,      2),
            "peak_in_event":      self.peak_in_event,
            "peak_delay_s":       round(self.peak_delay_s,    2),
            "subside_time_s":     round(self.subside_time_s,  2) if not np.isnan(self.subside_time_s) else np.nan,
            "returned_to_median": self.returned_to_median,
            "time_to_return_s":   round(self.time_to_return_s, 2) if not np.isnan(self.time_to_return_s) else np.nan,
            "post_event_median":  round(self.post_event_median, 4),
            "median_drift":       round(self.median_drift,    4),
        }


@dataclass
class ThresholdEvent:
    """A detected adaptive threshold crossing."""
    signal_name:    str
    start_s:        float
    end_s:          float
    duration_s:     float
    deviation_mad:  float   # how many MADs above the running median
    peak_value:     float
    running_median: float

    def to_dict(self):
        return {
            "signal":         self.signal_name,
            "start_s":        round(self.start_s, 2),
            "end_s":          round(self.end_s,   2),
            "duration_s":     round(self.duration_s, 2),
            "deviation_mad":  round(self.deviation_mad, 2),
            "peak_value":     round(self.peak_value,   4),
            "running_median": round(self.running_median, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

class SignalAnalyser:
    """
    Analyse raw signal temporal dynamics relative to annotated events.

    Parameters
    ----------
    threshold_window_s  : rolling window for adaptive median (default 300 s)
    threshold_mad_factor: deviation multiplier (default 3 × MAD)
    threshold_sustain_s : minimum sustained duration to flag (default 30 s)
    baseline_window_s   : pre-event window used as baseline (default 30 s)
    verbose             : print progress
    """

    def __init__(
        self,
        threshold_window_s:   float = DEFAULT_THRESHOLD_WINDOW_S,
        threshold_mad_factor: float = DEFAULT_THRESHOLD_MAD_FACTOR,
        threshold_sustain_s:  float = DEFAULT_THRESHOLD_SUSTAIN_S,
        baseline_window_s:    float = BASELINE_WINDOW_S,
        verbose:              bool  = True,
    ):
        self.threshold_window_s   = threshold_window_s
        self.threshold_mad_factor = threshold_mad_factor
        self.threshold_sustain_s  = threshold_sustain_s
        self.baseline_window_s    = baseline_window_s
        self.verbose              = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def analyse_all(
        self,
        signals: Dict[str, pd.DataFrame],
    ) -> Dict[str, object]:
        """
        Run all temporal analyses on all signal channels.

        Parameters
        ----------
        signals : {signal_name: DataFrame} from Module 3 cleaned output
                  Each DataFrame must have: timestamp_s, <value_col>,
                  target_label (annotation).

        Returns
        -------
        dict with keys:
          event_dynamics   : pd.DataFrame  — per-event × per-signal metrics
          median_drift     : pd.DataFrame  — running median over time per signal
          return_summary   : pd.DataFrame  — % events returning to median per signal
          threshold_events : pd.DataFrame  — adaptive threshold crossings
        """
        all_dynamics:   List[dict] = []
        all_thresholds: List[dict] = []
        median_drift_dfs: List[pd.DataFrame] = []

        for sig_name, df in signals.items():
            val_col = VALUE_COLS.get(sig_name) or self._find_val_col(df, sig_name)
            if val_col is None or val_col not in df.columns:
                continue
            if "timestamp_s" not in df.columns:
                continue

            self._log(f"  [SignalAnalyser] {sig_name} ({len(df):,} samples)")

            ts   = df["timestamp_s"].values.astype(float)
            vals = df[val_col].values.astype(float)

            # ── Event dynamics ─────────────────────────────────────────
            if "target_label" in df.columns:
                events = self._extract_events(df, ts)
                for ev in events:
                    d = self._analyse_event(
                        ev, ts, vals, sig_name, df
                    )
                    if d:
                        all_dynamics.append(d.to_dict())

            # ── Running median drift ───────────────────────────────────
            md_df = self._running_median(ts, vals, sig_name)
            median_drift_dfs.append(md_df)

            # ── Adaptive threshold crossings ───────────────────────────
            thr_events = self._detect_threshold_crossings(ts, vals, sig_name)
            all_thresholds.extend([t.to_dict() for t in thr_events])

        dynamics_df   = pd.DataFrame(all_dynamics) if all_dynamics else pd.DataFrame()
        thresholds_df = pd.DataFrame(all_thresholds) if all_thresholds else pd.DataFrame()

        return {
            "event_dynamics":   dynamics_df,
            "median_drift":     median_drift_dfs,
            "return_summary":   self._return_summary(dynamics_df),
            "threshold_events": thresholds_df,
        }

    # ── Event extraction ───────────────────────────────────────────────────

    def _extract_events(self, df, ts):
        """Extract event segments from annotated signal DataFrame."""
        labels = df["target_label"].values
        events = []
        i = 0
        while i < len(labels):
            lbl = labels[i]
            if lbl == "baseline":
                i += 1
                continue
            j = i
            while j < len(labels) and labels[j] == lbl:
                j += 1
            ev_id = df["event_id"].iloc[i] if "event_id" in df.columns else len(events) + 1
            events.append({
                "event_id": int(ev_id),
                "label":    lbl,
                "start_s":  float(ts[i]),
                "end_s":    float(ts[j - 1]),
                "start_idx":i,
                "end_idx":  j,
            })
            i = j
        return events

    # ── Single event analysis ──────────────────────────────────────────────

    def _analyse_event(self, ev, ts, vals, sig_name, df) -> Optional[EventDynamics]:
        """Compute temporal dynamics for one event."""
        s_idx, e_idx = ev["start_idx"], ev["end_idx"]
        start_s = ev["start_s"]

        # Pre-event baseline window
        baseline_mask = (ts >= start_s - self.baseline_window_s) & (ts < start_s)
        if baseline_mask.sum() < 3:
            baseline_mask = ts < start_s  # use all pre-event data
        if baseline_mask.sum() < 2:
            return None

        baseline_vals  = vals[baseline_mask]
        baseline_med   = float(np.median(baseline_vals))
        baseline_std   = float(np.std(baseline_vals))

        # Event segment
        event_vals = vals[s_idx:e_idx]
        event_ts   = ts[s_idx:e_idx]
        if len(event_vals) < 2:
            return None

        # Peak
        peak_dir = 1 if np.mean(event_vals) >= baseline_med else -1
        if peak_dir == 1:
            peak_idx_local = int(np.argmax(event_vals))
        else:
            peak_idx_local = int(np.argmin(event_vals))

        peak_val    = float(event_vals[peak_idx_local])
        peak_ts     = float(event_ts[peak_idx_local])
        peak_delay  = peak_ts - start_s
        change_pct  = ((peak_val - baseline_med) / (abs(baseline_med) + 1e-9)) * 100
        peak_in_ev  = s_idx <= s_idx + peak_idx_local < e_idx

        # Subside: time from peak to 50% return toward baseline
        post_peak_vals = event_vals[peak_idx_local:]
        post_peak_ts   = event_ts[peak_idx_local:]
        rise           = abs(peak_val - baseline_med)
        half_return    = baseline_med + peak_dir * rise * (1 - SUBSIDE_THRESHOLD)
        subside_s = np.nan
        if len(post_peak_vals) > 1:
            if peak_dir == 1:
                crossed = post_peak_vals <= half_return
            else:
                crossed = post_peak_vals >= half_return
            if crossed.any():
                subside_s = float(post_peak_ts[crossed.argmax()]) - peak_ts

        # Return to median (full return) — look up to 60 s after event
        end_s     = ev["end_s"]
        post_mask = (ts > end_s) & (ts <= end_s + 120)
        post_vals = vals[post_mask]
        post_ts_  = ts[post_mask]
        returned  = False
        t_return  = np.nan
        tol       = abs(baseline_med) * RETURN_TOLERANCE + 0.01
        if len(post_vals) > 1:
            at_baseline = np.abs(post_vals - baseline_med) <= tol
            if at_baseline.any():
                returned = True
                t_return = float(post_ts_[at_baseline.argmax()]) - end_s

        # Post-event median (30 s window after event)
        post_med_mask = (ts > end_s) & (ts <= end_s + 60)
        post_med = float(np.median(vals[post_med_mask])) if post_med_mask.sum() > 2 else baseline_med
        drift    = post_med - baseline_med

        return EventDynamics(
            event_id          = ev["event_id"],
            target_label      = ev["label"],
            signal_name       = sig_name,
            baseline_median   = baseline_med,
            baseline_std      = baseline_std,
            peak_value        = peak_val,
            peak_time_s       = peak_ts,
            change_pct        = change_pct,
            peak_in_event     = peak_in_ev,
            peak_delay_s      = peak_delay,
            subside_time_s    = subside_s,
            returned_to_median= returned,
            time_to_return_s  = t_return,
            post_event_median = post_med,
            median_drift      = drift,
        )

    # ── Running median ──────────────────────────────────────────────────────

    def _running_median(self, ts, vals, sig_name) -> pd.DataFrame:
        """Compute rolling median and MAD over time."""
        fs      = 1.0 / (np.median(np.diff(ts)) + 1e-9)
        win_n   = max(3, int(self.threshold_window_s * fs))
        ser     = pd.Series(vals)
        med     = ser.rolling(win_n, min_periods=3, center=True).median()
        mad     = ser.rolling(win_n, min_periods=3, center=True).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        return pd.DataFrame({
            "timestamp_s":      np.round(ts, 4),
            "signal":           sig_name,
            "value":            np.round(vals, 6),
            "running_median":   np.round(med.values, 6),
            "running_mad":      np.round(mad.values, 6),
        })

    # ── Adaptive threshold ──────────────────────────────────────────────────

    def _detect_threshold_crossings(
        self, ts, vals, sig_name
    ) -> List[ThresholdEvent]:
        """Detect sustained deviations beyond threshold_mad_factor × MAD."""
        fs    = 1.0 / (np.median(np.diff(ts)) + 1e-9)
        win_n = max(3, int(self.threshold_window_s * fs))
        ser   = pd.Series(vals)
        med   = ser.rolling(win_n, min_periods=3, center=True).median()
        mad   = ser.rolling(win_n, min_periods=3, center=True).apply(
            lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True
        )

        # Deviation in MAD units
        deviation  = np.abs(vals - med.values) / (mad.values * self.threshold_mad_factor + 1e-9)
        above_thr  = deviation >= 1.0
        sustain_n  = max(1, int(self.threshold_sustain_s * fs))

        events: List[ThresholdEvent] = []
        i = 0
        while i < len(above_thr):
            if above_thr[i]:
                j = i
                while j < len(above_thr) and above_thr[j]:
                    j += 1
                if (j - i) >= sustain_n:
                    seg      = vals[i:j]
                    seg_dev  = deviation[i:j]
                    events.append(ThresholdEvent(
                        signal_name    = sig_name,
                        start_s        = float(ts[i]),
                        end_s          = float(ts[j - 1]),
                        duration_s     = float(ts[j - 1]) - float(ts[i]),
                        deviation_mad  = float(np.max(seg_dev)),
                        peak_value     = float(seg[np.argmax(np.abs(seg - float(med.values[i])))]),
                        running_median = float(med.values[i]),
                    ))
                i = j
            else:
                i += 1
        return events

    # ── Summary helpers ─────────────────────────────────────────────────────

    def _return_summary(self, dynamics_df: pd.DataFrame) -> pd.DataFrame:
        """Compute % events returning to median per signal × target."""
        if dynamics_df.empty:
            return pd.DataFrame()
        grp = dynamics_df.groupby(["signal", "target_label"])
        summary = grp["returned_to_median"].agg(
            total="count",
            returned="sum",
        ).reset_index()
        summary["pct_returned"] = (summary["returned"] / summary["total"] * 100).round(1)
        summary["mean_return_time_s"] = dynamics_df.groupby(
            ["signal", "target_label"]
        )["time_to_return_s"].mean().round(2).values
        return summary

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _find_val_col(df, sig_name):
        candidates = {
            "EDA": "EDA_uS", "BVP": "BVP_nT",
            "IBI": "IBI_ms", "ST": "ST_degC",
            "ACC": "ACC_X_g",
        }
        return candidates.get(sig_name)

    def _log(self, msg):
        if self.verbose:
            print(msg)
