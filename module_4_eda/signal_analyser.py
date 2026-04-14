"""
=============================================================================
MODULE 4 - EXPLORATORY DATA ANALYSIS  |  signal_analyser.py
=============================================================================
Temporal event dynamics on combined multi-user raw physiological signals.

Section 8 of the EDA report:
  8.1 — Event duration analysis
  8.2 — Signal % change from baseline
  8.3 — Time to peak
  8.4 — Return-to-median analysis (rate)
  8.5 — Average time to return to median
  8.6 — Return count by signal and target
  8.7 — Median drift (post-event shift)
  8.8 — Adaptive threshold crossings
=============================================================================
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import (
    BASELINE_WINDOW_S, RETURN_TOLERANCE, SUBSIDE_THRESHOLD,
    POST_EVENT_WINDOW_S,
    DEFAULT_THRESHOLD_WINDOW_S, DEFAULT_THRESHOLD_MAD_FACTOR,
    DEFAULT_THRESHOLD_SUSTAIN_S, VALUE_COLS,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EventDynamics:
    """Temporal dynamics metrics for one event x one signal x one user."""
    user_id: str
    event_id: int
    target_label: str
    signal_name: str
    event_start_s: float
    event_end_s: float
    event_duration_s: float
    # Baseline
    baseline_median: float
    baseline_std: float
    # Peak
    peak_value: float
    peak_time_s: float
    change_pct: float
    # Peak alignment
    peak_in_event: bool
    peak_delay_s: float
    # Subside
    subside_time_s: float
    # Return to median
    returned_to_median: bool
    time_to_return_s: float
    # Drift
    post_event_median: float
    median_drift: float

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, float) and np.isnan(v):
                d[k] = np.nan
            elif isinstance(v, float):
                d[k] = round(v, 4)
            else:
                d[k] = v
        return d


@dataclass
class ThresholdEvent:
    """Detected adaptive threshold crossing."""
    signal_name: str
    user_id: str
    start_s: float
    end_s: float
    duration_s: float
    deviation_mad: float
    peak_value: float
    running_median: float

    def to_dict(self) -> dict:
        return {
            "signal": self.signal_name,
            "user_id": self.user_id,
            "start_s": round(self.start_s, 2),
            "end_s": round(self.end_s, 2),
            "duration_s": round(self.duration_s, 2),
            "deviation_mad": round(self.deviation_mad, 2),
            "peak_value": round(self.peak_value, 4),
            "running_median": round(self.running_median, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ANALYSER
# ─────────────────────────────────────────────────────────────────────────────

class SignalAnalyser:
    """
    Analyse temporal event dynamics across combined multi-user signals.

    Parameters
    ----------
    threshold_window_s  : rolling window for adaptive median
    threshold_mad_factor: deviation multiplier (N x MAD)
    threshold_sustain_s : minimum duration to flag crossing
    baseline_window_s   : pre-event baseline window
    verbose             : print progress
    """

    def __init__(
        self,
        threshold_window_s: float = DEFAULT_THRESHOLD_WINDOW_S,
        threshold_mad_factor: float = DEFAULT_THRESHOLD_MAD_FACTOR,
        threshold_sustain_s: float = DEFAULT_THRESHOLD_SUSTAIN_S,
        baseline_window_s: float = BASELINE_WINDOW_S,
        verbose: bool = True,
    ):
        self.threshold_window_s = threshold_window_s
        self.threshold_mad_factor = threshold_mad_factor
        self.threshold_sustain_s = threshold_sustain_s
        self.baseline_window_s = baseline_window_s
        self.verbose = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def analyse_all(
        self,
        signals: Dict[str, pd.DataFrame],
    ) -> Dict[str, object]:
        """
        Run all temporal analyses on combined multi-user signals.

        Processes each user's events independently within each signal
        to avoid cross-user contamination of baseline/post-event windows.

        Returns
        -------
        dict with keys:
          event_dynamics     : DataFrame (per-event x per-signal metrics)
          event_durations    : DataFrame (duration stats per target)
          return_summary     : DataFrame (% returning per signal x target)
          return_time_summary: DataFrame (mean return time per signal x target)
          return_counts      : DataFrame (count of returns per signal x target)
          median_drift_summary: DataFrame (mean drift per signal x target)
          threshold_events   : DataFrame (adaptive threshold crossings)
        """
        all_dynamics: List[dict] = []
        all_thresholds: List[dict] = []

        for sig_name, df in sorted(signals.items()):
            val_col = self._pick_val_col(sig_name, df)
            if val_col is None or "timestamp_s" not in df.columns:
                continue
            self._log(f"  [SignalAnalyser] {sig_name} ({len(df):,} rows)")

            # Process per-user to keep baselines correct
            user_col = "user_id" if "user_id" in df.columns else None
            user_groups = df.groupby("user_id") if user_col else [("all", df)]

            for uid, udf in user_groups:
                ts = udf["timestamp_s"].values.astype(float)
                vals = udf[val_col].values.astype(float)

                # Event dynamics
                if "target_label" in udf.columns:
                    events = self._extract_events(udf, ts)
                    for ev in events:
                        d = self._analyse_event(ev, ts, vals, sig_name, str(uid))
                        if d is not None:
                            all_dynamics.append(d.to_dict())

                # Adaptive threshold crossings
                thr = self._detect_threshold_crossings(ts, vals, sig_name, str(uid))
                all_thresholds.extend([t.to_dict() for t in thr])

        dynamics_df = pd.DataFrame(all_dynamics) if all_dynamics else pd.DataFrame()
        threshold_df = pd.DataFrame(all_thresholds) if all_thresholds else pd.DataFrame()

        return {
            "event_dynamics": dynamics_df,
            "event_durations": self._event_duration_stats(dynamics_df),
            "return_summary": self._return_summary(dynamics_df),
            "return_time_summary": self._return_time_summary(dynamics_df),
            "return_counts": self._return_counts(dynamics_df),
            "median_drift_summary": self._drift_summary(dynamics_df),
            "threshold_events": threshold_df,
        }

    # ── Event extraction ───────────────────────────────────────────────────

    def _extract_events(self, df: pd.DataFrame, ts: np.ndarray) -> List[dict]:
        """Extract contiguous labelled event segments."""
        labels = df["target_label"].values
        events = []
        i = 0
        while i < len(labels):
            lbl = str(labels[i])
            if lbl == "baseline":
                i += 1
                continue
            j = i
            while j < len(labels) and str(labels[j]) == lbl:
                j += 1
            ev_id = int(df["event_id"].iloc[i]) if "event_id" in df.columns else len(events) + 1
            events.append({
                "event_id": ev_id,
                "label": lbl,
                "start_s": float(ts[i]),
                "end_s": float(ts[j - 1]),
                "start_idx": i,
                "end_idx": j,
            })
            i = j
        return events

    # ── Single event analysis ──────────────────────────────────────────────

    def _analyse_event(
        self, ev: dict, ts: np.ndarray, vals: np.ndarray,
        sig_name: str, user_id: str,
    ) -> Optional[EventDynamics]:
        """Compute all temporal dynamics for one event."""
        s_idx, e_idx = ev["start_idx"], ev["end_idx"]
        start_s, end_s = ev["start_s"], ev["end_s"]
        duration = end_s - start_s

        # Pre-event baseline
        baseline_mask = (ts >= start_s - self.baseline_window_s) & (ts < start_s)
        if baseline_mask.sum() < 3:
            baseline_mask = ts < start_s
        if baseline_mask.sum() < 2:
            return None

        baseline_vals = vals[baseline_mask]
        baseline_med = float(np.nanmedian(baseline_vals))
        baseline_std = float(np.nanstd(baseline_vals))

        # Event segment
        event_vals = vals[s_idx:e_idx]
        event_ts = ts[s_idx:e_idx]
        if len(event_vals) < 2:
            return None

        # Peak detection (direction-aware)
        peak_dir = 1 if np.nanmean(event_vals) >= baseline_med else -1
        if peak_dir == 1:
            peak_idx = int(np.nanargmax(event_vals))
        else:
            peak_idx = int(np.nanargmin(event_vals))

        peak_val = float(event_vals[peak_idx])
        peak_ts = float(event_ts[peak_idx])
        peak_delay = peak_ts - start_s
        change_pct = ((peak_val - baseline_med) / (abs(baseline_med) + 1e-9)) * 100
        peak_in_ev = True  # always within event window by construction

        # Subside: time from peak to 50% return toward baseline
        post_peak_vals = event_vals[peak_idx:]
        post_peak_ts = event_ts[peak_idx:]
        rise = abs(peak_val - baseline_med)
        half_return = baseline_med + peak_dir * rise * (1 - SUBSIDE_THRESHOLD)
        subside_s = np.nan
        if len(post_peak_vals) > 1:
            if peak_dir == 1:
                crossed = post_peak_vals <= half_return
            else:
                crossed = post_peak_vals >= half_return
            if np.any(crossed):
                subside_s = float(post_peak_ts[np.argmax(crossed)]) - peak_ts

        # Return to median — look up to POST_EVENT_WINDOW_S after event end
        post_mask = (ts > end_s) & (ts <= end_s + POST_EVENT_WINDOW_S)
        post_vals = vals[post_mask]
        post_ts_arr = ts[post_mask]
        returned = False
        t_return = np.nan
        tol = abs(baseline_med) * RETURN_TOLERANCE + 0.01
        if len(post_vals) > 1:
            at_baseline = np.abs(post_vals - baseline_med) <= tol
            if np.any(at_baseline):
                returned = True
                t_return = float(post_ts_arr[np.argmax(at_baseline)]) - end_s

        # Post-event median (60 s window)
        post_med_mask = (ts > end_s) & (ts <= end_s + 60)
        post_med = float(np.nanmedian(vals[post_med_mask])) if post_med_mask.sum() > 2 else baseline_med
        drift = post_med - baseline_med

        return EventDynamics(
            user_id=user_id,
            event_id=ev["event_id"],
            target_label=ev["label"],
            signal_name=sig_name,
            event_start_s=start_s,
            event_end_s=end_s,
            event_duration_s=round(duration, 2),
            baseline_median=baseline_med,
            baseline_std=baseline_std,
            peak_value=peak_val,
            peak_time_s=peak_ts,
            change_pct=change_pct,
            peak_in_event=peak_in_ev,
            peak_delay_s=peak_delay,
            subside_time_s=subside_s,
            returned_to_median=returned,
            time_to_return_s=t_return,
            post_event_median=post_med,
            median_drift=drift,
        )

    # ── Adaptive threshold detection ───────────────────────────────────────

    def _detect_threshold_crossings(
        self, ts: np.ndarray, vals: np.ndarray,
        sig_name: str, user_id: str,
    ) -> List[ThresholdEvent]:
        """Detect sustained deviations > N x MAD from rolling median."""
        if len(ts) < 10:
            return []

        fs = 1.0 / (np.median(np.diff(ts)) + 1e-9)
        win_n = max(3, int(self.threshold_window_s * fs))

        ser = pd.Series(vals)
        med = ser.rolling(win_n, min_periods=3, center=True).median().values
        abs_dev = np.abs(vals - med)
        mad = pd.Series(abs_dev).rolling(win_n, min_periods=3, center=True).median().values

        deviation = np.abs(vals - med) / (mad * self.threshold_mad_factor + 1e-9)
        above_thr = deviation >= 1.0
        sustain_n = max(1, int(self.threshold_sustain_s * fs))

        events: List[ThresholdEvent] = []
        i = 0
        while i < len(above_thr):
            if above_thr[i]:
                j = i
                while j < len(above_thr) and above_thr[j]:
                    j += 1
                if (j - i) >= sustain_n:
                    seg_dev = deviation[i:j]
                    seg_vals = vals[i:j]
                    events.append(ThresholdEvent(
                        signal_name=sig_name,
                        user_id=user_id,
                        start_s=float(ts[i]),
                        end_s=float(ts[j - 1]),
                        duration_s=float(ts[j - 1]) - float(ts[i]),
                        deviation_mad=float(np.max(seg_dev)),
                        peak_value=float(seg_vals[np.argmax(np.abs(seg_vals - float(med[i])))]),
                        running_median=float(med[i]),
                    ))
                i = j
            else:
                i += 1
        return events

    # ── Summary helpers ─────────────────────────────────────────────────────

    def _event_duration_stats(self, dynamics_df: pd.DataFrame) -> pd.DataFrame:
        """Section 8.1 — Event duration stats per target label."""
        if dynamics_df.empty or "event_duration_s" not in dynamics_df.columns:
            return pd.DataFrame()
        # De-duplicate events (same event_id may appear for multiple signals)
        dedup = dynamics_df.drop_duplicates(subset=["user_id", "event_id", "target_label"])
        grp = dedup.groupby("target_label")["event_duration_s"]
        result = grp.agg(
            n_events="count",
            mean_duration_s="mean",
            std_duration_s="std",
            median_duration_s="median",
            min_duration_s="min",
            max_duration_s="max",
        ).round(2).reset_index()
        return result

    def _return_summary(self, dynamics_df: pd.DataFrame) -> pd.DataFrame:
        """Section 8.4 — % events where signal returned to median."""
        if dynamics_df.empty or "returned_to_median" not in dynamics_df.columns:
            return pd.DataFrame()
        grp = dynamics_df.groupby(["signal_name", "target_label"])
        summary = grp["returned_to_median"].agg(
            n_events="count",
            n_returned="sum",
        ).reset_index()
        summary["pct_returned"] = (summary["n_returned"] / summary["n_events"] * 100).round(1)
        return summary

    def _return_time_summary(self, dynamics_df: pd.DataFrame) -> pd.DataFrame:
        """Section 8.5 — Mean time to return to median per signal x target."""
        if dynamics_df.empty or "time_to_return_s" not in dynamics_df.columns:
            return pd.DataFrame()
        returned = dynamics_df[dynamics_df["returned_to_median"]].copy()
        if returned.empty:
            return pd.DataFrame()
        grp = returned.groupby(["signal_name", "target_label"])["time_to_return_s"]
        result = grp.agg(
            mean_return_s="mean",
            std_return_s="std",
            median_return_s="median",
            n_returned="count",
        ).round(2).reset_index()
        return result

    def _return_counts(self, dynamics_df: pd.DataFrame) -> pd.DataFrame:
        """Section 8.6 — Count of returns per signal x target."""
        if dynamics_df.empty:
            return pd.DataFrame()
        grp = dynamics_df.groupby(["signal_name", "target_label"])
        result = grp.agg(
            n_events=("returned_to_median", "count"),
            n_returned=("returned_to_median", "sum"),
            n_not_returned=("returned_to_median", lambda x: (~x).sum()),
        ).reset_index()
        return result

    def _drift_summary(self, dynamics_df: pd.DataFrame) -> pd.DataFrame:
        """Section 8.7 — Mean median drift per signal x target."""
        if dynamics_df.empty or "median_drift" not in dynamics_df.columns:
            return pd.DataFrame()
        grp = dynamics_df.groupby(["signal_name", "target_label"])["median_drift"]
        result = grp.agg(
            mean_drift="mean",
            std_drift="std",
            median_drift="median",
            n_events="count",
        ).round(4).reset_index()
        return result

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _pick_val_col(self, sig_name: str, df: pd.DataFrame) -> Optional[str]:
        """Pick the primary value column for a signal."""
        val_cols = VALUE_COLS.get(sig_name, [])
        if val_cols:
            vc = val_cols[0]
            return vc if vc in df.columns else None
        return None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: SIGNAL % CHANGE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def signal_pct_change_summary(dynamics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Section 8.2 — Summarise % change from baseline per signal x target.
    """
    if dynamics_df.empty or "change_pct" not in dynamics_df.columns:
        return pd.DataFrame()
    grp = dynamics_df.groupby(["signal_name", "target_label"])["change_pct"]
    return grp.agg(
        mean_change_pct="mean",
        std_change_pct="std",
        median_change_pct="median",
        min_change_pct="min",
        max_change_pct="max",
        n_events="count",
    ).round(2).reset_index()


def time_to_peak_summary(dynamics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Section 8.3 — Time to peak summary per signal x target.
    """
    if dynamics_df.empty or "peak_delay_s" not in dynamics_df.columns:
        return pd.DataFrame()
    grp = dynamics_df.groupby(["signal_name", "target_label"])["peak_delay_s"]
    return grp.agg(
        mean_peak_delay_s="mean",
        std_peak_delay_s="std",
        median_peak_delay_s="median",
        n_events="count",
    ).round(2).reset_index()
