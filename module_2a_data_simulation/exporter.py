"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  exporter.py
=============================================================================
CSV / JSON export engine.

Every CSV includes three target variable columns:
  target_label  – emotion / behaviour name, or 'baseline'
  event_id      – integer (1-based event number; 0 = baseline)
  category      – 'affective' | 'physiological_need' | 'baseline'

Per-signal CSVs (native sampling rate):
  EDA.csv       – timestamp_s, EDA_uS,          target_label, event_id, category
  BVP.csv       – timestamp_s, BVP_nT,          target_label, event_id, category
  IBI.csv       – timestamp_s, IBI_ms,          target_label, event_id, category
  ST.csv        – timestamp_s, ST_degC,         target_label, event_id, category
  ACC.csv       – timestamp_s, ACC_X/Y/Z_g,     target_label, event_id, category

Combined CSV (all channels at 64 Hz):
  combined_signals.csv – all above + IBI sparse column

Annotation CSVs + metadata JSON unchanged.
=============================================================================
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from scipy.interpolate import interp1d

from config import SAMPLING_RATES
from simulator import SimulationResult


class DataExporter:
    """
    Export a SimulationResult to CSV and JSON files.

    Parameters
    ----------
    result     : SimulationResult
    output_dir : destination directory (created if absent)
    """

    def __init__(self, result: SimulationResult, output_dir: str = "output"):
        self.result     = result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Master method ─────────────────────────────────────────────────────────

    def export_all(self) -> Dict[str, Path]:
        saved = {}
        print("[Exporter] Writing per-signal CSVs ...")
        saved.update(self.export_individual_signals())

        print("[Exporter] Writing combined CSV ...")
        saved["combined"] = self.export_combined()

        print("[Exporter] Writing annotation CSVs ...")
        saved.update(self.export_annotations())

        print("[Exporter] Writing metadata ...")
        saved["metadata"] = self.export_metadata()

        return saved

    # ── Target label helper ───────────────────────────────────────────────────

    def _labels_for_timestamps(self, timestamps: np.ndarray):
        """
        Map a timestamp array to (target_label, event_id, category) arrays.

        Each timestamp is compared against the scheduled event windows.
        Timestamps outside all events receive label='baseline', event_id=0.
        """
        n      = len(timestamps)
        labels = np.full(n, "baseline", dtype=object)
        ev_ids = np.zeros(n, dtype=int)
        cats   = np.full(n, "baseline", dtype=object)

        for i, ev in enumerate(self.result.events):
            mask = (timestamps >= ev.start_s) & (timestamps < ev.end_s)
            labels[mask] = ev.emotion
            ev_ids[mask] = i + 1
            cats[mask]   = ev.profile["category"]

        return labels, ev_ids, cats

    # ── Individual signal CSVs ────────────────────────────────────────────────

    def export_individual_signals(self) -> Dict[str, Path]:
        """
        Write one CSV per signal at its native sampling rate.

        Schema for every CSV:
          timestamp_s  | <signal_col(s)> | target_label | event_id | category
        """
        r     = self.result
        saved: Dict[str, Path] = {}

        # ── EDA  (4 Hz) ───────────────────────────────────────────────────
        ts = r.time_vectors["EDA"]
        lbl, eid, cat = self._labels_for_timestamps(ts)
        saved["EDA"] = self._write_df(pd.DataFrame({
            "timestamp_s":  np.round(ts, 4),
            "EDA_uS":       r.signals["EDA"],
            "target_label": lbl,
            "event_id":     eid,
            "category":     cat,
        }), "EDA.csv")

        # ── BVP  (64 Hz) ──────────────────────────────────────────────────
        ts = r.time_vectors["BVP"]
        lbl, eid, cat = self._labels_for_timestamps(ts)
        saved["BVP"] = self._write_df(pd.DataFrame({
            "timestamp_s":  np.round(ts, 5),
            "BVP_nT":       r.signals["BVP"],
            "target_label": lbl,
            "event_id":     eid,
            "category":     cat,
        }), "BVP.csv")

        # ── IBI  (event-based — one row per heartbeat) ────────────────────
        lbl, eid, cat = self._labels_for_timestamps(r.ibi_times_s)
        saved["IBI"] = self._write_df(pd.DataFrame({
            "timestamp_s":  np.round(r.ibi_times_s, 5),
            "IBI_ms":       r.ibi_values_ms,
            "target_label": lbl,
            "event_id":     eid,
            "category":     cat,
        }), "IBI.csv")

        # ── ST  (4 Hz) ────────────────────────────────────────────────────
        ts = r.time_vectors["ST"]
        lbl, eid, cat = self._labels_for_timestamps(ts)
        saved["ST"] = self._write_df(pd.DataFrame({
            "timestamp_s":  np.round(ts, 4),
            "ST_degC":      r.signals["ST"],
            "target_label": lbl,
            "event_id":     eid,
            "category":     cat,
        }), "ST.csv")

        # ── ACC  (32 Hz — all three axes in one file) ─────────────────────
        ts = r.time_vectors["ACC_X"]
        lbl, eid, cat = self._labels_for_timestamps(ts)
        saved["ACC"] = self._write_df(pd.DataFrame({
            "timestamp_s":  np.round(ts, 5),
            "ACC_X_g":      r.signals["ACC_X"],
            "ACC_Y_g":      r.signals["ACC_Y"],
            "ACC_Z_g":      r.signals["ACC_Z"],
            "target_label": lbl,
            "event_id":     eid,
            "category":     cat,
        }), "ACC.csv")

        return saved

    # ── Combined CSV ──────────────────────────────────────────────────────────

    def export_combined(self) -> Path:
        """
        All channels on a 64 Hz reference grid with target label columns.
        Lower-rate signals are linearly interpolated.
        IBI appears as a sparse column (NaN between beats).
        """
        r      = self.result
        fs_ref = SAMPLING_RATES["BVP"]
        n_ref  = len(r.signals["BVP"])
        t_ref  = r.time_vectors["BVP"]

        # Target labels at 64 Hz
        lbl, eid, cat = self._labels_for_timestamps(t_ref)

        df = pd.DataFrame({
            "timestamp_s":  np.round(t_ref, 5),
            "BVP_nT":       r.signals["BVP"],
            "target_label": lbl,
            "event_id":     eid,
            "category":     cat,
        })

        # Interpolate slower signals onto 64 Hz grid
        for sig_name, col in [("EDA","EDA_uS"), ("ST","ST_degC"),
                               ("ACC_X","ACC_X_g"), ("ACC_Y","ACC_Y_g"),
                               ("ACC_Z","ACC_Z_g")]:
            interp = interp1d(
                r.time_vectors[sig_name], r.signals[sig_name],
                kind="linear", bounds_error=False, fill_value="extrapolate",
            )
            df[col] = interp(t_ref)

        # IBI sparse column
        ibi_col = np.full(n_ref, np.nan)
        for bt, iv in zip(r.ibi_times_s, r.ibi_values_ms):
            idx = int(round(bt * fs_ref))
            if 0 <= idx < n_ref:
                ibi_col[idx] = iv
        df["IBI_ms"] = ibi_col

        # Reorder columns for readability
        col_order = ["timestamp_s", "target_label", "event_id", "category",
                     "EDA_uS", "BVP_nT", "IBI_ms", "ST_degC",
                     "ACC_X_g", "ACC_Y_g", "ACC_Z_g"]
        df = df[[c for c in col_order if c in df.columns]]

        return self._write_df(df, "combined_signals.csv")

    # ── Annotation CSVs ───────────────────────────────────────────────────────

    def export_annotations(self) -> Dict[str, Path]:
        saved: Dict[str, Path] = {}
        mapping = {
            "events":         "annotations_events.csv",
            "baseline_wins":  "annotations_baseline_windows.csv",
            "signal_quality": "annotations_signal_quality.csv",
            "sample_labels":  "annotations_sample_labels.csv",
        }
        for key, filename in mapping.items():
            ann = self.result.annotations
            if key in ann and len(ann[key]) > 0:
                saved[key] = self._write_df(ann[key], filename)
        return saved

    # ── Metadata JSON ─────────────────────────────────────────────────────────

    def export_metadata(self) -> Path:
        meta = dict(self.result.metadata)
        meta["events"] = [
            {
                "event_id":   i + 1,
                "emotion":    ev.emotion,
                "category":   ev.profile["category"],
                "start_s":    round(ev.start_s, 3),
                "end_s":      round(ev.end_s, 3),
                "duration_s": round(ev.duration_s, 3),
            }
            for i, ev in enumerate(self.result.events)
        ]
        meta["signal_stats"] = {
            name: {
                "n_samples": int(len(arr)),
                "mean":  round(float(np.mean(arr)), 4),
                "std":   round(float(np.std(arr)),  4),
                "min":   round(float(np.min(arr)),  4),
                "max":   round(float(np.max(arr)),  4),
            }
            for name, arr in self.result.signals.items()
        }
        meta["ibi_n_beats"] = int(len(self.result.ibi_times_s))

        out_path = self.output_dir / "metadata.json"
        with open(out_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"  [Exporter] Saved metadata.json")
        return out_path

    # ── Internal helper ───────────────────────────────────────────────────────

    def _write_df(self, df: pd.DataFrame, filename: str) -> Path:
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"  [Exporter] Saved {filename}  ({len(df):,} rows)")
        return out_path

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-USER CROSS-EXPORT  (module-level function, not a class method)
# ─────────────────────────────────────────────────────────────────────────────


def export_all_users_combined(
    results: list,          # List[SimulationResult]
    run_folder: "Path",
) -> _Path:
    """
    Stack per-user combined_signals.csv files into one cross-user CSV.

    Adds a `user_id` column as the first column.
    Saved as:  <run_folder>/all_users_combined.csv

    Also writes a run_summary.csv with one row per user showing
    key physiological parameters and event counts.
    """
    frames = []
    for result in results:
        uid = result.user_profile.user_id if result.user_profile else "user_001"
        # Re-build a lightweight combined frame from signals
        from scipy.interpolate import interp1d as _interp1d
        import numpy as _np

        fs_ref = SAMPLING_RATES["BVP"]
        t_ref  = result.time_vectors["BVP"]
        n_ref  = len(result.signals["BVP"])

        lbl, eid, cat = DataExporter(result)._labels_for_timestamps(t_ref)

        df = pd.DataFrame({
            "user_id":      uid,
            "timestamp_s":  _np.round(t_ref, 5),
            "target_label": lbl,
            "event_id":     eid,
            "category":     cat,
            "BVP_nT":       result.signals["BVP"],
        })

        for sig_name, col in [("EDA","EDA_uS"), ("ST","ST_degC"),
                               ("ACC_X","ACC_X_g"), ("ACC_Y","ACC_Y_g"),
                               ("ACC_Z","ACC_Z_g")]:
            interp = _interp1d(
                result.time_vectors[sig_name], result.signals[sig_name],
                kind="linear", bounds_error=False, fill_value="extrapolate")
            df[col] = interp(t_ref)

        ibi_col = _np.full(n_ref, _np.nan)
        for bt, iv in zip(result.ibi_times_s, result.ibi_values_ms):
            idx = int(round(bt * fs_ref))
            if 0 <= idx < n_ref:
                ibi_col[idx] = iv
        df["IBI_ms"] = ibi_col

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    out_path = Path(run_folder) / "all_users_combined.csv"
    combined.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  [Exporter] Saved all_users_combined.csv  ({len(combined):,} rows, "
          f"{len(results)} users)")

    # ── Run summary ───────────────────────────────────────────────────────────
    summary_rows = []
    for result in results:
        up  = result.user_profile
        row = {
            "user_id":          up.user_id if up else "user_001",
            "n_events":         len(result.events),
            "emotions":         ", ".join(ev.emotion for ev in result.events),
            "duration_s":       result.duration_s,
            "noise_level":      result.metadata.get("noise_level"),
            "seed":             result.metadata.get("seed"),
        }
        if up:
            row.update({
                "eda_tonic_mean":   round(up.eda_tonic_mean, 3),
                "hr_resting":       round(up.hr_resting, 1),
                "hrv_sdnn":         round(up.hrv_sdnn, 1),
                "st_baseline":      round(up.st_baseline, 2),
                "eda_reactivity":   round(up.eda_reactivity, 3),
                "hr_reactivity":    round(up.hr_reactivity, 3),
                "movement_scale":   round(up.movement_scale, 3),
                "reactivity_class": up.reactivity_class,
            })
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    sum_path   = Path(run_folder) / "run_summary.csv"
    summary_df.to_csv(sum_path, index=False)
    print(f"  [Exporter] Saved run_summary.csv  ({len(summary_df)} users)")

    return out_path
