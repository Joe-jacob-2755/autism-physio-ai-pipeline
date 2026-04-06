"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  exporter.py
=============================================================================
CSV export engine.

Outputs
-------
Per-signal CSVs (native sampling rate):
  EDA.csv       – timestamp_s, EDA_uS
  BVP.csv       – timestamp_s, BVP_nT
  IBI.csv       – timestamp_s, IBI_ms
  ST.csv        – timestamp_s, ST_degC
  ACC.csv       – timestamp_s, ACC_X_g, ACC_Y_g, ACC_Z_g

Combined CSV (BVP rate, 64 Hz – other signals interpolated):
  combined_signals.csv – timestamp_s, EDA, BVP, IBI, ST, ACC_X, ACC_Y, ACC_Z

Annotation CSVs:
  annotations_events.csv
  annotations_baseline_windows.csv
  annotations_signal_quality.csv
  annotations_sample_labels.csv (one row per BVP sample)

Metadata JSON:
  metadata.json

=============================================================================
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from scipy.interpolate import interp1d

from config import SAMPLING_RATES
from simulator import SimulationResult


class DataExporter:
    """
    Export a SimulationResult to CSV files.

    Parameters
    ----------
    result     : SimulationResult
    output_dir : destination directory (created if absent)
    """

    def __init__(self, result: SimulationResult, output_dir: str = "output"):
        self.result     = result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Master method ───────────────────────────────────────────────────────

    def export_all(self) -> Dict[str, Path]:
        """
        Export every file and return a dict of {label: Path}.
        """
        saved = {}
        print("[Exporter] Writing per-signal CSVs …")
        saved.update(self.export_individual_signals())

        print("[Exporter] Writing combined CSV …")
        saved["combined"] = self.export_combined()

        print("[Exporter] Writing annotation CSVs …")
        saved.update(self.export_annotations())

        print("[Exporter] Writing metadata …")
        saved["metadata"] = self.export_metadata()

        return saved

    # ── Individual signal CSVs ───────────────────────────────────────────────

    def export_individual_signals(self) -> Dict[str, Path]:
        r  = self.result
        saved: Dict[str, Path] = {}

        # EDA
        path = self._write_df(
            pd.DataFrame({
                "timestamp_s": r.time_vectors["EDA"],
                "EDA_uS":      r.signals["EDA"],
            }),
            "EDA.csv",
        )
        saved["EDA"] = path

        # BVP
        path = self._write_df(
            pd.DataFrame({
                "timestamp_s": r.time_vectors["BVP"],
                "BVP_nT":      r.signals["BVP"],
            }),
            "BVP.csv",
        )
        saved["BVP"] = path

        # IBI  (event-based – each row is a detected beat)
        path = self._write_df(
            pd.DataFrame({
                "timestamp_s": r.ibi_times_s,
                "IBI_ms":      r.ibi_values_ms,
            }),
            "IBI.csv",
        )
        saved["IBI"] = path

        # ST
        path = self._write_df(
            pd.DataFrame({
                "timestamp_s": r.time_vectors["ST"],
                "ST_degC":     r.signals["ST"],
            }),
            "ST.csv",
        )
        saved["ST"] = path

        # ACC  (3-axis in one file, native 32 Hz)
        path = self._write_df(
            pd.DataFrame({
                "timestamp_s": r.time_vectors["ACC_X"],
                "ACC_X_g":     r.signals["ACC_X"],
                "ACC_Y_g":     r.signals["ACC_Y"],
                "ACC_Z_g":     r.signals["ACC_Z"],
            }),
            "ACC.csv",
        )
        saved["ACC"] = path

        return saved

    # ── Combined CSV ─────────────────────────────────────────────────────────

    def export_combined(self) -> Path:
        """
        Produce a single CSV with all channels time-aligned to 64 Hz (BVP rate).
        Lower-rate signals are interpolated; IBI is NaN except at beat events.
        """
        r      = self.result
        fs_ref = SAMPLING_RATES["BVP"]   # 64 Hz reference grid
        n_ref  = len(r.signals["BVP"])
        t_ref  = r.time_vectors["BVP"]

        df = pd.DataFrame({"timestamp_s": np.round(t_ref, 5)})
        df["BVP_nT"] = r.signals["BVP"]

        # Interpolate slow signals onto 64 Hz grid
        for sig_name, col_label, unit in [
            ("EDA",   "EDA_uS",  None),
            ("ST",    "ST_degC", None),
            ("ACC_X", "ACC_X_g", None),
            ("ACC_Y", "ACC_Y_g", None),
            ("ACC_Z", "ACC_Z_g", None),
        ]:
            t_native  = r.time_vectors[sig_name]
            v_native  = r.signals[sig_name]
            interp    = interp1d(
                t_native, v_native,
                kind       = "linear",
                bounds_error = False,
                fill_value = "extrapolate",
            )
            df[col_label] = interp(t_ref)

        # IBI as sparse column (NaN between beats)
        ibi_col = np.full(n_ref, np.nan)
        for bt, iv in zip(r.ibi_times_s, r.ibi_values_ms):
            idx = int(round(bt * fs_ref))
            if 0 <= idx < n_ref:
                ibi_col[idx] = iv
        df["IBI_ms"] = ibi_col

        # Emotion label column (from sample_labels annotation)
        if "sample_labels" in r.annotations:
            lbl_df    = r.annotations["sample_labels"]
            # sample_labels is at 64 Hz too
            if len(lbl_df) == n_ref:
                df["label"]    = lbl_df["label"].values
                df["event_id"] = lbl_df["event_id"].values
                df["category"] = lbl_df["category"].values

        return self._write_df(df, "combined_signals.csv")

    # ── Annotation CSVs ──────────────────────────────────────────────────────

    def export_annotations(self) -> Dict[str, Path]:
        saved: Dict[str, Path] = {}
        ann = self.result.annotations

        mapping = {
            "events":         "annotations_events.csv",
            "baseline_wins":  "annotations_baseline_windows.csv",
            "signal_quality": "annotations_signal_quality.csv",
            "sample_labels":  "annotations_sample_labels.csv",
        }

        for key, filename in mapping.items():
            if key in ann and len(ann[key]) > 0:
                saved[key] = self._write_df(ann[key], filename)

        return saved

    # ── Metadata JSON ─────────────────────────────────────────────────────────

    def export_metadata(self) -> Path:
        meta = dict(self.result.metadata)

        # Add event summary
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

        # Add signal statistics
        meta["signal_stats"] = {}
        for name, arr in self.result.signals.items():
            meta["signal_stats"][name] = {
                "n_samples": int(len(arr)),
                "mean":      round(float(np.mean(arr)), 4),
                "std":       round(float(np.std(arr)),  4),
                "min":       round(float(np.min(arr)),  4),
                "max":       round(float(np.max(arr)),  4),
            }
        meta["ibi_n_beats"] = int(len(self.result.ibi_times_s))

        out_path = self.output_dir / "metadata.json"
        with open(out_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"  [Exporter] Saved {out_path.name}")
        return out_path

    # ── Internal helper ───────────────────────────────────────────────────────

    def _write_df(self, df: pd.DataFrame, filename: str) -> Path:
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False, float_format="%.6f")
        print(f"  [Exporter] Saved {filename}  ({len(df):,} rows)")
        return out_path
