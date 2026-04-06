"""
=============================================================================
MODULE 2 – DATA ACQUISITION  |  mode_2_1_import.py
=============================================================================
Mode 2.1 — Import Existing Data

Imports pre-existing physiological signal data from a designated folder
into the pipeline as a PipelinePacket.

Supported folder layouts
------------------------
Layout A — Module 1A output (recommended):
  <folder>/
    EDA.csv, BVP.csv, IBI.csv, ST.csv, ACC.csv
    combined_signals.csv
    metadata.json            (optional but strongly recommended)

Layout B — Individual signal CSVs only (no combined):
  <folder>/
    EDA.csv, BVP.csv, IBI.csv, ST.csv, ACC.csv

Layout C — Combined CSV only:
  <folder>/
    combined_signals.csv     (must contain timestamp_s + all signal cols)

Layout D — Single CSV:
  <file_path>.csv            (treated as combined signals)

Column requirements
-------------------
Each signal CSV must contain:
  - timestamp_s            (required)
  - <signal value column>  (EDA_uS / BVP_nT / IBI_ms / ST_degC / ACC_*_g)
  - target_label           (optional — if absent, is_annotated = False)
  - event_id               (optional)
  - category               (optional)

=============================================================================
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from pipeline_packet import (
    PipelinePacket, build_packet_from_dataframes,
    SOURCE_IMPORTED, ANNOTATION_COLS, REQUIRED_COL, _new_session_id,
)


# ─────────────────────────────────────────────────────────────────────────────
# KNOWN SIGNAL FILE NAMES AND THEIR VALUE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_FILES = {
    "EDA": {"file": "EDA.csv",  "value_cols": ["EDA_uS"]},
    "BVP": {"file": "BVP.csv",  "value_cols": ["BVP_nT"]},
    "IBI": {"file": "IBI.csv",  "value_cols": ["IBI_ms"]},
    "ST":  {"file": "ST.csv",   "value_cols": ["ST_degC"]},
    "ACC": {"file": "ACC.csv",  "value_cols": ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"]},
}

COMBINED_FILE = "combined_signals.csv"
METADATA_FILE = "metadata.json"


# ─────────────────────────────────────────────────────────────────────────────
# DATA IMPORTER
# ─────────────────────────────────────────────────────────────────────────────

class DataImporter:
    """
    Import existing physiological signal data into the pipeline.

    Parameters
    ----------
    source_path : path to a folder or single CSV file
    user_id     : participant identifier (auto-detected from metadata if present)
    session_id  : if None, auto-generated
    verbose     : print progress messages
    """

    def __init__(
        self,
        source_path: str | Path,
        user_id:     str           = "imported_user",
        session_id:  Optional[str] = None,
        verbose:     bool          = True,
    ):
        self.source_path = Path(source_path)
        self.user_id     = user_id
        self.session_id  = session_id or _new_session_id("IMP")
        self.verbose     = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> PipelinePacket:
        """
        Detect folder layout, load data, validate, and return a PipelinePacket.
        """
        self._log(f"[Importer 2.1] Loading data from:  {self.source_path}")

        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Source path not found: {self.source_path}"
            )

        # ── Detect and load ────────────────────────────────────────────────
        if self.source_path.is_file():
            signals, combined = self._load_single_csv(self.source_path)
        else:
            signals, combined = self._load_folder(self.source_path)

        # ── Detect annotation ──────────────────────────────────────────────
        is_annotated = self._detect_annotation(signals, combined)

        # ── Load metadata if available ─────────────────────────────────────
        extra_meta = self._load_metadata()
        # Use user_id from metadata if not overridden
        if "user" in extra_meta:
            uid = extra_meta["user"].get("user_id", self.user_id)
        else:
            uid = extra_meta.get("user_id", self.user_id)

        packet = build_packet_from_dataframes(
            signals      = signals,
            combined     = combined,
            source_type  = SOURCE_IMPORTED,
            is_annotated = is_annotated,
            user_id      = uid,
            session_id   = self.session_id,
            extra_meta   = {
                "source_path": str(self.source_path),
                **extra_meta,
            },
        )

        self._log(f"[Importer 2.1] Import complete.")
        self._log(packet.summary())
        return packet

    # ── Layout detection and loading ─────────────────────────────────────

    def _load_folder(self, folder: Path):
        """Auto-detect folder layout and load accordingly."""
        found_signals  = {}
        found_combined = None

        # Try loading each known signal file
        for sig_name, spec in SIGNAL_FILES.items():
            fpath = folder / spec["file"]
            if fpath.exists():
                df = self._read_csv(fpath)
                if self._validate_signal_df(df, sig_name, spec["value_cols"]):
                    found_signals[sig_name] = df
                    self._log(f"  Loaded {spec['file']}  ({len(df):,} rows)")

        # Try loading combined
        combined_path = folder / COMBINED_FILE
        if combined_path.exists():
            found_combined = self._read_csv(combined_path)
            self._log(f"  Loaded {COMBINED_FILE}  ({len(found_combined):,} rows)")

        if not found_signals and found_combined is None:
            raise ValueError(
                f"No recognisable signal files found in: {folder}\n"
                f"Expected: {[s['file'] for s in SIGNAL_FILES.values()]} "
                f"or {COMBINED_FILE}"
            )

        # If no individual signals but combined exists, split it
        if not found_signals and found_combined is not None:
            found_signals = self._split_combined(found_combined)

        # If no combined but individual signals exist, build combined
        if found_combined is None and found_signals:
            found_combined = self._build_combined_from_signals(found_signals)

        return found_signals, found_combined

    def _load_single_csv(self, fpath: Path):
        """Load a single CSV file as combined signals."""
        df = self._read_csv(fpath)
        self._log(f"  Loaded {fpath.name}  ({len(df):,} rows)")
        signals = self._split_combined(df)
        return signals, df

    def _read_csv(self, fpath: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(fpath)
        except Exception as e:
            raise IOError(f"Failed to read {fpath}: {e}")

    # ── Validation ─────────────────────────────────────────────────────────

    def _validate_signal_df(self, df: pd.DataFrame, name: str,
                             value_cols: list) -> bool:
        if REQUIRED_COL not in df.columns:
            self._log(f"  WARNING: {name}.csv missing '{REQUIRED_COL}' — skipped")
            return False
        missing = [c for c in value_cols if c not in df.columns]
        if missing:
            self._log(f"  WARNING: {name}.csv missing value columns {missing} — skipped")
            return False
        return True

    def _detect_annotation(self, signals: dict, combined: pd.DataFrame) -> bool:
        """Return True if target_label column exists in any loaded data."""
        if "target_label" in combined.columns:
            return True
        for df in signals.values():
            if "target_label" in df.columns:
                return True
        return False

    # ── Build/split helpers ────────────────────────────────────────────────

    def _split_combined(self, combined: pd.DataFrame) -> dict:
        """Extract per-signal DataFrames from a combined signals DataFrame."""
        signals = {}
        ts_col  = combined["timestamp_s"] if "timestamp_s" in combined.columns else None
        ann_cols = [c for c in ANNOTATION_COLS if c in combined.columns]

        col_map = {
            "EDA": ["EDA_uS"],
            "BVP": ["BVP_nT"],
            "IBI": ["IBI_ms"],
            "ST":  ["ST_degC"],
            "ACC": ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"],
        }

        for sig_name, val_cols in col_map.items():
            present = [c for c in val_cols if c in combined.columns]
            if not present:
                continue
            cols = (["timestamp_s"] + present + ann_cols
                    if "timestamp_s" in combined.columns else present + ann_cols)
            sub = combined[[c for c in cols if c in combined.columns]].copy()
            # Drop rows where ALL value columns are NaN (e.g. IBI sparse)
            sub = sub.dropna(subset=present, how="all").reset_index(drop=True)
            if len(sub) > 0:
                signals[sig_name] = sub

        return signals

    def _build_combined_from_signals(self, signals: dict) -> pd.DataFrame:
        """
        Build a combined DataFrame by merging all signals on timestamp_s.
        Uses BVP (highest rate) as the reference grid; others are forward-filled.
        """
        from scipy.interpolate import interp1d

        # Use the longest signal as reference time grid
        ref_sig = max(signals.items(), key=lambda x: len(x[1]))[1]
        t_ref   = ref_sig["timestamp_s"].values

        combined = pd.DataFrame({"timestamp_s": t_ref})

        # Carry annotation columns from BVP if available, else first signal
        ann_source = signals.get("BVP", list(signals.values())[0])
        for col in ANNOTATION_COLS:
            if col in ann_source.columns:
                combined[col] = ann_source[col].values[:len(t_ref)]

        col_map = {
            "EDA": "EDA_uS", "BVP": "BVP_nT", "IBI": "IBI_ms",
            "ST": "ST_degC",
        }
        acc_cols = {"ACC": ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"]}

        for sig_name, col in col_map.items():
            if sig_name not in signals:
                continue
            df  = signals[sig_name]
            ts  = df["timestamp_s"].values
            val = df[col].values
            if len(ts) == len(t_ref):
                combined[col] = val
            else:
                f = interp1d(ts, val, kind="linear",
                             bounds_error=False, fill_value="extrapolate")
                combined[col] = f(t_ref)

        for sig_name, cols in acc_cols.items():
            if sig_name not in signals:
                continue
            df = signals[sig_name]
            ts = df["timestamp_s"].values
            for col in cols:
                if col in df.columns:
                    if len(ts) == len(t_ref):
                        combined[col] = df[col].values
                    else:
                        f = interp1d(ts, df[col].values, kind="linear",
                                     bounds_error=False, fill_value="extrapolate")
                        combined[col] = f(t_ref)

        return combined

    def _load_metadata(self) -> dict:
        """Load metadata.json from source folder if it exists."""
        if self.source_path.is_dir():
            meta_path = self.source_path / METADATA_FILE
            if meta_path.exists():
                with open(meta_path) as fh:
                    return json.load(fh)
        return {}

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def import_data(
    source_path: str | Path,
    user_id:     str  = "imported_user",
    verbose:     bool = True,
) -> PipelinePacket:
    """
    Shorthand: import data from a folder or CSV file.

    Parameters
    ----------
    source_path : path to Module 1A output folder, or any signal CSV
    user_id     : participant label
    verbose     : print progress

    Returns
    -------
    PipelinePacket ready for Module 3 (Preprocessing)
    """
    return DataImporter(source_path=source_path,
                        user_id=user_id,
                        verbose=verbose).run()
