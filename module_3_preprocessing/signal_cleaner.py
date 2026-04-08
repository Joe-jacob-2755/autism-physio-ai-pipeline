"""
=============================================================================
MODULE 3 – DATA PREPROCESSING  |  signal_cleaner.py
=============================================================================
Raw signal cleaning.

Steps per signal channel:
  1. Detect and flag missing (NaN / blank) values
  2. If missing > MISSING_DISCARD_THRESHOLD (70%) → discard channel
  3. If missing ≤ 70% → interpolate using linear (or forward-fill for IBI)
  4. Detect and remove physiological range violations (clipping)
  5. Detect flatline segments (std < threshold)
  6. Return cleaned signal with a quality report

=============================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import (
    MISSING_DISCARD_THRESHOLD, MISSING_FILL_METHOD,
    FLAT_SIGNAL_STD_THRESHOLD, SAMPLING_RATES, SIGNAL_VALUE_COLUMNS,
)

# Physiological ranges for clipping detection
PHYS_RANGES: dict = {
    "EDA_uS": (0.00, 50.0),
    "BVP_nT": (-500, 500.0),
    "IBI_ms": (200.0, 2500.0),
    "ST_degC": (20.0, 42.0),
    "ACC_X_g": (-8.0, 8.0),
    "ACC_Y_g": (-8.0, 8.0),
    "ACC_Z_g": (-8.0, 8.0),
}


@dataclass
class CleaningReport:
    """Quality report for a single signal channel."""
    signal_name: str
    n_samples_original: int
    n_missing: int
    pct_missing: float
    n_out_of_range: int
    n_flatline_samples: int
    discarded: bool
    fill_method: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "signal": self.signal_name,
            "n_samples": self.n_samples_original,
            "n_missing": self.n_missing,
            "pct_missing": round(self.pct_missing * 100, 2),
            "n_out_of_range": self.n_out_of_range,
            "n_flatline_samples": self.n_flatline_samples,
            "discarded": self.discarded,
            "fill_method": self.fill_method,
            "notes": "; ".join(self.notes),
        }


class SignalCleaner:
    """
    Clean raw physiological signal DataFrames from Module 2.

    Parameters
    ----------
    discard_threshold : fraction of missing samples above which channel
                        is discarded (default: 0.70)
    fill_method       : interpolation method for missing < threshold
                        ('linear', 'time', 'nearest', 'ffill')
    verbose           : print cleaning summary
    """

    def __init__(
        self,
        discard_threshold: float = MISSING_DISCARD_THRESHOLD,
        fill_method: str = MISSING_FILL_METHOD,
        verbose: bool = True,
    ):
        self.discard_threshold = discard_threshold
        self.fill_method = fill_method
        self.verbose = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def clean_all(
        self, signals: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], List[CleaningReport]]:
        """
        Clean all signal DataFrames.

        Parameters
        ----------
        signals : {signal_name: DataFrame} — from Module 2 PipelinePacket

        Returns
        -------
        cleaned_signals : dict of cleaned DataFrames (discarded channels removed)
        reports         : list of CleaningReport per channel
        """
        cleaned = {}
        reports = []

        for name, df in signals.items():
            df_clean, report = self.clean_signal(df, name)
            reports.append(report)

            if report.discarded:
                self._log(
                    f"  [Cleaner] ✗ {name}: DISCARDED "
                    f"({report.pct_missing:.1f}% missing > "
                    f"{self.discard_threshold * 100:.0f}% threshold)"
                )
            else:
                self._log(
                    f"  [Cleaner] ✓ {name}: "
                    f"missing={report.pct_missing:.1f}%  "
                    f"out-of-range={report.n_out_of_range}  "
                    f"flatline={report.n_flatline_samples}"
                )
                cleaned[name] = df_clean

        return cleaned, reports

    def clean_signal(
        self, df: pd.DataFrame, signal_name: str
    ) -> Tuple[pd.DataFrame, CleaningReport]:
        """Clean a single signal DataFrame."""
        df_work = df.copy()
        notes = []

        # Identify value columns (not timestamp or annotation cols)
        val_cols = [c for c in df_work.columns
                    if c not in ("timestamp_s", "target_label",
                                 "event_id", "category", "user_id")
                    and df_work[c].dtype in (float, "float64", "float32")]

        if not val_cols:
            return df_work, CleaningReport(
                signal_name=signal_name, n_samples_original=len(df_work),
                n_missing=0, pct_missing=0.0, n_out_of_range=0,
                n_flatline_samples=0, discarded=False,
                fill_method="none", notes=["No value columns found"]
            )

        n_orig = len(df_work)
        total_val = n_orig * len(val_cols)

        # ── Step 1: Count missing ─────────────────────────────────────
        missing_mask = df_work[val_cols].isnull().any(axis=1)
        n_missing = int(missing_mask.sum())
        pct_missing = n_missing / n_orig if n_orig > 0 else 0.0

        # ── Step 2: Discard if > threshold ───────────────────────────
        if pct_missing > self.discard_threshold:
            return df_work, CleaningReport(
                signal_name=signal_name, n_samples_original=n_orig,
                n_missing=n_missing, pct_missing=pct_missing,
                n_out_of_range=0, n_flatline_samples=0,
                discarded=True, fill_method="none",
                notes=[f"Discarded: {pct_missing * 100:.1f}% missing > "
                       f"{self.discard_threshold * 100:.0f}% threshold"]
            )

        # ── Step 3: Mark out-of-range as NaN ─────────────────────────
        n_oor = 0
        for col in val_cols:
            if col in PHYS_RANGES:
                lo, hi = PHYS_RANGES[col]
                oor = (df_work[col] < lo) | (df_work[col] > hi)
                n_oor += int(oor.sum())
                df_work.loc[oor, col] = np.nan
                if oor.sum() > 0:
                    notes.append(
                        f"{col}: {oor.sum()} out-of-range values → NaN"
                    )

        # ── Step 4: Fill missing values ───────────────────────────────
        if n_missing > 0 or n_oor > 0:
            if self.fill_method == "ffill":
                df_work[val_cols] = (
                    df_work[val_cols]
                    .ffill()
                    .bfill()
                )
            else:
                # Linear interpolation on timestamp index if available
                if "timestamp_s" in df_work.columns:
                    df_tmp = df_work.set_index("timestamp_s")
                    df_tmp[val_cols] = (
                        df_tmp[val_cols]
                        .interpolate(method="index")
                        .ffill()
                        .bfill()
                    )
                    df_work = df_tmp.reset_index()
                else:
                    df_work[val_cols] = (
                        df_work[val_cols]
                        .interpolate(method="linear")
                        .ffill()
                        .bfill()
                    )
            notes.append(f"Filled {n_missing} missing via {self.fill_method}")

        # ── Step 5: Detect flatline segments ──────────────────────────
        n_flatline = 0
        for col in val_cols:
            window = max(3, min(32, len(df_work) // 10))
            rolling_std = df_work[col].rolling(window, min_periods=1).std()
            flat = (rolling_std < FLAT_SIGNAL_STD_THRESHOLD).sum()
            n_flatline += int(flat)
            if flat > 0:
                notes.append(f"{col}: {flat} flatline samples detected")

        # ── Step 6: Final NaN check ───────────────────────────────────
        residual_nan = df_work[val_cols].isnull().sum().sum()
        if residual_nan > 0:
            df_work[val_cols] = df_work[val_cols].fillna(
                df_work[val_cols].median()
            )
            notes.append(f"Residual {residual_nan} NaN → median-filled")

        report = CleaningReport(
            signal_name=signal_name,
            n_samples_original=n_orig,
            n_missing=n_missing,
            pct_missing=pct_missing,
            n_out_of_range=n_oor,
            n_flatline_samples=n_flatline,
            discarded=False,
            fill_method=self.fill_method,
            notes=notes,
        )
        return df_work, report

    def _log(self, msg):
        if self.verbose:
            print(msg)
