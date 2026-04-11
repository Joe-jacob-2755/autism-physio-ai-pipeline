"""
=============================================================================
MODULE 5 – DATA PREPROCESSING  |  normaliser.py
=============================================================================
Feature standardisation and fusion.

Why RobustScaler?
-----------------
Physiological signals from autistic children require robust normalisation:

1. OUTLIER RESISTANCE
   Physiological signals contain frequent legitimate outliers — SCR peaks
   in EDA (3–5× baseline), HR surges in Fear (+36 bpm), ACC impulses from
   startle responses. StandardScaler uses mean ± std, which is pulled toward
   these extremes, incorrectly reducing the apparent discriminative power of
   surrounding samples. RobustScaler uses median and IQR (Q3−Q1), both
   resistant to outliers, preserving the relative structure of the data.

2. INTER-SUBJECT VARIABILITY
   Children's physiological baselines vary widely (EDA: 0.3–18 µS across
   users; HR: 58–115 bpm). StandardScaler normalises across the full
   dataset, which conflates individual differences with emotion changes.
   RobustScaler centres each feature on the population median, separating
   within-subject emotional variation from between-subject baseline shifts.

3. NON-GAUSSIAN DISTRIBUTIONS
   Many physiological features (SCR rate, SVM, spectral powers) follow
   log-normal or heavy-tailed distributions. MinMaxScaler and StandardScaler
   assume approximately Gaussian distributions; RobustScaler makes no such
   assumption and handles heavy tails gracefully.

4. PRESERVING OUTLIER SIGNAL
   Unlike MinMaxScaler, RobustScaler does NOT clip outlier values to [0,1].
   This means extreme physiological responses (high Fear EDA, Anger ACC)
   remain visible beyond the ±1 range — important for models learning to
   detect intense emotional states.

5. CONSISTENCY WITH CLINICAL PRACTICE
   HRV analysis standards (Task Force 1996) and affective computing
   literature recommend IQR-based normalisation for physiological time
   series with known outlier contamination.

=============================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from config import (
    SCALER_TYPE, DEMOGRAPHIC_ENCODINGS,
)

# Columns that should never be scaled
META_COLS = {"session_id", "user_id", "window_start_s", "window_end_s",
             "target_label", "event_id", "category",
             # Demographics — encoded as integers but not scaled here
             "age", "gender", "gender_enc", "ethnicity", "ethnicity_enc",
             "autism_severity", "autism_severity_enc",
             "verbal_status", "verbal_status_enc",
             "comorbidity", "comorbidity_enc"}


class FeatureNormaliser:
    """
    Standardise feature DataFrames using RobustScaler.

    Supports fitting on training data and transforming new data (test/live),
    enabling consistent normalisation across pipeline stages.

    Parameters
    ----------
    scaler_type : 'robust' (default) | 'standard' | 'minmax'
    verbose     : print scaling statistics
    """

    def __init__(self, scaler_type: str = SCALER_TYPE, verbose: bool = True):
        self.scaler_type = scaler_type
        self.verbose = verbose
        self._scalers: Dict[str, object] = {}   # one scaler per signal
        self._feature_cols: Dict[str, List[str]] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def fit_transform(
        self, feature_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fit scalers on all signal feature DataFrames and return scaled copies.

        Parameters
        ----------
        feature_dfs : {signal_name: feature DataFrame}

        Returns
        -------
        scaled_dfs : same structure with numeric features scaled
        """
        scaled = {}
        for sig_name, df in feature_dfs.items():
            scaler, feat_cols, df_scaled = self._fit_transform_one(sig_name, df)
            self._scalers[sig_name] = scaler
            self._feature_cols[sig_name] = feat_cols
            scaled[sig_name] = df_scaled
            self._log(f"  [Normaliser] {sig_name}: scaled {len(feat_cols)} features "
                      f"({self.scaler_type})")
        return scaled

    def transform(
        self, feature_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Transform new data using fitted scalers (call fit_transform first)."""
        scaled = {}
        for sig_name, df in feature_dfs.items():
            if sig_name not in self._scalers:
                raise ValueError(
                    f"No fitted scaler for '{sig_name}'. Call fit_transform first."
                )
            feat_cols = self._feature_cols[sig_name]
            df_out = df.copy()
            present = [c for c in feat_cols if c in df_out.columns]
            if present:
                df_out[present] = self._scalers[sig_name].transform(
                    df_out[present].values
                )
            scaled[sig_name] = df_out
        return scaled

    def save(self, path: str | Path):
        """Persist fitted scalers to disk."""
        with open(path, "wb") as f:
            pickle.dump({"scalers": self._scalers,
                         "feature_cols": self._feature_cols,
                         "scaler_type": self.scaler_type}, f)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormaliser":
        """Load fitted scalers from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        n = cls(scaler_type=obj["scaler_type"])
        n._scalers = obj["scalers"]
        n._feature_cols = obj["feature_cols"]
        return n

    # ── Internal ────────────────────────────────────────────────────────────

    def _fit_transform_one(self, sig_name, df):
        feat_cols = [c for c in df.columns
                     if c not in META_COLS
                     and pd.api.types.is_numeric_dtype(df[c])]

        df_out = df.copy()

        if not feat_cols:
            return None, [], df_out

        X = df_out[feat_cols].values.astype(float)
        # Replace inf with nan, then fill nan with median
        X[~np.isfinite(X)] = np.nan
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[np.isnan(X[:, j]), j] = col_medians[j]

        scaler = self._make_scaler()
        X_scaled = scaler.fit_transform(X)
        df_out[feat_cols] = X_scaled

        return scaler, feat_cols, df_out

    def _make_scaler(self):
        if self.scaler_type == "robust":
            return RobustScaler(quantile_range=(25.0, 75.0))
        elif self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

    def _log(self, msg):
        if self.verbose:
            print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# DEMOGRAPHIC FEATURE ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class DemographicEncoder:
    """
    Encode participant demographic features for inclusion in feature matrices.

    Categorical demographics are ordinally encoded using clinically meaningful
    ordering (e.g. autism severity: Low=1, Medium=2, Severe=3).

    Continuous demographics (age) are kept as-is.
    """

    @staticmethod
    def encode(demographics: dict) -> dict:
        """
        Encode a single participant's demographics dict.

        Parameters
        ----------
        demographics : dict with keys: age, gender, ethnicity,
                       autism_severity, verbal_status, comorbidity

        Returns
        -------
        dict with original + encoded columns
        """
        enc = dict(demographics)
        for col, mapping in DEMOGRAPHIC_ENCODINGS.items():
            raw_val = demographics.get(col, None)
            enc[f"{col}_enc"] = mapping.get(raw_val, -1)
        return enc

    @staticmethod
    def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Encode demographic columns in a DataFrame."""
        out = df.copy()
        for col, mapping in DEMOGRAPHIC_ENCODINGS.items():
            if col in out.columns:
                out[f"{col}_enc"] = out[col].map(mapping).fillna(-1).astype(int)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE FUSER
# ─────────────────────────────────────────────────────────────────────────────

class FeatureFuser:
    """
    Fuse per-signal feature DataFrames into combined datasets.

    Produces:
      1. Per-signal feature CSVs (for unimodal models)
      2. Combined feature CSV (for early fusion multimodal model)

    Fusion strategy: align all signal features by window_start_s within the
    same session, merge on (session_id, user_id, window_start_s, target_label).
    """

    def fuse(
        self,
        feature_dfs: Dict[str, pd.DataFrame],
        demographics: Optional[dict] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Fuse all signal feature DataFrames.

        Parameters
        ----------
        feature_dfs  : {signal_name: feature DataFrame}
        demographics : optional demographics dict to append to all rows

        Returns
        -------
        per_signal_dfs : {signal_name: DataFrame with demographics if provided}
        combined_df    : all signals merged on window alignment
        """
        # Add demographics to each per-signal DataFrame
        per_signal = {}
        for sig_name, df in feature_dfs.items():
            if demographics:
                enc = DemographicEncoder.encode(demographics)
                for k, v in enc.items():
                    df = df.copy()
                    df[k] = v
            per_signal[sig_name] = df

        # Build combined: merge all signals on window alignment
        combined = self._merge_signals(feature_dfs, demographics)

        return per_signal, combined

    def _merge_signals(
        self,
        feature_dfs: Dict[str, pd.DataFrame],
        demographics: Optional[dict],
    ) -> pd.DataFrame:
        """Merge feature DataFrames by approximate window alignment."""
        if not feature_dfs:
            return pd.DataFrame()

        # Start with largest feature set
        ordered = sorted(feature_dfs.items(), key=lambda x: len(x[1]),
                         reverse=True)
        base_name, base_df = ordered[0]
        merged = base_df.copy()

        # Drop non-feature duplicate columns before merge
        for sig_name, df in ordered[1:]:
            # Get feature-only columns (no meta cols)
            feat_cols = [c for c in df.columns
                         if c not in META_COLS
                         and c not in merged.columns]
            keep = ["session_id", "user_id", "window_start_s"] + feat_cols
            keep = [c for c in keep if c in df.columns]
            sub = df[keep].copy()

            merged = pd.merge_asof(
                merged.sort_values("window_start_s"),
                sub.sort_values("window_start_s"),
                on="window_start_s",
                by=["session_id", "user_id"],
                direction="nearest",
                tolerance=self.window_s_tolerance,
            )

        if demographics:
            enc = DemographicEncoder.encode(demographics)
            for k, v in enc.items():
                merged[k] = v

        return merged

    window_s_tolerance: float = 30.0  # seconds tolerance for window alignment
