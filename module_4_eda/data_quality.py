"""
=============================================================================
MODULE 4 – EXPLORATORY DATA ANALYSIS  |  data_quality.py
=============================================================================
IBM EDA Best Practices — Data Understanding & Quality Assessment.

Implements IBM's recommended EDA methodology phases:
  Phase 1: Data Understanding — shape, types, summary statistics
  Phase 2: Data Quality — missing values, outliers, duplicates, consistency
  Phase 3: Distribution Analysis — skewness, kurtosis, normality tests

These assessments inform downstream preprocessing (M5) and feature
engineering (M6) decisions.
=============================================================================
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple

from config import VALUE_COLS, SIGNAL_UNITS

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: DATA UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────

def data_understanding(
    signals: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    IBM EDA Phase 1 — Understand the data.

    For each signal, reports:
      - Number of rows (samples) and columns
      - Sampling rate (inferred from timestamps)
      - Duration in seconds
      - Data types
      - Basic summary statistics

    Returns
    -------
    DataFrame with one row per signal, columns:
      signal, n_rows, n_cols, duration_s, sampling_rate_hz,
      value_col, mean, std, min, q25, median, q75, max
    """
    rows = []
    for sig_name, df in signals.items():
        if df is None or len(df) == 0:
            continue

        row = {
            "signal": sig_name,
            "n_rows": len(df),
            "n_cols": df.shape[1],
            "unit": SIGNAL_UNITS.get(sig_name, "unknown"),
        }

        # Infer duration and sampling rate from timestamp_s
        if "timestamp_s" in df.columns:
            ts = df["timestamp_s"].values
            row["duration_s"] = round(float(ts[-1] - ts[0]), 2)
            if len(ts) > 1:
                dt = np.median(np.diff(ts))
                row["sampling_rate_hz"] = round(1.0 / dt, 1) if dt > 0 else 0
            else:
                row["sampling_rate_hz"] = 0
        else:
            row["duration_s"] = 0
            row["sampling_rate_hz"] = 0

        # Summary statistics on the primary value column
        val_cols = VALUE_COLS.get(sig_name, [])
        if not val_cols:
            # ACC has multiple columns; find any numeric column
            val_cols = [c for c in df.columns
                        if c not in ("timestamp_s", "target_label", "event_id",
                                     "category")
                        and pd.api.types.is_numeric_dtype(df[c])]
        val_col = val_cols[0] if val_cols else None
        row["value_col"] = val_col

        if val_col and val_col in df.columns:
            vals = df[val_col].dropna().astype(float)
            row["mean"] = round(float(vals.mean()), 4)
            row["std"] = round(float(vals.std()), 4)
            row["min"] = round(float(vals.min()), 4)
            row["q25"] = round(float(np.percentile(vals, 25)), 4)
            row["median"] = round(float(np.median(vals)), 4)
            row["q75"] = round(float(np.percentile(vals, 75)), 4)
            row["max"] = round(float(vals.max()), 4)

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: DATA QUALITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

def data_quality_report(
    signals: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    IBM EDA Phase 2 — Assess data quality.

    For each signal, reports:
      - Missing value count and percentage
      - Duplicate row count and percentage
      - Out-of-range value count (physiological bounds)
      - Flatline segments (constant value runs)
      - Completeness score (0-1)

    Returns
    -------
    DataFrame with one row per signal, columns:
      signal, value_col, n_total, n_missing, pct_missing,
      n_duplicates, pct_duplicates, n_out_of_range, pct_out_of_range,
      n_flatline_segments, max_flatline_length, completeness_score,
      quality_grade
    """
    # Physiological bounds for out-of-range detection
    BOUNDS = {
        "EDA_uS": (0.01, 30.0),
        "BVP_nT": (-300.0, 300.0),
        "IBI_ms": (300.0, 1500.0),
        "ST_degC": (25.0, 40.0),
        "ACC_X_g": (-4.0, 4.0),
        "ACC_Y_g": (-4.0, 4.0),
        "ACC_Z_g": (-4.0, 4.0),
    }

    rows = []
    for sig_name, df in signals.items():
        if df is None or len(df) == 0:
            continue

        val_cols = VALUE_COLS.get(sig_name, [])
        if not val_cols:
            val_cols = [c for c in df.columns
                        if c not in ("timestamp_s", "target_label", "event_id",
                                     "category")
                        and pd.api.types.is_numeric_dtype(df[c])]

        for val_col in val_cols:
            if val_col not in df.columns:
                continue
            n_total = len(df)
            series = df[val_col]

            # Missing values
            n_missing = int(series.isna().sum())
            pct_missing = round(100.0 * n_missing / max(n_total, 1), 2)

            # Duplicates (consecutive identical values)
            n_duplicates = int((series.diff() == 0).sum())
            pct_duplicates = round(100.0 * n_duplicates / max(n_total, 1), 2)

            # Out-of-range values
            n_oor = 0
            lo, hi = BOUNDS.get(val_col, (None, None))
            if lo is not None and hi is not None:
                valid = series.dropna()
                n_oor = int(((valid < lo) | (valid > hi)).sum())
            pct_oor = round(100.0 * n_oor / max(n_total, 1), 2)

            # Flatline detection (runs of identical values)
            vals = series.dropna().values
            flatline_segs = 0
            max_flat_len = 0
            if len(vals) > 1:
                diffs = np.diff(vals)
                is_flat = np.abs(diffs) < 1e-8
                # Count transitions from non-flat to flat
                transitions = np.diff(is_flat.astype(int))
                flatline_segs = int((transitions == 1).sum())
                if is_flat[0]:
                    flatline_segs += 1
                # Longest flatline run
                if flatline_segs > 0:
                    runs = []
                    run_len = 1 if is_flat[0] else 0
                    for f in is_flat[1:]:
                        if f:
                            run_len += 1
                        else:
                            if run_len > 0:
                                runs.append(run_len)
                            run_len = 0
                    if run_len > 0:
                        runs.append(run_len)
                    max_flat_len = max(runs) if runs else 0

            # Completeness score (1.0 = perfect)
            completeness = round(1.0 - (n_missing / max(n_total, 1)), 4)

            # Quality grade
            if pct_missing > 30 or pct_oor > 10:
                grade = "POOR"
            elif pct_missing > 10 or pct_oor > 5:
                grade = "FAIR"
            elif pct_missing > 2 or pct_oor > 1:
                grade = "GOOD"
            else:
                grade = "EXCELLENT"

            rows.append({
                "signal": sig_name,
                "value_col": val_col,
                "n_total": n_total,
                "n_missing": n_missing,
                "pct_missing": pct_missing,
                "n_duplicates": n_duplicates,
                "pct_duplicates": pct_duplicates,
                "n_out_of_range": n_oor,
                "pct_out_of_range": pct_oor,
                "n_flatline_segments": flatline_segs,
                "max_flatline_length": max_flat_len,
                "completeness_score": completeness,
                "quality_grade": grade,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def distribution_analysis(
    signals: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    IBM EDA Phase 3 — Analyse distributions.

    For each signal's value column, reports:
      - Skewness and kurtosis
      - Shapiro-Wilk normality test (on a subsample if n > 5000)
      - Distribution shape classification
      - Recommended transformation (log, sqrt, none)

    Returns
    -------
    DataFrame with columns:
      signal, value_col, skewness, kurtosis, shapiro_W, shapiro_p,
      is_normal, distribution_shape, recommended_transform
    """
    rows = []
    for sig_name, df in signals.items():
        if df is None or len(df) == 0:
            continue

        val_cols = VALUE_COLS.get(sig_name, [])
        if not val_cols:
            val_cols = [c for c in df.columns
                        if c not in ("timestamp_s", "target_label", "event_id",
                                     "category")
                        and pd.api.types.is_numeric_dtype(df[c])]

        for val_col in val_cols:
            if val_col not in df.columns:
                continue
            vals = df[val_col].dropna().astype(float).values
            if len(vals) < 8:
                continue

            skew = float(stats.skew(vals))
            kurt = float(stats.kurtosis(vals))

            # Shapiro-Wilk on subsample (max 5000 for performance)
            sub = vals[:5000] if len(vals) > 5000 else vals
            try:
                w_stat, p_val = stats.shapiro(sub)
            except Exception:
                w_stat, p_val = np.nan, np.nan
            is_normal = bool(p_val > 0.05) if not np.isnan(p_val) else False

            # Distribution shape classification
            if abs(skew) < 0.5:
                shape = "symmetric"
            elif skew > 0.5:
                shape = "right-skewed"
            else:
                shape = "left-skewed"
            if kurt > 3:
                shape += " (leptokurtic)"
            elif kurt < -1:
                shape += " (platykurtic)"

            # Recommended transformation
            if is_normal:
                transform = "none"
            elif skew > 1.0 and np.all(vals > 0):
                transform = "log"
            elif skew > 0.5 and np.all(vals >= 0):
                transform = "sqrt"
            elif abs(skew) > 1.0:
                transform = "box-cox" if np.all(vals > 0) else "yeo-johnson"
            else:
                transform = "none"

            rows.append({
                "signal": sig_name,
                "value_col": val_col,
                "skewness": round(skew, 4),
                "kurtosis": round(kurt, 4),
                "shapiro_W": round(float(w_stat), 4) if not np.isnan(w_stat) else np.nan,
                "shapiro_p": round(float(p_val), 6) if not np.isnan(p_val) else np.nan,
                "is_normal": is_normal,
                "distribution_shape": shape,
                "recommended_transform": transform,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def outlier_detection(
    signals: Dict[str, pd.DataFrame],
    method: str = "iqr",
    iqr_factor: float = 1.5,
) -> pd.DataFrame:
    """
    IBM EDA — Outlier detection using IQR or z-score method.

    Parameters
    ----------
    signals    : {signal_name: DataFrame}
    method     : 'iqr' or 'zscore'
    iqr_factor : multiplier for IQR bounds (default 1.5)

    Returns
    -------
    DataFrame with columns:
      signal, value_col, method, n_total, n_outliers, pct_outliers,
      lower_bound, upper_bound, outlier_min, outlier_max
    """
    rows = []
    for sig_name, df in signals.items():
        if df is None or len(df) == 0:
            continue

        val_cols = VALUE_COLS.get(sig_name, [])
        if not val_cols:
            val_cols = [c for c in df.columns
                        if c not in ("timestamp_s", "target_label", "event_id",
                                     "category")
                        and pd.api.types.is_numeric_dtype(df[c])]

        for val_col in val_cols:
            if val_col not in df.columns:
                continue
            vals = df[val_col].dropna().astype(float)
            n = len(vals)
            if n < 4:
                continue

            if method == "iqr":
                q1, q3 = np.percentile(vals, [25, 75])
                iqr = q3 - q1
                lower = q1 - iqr_factor * iqr
                upper = q3 + iqr_factor * iqr
            else:
                mu, sigma = vals.mean(), vals.std()
                lower = mu - 3 * sigma
                upper = mu + 3 * sigma

            outlier_mask = (vals < lower) | (vals > upper)
            n_out = int(outlier_mask.sum())
            outlier_vals = vals[outlier_mask]

            rows.append({
                "signal": sig_name,
                "value_col": val_col,
                "method": method,
                "n_total": n,
                "n_outliers": n_out,
                "pct_outliers": round(100.0 * n_out / n, 2),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
                "outlier_min": round(float(outlier_vals.min()), 4) if n_out > 0 else np.nan,
                "outlier_max": round(float(outlier_vals.max()), 4) if n_out > 0 else np.nan,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# INTER-SIGNAL CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def inter_signal_correlation(
    signals: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    IBM EDA — Bivariate analysis: correlations between signals.

    Computes Spearman rank correlation between all pairs of signals
    at shared timestamps (nearest-neighbour join).

    Returns
    -------
    DataFrame with columns:
      signal_a, signal_b, spearman_rho, p_value, n_shared
    """
    # Build a common time-aligned DataFrame
    sig_series = {}
    for sig_name, df in signals.items():
        if df is None or len(df) == 0 or "timestamp_s" not in df.columns:
            continue
        val_cols = VALUE_COLS.get(sig_name, [])
        if not val_cols:
            val_cols = [c for c in df.columns
                        if c not in ("timestamp_s", "target_label", "event_id",
                                     "category")
                        and pd.api.types.is_numeric_dtype(df[c])]
        if val_cols and val_cols[0] in df.columns:
            sub = df[["timestamp_s", val_cols[0]]].dropna()
            sub = sub.set_index("timestamp_s")
            sig_series[sig_name] = sub.iloc[:, 0]

    names = sorted(sig_series.keys())
    rows = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            # Align on nearest timestamp
            sa = sig_series[a]
            sb = sig_series[b]
            merged = pd.merge_asof(
                sa.reset_index().sort_values("timestamp_s"),
                sb.reset_index().sort_values("timestamp_s"),
                on="timestamp_s",
                tolerance=0.5,
                direction="nearest",
            ).dropna()
            n_shared = len(merged)
            if n_shared < 10:
                continue
            rho, pval = stats.spearmanr(merged.iloc[:, 1], merged.iloc[:, 2])
            rows.append({
                "signal_a": a,
                "signal_b": b,
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(pval), 6),
                "n_shared": n_shared,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()
