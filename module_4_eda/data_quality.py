"""
=============================================================================
MODULE 4 - EXPLORATORY DATA ANALYSIS  |  data_quality.py
=============================================================================
Combined multi-user data understanding, quality assessment, distribution
analysis, and outlier detection.

Sections 2-4 of the EDA report:
  Section 2 — Data Understanding (signal overview, target distribution)
  Section 3 — Data Quality (missing, outliers, range, sampling rate)
  Section 4 — Distribution Analysis (normality, skewness, kurtosis)
=============================================================================
"""
from __future__ import annotations

import warnings
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple

from config import (
    VALUE_COLS, ALL_VALUE_COLS, SIGNAL_UNITS, SIGNAL_SAMPLING_RATES,
    SIGNAL_RANGES, META_COLS, LABEL_TO_CATEGORY, ALPHA,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA UNDERSTANDING
# ─────────────────────────────────────────────────────────────────────────────

def signal_overview(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Section 2.1 — Signal overview across all training users.

    Returns one row per signal with: total samples, n_users, duration,
    inferred sampling rate, value columns, and basic summary stats.
    """
    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or len(df) == 0:
            continue

        n_users = df["user_id"].nunique() if "user_id" in df.columns else 1
        row = {
            "signal": sig_name,
            "unit": SIGNAL_UNITS.get(sig_name, "unknown"),
            "expected_fs_hz": SIGNAL_SAMPLING_RATES.get(sig_name),
            "n_rows": len(df),
            "n_users": n_users,
        }

        # Duration and inferred sampling rate
        if "timestamp_s" in df.columns:
            ts = df["timestamp_s"].values
            row["duration_s"] = round(float(ts[-1] - ts[0]), 2)
            if len(ts) > 1:
                dt = np.median(np.diff(ts))
                row["inferred_fs_hz"] = round(1.0 / dt, 1) if dt > 0 else 0
            else:
                row["inferred_fs_hz"] = 0
        else:
            row["duration_s"] = 0
            row["inferred_fs_hz"] = 0

        # Summary stats on each value column
        val_cols = VALUE_COLS.get(sig_name, [])
        for vc in val_cols:
            if vc not in df.columns:
                continue
            vals = df[vc].dropna().astype(float)
            row[f"{vc}_mean"] = round(float(vals.mean()), 4)
            row[f"{vc}_std"] = round(float(vals.std()), 4)
            row[f"{vc}_min"] = round(float(vals.min()), 4)
            row[f"{vc}_max"] = round(float(vals.max()), 4)

        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def target_distribution(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Section 2.2 — Target label distribution across combined data.

    Uses the first signal with target_label to count per-label occurrences.
    Returns: target_label, category, n_samples, pct, n_users.
    """
    # Pick the signal with target_label column
    ref_df = None
    for df in signals.values():
        if df is not None and "target_label" in df.columns:
            ref_df = df
            break
    if ref_df is None or len(ref_df) == 0:
        return pd.DataFrame()

    rows = []
    total = len(ref_df)
    for label, grp in ref_df.groupby("target_label"):
        n = len(grp)
        n_users = grp["user_id"].nunique() if "user_id" in grp.columns else 1
        rows.append({
            "target_label": label,
            "category": LABEL_TO_CATEGORY.get(label, "unknown"),
            "n_samples": n,
            "pct": round(100.0 * n / total, 2),
            "n_users": n_users,
        })
    result = pd.DataFrame(rows).sort_values("n_samples", ascending=False)
    return result.reset_index(drop=True)


def user_label_matrix(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Section 2.2 — User × label presence heatmap data.

    Returns a pivot table: rows=user_id, columns=target_label, values=n_samples.
    """
    ref_df = None
    for df in signals.values():
        if df is not None and "target_label" in df.columns and "user_id" in df.columns:
            ref_df = df
            break
    if ref_df is None:
        return pd.DataFrame()

    pivot = ref_df.groupby(["user_id", "target_label"]).size().unstack(fill_value=0)
    return pivot


def per_user_summary(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Section 2.3 — Per-user data summary (rows per signal per user).
    """
    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or "user_id" not in df.columns:
            continue
        for uid, udf in df.groupby("user_id"):
            row = {"user_id": uid, "signal": sig_name, "n_rows": len(udf)}
            if "timestamp_s" in udf.columns and len(udf) > 1:
                row["duration_s"] = round(
                    float(udf["timestamp_s"].max() - udf["timestamp_s"].min()), 1
                )
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATA QUALITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

def missing_values_report(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Section 3.1 — Missing value analysis per signal × value column.

    Returns: signal, value_col, n_total, n_missing, pct_missing,
             n_users_affected, completeness_score.
    """
    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or len(df) == 0:
            continue
        val_cols = VALUE_COLS.get(sig_name, [])
        for vc in val_cols:
            if vc not in df.columns:
                continue
            n_total = len(df)
            n_missing = int(df[vc].isna().sum())
            pct_missing = round(100.0 * n_missing / max(n_total, 1), 4)
            n_users_affected = 0
            if "user_id" in df.columns and n_missing > 0:
                n_users_affected = int(
                    df[df[vc].isna()]["user_id"].nunique()
                )
            rows.append({
                "signal": sig_name,
                "value_col": vc,
                "n_total": n_total,
                "n_missing": n_missing,
                "pct_missing": pct_missing,
                "n_users_affected": n_users_affected,
                "completeness_score": round(1.0 - pct_missing / 100, 4),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def outlier_detection_iqr(
    signals: Dict[str, pd.DataFrame],
    iqr_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Section 3.2 — IQR-based outlier detection per signal × value column.

    Returns: signal, value_col, n_total, q1, q3, iqr, lower_fence,
             upper_fence, n_outliers, pct_outliers, outlier_min, outlier_max.
    """
    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or len(df) == 0:
            continue
        val_cols = VALUE_COLS.get(sig_name, [])
        for vc in val_cols:
            if vc not in df.columns:
                continue
            vals = df[vc].dropna().astype(float)
            n = len(vals)
            if n < 4:
                continue
            q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            iqr = q3 - q1
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            outlier_mask = (vals < lower) | (vals > upper)
            n_out = int(outlier_mask.sum())
            out_vals = vals[outlier_mask]
            rows.append({
                "signal": sig_name,
                "value_col": vc,
                "n_total": n,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(iqr, 4),
                "lower_fence": round(lower, 4),
                "upper_fence": round(upper, 4),
                "n_outliers": n_out,
                "pct_outliers": round(100.0 * n_out / n, 2),
                "outlier_min": round(float(out_vals.min()), 4) if n_out > 0 else np.nan,
                "outlier_max": round(float(out_vals.max()), 4) if n_out > 0 else np.nan,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def signal_range_check(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Section 3.3 — Signal range consistency (physiological plausibility).

    Checks each value column against known physiological bounds from config.
    Returns: signal, value_col, expected_min, expected_max, actual_min,
             actual_max, n_below, n_above, pct_out_of_range, verdict.
    """
    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or len(df) == 0:
            continue
        val_cols = VALUE_COLS.get(sig_name, [])
        for vc in val_cols:
            if vc not in df.columns or vc not in SIGNAL_RANGES:
                continue
            lo, hi = SIGNAL_RANGES[vc]
            vals = df[vc].dropna().astype(float)
            n = len(vals)
            if n == 0:
                continue
            n_below = int((vals < lo).sum())
            n_above = int((vals > hi).sum())
            pct_oor = round(100.0 * (n_below + n_above) / n, 4)
            verdict = "PASS" if pct_oor < 1.0 else ("WARNING" if pct_oor < 5.0 else "FAIL")
            rows.append({
                "signal": sig_name,
                "value_col": vc,
                "expected_min": lo,
                "expected_max": hi,
                "actual_min": round(float(vals.min()), 4),
                "actual_max": round(float(vals.max()), 4),
                "n_below": n_below,
                "n_above": n_above,
                "n_total": n,
                "pct_out_of_range": pct_oor,
                "verdict": verdict,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def sampling_rate_check(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Section 3.4 — Sampling rate verification per signal.

    Compares inferred sampling rate against expected rates from config.
    Returns: signal, expected_fs, inferred_fs, deviation_pct, verdict.
    """
    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or len(df) < 2 or "timestamp_s" not in df.columns:
            continue
        expected_fs = SIGNAL_SAMPLING_RATES.get(sig_name)
        if expected_fs is None:
            continue  # event-based (IBI)
        dt = np.median(np.diff(df["timestamp_s"].values))
        inferred_fs = 1.0 / dt if dt > 0 else 0
        deviation = abs(inferred_fs - expected_fs) / expected_fs * 100 if expected_fs > 0 else 0
        verdict = "PASS" if deviation < 5 else ("WARNING" if deviation < 15 else "FAIL")
        rows.append({
            "signal": sig_name,
            "expected_fs_hz": expected_fs,
            "inferred_fs_hz": round(inferred_fs, 2),
            "deviation_pct": round(deviation, 2),
            "verdict": verdict,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def normality_tests(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Section 4.1 — Normality testing: Shapiro-Wilk + D'Agostino-Pearson.

    For each value column:
      - Shapiro-Wilk on subsample (max 5000, required for scipy)
      - D'Agostino-Pearson omnibus (more robust for N > 5000)
      - Combined verdict

    Returns: signal, value_col, n, shapiro_W, shapiro_p, dagostino_K2,
             dagostino_p, is_normal_shapiro, is_normal_dagostino, verdict.
    """
    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or len(df) == 0:
            continue
        val_cols = VALUE_COLS.get(sig_name, [])
        for vc in val_cols:
            if vc not in df.columns:
                continue
            vals = df[vc].dropna().astype(float).values
            n = len(vals)
            if n < 8:
                continue

            row = {"signal": sig_name, "value_col": vc, "n": n}

            # Shapiro-Wilk (subsample for large N)
            sub = vals[:5000] if n > 5000 else vals
            try:
                w, p_sw = stats.shapiro(sub)
                row["shapiro_W"] = round(float(w), 6)
                row["shapiro_p"] = float(p_sw)
                row["is_normal_shapiro"] = bool(p_sw > ALPHA)
            except Exception:
                row["shapiro_W"] = np.nan
                row["shapiro_p"] = np.nan
                row["is_normal_shapiro"] = False

            # D'Agostino-Pearson (needs N >= 20)
            if n >= 20:
                try:
                    k2, p_da = stats.normaltest(vals)
                    row["dagostino_K2"] = round(float(k2), 4)
                    row["dagostino_p"] = float(p_da)
                    row["is_normal_dagostino"] = bool(p_da > ALPHA)
                except Exception:
                    row["dagostino_K2"] = np.nan
                    row["dagostino_p"] = np.nan
                    row["is_normal_dagostino"] = False
            else:
                row["dagostino_K2"] = np.nan
                row["dagostino_p"] = np.nan
                row["is_normal_dagostino"] = False

            # Combined verdict
            sw_normal = row.get("is_normal_shapiro", False)
            da_normal = row.get("is_normal_dagostino", False)
            if sw_normal and da_normal:
                row["verdict"] = "Normal"
            elif sw_normal or da_normal:
                row["verdict"] = "Borderline"
            else:
                row["verdict"] = "Non-normal"

            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def skewness_kurtosis(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Section 4.2 — Skewness and kurtosis per signal value column.

    Returns: signal, value_col, n, skewness, kurtosis,
             shape_classification, recommended_transform.
    """
    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or len(df) == 0:
            continue
        val_cols = VALUE_COLS.get(sig_name, [])
        for vc in val_cols:
            if vc not in df.columns:
                continue
            vals = df[vc].dropna().astype(float).values
            if len(vals) < 8:
                continue

            skew = float(stats.skew(vals))
            kurt = float(stats.kurtosis(vals))

            # Shape classification
            if abs(skew) < 0.5:
                shape = "symmetric"
            elif skew > 0:
                shape = "right-skewed"
            else:
                shape = "left-skewed"
            if kurt > 3:
                shape += " (leptokurtic)"
            elif kurt < -1:
                shape += " (platykurtic)"
            else:
                shape += " (mesokurtic)"

            # Recommended transformation
            if abs(skew) < 0.5:
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
                "value_col": vc,
                "n": len(vals),
                "skewness": round(skew, 4),
                "kurtosis": round(kurt, 4),
                "shape_classification": shape,
                "recommended_transform": transform,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: RUN ALL QUALITY ASSESSMENTS
# ─────────────────────────────────────────────────────────────────────────────

def run_all_quality(signals: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Run all data understanding and quality checks.

    Returns dict of DataFrames keyed by analysis name.
    """
    return {
        "signal_overview": signal_overview(signals),
        "target_distribution": target_distribution(signals),
        "user_label_matrix": user_label_matrix(signals),
        "per_user_summary": per_user_summary(signals),
        "missing_values": missing_values_report(signals),
        "outlier_iqr": outlier_detection_iqr(signals),
        "signal_range": signal_range_check(signals),
        "sampling_rate": sampling_rate_check(signals),
        "normality": normality_tests(signals),
        "skewness_kurtosis": skewness_kurtosis(signals),
    }
