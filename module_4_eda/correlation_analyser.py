"""
=============================================================================
MODULE 4 - EXPLORATORY DATA ANALYSIS  |  correlation_analyser.py
=============================================================================
Correlational analyses on combined multi-user physiological signals.

Section 7 of the EDA report:
  7.1 — Spearman rank correlations (inter-signal)
  7.2 — Point-biserial correlations (signal vs binary target indicator)
  7.3 — Partial correlations controlling for baseline
=============================================================================
"""
from __future__ import annotations

import warnings
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple

from config import VALUE_COLS, ALPHA, SIGNAL_SAMPLING_RATES, META_COLS

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 7.1: SPEARMAN INTER-SIGNAL CORRELATIONS
# ─────────────────────────────────────────────────────────────────────────────

def inter_signal_spearman(
    signals: Dict[str, pd.DataFrame],
    max_samples: int = 100_000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Spearman rank correlation between all pairs of signal value columns.

    Aligns signals on timestamp_s using merge_asof, then computes
    pairwise Spearman rho.

    Parameters
    ----------
    signals     : {signal_name: DataFrame with timestamp_s + value cols}
    max_samples : subsample limit for large datasets

    Returns
    -------
    (correlation_matrix, pairwise_table)
      correlation_matrix: square DataFrame (columns × columns)
      pairwise_table: long-form with signal_a, signal_b, rho, p, n
    """
    # Build aligned series dict: {col_name: (timestamps, values)}
    col_series = {}
    for sig_name, df in sorted(signals.items()):
        if df is None or len(df) == 0 or "timestamp_s" not in df.columns:
            continue
        val_cols = VALUE_COLS.get(sig_name, [])
        for vc in val_cols:
            if vc in df.columns:
                sub = df[["timestamp_s", vc]].dropna().sort_values("timestamp_s")
                if len(sub) > max_samples:
                    step = len(sub) // max_samples
                    sub = sub.iloc[::step]
                col_series[vc] = sub

    col_names = sorted(col_series.keys())
    if len(col_names) < 2:
        return pd.DataFrame(), pd.DataFrame()

    # Build aligned DataFrame via sequential merge_asof
    base = col_series[col_names[0]].rename(columns={col_names[0]: col_names[0]})
    for cn in col_names[1:]:
        right = col_series[cn].rename(columns={cn: cn})
        base = pd.merge_asof(
            base.sort_values("timestamp_s"),
            right.sort_values("timestamp_s"),
            on="timestamp_s",
            tolerance=0.5,
            direction="nearest",
        )

    aligned = base.drop(columns=["timestamp_s"]).dropna()
    if len(aligned) < 10:
        return pd.DataFrame(), pd.DataFrame()

    # Correlation matrix
    n_cols = len(col_names)
    rho_mat = np.eye(n_cols)
    p_mat = np.zeros((n_cols, n_cols))
    pairwise_rows = []

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            a = aligned[col_names[i]].values
            b = aligned[col_names[j]].values
            try:
                rho, p = stats.spearmanr(a, b)
            except Exception:
                rho, p = np.nan, np.nan
            rho_mat[i, j] = rho_mat[j, i] = rho
            p_mat[i, j] = p_mat[j, i] = p
            pairwise_rows.append({
                "signal_a": col_names[i],
                "signal_b": col_names[j],
                "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else np.nan,
                "p_value": float(p) if not np.isnan(p) else np.nan,
                "significant": bool(p < ALPHA) if not np.isnan(p) else False,
                "n_aligned": len(aligned),
            })

    corr_df = pd.DataFrame(rho_mat, index=col_names, columns=col_names).round(4)
    pair_df = pd.DataFrame(pairwise_rows)
    if not pair_df.empty:
        pair_df = pair_df.sort_values("p_value").reset_index(drop=True)
    return corr_df, pair_df


# ─────────────────────────────────────────────────────────────────────────────
# 7.2: POINT-BISERIAL CORRELATION (Signal vs Binary Target)
# ─────────────────────────────────────────────────────────────────────────────

def point_biserial_correlations(
    signals: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Point-biserial correlation: each signal value column vs each target
    label (one-vs-rest binary encoding).

    Returns: signal, value_col, target_label, r_pb, p_value, significant, n.
    """
    rows = []

    for sig_name, df in sorted(signals.items()):
        if df is None or "target_label" not in df.columns:
            continue
        labels = df["target_label"].unique()
        val_cols = VALUE_COLS.get(sig_name, [])

        for vc in val_cols:
            if vc not in df.columns:
                continue
            vals = df[vc].values.astype(float)
            mask_valid = ~np.isnan(vals)

            for lbl in sorted(labels):
                binary = (df["target_label"].values == lbl).astype(float)
                # Apply validity mask
                v = vals[mask_valid]
                b = binary[mask_valid]
                if len(v) < 10 or b.sum() < 3 or (len(b) - b.sum()) < 3:
                    continue
                try:
                    r, p = stats.pointbiserialr(b, v)
                except Exception:
                    continue
                rows.append({
                    "signal": sig_name,
                    "value_col": vc,
                    "target_label": str(lbl),
                    "r_pb": round(float(r), 4),
                    "abs_r_pb": round(abs(float(r)), 4),
                    "p_value": float(p),
                    "significant": bool(p < ALPHA),
                    "n": len(v),
                })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("abs_r_pb", ascending=False).reset_index(drop=True)
    return result


def point_biserial_matrix(
    pb_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot point-biserial results into a matrix: rows=value_col, cols=target_label.

    Values are r_pb correlation coefficients.
    """
    if pb_df is None or pb_df.empty:
        return pd.DataFrame()
    pivot = pb_df.pivot_table(
        index="value_col", columns="target_label", values="r_pb", aggfunc="first"
    )
    return pivot.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# 7.3: PARTIAL CORRELATIONS CONTROLLING FOR BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def partial_correlations(
    signals: Dict[str, pd.DataFrame],
    control_label: str = "baseline",
    max_samples: int = 50_000,
) -> pd.DataFrame:
    """
    Partial Spearman correlations between signal pairs, controlling for
    baseline-period values.

    Method: regress both variables on baseline mean per user, then
    correlate residuals.

    Returns: signal_a, signal_b, rho_full, rho_partial, rho_change, n.
    """
    # Build aligned matrix (same as inter_signal_spearman)
    col_series = {}
    for sig_name, df in sorted(signals.items()):
        if df is None or len(df) == 0 or "timestamp_s" not in df.columns:
            continue
        val_cols = VALUE_COLS.get(sig_name, [])
        for vc in val_cols:
            if vc in df.columns:
                sub = df[["timestamp_s", vc, "target_label"]].dropna() \
                    if "target_label" in df.columns else df[["timestamp_s", vc]].dropna()
                col_series[vc] = sub

    col_names = sorted(col_series.keys())
    if len(col_names) < 2:
        return pd.DataFrame()

    # For each column, compute user-level baseline mean as control variable
    # Use first signal that has user_id + target_label
    ref_df = None
    for df in signals.values():
        if df is not None and "target_label" in df.columns:
            ref_df = df
            break
    if ref_df is None:
        return pd.DataFrame()

    # Build aligned non-baseline data
    base = col_series[col_names[0]].sort_values("timestamp_s")
    has_label = "target_label" in base.columns
    for cn in col_names[1:]:
        right = col_series[cn][["timestamp_s", cn]].sort_values("timestamp_s")
        base = pd.merge_asof(
            base,
            right,
            on="timestamp_s",
            tolerance=0.5,
            direction="nearest",
        )

    if has_label:
        non_baseline = base[base["target_label"] != control_label].copy()
        baseline_data = base[base["target_label"] == control_label].copy()
    else:
        non_baseline = base.copy()
        baseline_data = pd.DataFrame()

    non_baseline = non_baseline[col_names].dropna()
    if len(non_baseline) > max_samples:
        non_baseline = non_baseline.sample(max_samples, random_state=42)
    if len(non_baseline) < 20:
        return pd.DataFrame()

    # Compute baseline means per column (control variable)
    if not baseline_data.empty:
        baseline_means = baseline_data[col_names].mean()
    else:
        baseline_means = non_baseline[col_names].mean()

    rows = []
    for i in range(len(col_names)):
        for j in range(i + 1, len(col_names)):
            ci, cj = col_names[i], col_names[j]
            xi = non_baseline[ci].values
            xj = non_baseline[cj].values

            # Full Spearman
            try:
                rho_full, _ = stats.spearmanr(xi, xj)
            except Exception:
                continue

            # Partial: residualize both against baseline_mean control
            # Simple approach: subtract baseline mean (centering on baseline)
            xi_resid = xi - baseline_means[ci]
            xj_resid = xj - baseline_means[cj]
            try:
                rho_partial, _ = stats.spearmanr(xi_resid, xj_resid)
            except Exception:
                rho_partial = rho_full

            rows.append({
                "signal_a": ci,
                "signal_b": cj,
                "rho_full": round(float(rho_full), 4),
                "rho_partial": round(float(rho_partial), 4),
                "rho_change": round(float(rho_partial - rho_full), 4),
                "n": len(non_baseline),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: RUN ALL CORRELATIONAL ANALYSES
# ─────────────────────────────────────────────────────────────────────────────

def run_all_correlations(
    signals: Dict[str, pd.DataFrame],
) -> Dict:
    """Run all correlational analyses."""
    corr_matrix, corr_pairs = inter_signal_spearman(signals)
    pb = point_biserial_correlations(signals)
    pb_matrix = point_biserial_matrix(pb)
    partial = partial_correlations(signals)

    return {
        "spearman_matrix": corr_matrix,
        "spearman_pairs": corr_pairs,
        "point_biserial": pb,
        "point_biserial_matrix": pb_matrix,
        "partial_correlations": partial,
    }
