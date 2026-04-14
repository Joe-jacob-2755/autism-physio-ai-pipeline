"""
=============================================================================
MODULE 4 - EXPLORATORY DATA ANALYSIS  |  statistical_analyser.py
=============================================================================
Descriptive and inferential statistics on combined multi-user raw signals.

Sections 5-6 of the EDA report:
  Section 5 — Univariate (descriptive by target, category, demographic)
  Section 6 — Bivariate (Kruskal-Wallis, Mann-Whitney U, demographic KW)
=============================================================================
"""
from __future__ import annotations

import warnings
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from itertools import combinations

from config import (
    VALUE_COLS, ALL_VALUE_COLS, META_COLS, ALPHA, MIN_SAMPLES_PER_GROUP,
    EFFECT_SIZE_THRESHOLDS, LABEL_TO_CATEGORY, DEMOGRAPHIC_FIELDS,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def _effect_label(magnitude: float) -> str:
    """Classify effect size as negligible/small/medium/large."""
    t = EFFECT_SIZE_THRESHOLDS
    if magnitude >= t["large"]:
        return "large"
    elif magnitude >= t["medium"]:
        return "medium"
    elif magnitude >= t["small"]:
        return "small"
    return "negligible"


def _get_value_series(df: pd.DataFrame, sig_name: str) -> List[Tuple[str, pd.Series]]:
    """Get all value column series from a signal DataFrame."""
    val_cols = VALUE_COLS.get(sig_name, [])
    result = []
    for vc in val_cols:
        if vc in df.columns:
            result.append((vc, df[vc]))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: UNIVARIATE ANALYSIS (Descriptive Statistics)
# ─────────────────────────────────────────────────────────────────────────────

def descriptive_by_target(
    signals: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Section 5.1 — Per-target descriptive stats for each signal value column.

    Returns long-form: signal, value_col, target_label, n, mean, std, median,
                       iqr, q25, q75, min, max, cv.
    """
    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or "target_label" not in df.columns:
            continue
        for vc, series in _get_value_series(df, sig_name):
            for label, grp in df.groupby("target_label"):
                vals = grp[vc].dropna().astype(float).values
                if len(vals) < 2:
                    continue
                q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1))
                rows.append({
                    "signal": sig_name,
                    "value_col": vc,
                    "target_label": str(label),
                    "n": len(vals),
                    "mean": round(mean_v, 4),
                    "std": round(std_v, 4),
                    "median": round(float(np.median(vals)), 4),
                    "iqr": round(q75 - q25, 4),
                    "q25": round(q25, 4),
                    "q75": round(q75, 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                    "cv": round(abs(std_v / (mean_v + 1e-9)), 4),
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def descriptive_by_category(
    signals: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Section 5.2 — Per-category (affective/physiological_need/behavioural/baseline)
    descriptive stats.
    """
    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or "target_label" not in df.columns:
            continue
        df_cat = df.copy()
        df_cat["category"] = df_cat["target_label"].map(LABEL_TO_CATEGORY)
        for vc, _ in _get_value_series(df, sig_name):
            for cat, grp in df_cat.groupby("category"):
                vals = grp[vc].dropna().astype(float).values
                if len(vals) < 2:
                    continue
                q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1))
                rows.append({
                    "signal": sig_name,
                    "value_col": vc,
                    "category": str(cat),
                    "n": len(vals),
                    "mean": round(mean_v, 4),
                    "std": round(std_v, 4),
                    "median": round(float(np.median(vals)), 4),
                    "iqr": round(q75 - q25, 4),
                    "q25": round(q25, 4),
                    "q75": round(q75, 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                    "cv": round(abs(std_v / (mean_v + 1e-9)), 4),
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def descriptive_by_demographic(
    signals: Dict[str, pd.DataFrame],
    demographics: pd.DataFrame,
    demo_field: str = "autism_severity",
) -> pd.DataFrame:
    """
    Section 5.3 — Descriptive stats grouped by demographic field.

    Joins signals with demographics on user_id, then groups by demo_field.
    """
    if demographics is None or demographics.empty or demo_field not in demographics.columns:
        return pd.DataFrame()

    rows = []
    demo_sub = demographics[["user_id", demo_field]].drop_duplicates()

    for sig_name, df in sorted(signals.items()):
        if df is None or "user_id" not in df.columns:
            continue
        merged = df.merge(demo_sub, on="user_id", how="left")
        for vc, _ in _get_value_series(df, sig_name):
            if vc not in merged.columns:
                continue
            for grp_val, grp in merged.groupby(demo_field):
                vals = grp[vc].dropna().astype(float).values
                if len(vals) < 2:
                    continue
                q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1))
                rows.append({
                    "signal": sig_name,
                    "value_col": vc,
                    "demographic_field": demo_field,
                    "demographic_value": str(grp_val),
                    "n": len(vals),
                    "mean": round(mean_v, 4),
                    "std": round(std_v, 4),
                    "median": round(float(np.median(vals)), 4),
                    "iqr": round(q75 - q25, 4),
                    "cv": round(abs(std_v / (mean_v + 1e-9)), 4),
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: BIVARIATE ANALYSIS (Group Comparisons)
# ─────────────────────────────────────────────────────────────────────────────

def kruskal_wallis(
    signals: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Section 6.2 — Kruskal-Wallis H-test per signal value column across targets.

    Returns: signal, value_col, H_stat, p_value, n_groups, N_total,
             eta_squared, effect_size, significant.
    """
    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or "target_label" not in df.columns:
            continue
        for vc, _ in _get_value_series(df, sig_name):
            groups = []
            group_labels = []
            for label, grp in df.groupby("target_label"):
                vals = grp[vc].dropna().astype(float).values
                if len(vals) >= MIN_SAMPLES_PER_GROUP:
                    groups.append(vals)
                    group_labels.append(label)
            if len(groups) < 2:
                continue
            try:
                H, p = stats.kruskal(*groups)
            except Exception:
                continue

            N = sum(len(g) for g in groups)
            k = len(groups)
            eta2 = max(0.0, (H - k + 1) / (N - k + 1e-9))

            rows.append({
                "signal": sig_name,
                "value_col": vc,
                "H_stat": round(float(H), 4),
                "p_value": float(p),
                "n_groups": k,
                "N_total": N,
                "eta_squared": round(eta2, 4),
                "effect_size": _effect_label(eta2),
                "significant": bool(p < ALPHA),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("p_value").reset_index(drop=True)
    return result


def pairwise_mann_whitney(
    signals: Dict[str, pd.DataFrame],
    significant_only: bool = True,
    kw_results: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Section 6.3 — Pairwise Mann-Whitney U with Bonferroni correction.

    Only tests signal/column pairs that were significant in KW (if provided).
    Returns: signal, value_col, group_a, group_b, U_stat, p_value,
             p_bonferroni, significant, rank_biserial_r, effect_size, n_a, n_b.
    """
    # Determine which signal+col combos to test
    if kw_results is not None and not kw_results.empty and significant_only:
        sig_kw = kw_results[kw_results["significant"]]
        test_pairs = set(zip(sig_kw["signal"], sig_kw["value_col"]))
    else:
        test_pairs = None  # test all

    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or "target_label" not in df.columns:
            continue
        labels = sorted(df["target_label"].unique())
        for vc, _ in _get_value_series(df, sig_name):
            if test_pairs is not None and (sig_name, vc) not in test_pairs:
                continue

            # Count total pairwise tests for Bonferroni
            valid_labels = [
                l for l in labels
                if len(df.loc[df["target_label"] == l, vc].dropna()) >= MIN_SAMPLES_PER_GROUP
            ]
            n_comparisons = len(valid_labels) * (len(valid_labels) - 1) // 2
            if n_comparisons < 1:
                continue

            for la, lb in combinations(valid_labels, 2):
                va = df.loc[df["target_label"] == la, vc].dropna().astype(float).values
                vb = df.loc[df["target_label"] == lb, vc].dropna().astype(float).values
                if len(va) < MIN_SAMPLES_PER_GROUP or len(vb) < MIN_SAMPLES_PER_GROUP:
                    continue
                try:
                    U, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
                except Exception:
                    continue

                # Rank-biserial correlation
                r = 1 - (2 * U) / (len(va) * len(vb) + 1e-9)
                p_bonf = min(p * n_comparisons, 1.0)

                rows.append({
                    "signal": sig_name,
                    "value_col": vc,
                    "group_a": la,
                    "group_b": lb,
                    "U_stat": round(float(U), 2),
                    "p_value": float(p),
                    "p_bonferroni": round(p_bonf, 6),
                    "significant": bool(p_bonf < ALPHA),
                    "rank_biserial_r": round(abs(float(r)), 4),
                    "effect_size": _effect_label(abs(float(r))),
                    "n_a": len(va),
                    "n_b": len(vb),
                })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("p_bonferroni").reset_index(drop=True)
    return result


def kruskal_wallis_demographic(
    signals: Dict[str, pd.DataFrame],
    demographics: pd.DataFrame,
    demo_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Section 6.4 — Kruskal-Wallis H-test for demographic group differences.

    Tests whether signal distributions differ across severity, verbal_status, etc.
    """
    if demographics is None or demographics.empty:
        return pd.DataFrame()
    if demo_fields is None:
        demo_fields = [f for f in ["autism_severity", "verbal_status"]
                       if f in demographics.columns]

    demo_sub = demographics[["user_id"] + [f for f in demo_fields
                                            if f in demographics.columns]].drop_duplicates()
    rows = []
    for sig_name, df in sorted(signals.items()):
        if df is None or "user_id" not in df.columns:
            continue
        merged = df.merge(demo_sub, on="user_id", how="left")
        for vc, _ in _get_value_series(df, sig_name):
            if vc not in merged.columns:
                continue
            for field in demo_fields:
                if field not in merged.columns:
                    continue
                groups = []
                group_labels = []
                for val, grp in merged.groupby(field):
                    v = grp[vc].dropna().astype(float).values
                    if len(v) >= MIN_SAMPLES_PER_GROUP:
                        groups.append(v)
                        group_labels.append(val)
                if len(groups) < 2:
                    continue
                try:
                    H, p = stats.kruskal(*groups)
                except Exception:
                    continue
                N = sum(len(g) for g in groups)
                k = len(groups)
                eta2 = max(0.0, (H - k + 1) / (N - k + 1e-9))
                rows.append({
                    "signal": sig_name,
                    "value_col": vc,
                    "demographic_field": field,
                    "groups": str(group_labels),
                    "H_stat": round(float(H), 4),
                    "p_value": float(p),
                    "eta_squared": round(eta2, 4),
                    "effect_size": _effect_label(eta2),
                    "significant": bool(p < ALPHA),
                })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("p_value").reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: RUN ALL STATISTICAL ANALYSES
# ─────────────────────────────────────────────────────────────────────────────

def run_all_statistics(
    signals: Dict[str, pd.DataFrame],
    demographics: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Run all descriptive and inferential statistical analyses."""
    kw = kruskal_wallis(signals)

    results = {
        "descriptive_by_target": descriptive_by_target(signals),
        "descriptive_by_category": descriptive_by_category(signals),
        "kruskal_wallis": kw,
        "pairwise_mann_whitney": pairwise_mann_whitney(signals, kw_results=kw),
    }

    if demographics is not None and not demographics.empty:
        results["descriptive_by_severity"] = descriptive_by_demographic(
            signals, demographics, "autism_severity"
        )
        results["descriptive_by_verbal"] = descriptive_by_demographic(
            signals, demographics, "verbal_status"
        )
        results["kw_demographic"] = kruskal_wallis_demographic(
            signals, demographics
        )

    return results
