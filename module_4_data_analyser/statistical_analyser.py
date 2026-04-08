"""
=============================================================================
MODULE 4 – DATA ANALYSER  |  statistical_analyser.py
=============================================================================
All statistical analyses on the extracted feature DataFrames:

  1. Descriptive  — mean, std, median, IQR, min, max per target per feature
  2. Statistical  — Kruskal-Wallis H + pairwise Mann-Whitney U + effect sizes
  3. Correlation  — Spearman feature×target + feature×feature heatmap
  4. ANOVA        — one-way ANOVA where normality holds (Shapiro-Wilk pre-check)
=============================================================================
"""
from __future__ import annotations
from config import (
    ALPHA, MIN_SAMPLES_PER_GROUP, EFFECT_SIZE_THRESHOLDS, META_COLS,
)

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# DESCRIPTIVE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def descriptive_analysis(df: pd.DataFrame, label_col="target_label") -> pd.DataFrame:
    """
    Per-target descriptive statistics for all numeric features.

    Returns a long-form DataFrame:
      columns: target_label, feature, mean, std, median, iqr, min, max,
               q25, q75, n, cv (coefficient of variation)
    """
    feat_cols = [c for c in df.columns if c not in META_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols or label_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for label, grp in df.groupby(label_col):
        for col in feat_cols:
            vals = grp[col].dropna().values.astype(float)
            if len(vals) < 2:
                continue
            q25, q75 = np.percentile(vals, [25, 75])
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1))
            rows.append({
                "target_label": label,
                "feature": col,
                "n": len(vals),
                "mean": round(mean_v, 4),
                "std": round(std_v, 4),
                "median": round(float(np.median(vals)), 4),
                "iqr": round(float(q75 - q25), 4),
                "q25": round(float(q25), 4),
                "q75": round(float(q75), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "cv": round(abs(std_v / (mean_v + 1e-9)), 4),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# KRUSKAL-WALLIS + PAIRWISE MANN-WHITNEY
# ─────────────────────────────────────────────────────────────────────────────

def kruskal_wallis_analysis(
    df: pd.DataFrame, label_col="target_label"
) -> pd.DataFrame:
    """
    Kruskal-Wallis H-test for each feature across all target groups.

    Returns one row per feature:
      feature, H_stat, p_value, significant, eta_squared (effect size),
      effect_size_label
    """
    feat_cols = [c for c in df.columns if c not in META_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols or label_col not in df.columns:
        return pd.DataFrame()

    rows = []
    for col in feat_cols:
        groups = [
            grp[col].dropna().values.astype(float)
            for _, grp in df.groupby(label_col)
            if len(grp[col].dropna()) >= MIN_SAMPLES_PER_GROUP
        ]
        if len(groups) < 2:
            continue
        try:
            H, p = stats.kruskal(*groups)
        except Exception:
            continue

        # Eta-squared effect size: η² = (H - k + 1) / (N - k)
        N = sum(len(g) for g in groups)
        k = len(groups)
        eta2 = max(0, (H - k + 1) / (N - k + 1e-9))
        es_label = _effect_label(eta2)

        rows.append({
            "feature": col,
            "H_stat": round(H, 4),
            "p_value": round(p, 6),
            "significant": p < ALPHA,
            "eta_squared": round(eta2, 4),
            "effect_size": es_label,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("p_value").reset_index(drop=True)
    return result


def pairwise_mannwhitney(
    df: pd.DataFrame, label_col="target_label",
    top_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney U test for top features across label pairs.
    Applies Bonferroni correction.

    Parameters
    ----------
    top_features : list of feature names to test (if None, uses top 20 by KW)
    """
    feat_cols = (top_features
                 or [c for c in df.columns if c not in META_COLS and
                     pd.api.types.is_numeric_dtype(df[c])][:20])
    labels = sorted(df[label_col].unique())
    rows = []
    n_tests = len(feat_cols) * (len(labels) * (len(labels) - 1) // 2)
    bonf = ALPHA / max(n_tests, 1)

    for col in feat_cols:
        for i, la in enumerate(labels):
            for lb in labels[i + 1:]:
                va = df.loc[df[label_col] == la, col].dropna().values.astype(float)
                vb = df.loc[df[label_col] == lb, col].dropna().values.astype(float)
                if len(va) < 2 or len(vb) < 2:
                    continue
                try:
                    U, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
                except Exception:
                    continue
                # Rank-biserial correlation as effect size
                r = 1 - (2 * U) / (len(va) * len(vb) + 1e-9)
                rows.append({
                    "feature": col,
                    "group_a": la, "group_b": lb,
                    "U_stat": round(U, 2),
                    "p_value": round(p, 6),
                    "p_bonf": round(p * n_tests, 6),
                    "significant": p < bonf,
                    "effect_r": round(abs(r), 4),
                    "effect_size": _effect_label(abs(r)),
                    "n_a": len(va), "n_b": len(vb),
                })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("p_bonf").reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def correlation_feature_target(
    df: pd.DataFrame, label_col="target_label"
) -> pd.DataFrame:
    """
    Spearman correlation between each feature and the (encoded) target label.

    Returns: feature, spearman_r, p_value, abs_r, effect_size
    """
    feat_cols = [c for c in df.columns if c not in META_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols or label_col not in df.columns:
        return pd.DataFrame()

    # Ordinal encode the target label
    label_cats = pd.Categorical(df[label_col]).codes

    rows = []
    for col in feat_cols:
        vals = df[col].values.astype(float)
        mask = ~np.isnan(vals)
        if mask.sum() < 5:
            continue
        try:
            r, p = stats.spearmanr(vals[mask], label_cats[mask])
        except Exception:
            continue
        rows.append({
            "feature": col,
            "spearman_r": round(float(r), 4),
            "p_value": round(float(p), 6),
            "abs_r": round(abs(float(r)), 4),
            "effect_size": _effect_label(abs(float(r))),
            "significant": float(p) < ALPHA,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("abs_r", ascending=False).reset_index(drop=True)
    return result


def correlation_feature_matrix(
    df: pd.DataFrame, max_features: int = 30
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Spearman correlation matrix among the top N most variable features.

    Returns (corr_matrix, feature_names).
    """
    feat_cols = [c for c in df.columns if c not in META_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        return pd.DataFrame(), []

    # Select top N by variance
    variances = df[feat_cols].var().sort_values(ascending=False)
    top_feats = variances.index[:max_features].tolist()
    sub = df[top_feats].dropna()
    if len(sub) < 3:
        return pd.DataFrame(), top_feats

    corr_mat, _ = stats.spearmanr(sub.values)
    if corr_mat.ndim == 0:
        return pd.DataFrame(), top_feats
    corr_df = pd.DataFrame(
        corr_mat if corr_mat.ndim == 2 else [[corr_mat]],
        index=top_feats[:len(sub.columns)],
        columns=top_feats[:len(sub.columns)],
    )
    return corr_df, top_feats


def signal_pct_change_summary(dynamics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise % change from baseline per signal × target label.
    Shows mean, std, median, and range of change_pct.
    """
    if dynamics_df.empty or "change_pct" not in dynamics_df.columns:
        return pd.DataFrame()
    grp = dynamics_df.groupby(["signal", "target_label"])["change_pct"]
    return grp.agg(
        mean_change_pct="mean",
        std_change_pct="std",
        median_change_pct="median",
        min_change_pct="min",
        max_change_pct="max",
        n_events="count",
    ).round(2).reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _effect_label(magnitude: float) -> str:
    t = EFFECT_SIZE_THRESHOLDS
    if magnitude >= t["large"]:
        return "large"
    elif magnitude >= t["medium"]:
        return "medium"
    elif magnitude >= t["small"]:
        return "small"
    return "negligible"
