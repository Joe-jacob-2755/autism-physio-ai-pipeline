"""
=============================================================================
MODULE 6 – FEATURE ENGINEERING  |  feature_selector.py
=============================================================================
Feature importance ranking and selection.

Supports multiple scoring methods:
  1. Mutual Information  — model-free, captures non-linear dependencies
  2. Random Forest       — tree-based importance (Gini / permutation)
  3. F-statistic (ANOVA) — linear discriminative power per class
  4. Variance Threshold  — remove near-constant features

The user chooses how many top features to keep. Both the full feature
set and the selected subset are saved as CSV.

=============================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.feature_selection import (
    mutual_info_classif,
    f_classif,
)
from sklearn.ensemble import RandomForestClassifier

from config import DEFAULT_TOP_N_FEATURES, META_COLS

# META_COLS already includes demographic columns
_SKIP_COLS = META_COLS


class FeatureSelector:
    """
    Rank features by importance and select the top N.

    Parameters
    ----------
    top_n   : number of top features to select (user-configurable)
    method  : scoring method ('mutual_information', 'random_forest',
              'f_statistic', 'variance_threshold', 'ensemble')
    seed    : random seed for reproducibility
    verbose : print progress
    """

    def __init__(
        self,
        top_n: int = DEFAULT_TOP_N_FEATURES,
        method: str = "ensemble",
        seed: int = 42,
        verbose: bool = True,
    ):
        self.top_n = top_n
        self.method = method
        self.seed = seed
        self.verbose = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def rank_and_select(
        self,
        df: pd.DataFrame,
        target_col: str = "target_label",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Rank all features and return selected subset.

        Parameters
        ----------
        df         : combined feature DataFrame (all signals fused)
        target_col : column name for the classification target

        Returns
        -------
        importance_df  : DataFrame with feature names, scores, and rank
        selected_df    : df subset with only top_n features + meta columns
        full_df        : original df unchanged (for reference)
        """
        feat_cols = self._get_feature_cols(df)
        if not feat_cols:
            self._log("[Selector] No feature columns found.")
            return pd.DataFrame(), df, df

        X = df[feat_cols].values.astype(float)
        # Handle NaN/inf in features
        X = np.where(np.isfinite(X), X, 0.0)

        # Target encoding
        if target_col not in df.columns:
            self._log(f"[Selector] WARNING: '{target_col}' not found. "
                      "Using variance-based ranking only.")
            scores = self._variance_scores(X, feat_cols)
            importance_df = self._build_importance_df(feat_cols, scores,
                                                      "variance")
        elif self.method == "ensemble":
            importance_df = self._ensemble_rank(X, df[target_col], feat_cols)
        else:
            y = self._encode_target(df[target_col])
            scores = self._score_features(X, y, feat_cols)
            importance_df = self._build_importance_df(feat_cols, scores,
                                                      self.method)

        # Sort by rank and select top_n
        importance_df = importance_df.sort_values("rank").reset_index(drop=True)
        top_features = importance_df["feature"].iloc[:self.top_n].tolist()

        # Build selected DataFrame: meta cols + top features
        meta_in_df = [c for c in df.columns if c in _SKIP_COLS or c == target_col]
        selected_cols = meta_in_df + top_features
        selected_df = df[[c for c in selected_cols if c in df.columns]].copy()

        self._log(f"[Selector] Ranked {len(feat_cols)} features, "
                  f"selected top {min(self.top_n, len(feat_cols))}  "
                  f"(method={self.method})")

        return importance_df, selected_df, df

    # ── Scoring methods ───────────────────────────────────────────────────

    def _score_features(
        self, X: np.ndarray, y: np.ndarray, feat_cols: list
    ) -> np.ndarray:
        """Score features using the configured method."""
        if self.method == "mutual_information":
            return self._mi_scores(X, y)
        elif self.method == "random_forest":
            return self._rf_scores(X, y)
        elif self.method == "f_statistic":
            return self._f_scores(X, y)
        elif self.method == "variance_threshold":
            return self._variance_scores(X, feat_cols)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _mi_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mutual information between each feature and target."""
        self._log("  Computing mutual information scores ...")
        return mutual_info_classif(
            X, y, discrete_features=False, random_state=self.seed
        )

    def _rf_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Random Forest Gini importance."""
        self._log("  Computing Random Forest importance ...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.seed,
            n_jobs=-1,
        )
        rf.fit(X, y)
        return rf.feature_importances_

    def _f_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ANOVA F-statistic."""
        self._log("  Computing F-statistic (ANOVA) scores ...")
        f_vals, _ = f_classif(X, y)
        return np.nan_to_num(f_vals, nan=0.0)

    def _variance_scores(
        self, X: np.ndarray, feat_cols: list
    ) -> np.ndarray:
        """Variance of each feature (higher = more informative)."""
        self._log("  Computing variance scores ...")
        return np.var(X, axis=0)

    def _ensemble_rank(
        self,
        X: np.ndarray,
        y_series: pd.Series,
        feat_cols: list,
    ) -> pd.DataFrame:
        """
        Ensemble ranking: average normalised ranks across all methods.

        This gives a robust importance ranking that doesn't depend on a
        single scoring method's assumptions.
        """
        y = self._encode_target(y_series)
        self._log("[Selector] Ensemble ranking (MI + RF + F-stat + Variance)")

        all_scores = {}
        all_scores["mutual_information"] = self._mi_scores(X, y)
        all_scores["random_forest"] = self._rf_scores(X, y)
        all_scores["f_statistic"] = self._f_scores(X, y)
        all_scores["variance"] = self._variance_scores(X, feat_cols)

        # Normalise each method's scores to [0, 1] ranks
        n = len(feat_cols)
        rank_matrix = np.zeros((n, len(all_scores)))
        for i, (method_name, scores) in enumerate(all_scores.items()):
            # Rank: highest score = rank 1
            order = np.argsort(-scores)
            ranks = np.zeros(n)
            ranks[order] = np.arange(1, n + 1)
            rank_matrix[:, i] = ranks

        # Average rank across methods
        avg_rank = np.mean(rank_matrix, axis=1)
        final_order = np.argsort(avg_rank)

        rows = []
        for rank_pos, idx in enumerate(final_order, 1):
            row = {
                "feature": feat_cols[idx],
                "rank": rank_pos,
                "avg_rank": round(float(avg_rank[idx]), 2),
            }
            for method_name, scores in all_scores.items():
                row[f"score_{method_name}"] = round(float(scores[idx]), 6)
            rows.append(row)

        return pd.DataFrame(rows)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """Return numeric feature columns (excluding meta/demographic cols)."""
        return [c for c in df.columns
                if c not in _SKIP_COLS
                and pd.api.types.is_numeric_dtype(df[c])]

    def _encode_target(self, y_series: pd.Series) -> np.ndarray:
        """Encode string labels to integers."""
        codes, _ = pd.factorize(y_series)
        return codes

    def _build_importance_df(
        self, feat_cols: list, scores: np.ndarray, method: str,
    ) -> pd.DataFrame:
        """Build a DataFrame of feature importance scores with rank."""
        order = np.argsort(-scores)
        rows = []
        for rank_pos, idx in enumerate(order, 1):
            rows.append({
                "feature": feat_cols[idx],
                "rank": rank_pos,
                f"score_{method}": round(float(scores[idx]), 6),
            })
        return pd.DataFrame(rows)

    def _log(self, msg):
        if self.verbose:
            print(msg)
