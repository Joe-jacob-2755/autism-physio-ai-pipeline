"""
=============================================================================
MODULE 6 – FEATURE ENGINEERING  |  dimensionality_reducer.py
=============================================================================
Dimensionality reduction for physiological feature matrices.

Methods implemented:
  1. PCA         — Principal Component Analysis (unsupervised, linear)
  2. PLS-VIP     — Partial Least Squares Variable Importance in Projection
                   (supervised, linear — best for identifying discriminative
                   features when labels are available)
  3. Truncated SVD — like PCA but works on sparse matrices; no centring
  4. Comparison  — run all methods, report best by retained variance and
                   downstream separability

VIP (Variable Importance in Projection) scores from PLS identify which
original features contribute most to class separation — useful for
clinical interpretability.

=============================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder

from config import (
    DEFAULT_PCA_VARIANCE, DEFAULT_N_PLS_COMPONENTS, META_COLS,
)


class DimensionalityReducer:
    """
    Apply and compare dimensionality reduction methods.

    Parameters
    ----------
    pca_variance    : fraction of variance to retain in PCA (default 0.95)
    n_pls_components: number of PLS components for VIP (default 10)
    seed            : random seed
    verbose         : print progress
    """

    def __init__(
        self,
        pca_variance: float = DEFAULT_PCA_VARIANCE,
        n_pls_components: int = DEFAULT_N_PLS_COMPONENTS,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.pca_variance = pca_variance
        self.n_pls_components = n_pls_components
        self.seed = seed
        self.verbose = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def reduce_and_compare(
        self,
        df: pd.DataFrame,
        target_col: str = "target_label",
    ) -> Dict[str, object]:
        """
        Run all dimensionality reduction methods and compare.

        Returns
        -------
        dict with keys:
          pca_result      : dict (transformed, components, explained_var, model)
          pls_vip_result  : dict (vip_scores, vip_ranking, model)
          svd_result      : dict (transformed, components, explained_var, model)
          comparison      : dict summarising each method
          best_method     : str name of recommended method
        """
        feat_cols = self._get_feature_cols(df)
        X = df[feat_cols].values.astype(float)
        X = np.where(np.isfinite(X), X, 0.0)

        results = {}

        # ── PCA ───────────────────────────────────────────────────────
        self._log("[DimReduce] Running PCA ...")
        results["pca_result"] = self._run_pca(X, feat_cols)

        # ── PLS-VIP (requires target labels) ──────────────────────────
        if target_col in df.columns:
            self._log("[DimReduce] Running PLS-VIP ...")
            y = LabelEncoder().fit_transform(df[target_col])
            results["pls_vip_result"] = self._run_pls_vip(X, y, feat_cols)
        else:
            self._log("[DimReduce] Skipping PLS-VIP (no target column)")
            results["pls_vip_result"] = None

        # ── Truncated SVD ─────────────────────────────────────────────
        self._log("[DimReduce] Running Truncated SVD ...")
        results["svd_result"] = self._run_svd(X, feat_cols)

        # ── Comparison ────────────────────────────────────────────────
        results["comparison"] = self._compare(results)
        results["best_method"] = results["comparison"].get("recommended", "pca")

        self._log(f"[DimReduce] Recommended method: {results['best_method']}")
        return results

    # ── PCA ────────────────────────────────────────────────────────────────

    def _run_pca(
        self, X: np.ndarray, feat_cols: list
    ) -> dict:
        """PCA retaining pca_variance fraction of total variance."""
        pca = PCA(n_components=self.pca_variance, random_state=self.seed)
        X_pca = pca.fit_transform(X)

        explained = pca.explained_variance_ratio_
        n_components = pca.n_components_
        total_var = float(np.sum(explained))

        self._log(f"  PCA: {n_components} components, "
                  f"{total_var:.1%} variance retained")

        # Component loadings
        comp_df = pd.DataFrame(
            pca.components_.T,
            index=feat_cols,
            columns=[f"PC{i+1}" for i in range(n_components)],
        )

        # Transformed data
        trans_df = pd.DataFrame(
            X_pca,
            columns=[f"PC{i+1}" for i in range(n_components)],
        )

        return {
            "transformed": trans_df,
            "components": comp_df,
            "explained_variance": explained,
            "n_components": n_components,
            "total_variance": total_var,
            "model": pca,
        }

    # ── PLS-VIP ───────────────────────────────────────────────────────────

    def _run_pls_vip(
        self, X: np.ndarray, y: np.ndarray, feat_cols: list
    ) -> dict:
        """
        PLS regression + VIP score calculation.

        VIP (Variable Importance in Projection) measures how much each
        original variable contributes to the PLS model. Features with
        VIP > 1.0 are considered important (Wold, 1993).
        """
        n_comp = min(self.n_pls_components, X.shape[1], X.shape[0] - 1)
        if n_comp < 1:
            return {"vip_scores": None, "vip_ranking": None, "model": None}

        pls = PLSRegression(n_components=n_comp)
        pls.fit(X, y)

        # VIP calculation
        # VIP_j = sqrt(p * sum_a(w_ja^2 * SS_a) / sum_a(SS_a))
        T = pls.x_scores_       # (n, n_comp)
        W = pls.x_weights_      # (p, n_comp)
        Q = pls.y_loadings_     # (1, n_comp)

        p = X.shape[1]
        ss = np.zeros(n_comp)
        for a in range(n_comp):
            ss[a] = float(Q[0, a] ** 2 * np.dot(T[:, a], T[:, a]))
        ss_total = np.sum(ss)

        vip = np.zeros(p)
        if ss_total > 0:
            for j in range(p):
                s = 0.0
                for a in range(n_comp):
                    s += W[j, a] ** 2 * ss[a]
                vip[j] = np.sqrt(p * s / ss_total)

        # Build VIP ranking DataFrame
        vip_df = pd.DataFrame({
            "feature": feat_cols,
            "vip_score": np.round(vip, 6),
        }).sort_values("vip_score", ascending=False).reset_index(drop=True)
        vip_df["vip_rank"] = range(1, len(vip_df) + 1)
        vip_df["important"] = vip_df["vip_score"] >= 1.0

        n_important = int(vip_df["important"].sum())
        self._log(f"  PLS-VIP: {n_important}/{p} features with VIP >= 1.0")

        return {
            "vip_scores": vip,
            "vip_ranking": vip_df,
            "model": pls,
        }

    # ── Truncated SVD ─────────────────────────────────────────────────────

    def _run_svd(
        self, X: np.ndarray, feat_cols: list
    ) -> dict:
        """Truncated SVD — similar to PCA but no centring requirement."""
        n_comp = min(50, X.shape[1] - 1, X.shape[0] - 1)
        if n_comp < 1:
            return {"transformed": None, "components": None,
                    "explained_variance": None, "model": None}

        svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
        X_svd = svd.fit_transform(X)

        explained = svd.explained_variance_ratio_
        # Find how many components needed for target variance
        cumvar = np.cumsum(explained)
        n_keep = int(np.searchsorted(cumvar, self.pca_variance) + 1)
        n_keep = min(n_keep, n_comp)
        total_var = float(cumvar[n_keep - 1])

        self._log(f"  SVD: {n_keep}/{n_comp} components for "
                  f"{total_var:.1%} variance")

        comp_df = pd.DataFrame(
            svd.components_[:n_keep].T,
            index=feat_cols,
            columns=[f"SVD{i+1}" for i in range(n_keep)],
        )
        trans_df = pd.DataFrame(
            X_svd[:, :n_keep],
            columns=[f"SVD{i+1}" for i in range(n_keep)],
        )

        return {
            "transformed": trans_df,
            "components": comp_df,
            "explained_variance": explained[:n_keep],
            "n_components": n_keep,
            "total_variance": total_var,
            "model": svd,
        }

    # ── Comparison ────────────────────────────────────────────────────────

    def _compare(self, results: dict) -> dict:
        """Compare methods and recommend the best one."""
        comparison = {}

        # PCA
        pca = results.get("pca_result", {})
        if pca:
            comparison["pca"] = {
                "n_components": pca.get("n_components", 0),
                "total_variance": round(pca.get("total_variance", 0), 4),
                "method_type": "unsupervised_linear",
            }

        # SVD
        svd = results.get("svd_result", {})
        if svd and svd.get("transformed") is not None:
            comparison["truncated_svd"] = {
                "n_components": svd.get("n_components", 0),
                "total_variance": round(svd.get("total_variance", 0), 4),
                "method_type": "unsupervised_linear",
            }

        # PLS-VIP
        vip = results.get("pls_vip_result")
        if vip and vip.get("vip_ranking") is not None:
            n_important = int(vip["vip_ranking"]["important"].sum())
            comparison["pls_vip"] = {
                "n_important_features": n_important,
                "method_type": "supervised_linear",
                "note": "Feature ranking (not dimensionality reduction)",
            }

        # Recommendation logic:
        # - PCA for general dimensionality reduction (unsupervised)
        # - PLS-VIP for feature importance when labels are available
        if vip and vip.get("vip_ranking") is not None:
            comparison["recommended"] = "pls_vip"
            comparison["rationale"] = (
                "PLS-VIP recommended: supervised method identifies features "
                "most discriminative for target labels. Use PCA for "
                "unsupervised dimensionality reduction."
            )
        else:
            comparison["recommended"] = "pca"
            comparison["rationale"] = (
                "PCA recommended: no target labels available for "
                "supervised methods."
            )

        return comparison

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """Return numeric feature columns (excluding meta/demographic cols)."""
        return [c for c in df.columns
                if c not in META_COLS
                and pd.api.types.is_numeric_dtype(df[c])]

    def _log(self, msg):
        if self.verbose:
            print(msg)
