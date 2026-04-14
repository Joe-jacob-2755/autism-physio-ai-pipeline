"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  statistical_comparator.py
=============================================================================
Statistical tests for model comparison on the held-out test set:
  - McNemar's test (pairwise): are two models' errors significantly different?
  - Cochran's Q test (omnibus): do all models have the same error rate?
  - DeLong's test (pairwise AUC): are two models' AUC-ROC significantly different?
=============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)


@dataclass
class McNemarResult:
    """McNemar's test between two models."""
    model_a: str
    model_b: str
    n_both_correct: int
    n_both_wrong: int
    n_a_correct_b_wrong: int
    n_a_wrong_b_correct: int
    chi2: float
    p_value: float
    significant: bool
    better_model: Optional[str] = None


@dataclass
class CochranQResult:
    """Cochran's Q omnibus test across all models."""
    q_statistic: float
    p_value: float
    df: int
    n_models: int
    significant: bool


@dataclass
class DeLongResult:
    """DeLong test comparing AUC-ROC of two models."""
    model_a: str
    model_b: str
    auc_a: float
    auc_b: float
    z_statistic: float
    p_value: float
    significant: bool
    better_model: Optional[str] = None


@dataclass
class StatisticalComparison:
    """Full statistical comparison results."""
    mcnemar_tests: List[McNemarResult] = field(default_factory=list)
    cochran_q: Optional[CochranQResult] = None
    delong_tests: List[DeLongResult] = field(default_factory=list)
    n_models: int = 0


class StatisticalComparator:
    """Statistical tests for pairwise and omnibus model comparison."""

    def __init__(self, alpha: float = 0.05, verbose: bool = True):
        self.alpha = alpha
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [M8-Stat] {msg}")

    def compare_all(
        self,
        eval_results: list,
        y_test: np.ndarray,
        class_names: List[str],
    ) -> StatisticalComparison:
        """
        Run all statistical comparison tests.

        Parameters
        ----------
        eval_results : List[EvalResult] (non-skipped only)
        y_test : (n_test,) encoded labels
        class_names : list of class names

        Returns
        -------
        StatisticalComparison
        """
        # Filter to non-skipped models with predictions
        valid = [er for er in eval_results if not er.skipped and er.y_pred is not None]
        comp = StatisticalComparison(n_models=len(valid))

        if len(valid) < 2:
            self._log("Need at least 2 models for comparison")
            return comp

        # McNemar's pairwise tests
        self._log(f"Running McNemar's tests ({len(valid)} models)...")
        for er_a, er_b in combinations(valid, 2):
            result = self._mcnemar(er_a, er_b, y_test)
            comp.mcnemar_tests.append(result)

        n_sig = sum(1 for t in comp.mcnemar_tests if t.significant)
        self._log(f"  {n_sig}/{len(comp.mcnemar_tests)} pairs significantly different")

        # Cochran's Q omnibus test
        if len(valid) >= 3:
            self._log("Running Cochran's Q test...")
            comp.cochran_q = self._cochran_q(valid, y_test)
            self._log(
                f"  Q={comp.cochran_q.q_statistic:.2f}, "
                f"p={comp.cochran_q.p_value:.4f}, "
                f"sig={comp.cochran_q.significant}"
            )

        # DeLong AUC comparison (only for models with probabilities)
        prob_models = [er for er in valid if er.y_prob is not None]
        if len(prob_models) >= 2 and len(class_names) == 2:
            self._log(f"Running DeLong AUC tests ({len(prob_models)} models)...")
            for er_a, er_b in combinations(prob_models, 2):
                result = self._delong(er_a, er_b, y_test)
                if result is not None:
                    comp.delong_tests.append(result)
        elif len(prob_models) >= 2:
            # Multi-class: compare macro AUC using bootstrap
            self._log("Multi-class: DeLong not directly applicable, using bootstrap AUC comparison")
            for er_a, er_b in combinations(prob_models, 2):
                result = self._bootstrap_auc_comparison(
                    er_a, er_b, y_test, class_names
                )
                if result is not None:
                    comp.delong_tests.append(result)

        return comp

    def _mcnemar(self, er_a, er_b, y_test: np.ndarray) -> McNemarResult:
        """
        McNemar's test comparing two classifiers.

        Builds a 2x2 contingency table:
            B correct | B wrong
        A correct:  n00      | n01
        A wrong:    n10      | n11

        Tests whether n01 and n10 are significantly different.
        """
        correct_a = (er_a.y_pred == y_test)
        correct_b = (er_b.y_pred == y_test)

        n_both_correct = int((correct_a & correct_b).sum())
        n_both_wrong = int((~correct_a & ~correct_b).sum())
        n_a_only = int((correct_a & ~correct_b).sum())  # A correct, B wrong
        n_b_only = int((~correct_a & correct_b).sum())  # B correct, A wrong

        # McNemar's chi-squared with continuity correction
        denom = n_a_only + n_b_only
        if denom > 0:
            chi2 = (abs(n_a_only - n_b_only) - 1) ** 2 / denom
            p_value = float(stats.chi2.sf(chi2, df=1))
        else:
            chi2 = 0.0
            p_value = 1.0

        better = None
        if p_value < self.alpha:
            if n_a_only > n_b_only:
                better = er_a.model_name
            else:
                better = er_b.model_name

        return McNemarResult(
            model_a=er_a.model_name,
            model_b=er_b.model_name,
            n_both_correct=n_both_correct,
            n_both_wrong=n_both_wrong,
            n_a_correct_b_wrong=n_a_only,
            n_a_wrong_b_correct=n_b_only,
            chi2=float(chi2),
            p_value=p_value,
            significant=p_value < self.alpha,
            better_model=better,
        )

    def _cochran_q(self, eval_results: list, y_test: np.ndarray) -> CochranQResult:
        """
        Cochran's Q test: omnibus test for whether all models have
        the same error rate.

        H0: all models have the same proportion of correct predictions.
        """
        k = len(eval_results)
        n = len(y_test)

        # Build binary correctness matrix (n_samples x k_models)
        correct_matrix = np.zeros((n, k), dtype=int)
        for j, er in enumerate(eval_results):
            correct_matrix[:, j] = (er.y_pred == y_test).astype(int)

        # Row sums (Ti) and column sums (Cj)
        T = correct_matrix.sum(axis=1)  # per-sample: how many models got it right
        C = correct_matrix.sum(axis=0)  # per-model: total correct

        T_sum = T.sum()
        T_sq_sum = (T ** 2).sum()
        C_sq_sum = (C ** 2).sum()

        # Cochran's Q statistic
        numerator = (k - 1) * (k * C_sq_sum - T_sum ** 2)
        denominator = k * T_sum - T_sq_sum

        if denominator > 0:
            q_stat = float(numerator / denominator)
        else:
            q_stat = 0.0

        df = k - 1
        p_value = float(stats.chi2.sf(q_stat, df=df))

        return CochranQResult(
            q_statistic=q_stat,
            p_value=p_value,
            df=df,
            n_models=k,
            significant=p_value < self.alpha,
        )

    def _delong(
        self, er_a, er_b, y_test: np.ndarray
    ) -> Optional[DeLongResult]:
        """
        DeLong's test for comparing two AUC-ROC values (binary case).

        Uses the non-parametric approach based on Mann-Whitney U statistic
        variance estimation.
        """
        if er_a.y_prob is None or er_b.y_prob is None:
            return None

        try:
            from sklearn.metrics import roc_auc_score

            # Binary: use column 1 (positive class) probabilities
            prob_a = er_a.y_prob[:, 1] if er_a.y_prob.ndim > 1 else er_a.y_prob
            prob_b = er_b.y_prob[:, 1] if er_b.y_prob.ndim > 1 else er_b.y_prob

            auc_a = float(roc_auc_score(y_test, prob_a))
            auc_b = float(roc_auc_score(y_test, prob_b))

            # Variance estimation using placement values
            se = self._delong_variance(y_test, prob_a, prob_b)

            if se > 0:
                z = (auc_a - auc_b) / se
                p_value = float(2 * stats.norm.sf(abs(z)))
            else:
                z = 0.0
                p_value = 1.0

            better = None
            if p_value < self.alpha:
                better = er_a.model_name if auc_a > auc_b else er_b.model_name

            return DeLongResult(
                model_a=er_a.model_name,
                model_b=er_b.model_name,
                auc_a=auc_a,
                auc_b=auc_b,
                z_statistic=float(z),
                p_value=p_value,
                significant=p_value < self.alpha,
                better_model=better,
            )
        except Exception as e:
            self._log(f"  DeLong test failed: {e}")
            return None

    def _delong_variance(
        self, y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray
    ) -> float:
        """Estimate standard error of AUC difference using DeLong's method."""
        pos_mask = y_true == 1
        neg_mask = y_true == 0

        m = pos_mask.sum()
        n = neg_mask.sum()

        if m == 0 or n == 0:
            return 0.0

        # Placement values for each model
        def placements(probs):
            V_pos = np.zeros(m)
            V_neg = np.zeros(n)
            pos_scores = probs[pos_mask]
            neg_scores = probs[neg_mask]

            for i in range(m):
                V_pos[i] = np.mean(pos_scores[i] > neg_scores) + \
                           0.5 * np.mean(pos_scores[i] == neg_scores)
            for j in range(n):
                V_neg[j] = np.mean(pos_scores > neg_scores[j]) + \
                           0.5 * np.mean(pos_scores == neg_scores[j])
            return V_pos, V_neg

        V10_a, V01_a = placements(prob_a)
        V10_b, V01_b = placements(prob_b)

        # Covariance matrix of AUC estimates
        S10 = np.cov(V10_a, V10_b)[0, 1] if m > 1 else 0
        S01 = np.cov(V01_a, V01_b)[0, 1] if n > 1 else 0

        var_a = np.var(V10_a) / m + np.var(V01_a) / n if m > 1 and n > 1 else 0
        var_b = np.var(V10_b) / m + np.var(V01_b) / n if m > 1 and n > 1 else 0
        cov_ab = S10 / m + S01 / n

        var_diff = var_a + var_b - 2 * cov_ab
        return float(np.sqrt(max(var_diff, 0)))

    def _bootstrap_auc_comparison(
        self, er_a, er_b, y_test: np.ndarray, class_names: List[str],
        n_bootstrap: int = 1000,
    ) -> Optional[DeLongResult]:
        """Bootstrap-based AUC comparison for multi-class case."""
        try:
            from sklearn.metrics import roc_auc_score
            from sklearn.preprocessing import label_binarize

            y_bin = label_binarize(y_test, classes=list(range(len(class_names))))

            def macro_auc(y_prob):
                aucs = []
                for i in range(len(class_names)):
                    if y_bin[:, i].sum() > 0 and y_prob.shape[1] > i:
                        try:
                            aucs.append(roc_auc_score(y_bin[:, i], y_prob[:, i]))
                        except ValueError:
                            pass
                return np.mean(aucs) if aucs else 0.0

            auc_a = macro_auc(er_a.y_prob)
            auc_b = macro_auc(er_b.y_prob)

            # Bootstrap difference
            rng = np.random.default_rng(42)
            diffs = []
            n = len(y_test)
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                y_b = y_test[idx]
                yb_bin = label_binarize(y_b, classes=list(range(len(class_names))))

                a_auc = macro_auc(er_a.y_prob[idx])
                b_auc = macro_auc(er_b.y_prob[idx])
                diffs.append(a_auc - b_auc)

            diffs = np.array(diffs)
            se = np.std(diffs)
            z = (auc_a - auc_b) / se if se > 0 else 0.0
            p = float(2 * stats.norm.sf(abs(z)))

            better = None
            if p < self.alpha:
                better = er_a.model_name if auc_a > auc_b else er_b.model_name

            return DeLongResult(
                model_a=er_a.model_name,
                model_b=er_b.model_name,
                auc_a=float(auc_a),
                auc_b=float(auc_b),
                z_statistic=float(z),
                p_value=p,
                significant=p < self.alpha,
                better_model=better,
            )
        except Exception as e:
            self._log(f"  Bootstrap AUC comparison failed: {e}")
            return None

    def build_mcnemar_matrix(
        self, comparison: StatisticalComparison, model_names: List[str]
    ) -> pd.DataFrame:
        """Build a pairwise p-value matrix for McNemar's test."""
        n = len(model_names)
        matrix = pd.DataFrame(
            np.ones((n, n)), index=model_names, columns=model_names
        )

        for t in comparison.mcnemar_tests:
            if t.model_a in model_names and t.model_b in model_names:
                matrix.loc[t.model_a, t.model_b] = t.p_value
                matrix.loc[t.model_b, t.model_a] = t.p_value

        # Diagonal = NaN (self-comparison)
        for m in model_names:
            matrix.loc[m, m] = np.nan

        return matrix
