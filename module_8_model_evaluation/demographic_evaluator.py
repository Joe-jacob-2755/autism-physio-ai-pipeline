"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  demographic_evaluator.py
=============================================================================
Per-subgroup performance analysis and equity testing.  Breaks down model
metrics by autism severity, verbal status, age group, and gender.  Runs
chi-squared / Fisher's exact tests to detect statistically significant
performance disparities.
=============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

from config import EQUITY_ALPHA, MIN_SUBGROUP_SIZE

log = logging.getLogger(__name__)

# Demographic columns we evaluate equity across
EQUITY_DIMENSIONS = [
    ("autism_severity", "Autism Severity"),
    ("verbal_status", "Verbal Status"),
    ("age_group", "Age Group"),
    ("gender", "Gender"),
]


@dataclass
class SubgroupMetrics:
    """Metrics for one subgroup of one demographic dimension."""
    dimension: str
    subgroup: str
    n_samples: int
    accuracy: float
    f1_weighted: float
    f1_macro: float


@dataclass
class EquityTestResult:
    """Statistical test for performance disparity across subgroups."""
    dimension: str
    test_name: str         # "chi2" | "fisher" | "kruskal"
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    subgroup_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class DemographicResult:
    """Full demographic analysis for one model."""
    model_name: str
    subgroup_metrics: List[SubgroupMetrics] = field(default_factory=list)
    equity_tests: List[EquityTestResult] = field(default_factory=list)
    has_demographics: bool = False
    flags: List[str] = field(default_factory=list)


class DemographicEvaluator:
    """Evaluate model performance across demographic subgroups."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [M8-Demo] {msg}")

    def analyse_all(
        self,
        eval_results: list,
        y_test: np.ndarray,
        demographics: Optional[pd.DataFrame],
    ) -> Dict[str, DemographicResult]:
        """
        Analyse demographic breakdown for all evaluated models.

        Parameters
        ----------
        eval_results : List[EvalResult]
        y_test : (n_test,) encoded labels
        demographics : DataFrame with demographic columns, or None

        Returns
        -------
        Dict[model_name, DemographicResult]
        """
        if demographics is None or demographics.empty:
            self._log("No demographics available -- skipping")
            return {}

        results: Dict[str, DemographicResult] = {}

        for er in eval_results:
            if er.skipped or er.y_pred is None:
                continue

            dr = self._analyse_one(er, y_test, demographics)
            results[er.model_name] = dr

            n_sig = sum(1 for t in dr.equity_tests if t.significant)
            self._log(
                f"  {er.model_name}: {len(dr.subgroup_metrics)} subgroups, "
                f"{n_sig} significant disparities"
            )

        return results

    def _analyse_one(
        self, er, y_test: np.ndarray, demographics: pd.DataFrame
    ) -> DemographicResult:
        """Analyse one model across all demographic dimensions."""
        dr = DemographicResult(
            model_name=er.model_name,
            has_demographics=True,
        )
        y_pred = er.y_pred

        for col, dim_label in EQUITY_DIMENSIONS:
            if col not in demographics.columns:
                continue

            values = demographics[col].values
            unique_groups = sorted(
                [g for g in pd.Series(values).dropna().unique()],
                key=str,
            )

            if len(unique_groups) < 2:
                continue

            # Per-subgroup metrics
            subgroup_f1s = {}
            for group in unique_groups:
                mask = values == group
                if mask.sum() < MIN_SUBGROUP_SIZE:
                    continue

                y_t = y_test[mask]
                y_p = y_pred[mask]

                acc = float(accuracy_score(y_t, y_p))
                f1w = float(f1_score(y_t, y_p, average="weighted", zero_division=0))
                f1m = float(f1_score(y_t, y_p, average="macro", zero_division=0))

                sm = SubgroupMetrics(
                    dimension=dim_label,
                    subgroup=str(group),
                    n_samples=int(mask.sum()),
                    accuracy=acc,
                    f1_weighted=f1w,
                    f1_macro=f1m,
                )
                dr.subgroup_metrics.append(sm)
                subgroup_f1s[str(group)] = f1w

            # Equity test across subgroups
            if len(subgroup_f1s) >= 2:
                eq_test = self._run_equity_test(
                    dim_label, y_test, y_pred, values, unique_groups
                )
                if eq_test is not None:
                    eq_test.subgroup_scores = subgroup_f1s
                    dr.equity_tests.append(eq_test)

                    if eq_test.significant:
                        best = max(subgroup_f1s, key=subgroup_f1s.get)
                        worst = min(subgroup_f1s, key=subgroup_f1s.get)
                        gap = subgroup_f1s[best] - subgroup_f1s[worst]
                        dr.flags.append(
                            f"{dim_label}: significant disparity "
                            f"(p={eq_test.p_value:.4f}), "
                            f"gap={gap:.3f} between {best} and {worst}"
                        )

        return dr

    def _run_equity_test(
        self,
        dimension: str,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        group_values: np.ndarray,
        unique_groups: list,
    ) -> Optional[EquityTestResult]:
        """
        Run statistical test for performance disparity.

        Uses Kruskal-Wallis on per-sample correctness across groups
        (non-parametric test for differences in distributions).
        """
        # Per-sample correctness (binary: 0/1)
        correct = (y_pred == y_test).astype(int)

        groups_data = []
        valid_groups = []
        for group in unique_groups:
            mask = group_values == group
            if mask.sum() >= MIN_SUBGROUP_SIZE:
                groups_data.append(correct[mask])
                valid_groups.append(group)

        if len(groups_data) < 2:
            return None

        try:
            if len(groups_data) == 2:
                # Mann-Whitney U test for 2 groups
                stat, p = stats.mannwhitneyu(
                    groups_data[0], groups_data[1], alternative="two-sided"
                )
                # Effect size: rank-biserial correlation
                n1, n2 = len(groups_data[0]), len(groups_data[1])
                effect = 1 - (2 * stat) / (n1 * n2) if (n1 * n2) > 0 else 0.0
                test_name = "mann_whitney"
            else:
                # Kruskal-Wallis for 3+ groups
                stat, p = stats.kruskal(*groups_data)
                # Effect size: epsilon-squared
                n_total = sum(len(g) for g in groups_data)
                k = len(groups_data)
                effect = (stat - k + 1) / (n_total - k) if (n_total - k) > 0 else 0.0
                test_name = "kruskal_wallis"

            return EquityTestResult(
                dimension=dimension,
                test_name=test_name,
                statistic=float(stat),
                p_value=float(p),
                significant=p < EQUITY_ALPHA,
                effect_size=float(effect),
            )
        except Exception as e:
            self._log(f"  Equity test failed for {dimension}: {e}")
            return None

    def build_subgroup_table(
        self, results: Dict[str, DemographicResult]
    ) -> pd.DataFrame:
        """Build a flat table of subgroup metrics across all models."""
        rows = []
        for name, dr in results.items():
            for sm in dr.subgroup_metrics:
                rows.append({
                    "model": name,
                    "dimension": sm.dimension,
                    "subgroup": sm.subgroup,
                    "n_samples": sm.n_samples,
                    "accuracy": sm.accuracy,
                    "f1_weighted": sm.f1_weighted,
                    "f1_macro": sm.f1_macro,
                })
        return pd.DataFrame(rows)

    def build_equity_summary(
        self, results: Dict[str, DemographicResult]
    ) -> pd.DataFrame:
        """Build equity test summary table."""
        rows = []
        for name, dr in results.items():
            for et in dr.equity_tests:
                rows.append({
                    "model": name,
                    "dimension": et.dimension,
                    "test": et.test_name,
                    "statistic": et.statistic,
                    "p_value": et.p_value,
                    "significant": et.significant,
                    "effect_size": et.effect_size,
                })
        return pd.DataFrame(rows)
