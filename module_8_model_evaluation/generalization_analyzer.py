"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  generalization_analyzer.py
=============================================================================
Compare train / validation / test metrics to detect overfitting, underfitting,
and generalisation gaps.  Flags models whose test performance drops more than
OVERFIT_THRESHOLD below validation performance.
=============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    OVERFIT_THRESHOLD, SEVERE_OVERFIT_THRESHOLD,
    PRIMARY_METRIC, COMPARISON_METRICS,
)

log = logging.getLogger(__name__)


@dataclass
class GeneralizationResult:
    """Generalization analysis for one model."""
    model_name: str
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    gaps: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    verdict: str = "OK"  # "OK" | "MILD_OVERFIT" | "SEVERE_OVERFIT" | "UNDERFIT"


class GeneralizationAnalyzer:
    """Detect overfitting by comparing val -> test metric drops."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [M8-Gen] {msg}")

    def analyse_all(
        self,
        eval_results: list,
        split_distributions: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> Dict[str, GeneralizationResult]:
        """
        Analyse generalisation gap for every evaluated model.

        Parameters
        ----------
        eval_results : List[EvalResult]
        split_distributions : {split: {class: count}} from M8DataLoader

        Returns
        -------
        Dict[model_name, GeneralizationResult]
        """
        results: Dict[str, GeneralizationResult] = {}

        for er in eval_results:
            if er.skipped:
                continue

            gr = self._analyse_one(er)
            results[er.model_name] = gr

            self._log(
                f"  {er.model_name}: {gr.verdict} "
                f"(gap={gr.gaps.get(PRIMARY_METRIC, 0):+.4f})"
            )

        # Distribution drift check
        if split_distributions:
            drift = self._check_distribution_drift(split_distributions)
            if drift:
                self._log(f"  Distribution drift detected: {drift}")

        return results

    def _analyse_one(self, er) -> GeneralizationResult:
        """Analyse a single model's generalisation."""
        val = er.val_metrics or {}
        test = er.test_metrics or {}

        gr = GeneralizationResult(
            model_name=er.model_name,
            val_metrics={k: float(val.get(k, 0)) for k in COMPARISON_METRICS
                         if k in val},
            test_metrics={k: float(test.get(k, 0)) for k in COMPARISON_METRICS
                          if k in test},
        )

        # Compute gaps: val - test (positive = overfit)
        for metric in COMPARISON_METRICS:
            v = val.get(metric)
            t = test.get(metric)
            if v is not None and t is not None:
                gap = float(v) - float(t)
                gr.gaps[metric] = gap

        # Determine verdict based on primary metric gap
        primary_gap = gr.gaps.get(PRIMARY_METRIC, 0)

        if primary_gap > SEVERE_OVERFIT_THRESHOLD:
            gr.verdict = "SEVERE_OVERFIT"
            gr.flags.append(
                f"{PRIMARY_METRIC} dropped {primary_gap:.1%} from val to test "
                f"(threshold: {SEVERE_OVERFIT_THRESHOLD:.0%})"
            )
        elif primary_gap > OVERFIT_THRESHOLD:
            gr.verdict = "MILD_OVERFIT"
            gr.flags.append(
                f"{PRIMARY_METRIC} dropped {primary_gap:.1%} from val to test "
                f"(threshold: {OVERFIT_THRESHOLD:.0%})"
            )
        else:
            gr.verdict = "OK"

        # Check individual metric gaps
        for metric, gap in gr.gaps.items():
            if metric == PRIMARY_METRIC:
                continue
            if gap > SEVERE_OVERFIT_THRESHOLD:
                gr.flags.append(
                    f"{metric} severe overfit: val->test drop = {gap:.1%}"
                )
            elif gap > OVERFIT_THRESHOLD:
                gr.flags.append(
                    f"{metric} mild overfit: val->test drop = {gap:.1%}"
                )

        # Underfit detection: both val and test are poor
        primary_test = test.get(PRIMARY_METRIC, 0)
        primary_val = val.get(PRIMARY_METRIC, 0)
        if primary_test < 0.3 and primary_val < 0.3:
            gr.verdict = "UNDERFIT"
            gr.flags.append(
                f"Both val ({primary_val:.3f}) and test ({primary_test:.3f}) "
                f"{PRIMARY_METRIC} below 0.3 -- model may be underfitting"
            )

        return gr

    def _check_distribution_drift(
        self, splits: Dict[str, Dict[str, int]]
    ) -> Optional[str]:
        """
        Check if class distribution differs significantly across splits.

        Returns a warning string if drift is detected, else None.
        """
        if "train" not in splits or "test" not in splits:
            return None

        train_dist = splits["train"]
        test_dist = splits["test"]

        all_classes = set(train_dist.keys()) | set(test_dist.keys())
        train_total = sum(train_dist.values())
        test_total = sum(test_dist.values())

        if train_total == 0 or test_total == 0:
            return "Empty split detected"

        max_drift = 0.0
        drift_class = ""
        for cls in all_classes:
            train_pct = train_dist.get(cls, 0) / train_total
            test_pct = test_dist.get(cls, 0) / test_total
            drift = abs(train_pct - test_pct)
            if drift > max_drift:
                max_drift = drift
                drift_class = cls

        if max_drift > 0.05:
            return (
                f"Class '{drift_class}' proportion differs by {max_drift:.1%} "
                f"between train and test splits"
            )
        return None

    def build_gap_table(
        self, results: Dict[str, GeneralizationResult]
    ) -> pd.DataFrame:
        """Build a DataFrame of val/test/gap per model per metric."""
        rows = []
        for name, gr in results.items():
            row = {"model": name, "verdict": gr.verdict}
            for metric in COMPARISON_METRICS:
                row[f"val_{metric}"] = gr.val_metrics.get(metric)
                row[f"test_{metric}"] = gr.test_metrics.get(metric)
                row[f"gap_{metric}"] = gr.gaps.get(metric)
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(f"gap_{PRIMARY_METRIC}", ascending=True)
        return df

    def build_gap_matrix(
        self, results: Dict[str, GeneralizationResult]
    ) -> pd.DataFrame:
        """
        Build a (models x metrics) matrix of gaps for heatmap plotting.
        Positive values = overfit, negative = test outperforms val.
        """
        models = sorted(results.keys())
        data = {}
        for metric in COMPARISON_METRICS:
            data[metric] = [
                results[m].gaps.get(metric, np.nan) for m in models
            ]
        return pd.DataFrame(data, index=models)
