"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  calibration_analyzer.py
=============================================================================
Assess confidence calibration of probabilistic classifiers on the test set.
Computes Expected Calibration Error (ECE), Brier score, reliability diagrams,
and confidence distribution statistics.
=============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import ECE_BINS, CALIBRATION_GOOD_THRESHOLD

log = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Calibration analysis for one model."""
    model_name: str
    ece: Optional[float] = None            # Expected Calibration Error
    mce: Optional[float] = None            # Maximum Calibration Error
    brier_score: Optional[float] = None    # Multi-class Brier score
    mean_confidence: Optional[float] = None
    median_confidence: Optional[float] = None
    overconfidence_frac: Optional[float] = None  # fraction where conf > acc
    # Reliability diagram data (for plotting)
    bin_midpoints: Optional[np.ndarray] = None
    bin_accuracies: Optional[np.ndarray] = None
    bin_confidences: Optional[np.ndarray] = None
    bin_counts: Optional[np.ndarray] = None
    # Per-class calibration
    per_class_ece: Dict[str, float] = field(default_factory=dict)
    verdict: str = "NO_PROBS"  # "WELL_CALIBRATED" | "MISCALIBRATED" | "NO_PROBS"
    skipped: bool = False
    skip_reason: str = ""


class CalibrationAnalyzer:
    """Analyse confidence calibration on test predictions."""

    def __init__(self, n_bins: int = ECE_BINS, verbose: bool = True):
        self.n_bins = n_bins
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [M8-Cal] {msg}")

    def analyse_all(
        self,
        eval_results: list,
        y_test: np.ndarray,
        class_names: List[str],
    ) -> Dict[str, CalibrationResult]:
        """
        Analyse calibration for every model that produced probabilities.

        Parameters
        ----------
        eval_results : List[EvalResult]
        y_test : (n_test,) encoded labels
        class_names : list of class names

        Returns
        -------
        Dict[model_name, CalibrationResult]
        """
        results: Dict[str, CalibrationResult] = {}

        for er in eval_results:
            if er.skipped:
                continue

            cr = self._analyse_one(er, y_test, class_names)
            results[er.model_name] = cr

            if cr.ece is not None:
                self._log(
                    f"  {er.model_name}: ECE={cr.ece:.4f}, "
                    f"Brier={cr.brier_score:.4f} -> {cr.verdict}"
                )
            else:
                self._log(f"  {er.model_name}: no probabilities available")

        return results

    def _analyse_one(
        self, er, y_test: np.ndarray, class_names: List[str]
    ) -> CalibrationResult:
        """Analyse calibration for a single model."""
        cr = CalibrationResult(model_name=er.model_name)

        if er.y_prob is None:
            cr.skipped = True
            cr.skip_reason = "No probability output"
            return cr

        y_prob = er.y_prob

        # Overall ECE + reliability diagram
        ece_data = self._compute_ece(y_test, y_prob)
        cr.ece = ece_data["ece"]
        cr.mce = ece_data["mce"]
        cr.bin_midpoints = ece_data["bin_midpoints"]
        cr.bin_accuracies = ece_data["bin_accuracies"]
        cr.bin_confidences = ece_data["bin_confidences"]
        cr.bin_counts = ece_data["bin_counts"]
        cr.overconfidence_frac = ece_data["overconfidence_frac"]

        # Brier score (multi-class)
        cr.brier_score = self._compute_brier(y_test, y_prob, len(class_names))

        # Confidence statistics
        max_probs = y_prob.max(axis=1)
        cr.mean_confidence = float(np.mean(max_probs))
        cr.median_confidence = float(np.median(max_probs))

        # Per-class ECE
        for i, cls_name in enumerate(class_names):
            mask = y_test == i
            if mask.sum() < 10:
                continue
            cls_probs = y_prob[mask]
            cls_labels = y_test[mask]
            cls_ece = self._compute_ece(cls_labels, cls_probs)
            cr.per_class_ece[cls_name] = cls_ece["ece"]

        # Verdict
        if cr.ece <= CALIBRATION_GOOD_THRESHOLD:
            cr.verdict = "WELL_CALIBRATED"
        else:
            cr.verdict = "MISCALIBRATED"

        return cr

    def _compute_ece(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute Expected Calibration Error using equal-width binning.

        For each sample, confidence = max(predicted probability).
        Samples are binned by confidence. ECE = weighted average of
        |accuracy - confidence| per bin.
        """
        confidences = y_prob.max(axis=1)
        predictions = y_prob.argmax(axis=1)
        accuracies = (predictions == y_true).astype(float)

        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

        bin_accs = np.zeros(self.n_bins)
        bin_confs = np.zeros(self.n_bins)
        bin_counts = np.zeros(self.n_bins, dtype=int)

        for b in range(self.n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            if b == self.n_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)

            if mask.sum() > 0:
                bin_accs[b] = accuracies[mask].mean()
                bin_confs[b] = confidences[mask].mean()
                bin_counts[b] = mask.sum()

        n_total = len(y_true)
        ece = 0.0
        mce = 0.0
        overconf_bins = 0
        total_bins_used = 0

        for b in range(self.n_bins):
            if bin_counts[b] > 0:
                gap = abs(bin_accs[b] - bin_confs[b])
                ece += (bin_counts[b] / n_total) * gap
                mce = max(mce, gap)
                total_bins_used += 1
                if bin_confs[b] > bin_accs[b]:
                    overconf_bins += 1

        overconf_frac = (
            overconf_bins / total_bins_used if total_bins_used > 0 else 0.0
        )

        return {
            "ece": float(ece),
            "mce": float(mce),
            "bin_midpoints": bin_midpoints,
            "bin_accuracies": bin_accs,
            "bin_confidences": bin_confs,
            "bin_counts": bin_counts,
            "overconfidence_frac": float(overconf_frac),
        }

    def _compute_brier(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_classes: int
    ) -> float:
        """
        Multi-class Brier score: mean squared difference between
        predicted probabilities and one-hot true labels.
        Lower is better; 0 = perfect, 1 = worst.
        """
        one_hot = np.zeros((len(y_true), n_classes))
        for i, label in enumerate(y_true):
            if 0 <= label < n_classes:
                one_hot[i, label] = 1.0

        # Ensure shapes match
        if y_prob.shape[1] != n_classes:
            return float("nan")

        brier = np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))
        return float(brier)

    def build_comparison_table(
        self, results: Dict[str, CalibrationResult]
    ) -> pd.DataFrame:
        """Build a calibration comparison table."""
        rows = []
        for name, cr in results.items():
            if cr.skipped:
                continue
            rows.append({
                "model": name,
                "ECE": cr.ece,
                "MCE": cr.mce,
                "Brier": cr.brier_score,
                "mean_conf": cr.mean_confidence,
                "median_conf": cr.median_confidence,
                "overconf_frac": cr.overconfidence_frac,
                "verdict": cr.verdict,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("ECE", ascending=True)
        return df
