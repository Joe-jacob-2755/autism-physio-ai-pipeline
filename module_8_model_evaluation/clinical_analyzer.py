"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  clinical_analyzer.py
=============================================================================
Clinical utility analysis: cost-weighted confusion matrix, Number Needed to
Screen (NNS), high-priority false-negative rates, and sensitivity-specificity
trade-offs for clinically important states.
=============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    CLINICAL_COST_WEIGHTS, HIGH_PRIORITY_STATES,
    NNS_PREVALENCE_ASSUMED,
)

log = logging.getLogger(__name__)


@dataclass
class ClinicalResult:
    """Clinical utility analysis for one model."""
    model_name: str
    total_clinical_cost: float = 0.0
    cost_per_sample: float = 0.0
    cost_by_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fnr_high_priority: Dict[str, float] = field(default_factory=dict)
    sensitivity_by_class: Dict[str, float] = field(default_factory=dict)
    specificity_by_class: Dict[str, float] = field(default_factory=dict)
    nns_by_class: Dict[str, float] = field(default_factory=dict)
    clinical_score: float = 0.0  # Composite clinical utility score


class ClinicalAnalyzer:
    """Assess clinical utility of test-set predictions."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [M8-Clin] {msg}")

    def analyse_all(
        self,
        eval_results: list,
        y_test: np.ndarray,
        class_names: List[str],
    ) -> Dict[str, ClinicalResult]:
        """
        Run clinical utility analysis for all evaluated models.

        Parameters
        ----------
        eval_results : List[EvalResult]
        y_test : (n_test,) encoded labels
        class_names : list of class names

        Returns
        -------
        Dict[model_name, ClinicalResult]
        """
        results: Dict[str, ClinicalResult] = {}

        for er in eval_results:
            if er.skipped or er.y_pred is None:
                continue

            cr = self._analyse_one(er, y_test, class_names)
            results[er.model_name] = cr

            self._log(
                f"  {er.model_name}: cost/sample={cr.cost_per_sample:.3f}, "
                f"clinical_score={cr.clinical_score:.3f}"
            )

        return results

    def _analyse_one(
        self, er, y_test: np.ndarray, class_names: List[str]
    ) -> ClinicalResult:
        """Analyse clinical utility for a single model."""
        cr = ClinicalResult(model_name=er.model_name)
        y_pred = er.y_pred
        n_classes = len(class_names)

        # Build confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))

        # Per-class analysis
        total_cost = 0.0

        for i, cls_name in enumerate(class_names):
            tp = cm[i, i] if i < cm.shape[0] else 0
            fn = cm[i, :].sum() - tp if i < cm.shape[0] else 0
            fp = cm[:, i].sum() - tp if i < cm.shape[1] else 0
            tn = cm.sum() - tp - fn - fp

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            cr.sensitivity_by_class[cls_name] = float(sensitivity)
            cr.specificity_by_class[cls_name] = float(specificity)

            # Clinical cost
            weights = CLINICAL_COST_WEIGHTS.get(
                cls_name, CLINICAL_COST_WEIGHTS["default"]
            )
            fn_cost = fn * weights["fn_cost"]
            fp_cost = fp * weights["fp_cost"]
            cls_cost = fn_cost + fp_cost
            total_cost += cls_cost

            cr.cost_by_class[cls_name] = {
                "fn_count": int(fn),
                "fp_count": int(fp),
                "fn_cost": float(fn_cost),
                "fp_cost": float(fp_cost),
                "total_cost": float(cls_cost),
            }

            # FNR for high-priority states
            if cls_name in HIGH_PRIORITY_STATES:
                fnr = 1.0 - sensitivity
                cr.fnr_high_priority[cls_name] = float(fnr)

            # Number Needed to Screen (NNS)
            # NNS = 1 / (Sensitivity x Prevalence)
            # Using assumed prevalence for each class
            if sensitivity > 0:
                prevalence = (tp + fn) / len(y_test) if len(y_test) > 0 else NNS_PREVALENCE_ASSUMED
                prevalence = max(prevalence, 0.001)  # Guard against zero
                nns = 1.0 / (sensitivity * prevalence)
                cr.nns_by_class[cls_name] = float(nns)
            else:
                cr.nns_by_class[cls_name] = float("inf")

        cr.total_clinical_cost = total_cost
        cr.cost_per_sample = total_cost / len(y_test) if len(y_test) > 0 else 0.0

        # Composite clinical score: weighted harmonic mean of
        # high-priority sensitivities (higher = better)
        hp_sens = [
            cr.sensitivity_by_class.get(s, 0.0)
            for s in HIGH_PRIORITY_STATES
            if s in cr.sensitivity_by_class
        ]
        if hp_sens and all(s > 0 for s in hp_sens):
            cr.clinical_score = float(len(hp_sens) / sum(1.0 / s for s in hp_sens))
        elif hp_sens:
            cr.clinical_score = float(np.mean(hp_sens))
        else:
            cr.clinical_score = 0.0

        return cr

    def build_cost_table(
        self, results: Dict[str, ClinicalResult], class_names: List[str]
    ) -> pd.DataFrame:
        """Build a cost comparison table (models x classes)."""
        rows = []
        for name, cr in results.items():
            row = {"model": name, "total_cost": cr.total_clinical_cost,
                   "cost_per_sample": cr.cost_per_sample,
                   "clinical_score": cr.clinical_score}
            for cls in class_names:
                if cls in cr.cost_by_class:
                    row[f"{cls}_cost"] = cr.cost_by_class[cls]["total_cost"]
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("total_cost", ascending=True)
        return df

    def build_fnr_table(
        self, results: Dict[str, ClinicalResult]
    ) -> pd.DataFrame:
        """Build FNR comparison for high-priority states."""
        rows = []
        for name, cr in results.items():
            row = {"model": name}
            for state in sorted(HIGH_PRIORITY_STATES):
                row[f"FNR_{state}"] = cr.fnr_high_priority.get(state)
                row[f"Sens_{state}"] = cr.sensitivity_by_class.get(state)
            row["clinical_score"] = cr.clinical_score
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("clinical_score", ascending=False)
        return df

    def build_sensitivity_specificity_table(
        self, results: Dict[str, ClinicalResult], class_names: List[str]
    ) -> pd.DataFrame:
        """Build sensitivity/specificity table for all models and classes."""
        rows = []
        for name, cr in results.items():
            for cls in class_names:
                rows.append({
                    "model": name,
                    "class": cls,
                    "sensitivity": cr.sensitivity_by_class.get(cls),
                    "specificity": cr.specificity_by_class.get(cls),
                    "is_high_priority": cls in HIGH_PRIORITY_STATES,
                })
        return pd.DataFrame(rows)
