"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  test_evaluator.py
=============================================================================
Run all loaded models on the held-out test set and compute full metric suites.
=============================================================================
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, confusion_matrix, classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from config import PRIMARY_METRIC, HIGH_PRIORITY_STATES, COMPARISON_METRICS

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of evaluating one model on the test set."""
    model_name: str
    model_type: str
    framework: str
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    y_pred: Optional[np.ndarray] = None
    y_prob: Optional[np.ndarray] = None
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    inference_time_s: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


class TestEvaluator:
    """Evaluate models on held-out test data."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [M8-Eval] {msg}")

    def evaluate_all(
        self,
        models: dict,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str],
        val_metrics_by_model: Dict[str, dict],
    ) -> List[EvalResult]:
        """
        Run every loaded model on the test set.

        Parameters
        ----------
        models : dict of {name: LoadedModel}
        X_test : (n_test, n_features) feature matrix
        y_test : (n_test,) encoded labels
        class_names : list of class names (index = label)
        val_metrics_by_model : {name: metrics.json dict} from M7

        Returns
        -------
        List[EvalResult]
        """
        results = []

        for name, model in models.items():
            self._log(f"Evaluating: {name}")
            result = self._evaluate_one(
                model, X_test, y_test, class_names,
                val_metrics_by_model.get(name, {}),
            )
            results.append(result)

        # Sort by primary metric descending
        results.sort(
            key=lambda r: r.test_metrics.get(PRIMARY_METRIC, 0),
            reverse=True,
        )
        return results

    def _evaluate_one(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str],
        val_metrics: Dict[str, Any],
    ) -> EvalResult:
        """Evaluate a single model."""
        result = EvalResult(
            model_name=model.name,
            model_type=model.model_type,
            framework=model.framework,
            val_metrics=val_metrics,
        )

        # Skip non-tabular models (sequence/tft/tpp need special data)
        if model.data_format not in ("tabular",):
            result.skipped = True
            result.skip_reason = (
                f"Data format '{model.data_format}' requires specialised "
                f"input -- tabular evaluation only in M8 v1.0"
            )
            self._log(f"  {model.name}: SKIPPED ({result.skip_reason})")
            return result

        try:
            # Predict
            t0 = time.time()
            y_pred = model.predict(X_test)
            result.inference_time_s = time.time() - t0

            # Probabilities
            y_prob = model.predict_proba(X_test)

            result.y_pred = y_pred
            result.y_prob = y_prob

            # Compute metrics
            result.test_metrics = self._compute_metrics(
                y_test, y_pred, y_prob, class_names
            )

            f1 = result.test_metrics.get("f1_weighted", 0)
            acc = result.test_metrics.get("accuracy", 0)
            self._log(f"  {model.name}: F1w={f1:.4f}, Acc={acc:.4f} "
                      f"({result.inference_time_s:.3f}s)")

        except Exception as e:
            result.skipped = True
            result.skip_reason = str(e)
            self._log(f"  {model.name}: ERROR -- {e}")

        return result

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        class_names: List[str],
    ) -> Dict[str, Any]:
        """Compute full metric suite on test predictions."""
        metrics: Dict[str, Any] = {}

        # Global metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["cohens_kappa"] = float(cohen_kappa_score(y_true, y_pred))

        # Confusion matrix -- use explicit labels so shape matches class_names
        all_labels = list(range(len(class_names)))
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["confusion_labels"] = class_names

        # Per-class metrics
        per_class = {}
        report = classification_report(
            y_true, y_pred, labels=all_labels,
            target_names=class_names,
            output_dict=True, zero_division=0,
        )
        for cls_name in class_names:
            if cls_name in report:
                cls_report = report[cls_name]
                idx = class_names.index(cls_name)
                # Sensitivity = recall = TP / (TP + FN)
                tp = cm[idx, idx] if idx < cm.shape[0] else 0
                fn = cm[idx, :].sum() - tp if idx < cm.shape[0] else 0
                fp = cm[:, idx].sum() - tp if idx < cm.shape[1] else 0
                tn = cm.sum() - tp - fn - fp
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                per_class[cls_name] = {
                    "precision": cls_report["precision"],
                    "recall": cls_report["recall"],
                    "f1": cls_report["f1-score"],
                    "support": int(cls_report["support"]),
                    "sensitivity": float(sensitivity),
                    "specificity": float(specificity),
                }
        metrics["per_class"] = per_class

        # High-priority false negative rates
        fnr_hp = {}
        for state in HIGH_PRIORITY_STATES:
            if state in per_class:
                fnr_hp[state] = 1.0 - per_class[state]["sensitivity"]
        metrics["fnr_high_priority"] = fnr_hp

        # AUC-ROC (requires probabilities)
        if y_prob is not None and len(class_names) > 2:
            try:
                y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
                # Handle case where probabilities have different shape
                if y_prob.shape[1] == y_bin.shape[1]:
                    auc_per_class = {}
                    for i, cls_name in enumerate(class_names):
                        if y_bin[:, i].sum() > 0:
                            try:
                                auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
                                auc_per_class[cls_name] = float(auc)
                            except ValueError:
                                auc_per_class[cls_name] = None
                    metrics["auc_roc_per_class"] = auc_per_class
                    # Macro AUC
                    valid_aucs = [v for v in auc_per_class.values() if v is not None]
                    metrics["auc_roc_macro"] = float(np.mean(valid_aucs)) if valid_aucs else None
            except Exception:
                metrics["auc_roc_per_class"] = {}
                metrics["auc_roc_macro"] = None

        return metrics

    def build_comparison_table(
        self, results: List[EvalResult]
    ) -> pd.DataFrame:
        """Build a comparison table of all models on test metrics."""
        rows = []
        for r in results:
            if r.skipped:
                continue
            row = {"model": r.model_name, "framework": r.framework}
            for metric in COMPARISON_METRICS:
                row[metric] = r.test_metrics.get(metric)
            row["inference_s"] = r.inference_time_s
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(PRIMARY_METRIC, ascending=False)
        return df
