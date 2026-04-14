"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  error_analyzer.py
=============================================================================
Characterise model errors: most confused class pairs, per-user error rates,
failure mode analysis, and misclassification pattern discovery.
=============================================================================
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

log = logging.getLogger(__name__)


@dataclass
class ConfusedPair:
    """A frequently confused class pair."""
    true_class: str
    pred_class: str
    count: int
    fraction_of_errors: float


@dataclass
class UserErrorProfile:
    """Error profile for one user."""
    user_id: str
    n_samples: int
    n_errors: int
    error_rate: float
    most_confused: Optional[str] = None  # Most common misclassification


@dataclass
class FailureMode:
    """A characterised failure mode."""
    description: str
    affected_classes: List[str]
    n_errors: int
    feature_signature: Dict[str, float] = field(default_factory=dict)


@dataclass
class ErrorAnalysis:
    """Full error analysis for one model."""
    model_name: str
    n_total: int = 0
    n_errors: int = 0
    error_rate: float = 0.0
    confused_pairs: List[ConfusedPair] = field(default_factory=list)
    user_profiles: List[UserErrorProfile] = field(default_factory=list)
    failure_modes: List[FailureMode] = field(default_factory=list)
    error_by_class: Dict[str, float] = field(default_factory=dict)
    # For Sankey / flow visualisation
    misclassification_flow: List[Tuple[str, str, int]] = field(default_factory=list)


class ErrorAnalyzer:
    """Characterise errors for models on the test set."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [M8-Err] {msg}")

    def analyse_all(
        self,
        eval_results: list,
        y_test: np.ndarray,
        class_names: List[str],
        X_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        user_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, ErrorAnalysis]:
        """
        Run error analysis for all evaluated models.

        Parameters
        ----------
        eval_results : List[EvalResult]
        y_test : (n_test,) encoded labels
        class_names : list of class names
        X_test : (n_test, n_features) for feature-level analysis
        feature_names : feature column names
        user_ids : (n_test,) user_id per sample

        Returns
        -------
        Dict[model_name, ErrorAnalysis]
        """
        results: Dict[str, ErrorAnalysis] = {}

        for er in eval_results:
            if er.skipped or er.y_pred is None:
                continue

            ea = self._analyse_one(
                er, y_test, class_names, X_test, feature_names, user_ids
            )
            results[er.model_name] = ea

            self._log(
                f"  {er.model_name}: {ea.n_errors}/{ea.n_total} errors "
                f"({ea.error_rate:.1%}), top pair: "
                f"{ea.confused_pairs[0].true_class}->{ea.confused_pairs[0].pred_class}"
                if ea.confused_pairs else
                f"  {er.model_name}: {ea.n_errors}/{ea.n_total} errors"
            )

        return results

    def _analyse_one(
        self, er, y_test: np.ndarray, class_names: List[str],
        X_test: Optional[np.ndarray],
        feature_names: Optional[List[str]],
        user_ids: Optional[np.ndarray],
    ) -> ErrorAnalysis:
        """Analyse errors for a single model."""
        y_pred = er.y_pred
        n_total = len(y_test)
        error_mask = y_pred != y_test
        n_errors = int(error_mask.sum())

        ea = ErrorAnalysis(
            model_name=er.model_name,
            n_total=n_total,
            n_errors=n_errors,
            error_rate=n_errors / n_total if n_total > 0 else 0.0,
        )

        # 1) Confused pairs
        ea.confused_pairs = self._find_confused_pairs(
            y_test, y_pred, class_names, n_errors
        )

        # 2) Misclassification flow (for Sankey diagram)
        ea.misclassification_flow = self._build_flow(
            y_test, y_pred, class_names
        )

        # 3) Per-class error rates
        for i, cls in enumerate(class_names):
            cls_mask = y_test == i
            if cls_mask.sum() > 0:
                cls_errors = (y_pred[cls_mask] != y_test[cls_mask]).sum()
                ea.error_by_class[cls] = float(cls_errors / cls_mask.sum())

        # 4) Per-user error profiles
        if user_ids is not None:
            ea.user_profiles = self._user_error_profiles(
                y_test, y_pred, user_ids, class_names
            )

        # 5) Failure mode characterisation
        if X_test is not None and feature_names is not None:
            ea.failure_modes = self._characterise_failures(
                y_test, y_pred, X_test, feature_names, class_names, error_mask
            )

        return ea

    def _find_confused_pairs(
        self, y_test: np.ndarray, y_pred: np.ndarray,
        class_names: List[str], n_errors: int,
    ) -> List[ConfusedPair]:
        """Find the most frequently confused class pairs."""
        error_mask = y_pred != y_test
        if not error_mask.any():
            return []

        true_err = y_test[error_mask]
        pred_err = y_pred[error_mask]

        pair_counts = Counter(zip(true_err, pred_err))
        pairs = []
        for (t, p), count in pair_counts.most_common(20):
            t_name = class_names[t] if t < len(class_names) else str(t)
            p_name = class_names[p] if p < len(class_names) else str(p)
            pairs.append(ConfusedPair(
                true_class=t_name,
                pred_class=p_name,
                count=count,
                fraction_of_errors=count / n_errors if n_errors > 0 else 0.0,
            ))

        return pairs

    def _build_flow(
        self, y_test: np.ndarray, y_pred: np.ndarray,
        class_names: List[str],
    ) -> List[Tuple[str, str, int]]:
        """Build misclassification flow for Sankey diagram."""
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
        flows = []
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    flows.append((class_names[i], class_names[j], int(cm[i, j])))
        # Sort by count descending
        flows.sort(key=lambda x: x[2], reverse=True)
        return flows

    def _user_error_profiles(
        self, y_test: np.ndarray, y_pred: np.ndarray,
        user_ids: np.ndarray, class_names: List[str],
    ) -> List[UserErrorProfile]:
        """Compute per-user error profiles."""
        profiles = []
        unique_users = sorted(set(user_ids))

        for uid in unique_users:
            mask = user_ids == uid
            n = int(mask.sum())
            if n == 0:
                continue

            errors = int((y_pred[mask] != y_test[mask]).sum())

            # Most common misclassification for this user
            err_mask = mask & (y_pred != y_test)
            most_confused = None
            if err_mask.any():
                pred_errors = y_pred[err_mask]
                if len(pred_errors) > 0:
                    mc_idx = Counter(pred_errors).most_common(1)[0][0]
                    most_confused = class_names[mc_idx] if mc_idx < len(class_names) else str(mc_idx)

            profiles.append(UserErrorProfile(
                user_id=str(uid),
                n_samples=n,
                n_errors=errors,
                error_rate=errors / n if n > 0 else 0.0,
                most_confused=most_confused,
            ))

        # Sort by error rate descending
        profiles.sort(key=lambda p: p.error_rate, reverse=True)
        return profiles

    def _characterise_failures(
        self, y_test: np.ndarray, y_pred: np.ndarray,
        X_test: np.ndarray, feature_names: List[str],
        class_names: List[str], error_mask: np.ndarray,
    ) -> List[FailureMode]:
        """
        Characterise failure modes by comparing feature distributions
        of correctly vs incorrectly classified samples.
        """
        modes = []

        if not error_mask.any():
            return modes

        X_correct = X_test[~error_mask]
        X_wrong = X_test[error_mask]

        # Find features where errors have significantly different values
        diff_features = []
        for j, fname in enumerate(feature_names):
            if X_correct.shape[0] < 5 or X_wrong.shape[0] < 5:
                continue
            correct_vals = X_correct[:, j]
            wrong_vals = X_wrong[:, j]

            # Skip if all same value
            if np.std(correct_vals) < 1e-10 and np.std(wrong_vals) < 1e-10:
                continue

            try:
                stat, p = stats_mannwhitney(correct_vals, wrong_vals)
                if p < 0.01:  # Strict threshold for feature-level
                    effect = (np.mean(wrong_vals) - np.mean(correct_vals))
                    std_pool = np.std(np.concatenate([correct_vals, wrong_vals]))
                    if std_pool > 0:
                        cohens_d = effect / std_pool
                    else:
                        cohens_d = 0.0
                    diff_features.append((fname, cohens_d, p))
            except Exception:
                continue

        # Sort by absolute effect size
        diff_features.sort(key=lambda x: abs(x[1]), reverse=True)

        if diff_features:
            top_features = diff_features[:5]
            sig = {f: d for f, d, _ in top_features}
            modes.append(FailureMode(
                description=(
                    f"Errors associated with {len(diff_features)} features "
                    f"having significantly different distributions. "
                    f"Top: {top_features[0][0]} (d={top_features[0][1]:.2f})"
                ),
                affected_classes=[],
                n_errors=int(error_mask.sum()),
                feature_signature=sig,
            ))

        # Per-class failure modes
        for i, cls in enumerate(class_names):
            cls_errors = error_mask & (y_test == i)
            if cls_errors.sum() < 5:
                continue

            cls_correct = (~error_mask) & (y_test == i)
            if cls_correct.sum() < 5:
                continue

            # What do errors on this class get predicted as?
            pred_dist = Counter(y_pred[cls_errors])
            top_pred = pred_dist.most_common(1)[0]
            top_pred_name = class_names[top_pred[0]] if top_pred[0] < len(class_names) else str(top_pred[0])

            modes.append(FailureMode(
                description=(
                    f"'{cls}' misclassified as '{top_pred_name}' "
                    f"({top_pred[1]}/{cls_errors.sum()} errors)"
                ),
                affected_classes=[cls, top_pred_name],
                n_errors=int(cls_errors.sum()),
            ))

        return modes

    def build_confused_pairs_table(
        self, results: Dict[str, ErrorAnalysis]
    ) -> pd.DataFrame:
        """Build a table of top confused pairs across all models."""
        rows = []
        for name, ea in results.items():
            for cp in ea.confused_pairs[:10]:
                rows.append({
                    "model": name,
                    "true_class": cp.true_class,
                    "pred_class": cp.pred_class,
                    "count": cp.count,
                    "frac_of_errors": cp.fraction_of_errors,
                })
        return pd.DataFrame(rows)

    def build_user_error_table(
        self, results: Dict[str, ErrorAnalysis]
    ) -> pd.DataFrame:
        """Build per-user error rate table."""
        rows = []
        for name, ea in results.items():
            for up in ea.user_profiles:
                rows.append({
                    "model": name,
                    "user_id": up.user_id,
                    "n_samples": up.n_samples,
                    "n_errors": up.n_errors,
                    "error_rate": up.error_rate,
                    "most_confused": up.most_confused,
                })
        return pd.DataFrame(rows)


def stats_mannwhitney(a: np.ndarray, b: np.ndarray):
    """Wrapper for scipy Mann-Whitney U test."""
    from scipy.stats import mannwhitneyu
    return mannwhitneyu(a, b, alternative="two-sided")
