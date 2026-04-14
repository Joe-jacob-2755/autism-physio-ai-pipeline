"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  class_balancer.py
=============================================================================
Handle class imbalance in training data.
Strategies: SMOTE, random oversampling, class weights.
Only applied to training set -- never val/test.
=============================================================================
"""
from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Dict, Optional, Tuple

from config import (
    BALANCE_STRATEGY, SMOTE_K_NEIGHBORS, MIN_SAMPLES_FOR_SMOTE,
    RANDOM_SEED,
)


class ClassBalancer:
    """
    Balance class distribution in training data.

    Parameters
    ----------
    strategy : "smote" | "class_weight" | "oversample" | "none"
    seed     : random seed
    """

    def __init__(self, strategy: str = BALANCE_STRATEGY,
                 seed: int = RANDOM_SEED, verbose: bool = True):
        self.strategy = strategy
        self.seed = seed
        self.verbose = verbose
        self._before: Optional[Dict] = None
        self._after: Optional[Dict] = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [ClassBalancer] {msg}")

    def balance_tabular(
        self, X: np.ndarray, y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance tabular (2D) data. Returns (X_balanced, y_balanced)."""
        self._before = dict(Counter(y))

        if self.strategy == "none":
            self._after = self._before.copy()
            self._log("No balancing applied (strategy=none)")
            return X, y

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self._log("Only 1 class present -- cannot balance")
            self._after = self._before.copy()
            return X, y

        min_count = min(Counter(y).values())

        if self.strategy == "smote" and min_count >= MIN_SAMPLES_FOR_SMOTE:
            return self._apply_smote(X, y)
        elif self.strategy == "smote":
            self._log(f"SMOTE requires >= {MIN_SAMPLES_FOR_SMOTE} per class "
                       f"(min={min_count}). Falling back to oversample.")
            return self._apply_oversample(X, y)
        elif self.strategy == "oversample":
            return self._apply_oversample(X, y)
        elif self.strategy == "class_weight":
            # Class weights don't change data -- just log
            self._after = self._before.copy()
            weights = self.compute_class_weights(y)
            self._log(f"Class weights: {weights}")
            return X, y
        else:
            self._after = self._before.copy()
            return X, y

    def balance_sequences(
        self, X: np.ndarray, y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Balance sequence (3D) data by random oversampling."""
        self._before = dict(Counter(y))
        unique_classes = np.unique(y)

        if len(unique_classes) < 2:
            self._after = self._before.copy()
            return X, y

        max_count = max(Counter(y).values())
        rng = np.random.default_rng(self.seed)

        X_parts, y_parts = [X], [y]
        for cls in unique_classes:
            cls_idx = np.where(y == cls)[0]
            n_needed = max_count - len(cls_idx)
            if n_needed > 0:
                oversample_idx = rng.choice(cls_idx, size=n_needed,
                                            replace=True)
                X_parts.append(X[oversample_idx])
                y_parts.append(y[oversample_idx])

        X_bal = np.concatenate(X_parts, axis=0)
        y_bal = np.concatenate(y_parts, axis=0)
        shuffle_idx = rng.permutation(len(y_bal))
        X_bal, y_bal = X_bal[shuffle_idx], y_bal[shuffle_idx]

        self._after = dict(Counter(y_bal))
        self._log(f"Sequence oversample: {self._before} -> {self._after}")
        return X_bal, y_bal

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute inverse-frequency class weights for loss functions."""
        counts = Counter(y)
        n_samples = len(y)
        n_classes = len(counts)
        weights = {}
        for cls, count in counts.items():
            weights[cls] = n_samples / (n_classes * count)
        return weights

    def report(self) -> dict:
        """Return before/after class distribution."""
        return {"before": self._before, "after": self._after,
                "strategy": self.strategy}

    # ── Private ──────────────────────────────────────────────────────────

    def _apply_smote(self, X, y):
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            self._log("imbalanced-learn not installed. "
                       "Falling back to oversample.")
            return self._apply_oversample(X, y)

        k = min(SMOTE_K_NEIGHBORS,
                min(Counter(y).values()) - 1)
        k = max(1, k)
        sm = SMOTE(k_neighbors=k, random_state=self.seed)
        X_bal, y_bal = sm.fit_resample(X, y)
        self._after = dict(Counter(y_bal))
        self._log(f"SMOTE (k={k}): {self._before} -> {self._after}")
        return X_bal, y_bal

    def _apply_oversample(self, X, y):
        counts = Counter(y)
        max_count = max(counts.values())
        rng = np.random.default_rng(self.seed)

        X_parts, y_parts = [X], [y]
        for cls, count in counts.items():
            if count < max_count:
                cls_idx = np.where(y == cls)[0]
                n_needed = max_count - count
                extra_idx = rng.choice(cls_idx, size=n_needed, replace=True)
                X_parts.append(X[extra_idx])
                y_parts.append(y[extra_idx])

        X_bal = np.concatenate(X_parts, axis=0)
        y_bal = np.concatenate(y_parts, axis=0)
        shuffle_idx = rng.permutation(len(y_bal))
        X_bal, y_bal = X_bal[shuffle_idx], y_bal[shuffle_idx]

        self._after = dict(Counter(y_bal))
        self._log(f"Oversample: {self._before} -> {self._after}")
        return X_bal, y_bal
