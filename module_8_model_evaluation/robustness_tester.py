"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  robustness_tester.py
=============================================================================
Test model robustness under realistic perturbations: Gaussian noise injection,
random feature dropout (simulating missing data), and signal-channel dropout
(simulating sensor failure).  Only runs on top-N models to control runtime.
=============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import (
    NOISE_LEVELS, MISSING_FRACTIONS, CHANNEL_DROPOUT_CONFIGS,
    SIGNAL_FEATURE_PREFIXES, ROBUSTNESS_TOP_N, PRIMARY_METRIC,
    RANDOM_SEED,
)

log = logging.getLogger(__name__)


@dataclass
class NoiseResult:
    """Model performance under Gaussian noise at various levels."""
    noise_level: float
    f1_weighted: float
    f1_drop: float  # Absolute drop from clean F1


@dataclass
class MissingResult:
    """Model performance with random missing features."""
    missing_fraction: float
    f1_weighted: float
    f1_drop: float


@dataclass
class ChannelDropResult:
    """Model performance with one or more signal channels zeroed out."""
    dropped_channels: List[str]
    n_features_dropped: int
    f1_weighted: float
    f1_drop: float


@dataclass
class RobustnessResult:
    """Full robustness analysis for one model."""
    model_name: str
    clean_f1: float
    noise_results: List[NoiseResult] = field(default_factory=list)
    missing_results: List[MissingResult] = field(default_factory=list)
    channel_results: List[ChannelDropResult] = field(default_factory=list)
    # Summary scores
    noise_auc: float = 0.0       # Area under noise-degradation curve
    missing_auc: float = 0.0     # Area under missing-degradation curve
    worst_channel_drop: float = 0.0
    robustness_score: float = 0.0  # Composite robustness score (0-1)


class RobustnessTester:
    """Test model robustness under perturbations."""

    def __init__(self, seed: int = RANDOM_SEED, verbose: bool = True):
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [M8-Rob] {msg}")

    def test_all(
        self,
        eval_results: list,
        models: dict,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, RobustnessResult]:
        """
        Run robustness tests on top-N models.

        Parameters
        ----------
        eval_results : sorted List[EvalResult] (best first)
        models : dict of {name: LoadedModel}
        X_test : (n_test, n_features)
        y_test : (n_test,)
        feature_names : list of feature column names

        Returns
        -------
        Dict[model_name, RobustnessResult]
        """
        # Select top-N non-skipped models
        top_models = []
        for er in eval_results:
            if not er.skipped and er.model_name in models:
                top_models.append(er)
                if len(top_models) >= ROBUSTNESS_TOP_N:
                    break

        self._log(f"Testing {len(top_models)} models for robustness")
        results: Dict[str, RobustnessResult] = {}

        for er in top_models:
            model = models[er.model_name]
            clean_f1 = er.test_metrics.get("f1_weighted", 0.0)

            rr = RobustnessResult(
                model_name=er.model_name,
                clean_f1=clean_f1,
            )

            # 1) Noise injection
            self._log(f"  {er.model_name}: noise injection...")
            rr.noise_results = self._test_noise(
                model, X_test, y_test, clean_f1
            )

            # 2) Missing features
            self._log(f"  {er.model_name}: missing features...")
            rr.missing_results = self._test_missing(
                model, X_test, y_test, feature_names, clean_f1
            )

            # 3) Channel dropout
            self._log(f"  {er.model_name}: channel dropout...")
            rr.channel_results = self._test_channel_dropout(
                model, X_test, y_test, feature_names, clean_f1
            )

            # Summary scores
            rr.noise_auc = self._degradation_auc(
                [0.0] + NOISE_LEVELS,
                [clean_f1] + [nr.f1_weighted for nr in rr.noise_results],
            )
            rr.missing_auc = self._degradation_auc(
                [0.0] + MISSING_FRACTIONS,
                [clean_f1] + [mr.f1_weighted for mr in rr.missing_results],
            )
            if rr.channel_results:
                rr.worst_channel_drop = max(
                    cr.f1_drop for cr in rr.channel_results
                )

            # Composite robustness score (higher = more robust)
            # Normalised AUC: perfect model retains F1=clean_f1 at all levels
            max_noise_auc = clean_f1 * max(NOISE_LEVELS)
            max_missing_auc = clean_f1 * max(MISSING_FRACTIONS)
            norm_noise = rr.noise_auc / max_noise_auc if max_noise_auc > 0 else 0
            norm_missing = rr.missing_auc / max_missing_auc if max_missing_auc > 0 else 0
            channel_score = 1.0 - (rr.worst_channel_drop / clean_f1 if clean_f1 > 0 else 1.0)
            channel_score = max(channel_score, 0.0)
            rr.robustness_score = float(np.mean([norm_noise, norm_missing, channel_score]))

            self._log(
                f"  {er.model_name}: robustness_score={rr.robustness_score:.3f}"
            )
            results[er.model_name] = rr

        return results

    def _test_noise(
        self, model, X_test: np.ndarray, y_test: np.ndarray, clean_f1: float
    ) -> List[NoiseResult]:
        """Test with additive Gaussian noise at various levels."""
        results = []
        feature_stds = np.std(X_test, axis=0)
        # Guard: if std is 0 for some features, use 1.0
        feature_stds = np.where(feature_stds > 0, feature_stds, 1.0)

        for level in NOISE_LEVELS:
            noise = self.rng.normal(0, level, size=X_test.shape) * feature_stds
            X_noisy = X_test + noise

            try:
                y_pred = model.predict(X_noisy)
                f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            except Exception:
                f1 = 0.0

            results.append(NoiseResult(
                noise_level=level,
                f1_weighted=f1,
                f1_drop=clean_f1 - f1,
            ))

        return results

    def _test_missing(
        self, model, X_test: np.ndarray, y_test: np.ndarray,
        feature_names: List[str], clean_f1: float,
    ) -> List[MissingResult]:
        """Test with random features set to zero (simulating missing data)."""
        results = []

        for frac in MISSING_FRACTIONS:
            X_missing = X_test.copy()
            n_features = X_test.shape[1]
            n_drop = max(1, int(frac * n_features))

            # Random mask: each sample gets different features dropped
            mask = np.zeros_like(X_missing, dtype=bool)
            for i in range(X_missing.shape[0]):
                drop_idx = self.rng.choice(n_features, size=n_drop, replace=False)
                mask[i, drop_idx] = True

            X_missing[mask] = 0.0

            try:
                y_pred = model.predict(X_missing)
                f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            except Exception:
                f1 = 0.0

            results.append(MissingResult(
                missing_fraction=frac,
                f1_weighted=f1,
                f1_drop=clean_f1 - f1,
            ))

        return results

    def _test_channel_dropout(
        self, model, X_test: np.ndarray, y_test: np.ndarray,
        feature_names: List[str], clean_f1: float,
    ) -> List[ChannelDropResult]:
        """Test with entire signal channels zeroed out."""
        results = []

        for channels in CHANNEL_DROPOUT_CONFIGS:
            X_dropped = X_test.copy()
            n_dropped = 0

            for ch in channels:
                prefixes = SIGNAL_FEATURE_PREFIXES.get(ch, [])
                for j, fname in enumerate(feature_names):
                    if any(fname.startswith(p) for p in prefixes):
                        X_dropped[:, j] = 0.0
                        n_dropped += 1

            if n_dropped == 0:
                continue

            try:
                y_pred = model.predict(X_dropped)
                f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            except Exception:
                f1 = 0.0

            results.append(ChannelDropResult(
                dropped_channels=channels,
                n_features_dropped=n_dropped,
                f1_weighted=f1,
                f1_drop=clean_f1 - f1,
            ))

        return results

    def _degradation_auc(
        self, x_values: List[float], y_values: List[float]
    ) -> float:
        """Compute area under the degradation curve using trapezoidal rule."""
        if len(x_values) < 2:
            return 0.0
        _trap = getattr(np, "trapezoid", np.trapz)
        return float(_trap(y_values, x_values))

    def build_noise_table(
        self, results: Dict[str, RobustnessResult]
    ) -> pd.DataFrame:
        """Build noise degradation table."""
        rows = []
        for name, rr in results.items():
            for nr in rr.noise_results:
                rows.append({
                    "model": name,
                    "noise_level": nr.noise_level,
                    "f1_weighted": nr.f1_weighted,
                    "f1_drop": nr.f1_drop,
                })
        return pd.DataFrame(rows)

    def build_channel_table(
        self, results: Dict[str, RobustnessResult]
    ) -> pd.DataFrame:
        """Build channel dropout table."""
        rows = []
        for name, rr in results.items():
            for cr in rr.channel_results:
                rows.append({
                    "model": name,
                    "dropped": "+".join(cr.dropped_channels),
                    "n_features": cr.n_features_dropped,
                    "f1_weighted": cr.f1_weighted,
                    "f1_drop": cr.f1_drop,
                })
        return pd.DataFrame(rows)
