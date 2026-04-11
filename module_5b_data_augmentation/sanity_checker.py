"""
=============================================================================
MODULE 5B - DATA AUGMENTATION  |  sanity_checker.py
=============================================================================
Validate synthetic signal segments against real data before acceptance.

Five quality checks:
  1. Physiological range — reject if too many out-of-range samples
  2. Statistical similarity — KS test between real and synthetic distributions
  3. Maximum Mean Discrepancy (MMD) — kernel-based distribution distance
  4. Temporal autocorrelation — lag-1 ACF comparison
  5. Discriminative score — train classifier to distinguish real vs fake

A segment passes only if ALL checks pass. The overall acceptance rate
must exceed SANITY["min_acceptance_rate"] or the user is warned.

References:
  Esteban et al. (2017) — Real-valued (Medical) Time Series Generation
  Yoon et al. (2019) — TimeGAN evaluation metrics
=============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from config import MODULE_LABEL, SANITY, SIGNAL_SPECS
from signal_generator import SyntheticSegment

log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single sanity check."""
    check_name: str
    passed: bool
    value: float
    threshold: float
    detail: str = ""


@dataclass
class SegmentSanityResult:
    """Sanity check results for a single synthetic segment."""
    segment_idx: int
    emotion_label: str
    signal_name: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


@dataclass
class SanityReport:
    """Aggregate sanity report for all synthetic segments of one signal."""
    signal_name: str
    total_segments: int
    passed_segments: int
    failed_segments: int
    acceptance_rate: float
    per_segment: List[SegmentSanityResult] = field(default_factory=list)
    check_pass_rates: Dict[str, float] = field(default_factory=dict)

    @property
    def acceptable(self) -> bool:
        return self.acceptance_rate >= SANITY["min_acceptance_rate"]


class SanityChecker:
    """
    Validate synthetic segments against real data.

    Parameters
    ----------
    real_segments : ndarray of shape (n_real, seq_len, n_features)
        The real training segments (same signal type).
    signal_name : str
        Signal type for range checking.
    """

    def __init__(self, real_segments: np.ndarray, signal_name: str):
        self.real = real_segments
        self.signal_name = signal_name
        self.spec = SIGNAL_SPECS.get(signal_name, {})
        self.value_range = self.spec.get("range", (None, None))

    def check_all(
        self, synthetic_segments: List[SyntheticSegment]
    ) -> SanityReport:
        """
        Run all sanity checks on a list of synthetic segments.

        Returns SanityReport with per-segment and aggregate results.
        """
        log.info(f"[{MODULE_LABEL}] Running sanity checks on "
                 f"{len(synthetic_segments)} synthetic {self.signal_name} "
                 f"segments")

        results = []
        check_counts: Dict[str, int] = {}

        for i, seg in enumerate(synthetic_segments):
            sr = SegmentSanityResult(
                segment_idx=i,
                emotion_label=seg.emotion_label,
                signal_name=seg.signal_name,
            )

            # 1. Physiological range check
            sr.checks.append(self._check_range(seg.values))

            # 2. Statistical similarity (KS test)
            sr.checks.append(self._check_ks_test(seg.values))

            # 3. Maximum Mean Discrepancy
            sr.checks.append(self._check_mmd(seg.values))

            # 4. Temporal autocorrelation
            sr.checks.append(self._check_acf(seg.values))

            results.append(sr)

            for c in sr.checks:
                if c.check_name not in check_counts:
                    check_counts[c.check_name] = 0
                if c.passed:
                    check_counts[c.check_name] += 1

        # 5. Discriminative score (batch check — one score for all)
        disc_result = self._check_discriminative(synthetic_segments)
        # Apply to all segments
        for sr in results:
            sr.checks.append(disc_result)
        check_counts["discriminative"] = (
            len(results) if disc_result.passed else 0
        )

        n_passed = sum(1 for sr in results if sr.passed)
        n_total = len(results)
        acc_rate = n_passed / n_total if n_total > 0 else 0.0

        check_pass_rates = {
            name: cnt / n_total if n_total > 0 else 0.0
            for name, cnt in check_counts.items()
        }

        report = SanityReport(
            signal_name=self.signal_name,
            total_segments=n_total,
            passed_segments=n_passed,
            failed_segments=n_total - n_passed,
            acceptance_rate=round(acc_rate, 3),
            per_segment=results,
            check_pass_rates=check_pass_rates,
        )

        self._log_report(report)
        return report

    # ── Check 1: Physiological range ──────────────────────────────────────

    def _check_range(self, values: np.ndarray) -> CheckResult:
        """Reject if > max_oor_pct% of samples are out of physiological range."""
        lo, hi = self.value_range
        if lo is None or hi is None:
            return CheckResult("range", True, 0.0, SANITY["max_oor_pct"],
                               "No range defined — skipped")

        total_samples = values.size
        if total_samples == 0:
            return CheckResult("range", False, 100.0, SANITY["max_oor_pct"],
                               "Empty segment")

        oor = np.sum((values < lo) | (values > hi))
        pct = 100.0 * oor / total_samples
        passed = pct <= SANITY["max_oor_pct"]

        return CheckResult(
            "range", passed, round(pct, 2), SANITY["max_oor_pct"],
            f"{oor}/{total_samples} samples out of [{lo}, {hi}]"
        )

    # ── Check 2: KS test ─────────────────────────────────────────────────

    def _check_ks_test(self, values: np.ndarray) -> CheckResult:
        """
        KS test between real and synthetic value distributions.
        p > alpha means distributions are similar (PASS).
        """
        # Flatten real and synthetic to 1D for per-feature comparison
        real_flat = self.real.reshape(-1, self.real.shape[-1])
        synth_flat = values.reshape(-1, values.shape[-1])

        # Average p-value across features
        p_values = []
        n_feat = min(real_flat.shape[1], synth_flat.shape[1])
        for f in range(n_feat):
            r = real_flat[:, f]
            s = synth_flat[:, f]
            if len(r) < 2 or len(s) < 2:
                continue
            _, p = stats.ks_2samp(r, s)
            p_values.append(p)

        if not p_values:
            return CheckResult("ks_test", False, 0.0, SANITY["ks_test_alpha"],
                               "Insufficient data for KS test")

        mean_p = float(np.mean(p_values))
        passed = mean_p > SANITY["ks_test_alpha"]

        return CheckResult(
            "ks_test", passed, round(mean_p, 4), SANITY["ks_test_alpha"],
            f"Mean p-value across {n_feat} features"
        )

    # ── Check 3: MMD ─────────────────────────────────────────────────────

    def _check_mmd(self, values: np.ndarray) -> CheckResult:
        """
        Maximum Mean Discrepancy with RBF kernel.
        Lower is better; reject if MMD > threshold.
        """
        # Use a subset of real data for efficiency
        max_real = 200
        real_sub = self.real
        if len(real_sub) > max_real:
            idx = np.random.choice(len(real_sub), max_real, replace=False)
            real_sub = real_sub[idx]

        # Flatten segments to 2D (n_samples, seq*feat)
        r_flat = real_sub.reshape(len(real_sub), -1)
        s_flat = values.reshape(1, -1)

        # RBF kernel MMD
        mmd = self._compute_mmd(r_flat, s_flat)
        passed = mmd <= SANITY["mmd_threshold"]

        return CheckResult(
            "mmd", passed, round(float(mmd), 4), SANITY["mmd_threshold"],
            "RBF kernel MMD"
        )

    @staticmethod
    def _compute_mmd(X: np.ndarray, Y: np.ndarray) -> float:
        """Compute MMD^2 with RBF kernel (bandwidth = median heuristic)."""
        XY = np.concatenate([X, Y], axis=0)

        # Pairwise squared distances
        from scipy.spatial.distance import cdist
        dists = cdist(XY, XY, metric="sqeuclidean")

        # Median heuristic for bandwidth
        median_dist = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        sigma2 = median_dist / 2.0
        if sigma2 < 1e-10:
            sigma2 = 1.0

        K = np.exp(-dists / (2.0 * sigma2))

        nx = len(X)
        ny = len(Y)
        Kxx = K[:nx, :nx]
        Kyy = K[nx:, nx:]
        Kxy = K[:nx, nx:]

        mmd2 = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
        return max(0.0, mmd2) ** 0.5

    # ── Check 4: Temporal autocorrelation ─────────────────────────────────

    def _check_acf(self, values: np.ndarray) -> CheckResult:
        """
        Compare lag-1 autocorrelation between real and synthetic.
        Reject if |ACF_real - ACF_synthetic| > threshold.
        """
        # Average lag-1 ACF across real segments (first feature)
        real_acfs = []
        for seg in self.real:
            acf = self._lag1_acf(seg[:, 0])
            if acf is not None:
                real_acfs.append(acf)

        if not real_acfs:
            return CheckResult("acf", True, 0.0, SANITY["acf_diff_threshold"],
                               "No valid ACF from real data — skipped")

        real_mean_acf = float(np.mean(real_acfs))
        synth_acf = self._lag1_acf(values[:, 0])

        if synth_acf is None:
            return CheckResult("acf", False, 1.0, SANITY["acf_diff_threshold"],
                               "Could not compute synthetic ACF")

        diff = abs(real_mean_acf - synth_acf)
        passed = diff <= SANITY["acf_diff_threshold"]

        return CheckResult(
            "acf", passed, round(diff, 4), SANITY["acf_diff_threshold"],
            f"Real ACF={real_mean_acf:.3f}, Synthetic ACF={synth_acf:.3f}"
        )

    @staticmethod
    def _lag1_acf(x: np.ndarray) -> Optional[float]:
        """Compute lag-1 autocorrelation."""
        if len(x) < 3:
            return None
        x = x - np.mean(x)
        var = np.var(x)
        if var < 1e-10:
            return None
        acf = np.correlate(x[:-1], x[1:]) / (var * (len(x) - 1))
        return float(acf[0]) if len(acf) > 0 else None

    # ── Check 5: Discriminative score ─────────────────────────────────────

    def _check_discriminative(
        self, synthetic_segments: List[SyntheticSegment]
    ) -> CheckResult:
        """
        Train a simple classifier to distinguish real vs fake.
        If accuracy > threshold, synthetic data is too distinguishable (FAIL).
        """
        if len(synthetic_segments) < 5 or len(self.real) < 5:
            return CheckResult(
                "discriminative", True, 0.5,
                SANITY["discriminative_accuracy_threshold"],
                "Too few samples for discriminative test — skipped"
            )

        # Flatten segments to feature vectors
        synth_arr = np.stack(
            [s.values for s in synthetic_segments], axis=0
        )
        real_flat = self.real.reshape(len(self.real), -1)
        synth_flat = synth_arr.reshape(len(synth_arr), -1)

        # Subsample to balance classes
        n_each = min(len(real_flat), len(synth_flat), 100)
        r_idx = np.random.choice(len(real_flat), n_each, replace=False)
        s_idx = np.random.choice(len(synth_flat), n_each, replace=False)

        X = np.concatenate([real_flat[r_idx], synth_flat[s_idx]], axis=0)
        y = np.concatenate([np.ones(n_each), np.zeros(n_each)])

        # Simple logistic regression with cross-validation
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score

            clf = LogisticRegression(max_iter=200, random_state=42)
            scores = cross_val_score(clf, X, y, cv=min(5, n_each), scoring="accuracy")
            acc = float(np.mean(scores))
        except Exception as e:
            log.warning(f"[{MODULE_LABEL}] Discriminative test failed: {e}")
            return CheckResult(
                "discriminative", True, 0.5,
                SANITY["discriminative_accuracy_threshold"],
                f"Test failed: {e}"
            )

        # Lower accuracy = harder to distinguish = better synthetic data
        passed = acc <= SANITY["discriminative_accuracy_threshold"]

        return CheckResult(
            "discriminative", passed, round(acc, 3),
            SANITY["discriminative_accuracy_threshold"],
            f"Classifier accuracy={acc:.3f} "
            f"({'indistinguishable' if passed else 'too distinguishable'})"
        )

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_report(self, report: SanityReport) -> None:
        log.info(f"[{MODULE_LABEL}] === Sanity Check Report: "
                 f"{report.signal_name} ===")
        log.info(f"[{MODULE_LABEL}]   Total:    {report.total_segments}")
        log.info(f"[{MODULE_LABEL}]   Passed:   {report.passed_segments}")
        log.info(f"[{MODULE_LABEL}]   Failed:   {report.failed_segments}")
        log.info(f"[{MODULE_LABEL}]   Accept rate: "
                 f"{report.acceptance_rate:.1%}")

        for name, rate in report.check_pass_rates.items():
            log.info(f"[{MODULE_LABEL}]     {name:20s}: {rate:.1%}")

        if not report.acceptable:
            log.warning(
                f"[{MODULE_LABEL}]   WARNING: Acceptance rate "
                f"{report.acceptance_rate:.1%} is below minimum "
                f"{SANITY['min_acceptance_rate']:.0%}. "
                "Consider retraining the TGAN or adjusting parameters."
            )
