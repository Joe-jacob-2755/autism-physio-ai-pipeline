"""
=============================================================================
MODULE 5B - DATA AUGMENTATION  |  smote_augmenter.py
=============================================================================
SMOTE-based feature-level augmentation as an alternative/complement to
CT-TimeGAN signal-level augmentation.

SMOTE operates on extracted feature vectors (post M6) rather than raw
signals. It is faster and simpler than CT-TimeGAN but cannot preserve
temporal signal morphology.

Feasibility analysis:
  - Checks per-class sample counts against SMOTE requirements
  - Reports to the researcher which classes can/cannot be augmented
  - Suggests optimal k_neighbors per class
  - Falls back to random oversampling for classes below minimum

References:
  Chawla et al. (2002) — SMOTE: Synthetic Minority Over-sampling Technique
  He et al. (2008) — ADASYN: Adaptive Synthetic Sampling
=============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import MODULE_LABEL

log = logging.getLogger(__name__)

# ── Availability checks ──────────────────────────────────────────────────

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False


# ═════════════════════════════════════════════════════════════════════════════
# Configuration (also added to config.py)
# ═════════════════════════════════════════════════════════════════════════════

SMOTE_CONFIG = {
    "default_k_neighbors": 5,
    "min_k_neighbors": 1,
    "min_samples_for_smote": 2,      # Absolute minimum (need k+1 >= 2)
    "min_samples_for_adasyn": 6,     # ADASYN needs more diversity
    "min_samples_for_borderline": 6, # BorderlineSMOTE needs boundary samples
    "methods": ["smote", "borderline_smote", "adasyn", "random_oversample"],
    "default_method": "smote",
}


# ═════════════════════════════════════════════════════════════════════════════
# Feasibility Report
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ClassFeasibility:
    """Feasibility assessment for a single class."""
    class_name: str
    current_count: int
    target_count: int
    synthetic_needed: int
    smote_feasible: bool
    adasyn_feasible: bool
    borderline_feasible: bool
    recommended_method: str
    recommended_k: int
    reason: str


@dataclass
class SMOTEFeasibilityReport:
    """Full feasibility report for the researcher."""
    per_class: Dict[str, ClassFeasibility]
    overall_feasible: bool
    n_classes_smote_ok: int
    n_classes_fallback: int
    n_classes_impossible: int
    imblearn_installed: bool
    summary: str
    recommendations: List[str]

    def print_report(self):
        """Print a formatted report for the researcher."""
        print("\n" + "=" * 70)
        print(f"  [{MODULE_LABEL}] SMOTE FEASIBILITY ANALYSIS")
        print("=" * 70)

        if not self.imblearn_installed:
            print("\n  [ERROR] imbalanced-learn is NOT installed.")
            print("  Install with: pip install imbalanced-learn>=0.11.0")
            print("=" * 70)
            return

        print(f"\n  Overall feasible: {'YES' if self.overall_feasible else 'PARTIAL'}")
        print(f"  Classes suitable for SMOTE:      {self.n_classes_smote_ok}")
        print(f"  Classes needing fallback:         {self.n_classes_fallback}")
        print(f"  Classes impossible to augment:    {self.n_classes_impossible}")

        print(f"\n  {'Class':14s} {'Count':>5s} {'Target':>6s} {'Need':>5s} "
              f"{'Method':20s} {'k':>3s}  {'Status'}")
        print(f"  {'-' * 72}")

        for cls in sorted(self.per_class.keys()):
            cf = self.per_class[cls]
            status = "OK" if cf.smote_feasible else (
                "FALLBACK" if cf.recommended_method == "random_oversample"
                else "SKIP")
            print(f"  {cf.class_name:14s} {cf.current_count:5d} "
                  f"{cf.target_count:6d} {cf.synthetic_needed:5d} "
                  f"{cf.recommended_method:20s} {cf.recommended_k:3d}  "
                  f"{status}")
            if cf.reason:
                print(f"    -> {cf.reason}")

        if self.recommendations:
            print(f"\n  Recommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"    {i}. {rec}")

        print("\n" + "=" * 70)

    def to_dict(self) -> dict:
        """Serialise for JSON export."""
        return {
            "overall_feasible": self.overall_feasible,
            "imblearn_installed": self.imblearn_installed,
            "n_classes_smote_ok": self.n_classes_smote_ok,
            "n_classes_fallback": self.n_classes_fallback,
            "n_classes_impossible": self.n_classes_impossible,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "per_class": {
                cls: {
                    "current_count": cf.current_count,
                    "target_count": cf.target_count,
                    "synthetic_needed": cf.synthetic_needed,
                    "smote_feasible": cf.smote_feasible,
                    "recommended_method": cf.recommended_method,
                    "recommended_k": cf.recommended_k,
                    "reason": cf.reason,
                }
                for cls, cf in self.per_class.items()
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
# SMOTE Augmenter
# ═════════════════════════════════════════════════════════════════════════════

class SMOTEAugmenter:
    """
    Feature-level augmentation using SMOTE and variants.

    Operates on feature DataFrames (post M6 feature extraction).
    Includes feasibility analysis that reports to the researcher
    whether SMOTE is viable for each class.

    Parameters
    ----------
    method : str
        "smote", "borderline_smote", "adasyn", or "random_oversample".
    k_neighbors : int
        Default k for SMOTE. Auto-adjusted per class if needed.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print progress and feasibility report.
    """

    def __init__(
        self,
        method: str = SMOTE_CONFIG["default_method"],
        k_neighbors: int = SMOTE_CONFIG["default_k_neighbors"],
        seed: int = 42,
        verbose: bool = True,
    ):
        self.method = method
        self.k_neighbors = k_neighbors
        self.seed = seed
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [{MODULE_LABEL}] [SMOTE] {msg}")

    # ── Feasibility Analysis ─────────────────────────────────────────────

    def assess_feasibility(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[List[str]] = None,
        target_counts: Optional[Dict[str, int]] = None,
    ) -> SMOTEFeasibilityReport:
        """
        Analyse whether SMOTE is feasible for the given data.

        This is the primary advisory function — it tells the researcher
        which classes can be augmented with SMOTE, which need fallbacks,
        and what parameters to use.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Feature matrix.
        y : array of shape (n_samples,)
            Class labels (integer-encoded or string).
        class_names : list of str, optional
            Human-readable class names (indexed by label value).
        target_counts : dict, optional
            {class_name: desired_count} for augmentation targets.

        Returns
        -------
        SMOTEFeasibilityReport
        """
        unique_classes, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique_classes, counts))
        majority_count = max(counts)

        per_class: Dict[str, ClassFeasibility] = {}
        n_smote_ok = 0
        n_fallback = 0
        n_impossible = 0
        recommendations = []

        for cls, cnt in class_dist.items():
            cls_name = (class_names[cls] if class_names and isinstance(cls, int)
                        and cls < len(class_names) else str(cls))

            target = (target_counts.get(cls_name, majority_count)
                      if target_counts else majority_count)
            needed = max(0, target - cnt)

            if needed == 0:
                per_class[cls_name] = ClassFeasibility(
                    class_name=cls_name, current_count=cnt,
                    target_count=target, synthetic_needed=0,
                    smote_feasible=True, adasyn_feasible=True,
                    borderline_feasible=True,
                    recommended_method="none_needed",
                    recommended_k=0,
                    reason="Already at or above target count",
                )
                continue

            # SMOTE needs k_neighbors + 1 samples minimum
            smote_ok = cnt >= SMOTE_CONFIG["min_samples_for_smote"]
            adasyn_ok = cnt >= SMOTE_CONFIG["min_samples_for_adasyn"]
            borderline_ok = cnt >= SMOTE_CONFIG["min_samples_for_borderline"]

            # Optimal k: min(default_k, n_samples - 1)
            optimal_k = min(self.k_neighbors, cnt - 1) if cnt > 1 else 0

            if smote_ok and optimal_k >= SMOTE_CONFIG["min_k_neighbors"]:
                method = self.method
                if method == "adasyn" and not adasyn_ok:
                    method = "smote"
                if method == "borderline_smote" and not borderline_ok:
                    method = "smote"
                reason = (f"SMOTE viable with k={optimal_k}"
                          if optimal_k < self.k_neighbors
                          else "")
                n_smote_ok += 1
            elif cnt >= 1:
                method = "random_oversample"
                optimal_k = 0
                reason = (f"Only {cnt} sample(s) — below SMOTE minimum "
                          f"(need >= {SMOTE_CONFIG['min_samples_for_smote']}). "
                          f"Using random oversampling.")
                n_fallback += 1
            else:
                method = "impossible"
                optimal_k = 0
                reason = "No samples — cannot generate synthetic data"
                n_impossible += 1

            per_class[cls_name] = ClassFeasibility(
                class_name=cls_name, current_count=cnt,
                target_count=target, synthetic_needed=needed,
                smote_feasible=smote_ok and optimal_k >= 1,
                adasyn_feasible=adasyn_ok,
                borderline_feasible=borderline_ok,
                recommended_method=method,
                recommended_k=optimal_k,
                reason=reason,
            )

        # Build summary and recommendations
        overall = n_smote_ok > 0
        if n_fallback > 0:
            recommendations.append(
                f"{n_fallback} class(es) will use random oversampling instead "
                f"of SMOTE. Consider generating more data (more users or "
                f"longer sessions) to enable SMOTE for all classes."
            )
        if n_impossible > 0:
            recommendations.append(
                f"{n_impossible} class(es) have zero samples and cannot be "
                f"augmented. Use CT-TimeGAN (M5B) at the signal level "
                f"or simulate more events in M2A."
            )
        if not _HAS_IMBLEARN:
            recommendations.append(
                "Install imbalanced-learn: pip install imbalanced-learn>=0.11.0"
            )
        if overall and n_fallback == 0:
            recommendations.append(
                "All minority classes are SMOTE-eligible. Feature-level "
                "SMOTE is recommended as a fast, deterministic augmentation."
            )
        if any(cf.recommended_k < 3 for cf in per_class.values()
               if cf.smote_feasible and cf.synthetic_needed > 0):
            recommendations.append(
                "Some classes have very low k (< 3). Synthetic samples "
                "will have limited diversity. Consider combining with "
                "CT-TimeGAN for better coverage."
            )

        summary = (
            f"SMOTE feasibility: {n_smote_ok} classes viable, "
            f"{n_fallback} need fallback, {n_impossible} impossible. "
            f"{'imbalanced-learn installed.' if _HAS_IMBLEARN else 'imbalanced-learn NOT installed.'}"
        )

        report = SMOTEFeasibilityReport(
            per_class=per_class,
            overall_feasible=overall,
            n_classes_smote_ok=n_smote_ok,
            n_classes_fallback=n_fallback,
            n_classes_impossible=n_impossible,
            imblearn_installed=_HAS_IMBLEARN,
            summary=summary,
            recommendations=recommendations,
        )

        return report

    # ── Augmentation ─────────────────────────────────────────────────────

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_counts: Optional[Dict[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, SMOTEFeasibilityReport]:
        """
        Apply SMOTE augmentation with automatic feasibility handling.

        For classes where SMOTE is not feasible, falls back to random
        oversampling. Reports the feasibility analysis to the caller.

        Parameters
        ----------
        X : array (n_samples, n_features)
        y : array (n_samples,)
        target_counts : dict {class_label_int: desired_total_count}, optional
            If None, equalises all classes to the majority count.

        Returns
        -------
        X_aug : augmented feature matrix
        y_aug : augmented labels
        report : SMOTEFeasibilityReport
        """
        report = self.assess_feasibility(X, y)

        if self.verbose:
            report.print_report()

        if not _HAS_IMBLEARN:
            self._log("imbalanced-learn not installed — using random oversample")
            X_aug, y_aug = self._random_oversample(X, y, target_counts)
            return X_aug, y_aug, report

        unique_classes, counts = np.unique(y, return_counts=True)
        majority_count = max(counts)

        # Build sampling strategy
        if target_counts is None:
            target_counts = {cls: majority_count for cls in unique_classes}

        # Check which classes can use SMOTE
        smote_classes = set()
        fallback_classes = set()
        for cls, cnt in zip(unique_classes, counts):
            needed = target_counts.get(cls, majority_count) - cnt
            if needed <= 0:
                continue
            if cnt >= SMOTE_CONFIG["min_samples_for_smote"] and cnt > 1:
                smote_classes.add(cls)
            else:
                fallback_classes.add(cls)

        # Step 1: Apply SMOTE for eligible classes
        X_result, y_result = X.copy(), y.copy()

        if smote_classes:
            # Build SMOTE sampling strategy (only SMOTE-eligible classes)
            smote_strategy = {}
            for cls in smote_classes:
                smote_strategy[cls] = target_counts.get(cls, majority_count)

            # Determine safe k_neighbors
            min_minority = min(
                np.sum(y == cls) for cls in smote_classes)
            safe_k = min(self.k_neighbors, min_minority - 1)
            safe_k = max(safe_k, SMOTE_CONFIG["min_k_neighbors"])

            try:
                if self.method == "borderline_smote" and all(
                        np.sum(y == c) >= SMOTE_CONFIG["min_samples_for_borderline"]
                        for c in smote_classes):
                    sampler = BorderlineSMOTE(
                        sampling_strategy=smote_strategy,
                        k_neighbors=safe_k,
                        random_state=self.seed,
                    )
                elif self.method == "adasyn" and all(
                        np.sum(y == c) >= SMOTE_CONFIG["min_samples_for_adasyn"]
                        for c in smote_classes):
                    sampler = ADASYN(
                        sampling_strategy=smote_strategy,
                        n_neighbors=safe_k,
                        random_state=self.seed,
                    )
                else:
                    sampler = SMOTE(
                        sampling_strategy=smote_strategy,
                        k_neighbors=safe_k,
                        random_state=self.seed,
                    )

                X_result, y_result = sampler.fit_resample(X_result, y_result)
                self._log(f"SMOTE applied: {len(X)} -> {len(X_result)} samples "
                          f"(k={safe_k}, {len(smote_classes)} classes)")
            except Exception as e:
                self._log(f"SMOTE failed: {e}. Falling back to random oversample.")
                fallback_classes.update(smote_classes)
                X_result, y_result = X.copy(), y.copy()

        # Step 2: Random oversample for non-eligible classes
        if fallback_classes:
            rng = np.random.default_rng(self.seed)
            new_X, new_y = [], []
            for cls in fallback_classes:
                target = target_counts.get(cls, majority_count)
                current = np.sum(y_result == cls)
                needed = target - current
                if needed <= 0:
                    continue
                cls_indices = np.where(y_result == cls)[0]
                if len(cls_indices) == 0:
                    continue
                chosen = rng.choice(cls_indices, size=needed, replace=True)
                new_X.append(X_result[chosen])
                new_y.append(y_result[chosen])
                self._log(f"Random oversample for class {cls}: "
                          f"{current} -> {current + needed}")

            if new_X:
                X_result = np.concatenate([X_result] + new_X, axis=0)
                y_result = np.concatenate([y_result] + new_y, axis=0)

        self._log(f"Augmentation complete: {len(X)} -> {len(X_result)} samples")
        return X_result, y_result, report

    # ── Random Oversample Fallback ───────────────────────────────────────

    def _random_oversample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_counts: Optional[Dict[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple random oversampling (duplicate minority samples)."""
        rng = np.random.default_rng(self.seed)
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_count = max(counts)

        if target_counts is None:
            target_counts = {cls: majority_count for cls in unique_classes}

        new_X, new_y = [X], [y]
        for cls in unique_classes:
            target = target_counts.get(cls, majority_count)
            current = np.sum(y == cls)
            needed = target - current
            if needed <= 0:
                continue
            cls_indices = np.where(y == cls)[0]
            chosen = rng.choice(cls_indices, size=needed, replace=True)
            new_X.append(X[chosen])
            new_y.append(y[chosen])

        return np.concatenate(new_X), np.concatenate(new_y)

    # ── DataFrame interface ──────────────────────────────────────────────

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_label",
        target_counts: Optional[Dict[str, int]] = None,
    ) -> Tuple[pd.DataFrame, SMOTEFeasibilityReport]:
        """
        Apply SMOTE on a feature DataFrame (e.g. from M6 output).

        Parameters
        ----------
        df : DataFrame with features and target column
        feature_cols : list of feature column names
        target_col : name of the target label column
        target_counts : {class_name: target_count}, optional

        Returns
        -------
        df_aug : augmented DataFrame
        report : SMOTEFeasibilityReport
        """
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(df[target_col].values)
        X = df[feature_cols].values.astype(np.float64)
        class_names = list(le.classes_)

        # Convert string target_counts to int-encoded
        int_targets = None
        if target_counts:
            int_targets = {}
            for cls_name, count in target_counts.items():
                if cls_name in le.classes_:
                    int_targets[le.transform([cls_name])[0]] = count

        # Run feasibility with class names for the report
        report = self.assess_feasibility(X, y, class_names=class_names,
                                         target_counts=target_counts)

        # Augment
        X_aug, y_aug, _ = self.augment(X, y, target_counts=int_targets)

        # Rebuild DataFrame
        df_aug = pd.DataFrame(X_aug, columns=feature_cols)
        df_aug[target_col] = le.inverse_transform(y_aug)

        # Mark synthetic rows
        n_original = len(df)
        source_col = np.array(["real"] * n_original +
                              ["smote_synthetic"] * (len(df_aug) - n_original))
        df_aug["augmentation_source"] = source_col

        return df_aug, report

    # ── Export ────────────────────────────────────────────────────────────

    def save_feasibility_report(
        self, report: SMOTEFeasibilityReport, output_dir: Path
    ):
        """Save feasibility report as JSON and human-readable text."""
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        with open(output_dir / "smote_feasibility.json", "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        # Human-readable text
        lines = [
            "SMOTE FEASIBILITY REPORT",
            "=" * 50,
            f"Overall feasible: {report.overall_feasible}",
            f"imbalanced-learn installed: {report.imblearn_installed}",
            f"Classes SMOTE-eligible: {report.n_classes_smote_ok}",
            f"Classes needing fallback: {report.n_classes_fallback}",
            f"Classes impossible: {report.n_classes_impossible}",
            "",
            f"{'Class':14s} {'Count':>5s} {'Target':>6s} {'Method':20s} {'k':>3s}",
            "-" * 55,
        ]
        for cls in sorted(report.per_class.keys()):
            cf = report.per_class[cls]
            lines.append(
                f"{cf.class_name:14s} {cf.current_count:5d} "
                f"{cf.target_count:6d} {cf.recommended_method:20s} "
                f"{cf.recommended_k:3d}"
            )
            if cf.reason:
                lines.append(f"  -> {cf.reason}")
        lines.append("")
        lines.append("Recommendations:")
        for rec in report.recommendations:
            lines.append(f"  - {rec}")

        with open(output_dir / "smote_feasibility.txt", "w") as f:
            f.write("\n".join(lines))

        self._log(f"Feasibility report saved to {output_dir}")
