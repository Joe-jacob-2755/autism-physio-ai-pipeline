"""
=============================================================================
MODULE 5B - DATA AUGMENTATION  |  imbalance_detector.py
=============================================================================
Detect class imbalance in training data by analysing event distributions
across cleaned signal CSVs from Module 5 (Preprocessing).

Reads: module_5_preprocessing/{user_id}/cleaned_signals/*.csv
Counts: unique (event_id, target_label) pairs per user
Reports: imbalance ratio, per-class counts, severity classification

References:
  He & Garcia (2009) — Learning from Imbalanced Data
  Branco et al. (2016) — Survey of Predictive Modelling under Imbalanced Domains
=============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    IMBALANCE_THRESHOLDS,
    MIN_SAMPLES_PER_CLASS,
    MIN_TOTAL_EVENTS_FOR_TGAN,
    MODULE_LABEL,
)

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ClassDistribution:
    """Distribution of event classes across training users."""
    class_counts: Dict[str, int]          # {emotion: count}
    per_user_counts: Dict[str, Dict[str, int]]  # {user_id: {emotion: count}}
    total_events: int
    n_classes: int
    n_users: int

    @property
    def majority_class(self) -> str:
        return max(self.class_counts, key=self.class_counts.get)

    @property
    def minority_class(self) -> str:
        return min(self.class_counts, key=self.class_counts.get)

    @property
    def majority_count(self) -> int:
        return max(self.class_counts.values())

    @property
    def minority_count(self) -> int:
        return min(self.class_counts.values())


@dataclass
class ImbalanceReport:
    """Complete imbalance analysis report."""
    distribution: ClassDistribution
    imbalance_ratio: float                # N_majority / N_minority
    severity: str                         # "balanced" | "mild" | "moderate" | "severe"
    below_minimum: List[str]              # classes below MIN_SAMPLES_PER_CLASS
    enough_for_tgan: bool                 # total events >= MIN_TOTAL_EVENTS_FOR_TGAN
    augmentation_recommended: bool
    reason: str                           # human-readable explanation
    per_class_ir: Dict[str, float] = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising per-class counts and IR."""
        rows = []
        dist = self.distribution
        maj_count = dist.majority_count
        for cls in sorted(dist.class_counts.keys()):
            count = dist.class_counts[cls]
            ir = maj_count / count if count > 0 else float("inf")
            rows.append({
                "class": cls,
                "count": count,
                "imbalance_ratio": round(ir, 2),
                "below_minimum": count < MIN_SAMPLES_PER_CLASS,
            })
        return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Imbalance Detector
# ═════════════════════════════════════════════════════════════════════════════

class ImbalanceDetector:
    """
    Analyse class distribution in M5 preprocessed training data.

    Reads cleaned signal CSVs to count unique events per target_label.
    Events are identified by unique (event_id, target_label) pairs —
    a single event appears in multiple signals but is counted once.

    Parameters
    ----------
    source_dir : Path
        Root of M5 output, containing user subdirectories.
        Expected structure: source_dir/{user_id}/cleaned_signals/*.csv
    user_ids : list of str, optional
        Specific user IDs to analyse. If None, auto-detect from subdirectories.
    """

    def __init__(self, source_dir: Path, user_ids: Optional[List[str]] = None):
        self.source_dir = Path(source_dir)
        self.user_ids = user_ids or self._detect_users()

    # ── Public API ────────────────────────────────────────────────────────

    def analyse(self) -> ImbalanceReport:
        """Run full imbalance analysis. Returns ImbalanceReport."""
        log.info(f"[{MODULE_LABEL}] Analysing class distribution across "
                 f"{len(self.user_ids)} training users")

        distribution = self._count_events()
        ir, severity = self._compute_imbalance_ratio(distribution)
        below_min = self._check_minimum_samples(distribution)
        enough = distribution.total_events >= MIN_TOTAL_EVENTS_FOR_TGAN
        per_class_ir = self._per_class_ir(distribution)

        # Decide recommendation
        recommended, reason = self._recommend(
            ir, severity, below_min, enough, distribution
        )

        report = ImbalanceReport(
            distribution=distribution,
            imbalance_ratio=round(ir, 2),
            severity=severity,
            below_minimum=below_min,
            enough_for_tgan=enough,
            augmentation_recommended=recommended,
            reason=reason,
            per_class_ir=per_class_ir,
        )

        self._log_report(report)
        return report

    # ── Event counting ────────────────────────────────────────────────────

    def _count_events(self) -> ClassDistribution:
        """
        Count unique events across all training users.

        An event is a unique (event_id, target_label) pair.
        We read ONE signal CSV per user (whichever exists first) since
        all signals share the same event annotations.
        """
        class_counts: Dict[str, int] = {}
        per_user: Dict[str, Dict[str, int]] = {}

        for uid in self.user_ids:
            user_dir = self.source_dir / uid / "cleaned_signals"
            if not user_dir.exists():
                # Try without cleaned_signals subdirectory
                user_dir = self.source_dir / uid
            if not user_dir.exists():
                log.warning(f"[{MODULE_LABEL}] User dir not found: {uid}")
                continue

            events = self._extract_events_from_user(user_dir)
            user_counts: Dict[str, int] = {}
            for label in events:
                user_counts[label] = user_counts.get(label, 0) + 1
                class_counts[label] = class_counts.get(label, 0) + 1
            per_user[uid] = user_counts

        total = sum(class_counts.values())
        n_classes = len(class_counts)
        log.info(f"[{MODULE_LABEL}] Found {total} events across "
                 f"{n_classes} classes from {len(per_user)} users")

        return ClassDistribution(
            class_counts=class_counts,
            per_user_counts=per_user,
            total_events=total,
            n_classes=n_classes,
            n_users=len(per_user),
        )

    def _extract_events_from_user(self, user_dir: Path) -> List[str]:
        """
        Extract unique event labels from a user's cleaned signal CSVs.

        Reads the first available CSV and extracts unique (event_id, target_label)
        pairs where target_label is not 'baseline' or NaN.

        Returns list of target_label strings (one per unique event).
        """
        csv_priority = ["EDA.csv", "BVP.csv", "ST.csv", "ACC.csv", "IBI.csv"]
        df = None

        for fname in csv_priority:
            fpath = user_dir / fname
            if fpath.exists():
                try:
                    df = pd.read_csv(fpath)
                    break
                except Exception as e:
                    log.warning(f"[{MODULE_LABEL}] Error reading {fpath}: {e}")
                    continue

        if df is None:
            log.warning(f"[{MODULE_LABEL}] No signal CSVs found in {user_dir}")
            return []

        # Need both target_label and event_id columns
        if "target_label" not in df.columns:
            log.warning(f"[{MODULE_LABEL}] No target_label column in {user_dir}")
            return []

        # Filter out baseline / NaN labels
        mask = df["target_label"].notna()
        if "category" in df.columns:
            mask &= df["category"] != "baseline"
        # Also exclude literal "baseline" in target_label
        mask &= df["target_label"].astype(str).str.lower() != "baseline"

        labelled = df.loc[mask].copy()
        if labelled.empty:
            return []

        # Get unique events by (event_id, target_label) if event_id exists
        if "event_id" in labelled.columns:
            events = (
                labelled.dropna(subset=["event_id"])
                .groupby(["event_id", "target_label"])
                .size()
                .reset_index()
            )
            return events["target_label"].tolist()
        else:
            # Fallback: count unique labels (each contiguous block = 1 event)
            labels = labelled["target_label"].values
            events = []
            prev = None
            for lbl in labels:
                if lbl != prev:
                    events.append(str(lbl))
                    prev = lbl
            return events

    # ── Imbalance computation ─────────────────────────────────────────────

    def _compute_imbalance_ratio(
        self, dist: ClassDistribution
    ) -> Tuple[float, str]:
        """
        Compute global imbalance ratio (IR = N_majority / N_minority)
        and classify severity.
        """
        if dist.n_classes < 2:
            return 1.0, "balanced"

        if dist.minority_count == 0:
            return float("inf"), "severe"

        ir = dist.majority_count / dist.minority_count

        # Classify severity
        if ir < IMBALANCE_THRESHOLDS["balanced"]:
            severity = "balanced"
        elif ir < IMBALANCE_THRESHOLDS["mild"]:
            severity = "mild"
        elif ir < IMBALANCE_THRESHOLDS["moderate"]:
            severity = "moderate"
        else:
            severity = "severe"

        return ir, severity

    def _per_class_ir(self, dist: ClassDistribution) -> Dict[str, float]:
        """Compute IR for each class relative to the majority class."""
        maj = dist.majority_count
        return {
            cls: round(maj / cnt, 2) if cnt > 0 else float("inf")
            for cls, cnt in dist.class_counts.items()
        }

    def _check_minimum_samples(self, dist: ClassDistribution) -> List[str]:
        """Return classes with fewer than MIN_SAMPLES_PER_CLASS events."""
        return [
            cls for cls, cnt in dist.class_counts.items()
            if cnt < MIN_SAMPLES_PER_CLASS
        ]

    # ── Recommendation logic ──────────────────────────────────────────────

    def _recommend(
        self,
        ir: float,
        severity: str,
        below_min: List[str],
        enough: bool,
        dist: ClassDistribution,
    ) -> Tuple[bool, str]:
        """Determine whether augmentation is recommended and why."""

        if dist.total_events == 0:
            return False, "No events found in training data."

        if dist.n_classes < 2:
            return False, (
                f"Only {dist.n_classes} class found. Need at least 2 classes "
                "for imbalance analysis."
            )

        if not enough:
            return False, (
                f"Only {dist.total_events} total events (minimum "
                f"{MIN_TOTAL_EVENTS_FOR_TGAN} needed for TGAN training). "
                "Collect more real data before attempting augmentation."
            )

        reasons = []

        if severity == "balanced" and not below_min:
            return False, (
                f"Data is balanced (IR={ir:.1f}, threshold={IMBALANCE_THRESHOLDS['balanced']}). "
                "No augmentation needed."
            )

        if severity in ("moderate", "severe"):
            reasons.append(
                f"Imbalance ratio is {ir:.1f} ({severity}). "
                f"Majority class '{dist.majority_class}' has {dist.majority_count} events, "
                f"minority class '{dist.minority_class}' has {dist.minority_count}."
            )

        if severity == "mild":
            reasons.append(
                f"Mild imbalance detected (IR={ir:.1f}). "
                "Augmentation is optional but may improve minority class performance."
            )

        if below_min:
            reasons.append(
                f"Classes below minimum sample threshold ({MIN_SAMPLES_PER_CLASS}): "
                f"{', '.join(below_min)}. Augmentation recommended regardless of IR."
            )

        recommended = severity in ("moderate", "severe") or len(below_min) > 0
        reason = " ".join(reasons)
        return recommended, reason

    # ── Utilities ─────────────────────────────────────────────────────────

    def _detect_users(self) -> List[str]:
        """Auto-detect user subdirectories under source_dir."""
        if not self.source_dir.exists():
            log.warning(f"[{MODULE_LABEL}] Source dir not found: {self.source_dir}")
            return []
        return sorted(
            d.name for d in self.source_dir.iterdir()
            if d.is_dir() and d.name.startswith("user_")
        )

    def _log_report(self, report: ImbalanceReport) -> None:
        """Log a human-readable summary of the imbalance report."""
        dist = report.distribution
        log.info(f"[{MODULE_LABEL}] === Imbalance Analysis Report ===")
        log.info(f"[{MODULE_LABEL}]   Users analysed:    {dist.n_users}")
        log.info(f"[{MODULE_LABEL}]   Total events:      {dist.total_events}")
        log.info(f"[{MODULE_LABEL}]   Classes found:     {dist.n_classes}")
        log.info(f"[{MODULE_LABEL}]   Imbalance ratio:   {report.imbalance_ratio}")
        log.info(f"[{MODULE_LABEL}]   Severity:          {report.severity}")

        for cls in sorted(dist.class_counts.keys()):
            cnt = dist.class_counts[cls]
            ir = report.per_class_ir.get(cls, 0)
            flag = " *" if cls in report.below_minimum else ""
            log.info(f"[{MODULE_LABEL}]     {cls:12s}: {cnt:4d} events  "
                     f"(IR={ir:.1f}){flag}")

        if report.below_minimum:
            log.info(f"[{MODULE_LABEL}]   * Below minimum sample threshold "
                     f"({MIN_SAMPLES_PER_CLASS})")

        log.info(f"[{MODULE_LABEL}]   Recommendation: "
                 f"{'AUGMENT' if report.augmentation_recommended else 'NO AUGMENTATION'}")
        log.info(f"[{MODULE_LABEL}]   Reason: {report.reason}")
