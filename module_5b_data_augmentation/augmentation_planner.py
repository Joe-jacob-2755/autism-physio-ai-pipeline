"""
=============================================================================
MODULE 5B - DATA AUGMENTATION  |  augmentation_planner.py
=============================================================================
Compute how many synthetic samples to generate per class based on the
imbalance report from ImbalanceDetector.

Supports four strategies:
  1. "equalize"       - bring all classes to majority count
  2. "median"         - bring below-median classes to median count
  3. "minimum_ratio"  - bring all classes within MAX_RATIO of majority
  4. "custom"         - user specifies per-class targets

Guards against over-generation with MAX_SYNTHETIC_RATIO.
=============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

from config import (
    DEFAULT_STRATEGY,
    MAX_RATIO_AFTER_AUGMENTATION,
    MAX_SYNTHETIC_RATIO,
    MODULE_LABEL,
)
from imbalance_detector import ClassDistribution, ImbalanceReport

log = logging.getLogger(__name__)


@dataclass
class AugmentationPlan:
    """Plan describing how many synthetic samples to generate per class."""
    strategy: str
    targets_per_class: Dict[str, int]       # {emotion: target_count}
    synthetic_needed: Dict[str, int]        # {emotion: n_to_generate}
    current_counts: Dict[str, int]          # {emotion: current_real_count}
    total_synthetic: int
    capped_classes: Dict[str, str] = field(default_factory=dict)
    # {class: reason} for classes where generation was capped

    @property
    def classes_to_augment(self) -> list:
        """Classes that need synthetic samples (n_to_generate > 0)."""
        return [c for c, n in self.synthetic_needed.items() if n > 0]

    def summary_table(self):
        """Return a summary DataFrame."""
        import pandas as pd
        rows = []
        for cls in sorted(self.current_counts.keys()):
            real = self.current_counts[cls]
            target = self.targets_per_class.get(cls, real)
            synth = self.synthetic_needed.get(cls, 0)
            cap = self.capped_classes.get(cls, "")
            rows.append({
                "class": cls,
                "real_count": real,
                "target_count": target,
                "synthetic_needed": synth,
                "total_after": real + synth,
                "synthetic_ratio": round(synth / real, 2) if real > 0 else 0,
                "capped": cap,
            })
        return pd.DataFrame(rows)


class AugmentationPlanner:
    """
    Compute per-class augmentation targets from an ImbalanceReport.

    Parameters
    ----------
    strategy : str
        One of "equalize", "median", "minimum_ratio", "custom".
    max_synthetic_ratio : float
        Maximum ratio of synthetic-to-real samples per class.
    custom_targets : dict, optional
        Per-class target counts for "custom" strategy.
    """

    def __init__(
        self,
        strategy: str = DEFAULT_STRATEGY,
        max_synthetic_ratio: float = MAX_SYNTHETIC_RATIO,
        custom_targets: Optional[Dict[str, int]] = None,
    ):
        self.strategy = strategy
        self.max_synthetic_ratio = max_synthetic_ratio
        self.custom_targets = custom_targets or {}

    def plan(self, report: ImbalanceReport) -> AugmentationPlan:
        """
        Create an augmentation plan from an imbalance report.

        Parameters
        ----------
        report : ImbalanceReport
            Output from ImbalanceDetector.analyse().

        Returns
        -------
        AugmentationPlan
        """
        dist = report.distribution
        counts = dict(dist.class_counts)

        log.info(f"[{MODULE_LABEL}] Planning augmentation with "
                 f"strategy='{self.strategy}'")

        # Compute raw targets based on strategy
        if self.strategy == "equalize":
            targets = self._equalize(counts)
        elif self.strategy == "median":
            targets = self._median(counts)
        elif self.strategy == "minimum_ratio":
            targets = self._minimum_ratio(counts)
        elif self.strategy == "custom":
            targets = self._custom(counts)
        else:
            raise ValueError(
                f"Unknown strategy '{self.strategy}'. "
                "Choose from: equalize, median, minimum_ratio, custom"
            )

        # Compute synthetic needed and apply caps
        synthetic = {}
        capped = {}
        for cls, target in targets.items():
            real = counts.get(cls, 0)
            needed = max(0, target - real)

            # Cap at max_synthetic_ratio * real
            if real > 0:
                cap = int(self.max_synthetic_ratio * real)
                if needed > cap:
                    capped[cls] = (
                        f"Capped from {needed} to {cap} "
                        f"(max synthetic ratio = {self.max_synthetic_ratio}x)"
                    )
                    needed = cap
            elif needed > 0:
                # No real data at all — can't generate from nothing
                capped[cls] = "Cannot augment: no real samples to learn from"
                needed = 0

            synthetic[cls] = needed

        total_synth = sum(synthetic.values())

        plan = AugmentationPlan(
            strategy=self.strategy,
            targets_per_class=targets,
            synthetic_needed=synthetic,
            current_counts=counts,
            total_synthetic=total_synth,
            capped_classes=capped,
        )

        self._log_plan(plan)
        return plan

    # ── Strategies ────────────────────────────────────────────────────────

    def _equalize(self, counts: Dict[str, int]) -> Dict[str, int]:
        """Bring all classes to the majority count."""
        majority = max(counts.values()) if counts else 0
        return {cls: majority for cls in counts}

    def _median(self, counts: Dict[str, int]) -> Dict[str, int]:
        """Bring below-median classes up to the median count."""
        if not counts:
            return {}
        import numpy as np
        values = list(counts.values())
        med = int(np.median(values))
        return {cls: max(cnt, med) for cls, cnt in counts.items()}

    def _minimum_ratio(self, counts: Dict[str, int]) -> Dict[str, int]:
        """Bring all classes within MAX_RATIO_AFTER_AUGMENTATION of majority."""
        if not counts:
            return {}
        majority = max(counts.values())
        min_target = int(majority / MAX_RATIO_AFTER_AUGMENTATION)
        return {cls: max(cnt, min_target) for cls, cnt in counts.items()}

    def _custom(self, counts: Dict[str, int]) -> Dict[str, int]:
        """Apply user-specified per-class targets."""
        targets = {}
        for cls in counts:
            if cls in self.custom_targets:
                targets[cls] = max(counts[cls], self.custom_targets[cls])
            else:
                targets[cls] = counts[cls]  # no change for unspecified classes
        return targets

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_plan(self, plan: AugmentationPlan) -> None:
        log.info(f"[{MODULE_LABEL}] === Augmentation Plan ===")
        log.info(f"[{MODULE_LABEL}]   Strategy:         {plan.strategy}")
        log.info(f"[{MODULE_LABEL}]   Total synthetic:   {plan.total_synthetic}")

        for cls in sorted(plan.current_counts.keys()):
            real = plan.current_counts[cls]
            synth = plan.synthetic_needed.get(cls, 0)
            target = plan.targets_per_class.get(cls, real)
            cap = " [CAPPED]" if cls in plan.capped_classes else ""
            log.info(
                f"[{MODULE_LABEL}]     {cls:12s}: "
                f"{real:4d} real + {synth:4d} synthetic = "
                f"{real + synth:4d} (target {target}){cap}"
            )

        if plan.capped_classes:
            for cls, reason in plan.capped_classes.items():
                log.info(f"[{MODULE_LABEL}]     Cap: {cls} — {reason}")
