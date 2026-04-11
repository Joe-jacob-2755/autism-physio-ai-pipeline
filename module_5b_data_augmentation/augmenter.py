"""
=============================================================================
MODULE 5B - DATA AUGMENTATION  |  augmenter.py
=============================================================================
Master orchestrator for the data augmentation pipeline.

Pipeline:
  1. Detect imbalance (ImbalanceDetector)
  2. CHECKPOINT: show imbalance report, ask user whether to proceed
  3. Plan augmentation (AugmentationPlanner)
  4. CHECKPOINT: show plan, ask user to confirm/modify
  5. Extract segments + train CT-TimeGAN per signal
  6. Generate synthetic segments
  7. Sanity-check synthetic data
  8. CHECKPOINT: show sanity report, ask user to accept/reject
  9. Export augmented training data

Modes:
  interactive=True  -> prompt user at each checkpoint
  interactive=False -> use AUTO_* thresholds from config for decisions
=============================================================================
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    AUTO_ACCEPT_SANITY_RATE,
    AUTO_AUGMENT_THRESHOLD,
    IMBALANCE_THRESHOLDS,
    MODULE_LABEL,
    MODULE_VERSION,
    SIGNAL_SPECS,
    TGAN,
)
from imbalance_detector import ImbalanceDetector, ImbalanceReport
from augmentation_planner import AugmentationPlan, AugmentationPlanner
from signal_generator import SegmentData, SegmentExtractor, SignalGenerator, SyntheticSegment
from sanity_checker import SanityChecker, SanityReport

log = logging.getLogger(__name__)


@dataclass
class AugmentationResult:
    """Result of the full augmentation pipeline."""
    imbalance_report: ImbalanceReport
    augmentation_plan: Optional[AugmentationPlan]
    sanity_reports: Dict[str, SanityReport]
    accepted_segments: Dict[str, Dict[str, List[SyntheticSegment]]]
    # {signal_name: {emotion: [segments]}}
    skipped: bool
    skip_reason: str = ""
    elapsed_s: float = 0.0
    output_dir: Optional[Path] = None


class DataAugmenter:
    """
    Master orchestrator for M5B data augmentation.

    Parameters
    ----------
    source_dir : Path
        Root of M5 preprocessing output (contains user_NNN/ subdirs).
    output_dir : Path
        Where to write augmented data.
    user_ids : list of str, optional
        Training user IDs. Auto-detected if not provided.
    demographics : dict, optional
        {user_id: {autism_severity, verbal_status, ...}}
    strategy : str
        Augmentation strategy (equalize, median, minimum_ratio, custom).
    interactive : bool
        If True, prompt user at checkpoints. If False, use auto thresholds.
    device : str
        "cuda" or "cpu" for TGAN training.
    seed : int
        Master random seed.
    """

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        user_ids: Optional[List[str]] = None,
        demographics: Optional[Dict[str, dict]] = None,
        strategy: str = "median",
        interactive: bool = True,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.user_ids = user_ids
        self.demographics = demographics or {}
        self.strategy = strategy
        self.interactive = interactive
        self.device = device
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ═════════════════════════════════════════════════════════════════════
    # Public API
    # ═════════════════════════════════════════════════════════════════════

    def run(self) -> AugmentationResult:
        """
        Run the full augmentation pipeline with human-in-the-loop checkpoints.

        Returns AugmentationResult (which may have skipped=True if user
        declined or data is already balanced).
        """
        t_start = time.time()

        # ─── Step 1: Detect imbalance ────────────────────────────────
        log.info(f"[{MODULE_LABEL}] Step 1: Detecting class imbalance")
        detector = ImbalanceDetector(self.source_dir, self.user_ids)
        report = detector.analyse()

        # ─── Checkpoint 1: Should we augment? ────────────────────────
        proceed = self._checkpoint_imbalance(report)
        if not proceed:
            return AugmentationResult(
                imbalance_report=report,
                augmentation_plan=None,
                sanity_reports={},
                accepted_segments={},
                skipped=True,
                skip_reason="User declined augmentation or data is balanced",
                elapsed_s=time.time() - t_start,
            )

        # ─── Step 2: Plan augmentation ───────────────────────────────
        log.info(f"[{MODULE_LABEL}] Step 2: Planning augmentation")
        planner = AugmentationPlanner(strategy=self.strategy)
        plan = planner.plan(report)

        if plan.total_synthetic == 0:
            return AugmentationResult(
                imbalance_report=report,
                augmentation_plan=plan,
                sanity_reports={},
                accepted_segments={},
                skipped=True,
                skip_reason="No synthetic samples needed after planning",
                elapsed_s=time.time() - t_start,
            )

        # ─── Checkpoint 2: Confirm plan ──────────────────────────────
        proceed = self._checkpoint_plan(plan)
        if not proceed:
            return AugmentationResult(
                imbalance_report=report,
                augmentation_plan=plan,
                sanity_reports={},
                accepted_segments={},
                skipped=True,
                skip_reason="User declined augmentation plan",
                elapsed_s=time.time() - t_start,
            )

        # ─── Step 3-6: Per-signal TGAN training + generation ─────────
        log.info(f"[{MODULE_LABEL}] Step 3: Training + generating per signal")

        all_accepted: Dict[str, Dict[str, List[SyntheticSegment]]] = {}
        all_sanity: Dict[str, SanityReport] = {}

        # Build user_dirs mapping
        user_dirs = {}
        for uid in (self.user_ids or detector.user_ids):
            cleaned_dir = self.source_dir / uid / "cleaned_signals"
            if cleaned_dir.exists():
                user_dirs[uid] = cleaned_dir
            elif (self.source_dir / uid).exists():
                user_dirs[uid] = self.source_dir / uid

        extractor = SegmentExtractor()

        for sig_name in SIGNAL_SPECS:
            log.info(f"[{MODULE_LABEL}] === Processing signal: {sig_name} ===")

            # Extract real segments
            seg_data = extractor.extract_for_signal(
                sig_name, user_dirs, self.demographics
            )
            if seg_data is None or len(seg_data.segments) < 5:
                log.warning(f"[{MODULE_LABEL}] Insufficient segments for "
                            f"{sig_name} — skipping TGAN training")
                continue

            # Train CT-TimeGAN
            accepted, sanity = self._train_and_generate(
                sig_name, seg_data, plan
            )

            if accepted:
                all_accepted[sig_name] = accepted
            if sanity:
                all_sanity[sig_name] = sanity

        # ─── Checkpoint 3: Accept synthetic data ─────────────────────
        proceed = self._checkpoint_sanity(all_sanity)
        if not proceed:
            return AugmentationResult(
                imbalance_report=report,
                augmentation_plan=plan,
                sanity_reports=all_sanity,
                accepted_segments={},
                skipped=True,
                skip_reason="User rejected synthetic data after sanity check",
                elapsed_s=time.time() - t_start,
            )

        # ─── Step 7: Export ──────────────────────────────────────────
        log.info(f"[{MODULE_LABEL}] Step 7: Exporting augmented data")
        from exporter import AugmentationExporter
        exporter_inst = AugmentationExporter(self.output_dir)
        exporter_inst.export(
            all_accepted, report, plan, all_sanity
        )

        elapsed = time.time() - t_start
        log.info(f"[{MODULE_LABEL}] Augmentation complete in {elapsed:.1f}s")

        return AugmentationResult(
            imbalance_report=report,
            augmentation_plan=plan,
            sanity_reports=all_sanity,
            accepted_segments=all_accepted,
            skipped=False,
            elapsed_s=elapsed,
            output_dir=self.output_dir,
        )

    # ═════════════════════════════════════════════════════════════════════
    # Per-signal TGAN training + generation
    # ═════════════════════════════════════════════════════════════════════

    def _train_and_generate(
        self,
        signal_name: str,
        seg_data: SegmentData,
        plan: AugmentationPlan,
    ) -> Tuple[Optional[Dict[str, List[SyntheticSegment]]], Optional[SanityReport]]:
        """
        Train TGAN on real segments, generate synthetic, sanity-check.

        Returns (accepted_segments_by_emotion, sanity_report) or (None, None).
        """
        import torch
        from tgan_model import CTTimeGAN
        from tgan_trainer import TGANTrainer

        n_features = seg_data.segments.shape[2]

        # Normalise to [0, 1]
        normalised = SignalGenerator.normalise(
            seg_data.segments, seg_data.min_vals, seg_data.max_vals
        )

        # Build model
        model = CTTimeGAN(n_features=n_features)
        log.info(f"[{MODULE_LABEL}] CT-TimeGAN for {signal_name}: "
                 f"{model.count_parameters()['total']:,} parameters")

        # Train
        trainer = TGANTrainer(model, device=self.device, seed=self.seed)
        conditions = {
            "emotion": seg_data.emotions,
            "severity": seg_data.severities,
            "verbal": seg_data.verbals,
        }
        trainer.train(normalised, conditions)

        # Save checkpoint
        ckpt_path = self.output_dir / f"tgan_{signal_name}.pt"
        trainer.save_checkpoint(ckpt_path)

        # Generate
        generator = SignalGenerator(
            model, seg_data, device=self.device, seed=self.seed
        )
        synthetic_by_emotion = generator.generate_for_plan(
            plan.synthetic_needed
        )

        # Flatten for sanity checking
        all_synthetic = []
        for segments in synthetic_by_emotion.values():
            all_synthetic.extend(segments)

        if not all_synthetic:
            return None, None

        # Sanity check
        checker = SanityChecker(seg_data.segments, signal_name)
        sanity = checker.check_all(all_synthetic)

        # Filter to passed segments only
        accepted: Dict[str, List[SyntheticSegment]] = {}
        passed_idx = {
            sr.segment_idx for sr in sanity.per_segment if sr.passed
        }

        flat_idx = 0
        for emo, segments in synthetic_by_emotion.items():
            accepted_list = []
            for seg in segments:
                if flat_idx in passed_idx:
                    accepted_list.append(seg)
                flat_idx += 1
            if accepted_list:
                accepted[emo] = accepted_list

        total_accepted = sum(len(v) for v in accepted.values())
        log.info(f"[{MODULE_LABEL}] {signal_name}: {total_accepted}/"
                 f"{len(all_synthetic)} synthetic segments passed sanity")

        return accepted, sanity

    # ═════════════════════════════════════════════════════════════════════
    # Human-in-the-loop checkpoints
    # ═════════════════════════════════════════════════════════════════════

    def _checkpoint_imbalance(self, report: ImbalanceReport) -> bool:
        """Checkpoint 1: Should we proceed with augmentation?"""
        if self.interactive:
            print("\n" + "=" * 70)
            print(f"  [{MODULE_LABEL}] CHECKPOINT 1: Imbalance Analysis")
            print("=" * 70)
            print(f"\n  Imbalance ratio: {report.imbalance_ratio}")
            print(f"  Severity:        {report.severity}")
            print(f"  Total events:    {report.distribution.total_events}")
            print(f"  Classes:         {report.distribution.n_classes}")
            print(f"\n  Per-class counts:")
            for cls in sorted(report.distribution.class_counts.keys()):
                cnt = report.distribution.class_counts[cls]
                ir = report.per_class_ir.get(cls, 0)
                flag = " ** BELOW MIN" if cls in report.below_minimum else ""
                print(f"    {cls:12s}: {cnt:4d}  (IR={ir:.1f}){flag}")
            print(f"\n  Recommendation: {report.reason}")
            print()

            if not report.augmentation_recommended:
                print("  Augmentation is NOT recommended.")
                resp = input("  Proceed anyway? [y/N]: ").strip().lower()
                return resp in ("y", "yes")
            else:
                resp = input("  Proceed with augmentation? [Y/n]: ").strip().lower()
                return resp not in ("n", "no")
        else:
            # Auto mode: use threshold
            if not report.augmentation_recommended:
                return False
            severity_order = ["balanced", "mild", "moderate", "severe"]
            report_idx = severity_order.index(report.severity)
            auto_idx = severity_order.index(AUTO_AUGMENT_THRESHOLD)
            return report_idx >= auto_idx

    def _checkpoint_plan(self, plan: AugmentationPlan) -> bool:
        """Checkpoint 2: Confirm the augmentation plan."""
        if self.interactive:
            print("\n" + "=" * 70)
            print(f"  [{MODULE_LABEL}] CHECKPOINT 2: Augmentation Plan")
            print("=" * 70)
            print(f"\n  Strategy: {plan.strategy}")
            print(f"  Total synthetic samples: {plan.total_synthetic}")
            print(f"\n  {'Class':12s}  {'Real':>5s}  {'Synth':>5s}  "
                  f"{'Total':>5s}  {'Ratio':>5s}")
            print(f"  {'-'*48}")
            for cls in sorted(plan.current_counts.keys()):
                real = plan.current_counts[cls]
                synth = plan.synthetic_needed.get(cls, 0)
                ratio = synth / real if real > 0 else 0
                cap = " *" if cls in plan.capped_classes else ""
                print(f"  {cls:12s}  {real:5d}  {synth:5d}  "
                      f"{real + synth:5d}  {ratio:5.1f}x{cap}")

            if plan.capped_classes:
                print(f"\n  * Capped classes:")
                for cls, reason in plan.capped_classes.items():
                    print(f"    {cls}: {reason}")

            print()
            resp = input("  Accept this plan? [Y/n]: ").strip().lower()
            return resp not in ("n", "no")
        else:
            return True  # auto mode always accepts plan

    def _checkpoint_sanity(self, sanity_reports: Dict[str, SanityReport]) -> bool:
        """Checkpoint 3: Accept or reject synthetic data."""
        if not sanity_reports:
            log.warning(f"[{MODULE_LABEL}] No sanity reports — nothing to accept")
            return False

        if self.interactive:
            print("\n" + "=" * 70)
            print(f"  [{MODULE_LABEL}] CHECKPOINT 3: Sanity Check Results")
            print("=" * 70)
            for sig, sr in sanity_reports.items():
                status = "PASS" if sr.acceptable else "FAIL"
                print(f"\n  {sig}: {sr.passed_segments}/{sr.total_segments} "
                      f"passed ({sr.acceptance_rate:.0%}) [{status}]")
                for name, rate in sr.check_pass_rates.items():
                    print(f"    {name:20s}: {rate:.0%}")

            all_acceptable = all(sr.acceptable for sr in sanity_reports.values())
            if all_acceptable:
                print("\n  All signals passed minimum acceptance threshold.")
                resp = input("  Accept synthetic data? [Y/n]: ").strip().lower()
                return resp not in ("n", "no")
            else:
                print("\n  WARNING: Some signals failed acceptance threshold.")
                resp = input("  Accept anyway? [y/N]: ").strip().lower()
                return resp in ("y", "yes")
        else:
            # Auto mode
            rates = [sr.acceptance_rate for sr in sanity_reports.values()]
            avg_rate = np.mean(rates) if rates else 0.0
            return avg_rate >= AUTO_ACCEPT_SANITY_RATE
