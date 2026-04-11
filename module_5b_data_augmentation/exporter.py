"""
=============================================================================
MODULE 5B - DATA AUGMENTATION  |  exporter.py
=============================================================================
Export augmentation results:
  - Synthetic signal segments as CSVs (per signal, per emotion)
  - Augmentation metadata JSON
  - Imbalance report CSV
  - Augmentation plan CSV
  - Sanity check summary CSV
=============================================================================
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import FLOAT_FMT, MODULE_LABEL, MODULE_VERSION
from imbalance_detector import ImbalanceReport
from augmentation_planner import AugmentationPlan
from sanity_checker import SanityReport
from signal_generator import SyntheticSegment

log = logging.getLogger(__name__)


class AugmentationExporter:
    """
    Export augmentation outputs to disk.

    Output structure:
      output_dir/
        synthetic_segments/
          EDA/
            Happy_001.csv, Happy_002.csv, ...
            Anger_001.csv, ...
          BVP/ ...
        imbalance_report.csv
        augmentation_plan.csv
        sanity_summary.csv
        augmentation_metadata.json
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        accepted_segments: Dict[str, Dict[str, List[SyntheticSegment]]],
        report: ImbalanceReport,
        plan: AugmentationPlan,
        sanity_reports: Dict[str, SanityReport],
    ) -> Path:
        """Export all augmentation outputs. Returns output_dir."""

        # 1. Synthetic segments as CSVs
        n_exported = self._export_segments(accepted_segments)

        # 2. Imbalance report
        report_df = report.summary_table()
        report_df.to_csv(
            self.output_dir / "imbalance_report.csv", index=False
        )

        # 3. Augmentation plan
        plan_df = plan.summary_table()
        plan_df.to_csv(
            self.output_dir / "augmentation_plan.csv", index=False
        )

        # 4. Sanity summary
        self._export_sanity_summary(sanity_reports)

        # 5. Metadata
        self._export_metadata(
            accepted_segments, report, plan, sanity_reports, n_exported
        )

        log.info(f"[{MODULE_LABEL}] Exported {n_exported} synthetic segments "
                 f"to {self.output_dir}")
        return self.output_dir

    def _export_segments(
        self,
        accepted: Dict[str, Dict[str, List[SyntheticSegment]]],
    ) -> int:
        """Write synthetic segments as individual CSVs."""
        seg_dir = self.output_dir / "synthetic_segments"
        seg_dir.mkdir(exist_ok=True)
        count = 0

        for sig_name, by_emotion in accepted.items():
            sig_dir = seg_dir / sig_name
            sig_dir.mkdir(exist_ok=True)

            for emotion, segments in by_emotion.items():
                for i, seg in enumerate(segments):
                    fname = f"{emotion}_{i + 1:03d}.csv"

                    # Build DataFrame
                    from config import SIGNAL_SPECS
                    spec = SIGNAL_SPECS.get(sig_name, {})
                    value_cols = spec.get("value_cols", [f"value_{j}" for j in range(seg.values.shape[1])])

                    data = {"timestamp_s": seg.timestamps}
                    for col_idx, col_name in enumerate(value_cols):
                        if col_idx < seg.values.shape[1]:
                            data[col_name] = seg.values[:, col_idx]

                    data["target_label"] = emotion
                    data["event_id"] = f"synthetic_{emotion}_{i + 1:03d}"
                    data["category"] = "synthetic"
                    data["source"] = "CT-TimeGAN"

                    df = pd.DataFrame(data)
                    df.to_csv(
                        sig_dir / fname, index=False,
                        float_format=FLOAT_FMT
                    )
                    count += 1

        return count

    def _export_sanity_summary(
        self, sanity_reports: Dict[str, SanityReport]
    ) -> None:
        """Write sanity check summary CSV."""
        rows = []
        for sig_name, sr in sanity_reports.items():
            row = {
                "signal": sig_name,
                "total": sr.total_segments,
                "passed": sr.passed_segments,
                "failed": sr.failed_segments,
                "acceptance_rate": sr.acceptance_rate,
                "acceptable": sr.acceptable,
            }
            for check_name, rate in sr.check_pass_rates.items():
                row[f"pass_rate_{check_name}"] = rate
            rows.append(row)

        if rows:
            pd.DataFrame(rows).to_csv(
                self.output_dir / "sanity_summary.csv", index=False
            )

    def _export_metadata(
        self,
        accepted: Dict[str, Dict[str, List[SyntheticSegment]]],
        report: ImbalanceReport,
        plan: AugmentationPlan,
        sanity_reports: Dict[str, SanityReport],
        n_exported: int,
    ) -> None:
        """Write augmentation_metadata.json."""
        # Count accepted per signal per emotion
        seg_counts = {}
        for sig, by_emo in accepted.items():
            seg_counts[sig] = {emo: len(segs) for emo, segs in by_emo.items()}

        metadata = {
            "module": f"{MODULE_LABEL} v{MODULE_VERSION}",
            "imbalance": {
                "ratio": report.imbalance_ratio,
                "severity": report.severity,
                "total_events": report.distribution.total_events,
                "n_classes": report.distribution.n_classes,
                "n_users": report.distribution.n_users,
                "class_counts": report.distribution.class_counts,
            },
            "plan": {
                "strategy": plan.strategy,
                "total_synthetic_planned": plan.total_synthetic,
                "synthetic_needed": plan.synthetic_needed,
            },
            "sanity": {
                sig: {
                    "total": sr.total_segments,
                    "passed": sr.passed_segments,
                    "acceptance_rate": sr.acceptance_rate,
                }
                for sig, sr in sanity_reports.items()
            },
            "output": {
                "n_segments_exported": n_exported,
                "segments_per_signal": seg_counts,
            },
        }

        with open(self.output_dir / "augmentation_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
