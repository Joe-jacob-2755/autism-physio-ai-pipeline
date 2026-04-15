"""
=============================================================================
MODULE 2B -- DATA ANNOTATION  |  exporter.py
=============================================================================
Write all output files for a completed annotation run.

Produces:
  - Per-signal CSVs with annotation columns
  - combined_signals.csv with annotation columns
  - annotations_events.csv (event-level metadata)
  - annotations_sample_labels.csv (per-sample at 64 Hz)
  - signal_overview.png + signal_annotated.png
  - packet_metadata.json

Author  : Autism Physio-AI Pipeline
Version : 1.0.0
=============================================================================
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import MODULE_LABEL, MODULE_VERSION, OUTPUT_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-NUMBERED OUTPUT FOLDER
# ─────────────────────────────────────────────────────────────────────────────

def next_run_folder(base_dir: Optional[Path] = None) -> Path:
    """Create the next auto-numbered output folder.

    Pattern: M2B_v1.0.0_run_NNN/
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent / OUTPUT_ROOT

    base_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{MODULE_LABEL}_v{MODULE_VERSION}_run_"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    existing = []
    if base_dir.exists():
        for d in base_dir.iterdir():
            if d.is_dir():
                m = pattern.match(d.name)
                if m:
                    existing.append(int(m.group(1)))

    next_num = (max(existing) + 1) if existing else 1
    run_folder = base_dir / f"{prefix}{next_num:03d}"
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTER
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationExporter:
    """Write all output files for a completed annotation session.

    Parameters
    ----------
    verbose : bool
        Print status messages to stdout.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def export_all(
        self,
        packet,
        annotator,
        visualizer,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """Export all annotation outputs.

        Parameters
        ----------
        packet : PipelinePacket-like
            Annotated packet (after apply_to_packet()).
        annotator : ManualAnnotator
            Annotator with finalised events.
        visualizer : AnnotationVisualizer
            For generating plots.
        output_dir : Path
            Destination folder (already created).

        Returns
        -------
        dict mapping output name to file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: Dict[str, Path] = {}

        # 1. Signal CSVs
        signal_paths = self._export_signals(packet, output_dir)
        outputs.update(signal_paths)

        # 2. Combined signals
        combined_path = self._export_combined(packet, output_dir)
        if combined_path:
            outputs["combined_signals"] = combined_path

        # 3. Annotation events CSV
        events_path = self._export_events(annotator, output_dir)
        if events_path:
            outputs["annotations_events"] = events_path

        # 4. Sample-level labels CSV
        labels_path = self._export_sample_labels(annotator, output_dir)
        if labels_path:
            outputs["annotations_sample_labels"] = labels_path

        # 5. Plots
        try:
            overview_path = output_dir / "signal_overview.png"
            visualizer.plot_overview(overview_path)
            outputs["signal_overview"] = overview_path

            annotated_path = output_dir / "signal_annotated.png"
            visualizer.plot_annotated(annotator.events, annotated_path)
            outputs["signal_annotated"] = annotated_path
        except Exception as e:
            if self.verbose:
                print(f"  [WARN] Plot generation failed: {e}")

        # 6. Metadata
        meta_path = self._export_metadata(
            packet, annotator, output_dir
        )
        outputs["packet_metadata"] = meta_path

        if self.verbose:
            print(f"\n  [EXPORT] {len(outputs)} files written to:")
            print(f"           {output_dir}")
            for name, path in sorted(outputs.items()):
                print(f"             {path.name}")

        return outputs

    # ── Signal CSVs ───────────────────────────────────────────────────────

    def _export_signals(
        self, packet, output_dir: Path,
    ) -> Dict[str, Path]:
        """Write per-signal CSVs with annotation columns."""
        paths = {}
        for name, df in packet.signals.items():
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False, float_format="%.6f")
            paths[name] = path
        return paths

    def _export_combined(
        self, packet, output_dir: Path,
    ) -> Optional[Path]:
        """Write combined_signals.csv with annotation columns."""
        if packet.combined is None or len(packet.combined) == 0:
            return None
        path = output_dir / "combined_signals.csv"
        packet.combined.to_csv(path, index=False, float_format="%.6f")
        return path

    # ── Annotation CSVs ───────────────────────────────────────────────────

    def _export_events(
        self, annotator, output_dir: Path,
    ) -> Optional[Path]:
        """Write annotations_events.csv (event-level metadata)."""
        df = annotator.to_event_dataframe()
        if df.empty:
            return None
        path = output_dir / "annotations_events.csv"
        df.to_csv(path, index=False, float_format="%.6f")
        return path

    def _export_sample_labels(
        self, annotator, output_dir: Path,
    ) -> Optional[Path]:
        """Write annotations_sample_labels.csv (per-sample at 64 Hz)."""
        df = annotator.to_sample_labels_dataframe()
        if df.empty:
            return None
        path = output_dir / "annotations_sample_labels.csv"
        df.to_csv(path, index=False, float_format="%.6f")
        return path

    # ── Metadata ──────────────────────────────────────────────────────────

    def _export_metadata(
        self,
        packet,
        annotator,
        output_dir: Path,
    ) -> Path:
        """Write packet_metadata.json with updated annotation info."""
        # Start from existing metadata if available
        meta = dict(getattr(packet, "metadata", {}))

        # Update with annotation info
        meta.update({
            "module": MODULE_LABEL,
            "module_version": MODULE_VERSION,
            "source_type": getattr(packet, "source_type", "imported"),
            "is_annotated": True,
            "session_id": getattr(packet, "session_id", "unknown"),
            "user_id": getattr(packet, "user_id", "unknown"),
            "n_events": len(annotator.events),
            "unique_labels": sorted(
                set(ev.emotion for ev in annotator.events)
            ),
            "annotation_density": annotator.validator.compute_annotation_density(
                annotator.events, annotator.duration_s
            ),
            "duration_s": round(annotator.duration_s, 3),
            "signals_present": list(packet.signals.keys()),
            "annotated_at": datetime.now().isoformat(),
        })

        path = output_dir / "packet_metadata.json"
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        return path
