"""
=============================================================================
MODULE 2B -- DATA ANNOTATION  |  loader.py
=============================================================================
SignalLoader -- load raw (unannotated) physiological data from a
PipelinePacket object or a folder on disk.

Handles:
  - PipelinePacket passed directly (in-memory)
  - Folder containing per-signal CSVs + combined_signals.csv + metadata
  - Re-annotation: detects existing annotation columns and can strip them

Author  : Autism Physio-AI Pipeline
Version : 1.0.0
=============================================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from config import ANNOTATION_COLS


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL LOADER
# ─────────────────────────────────────────────────────────────────────────────

class SignalLoader:
    """Load physiological signal data for annotation.

    Accepts either a PipelinePacket object (in-memory) or a folder path
    containing the standard output structure (EDA.csv, BVP.csv, ...,
    combined_signals.csv, packet_metadata.json).

    Parameters
    ----------
    verbose : bool
        Print status messages to stdout.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def load(self, source) -> Tuple:
        """Load data from a PipelinePacket or folder path.

        Parameters
        ----------
        source : PipelinePacket | str | Path
            Either an in-memory packet or path to a folder.

        Returns
        -------
        (packet, duration_s, existing_events_df_or_None, warnings)
            - packet : PipelinePacket with annotation columns stripped
            - duration_s : recording duration in seconds
            - existing_events_df : DataFrame from annotations_events.csv
              (if present), else None
            - warnings : list of warning strings
        """
        warnings: List[str] = []
        existing_events_df = None

        # Determine input type
        if hasattr(source, "signals") and hasattr(source, "combined"):
            # In-memory PipelinePacket
            packet = source
            if self.verbose:
                print("  [LOAD] Using in-memory PipelinePacket")
        else:
            # Folder path
            folder = Path(source)
            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Source folder not found: {folder}"
                )
            packet = self._load_from_folder(folder)
            # Check for existing annotations_events.csv
            events_path = folder / "annotations_events.csv"
            if events_path.exists():
                existing_events_df = pd.read_csv(events_path)
                if self.verbose:
                    n = len(existing_events_df)
                    print(f"  [LOAD] Found existing annotations_events.csv "
                          f"({n} events)")

        # Detect duration
        duration_s = self._detect_duration(packet)
        if self.verbose:
            print(f"  [LOAD] Recording duration: {duration_s:.2f}s")
            print(f"  [LOAD] Signals: {', '.join(packet.signals.keys())}")

        # Check if already annotated
        was_annotated = self._check_already_annotated(packet)
        if was_annotated:
            warnings.append(
                "Data already has annotation columns. "
                "These will be stripped for re-annotation."
            )
            if self.verbose:
                print("  [LOAD] Existing annotation columns detected "
                      "-- stripping for re-annotation")
            self._strip_annotation_cols(packet)

        return packet, duration_s, existing_events_df, warnings

    # ── Folder loading ────────────────────────────────────────────────────

    def _load_from_folder(self, folder: Path):
        """Load a PipelinePacket from a folder on disk.

        Uses PipelinePacket.load() via isolated import from M1.
        Falls back to manual CSV loading if PipelinePacket is not
        available.
        """
        # Try loading via PipelinePacket.load()
        meta_path = folder / "packet_metadata.json"
        combined_path = folder / "combined_signals.csv"

        if meta_path.exists() and combined_path.exists():
            return self._load_via_pipeline_packet(folder)

        # Fallback: manual CSV loading
        return self._load_manual(folder)

    def _load_via_pipeline_packet(self, folder: Path):
        """Load using PipelinePacket.load() from M1."""
        repo_root = Path(__file__).resolve().parent.parent
        m1_dir = repo_root / "module_1_data_acquisition"

        if not m1_dir.exists():
            return self._load_manual(folder)

        saved_path = list(sys.path)
        try:
            sys.path.insert(0, str(m1_dir))
            from pipeline_packet import PipelinePacket
            packet = PipelinePacket.load(folder)
        finally:
            sys.path[:] = saved_path

        return packet

    def _load_manual(self, folder: Path):
        """Manual fallback: load CSVs into a lightweight packet-like object."""
        signals = {}
        signal_names = ["EDA", "BVP", "IBI", "ST", "ACC"]
        for name in signal_names:
            csv_path = folder / f"{name}.csv"
            if csv_path.exists():
                signals[name] = pd.read_csv(csv_path)

        combined_path = folder / "combined_signals.csv"
        if combined_path.exists():
            combined = pd.read_csv(combined_path)
        elif signals:
            # Use the largest signal DataFrame as combined
            combined = max(signals.values(), key=len).copy()
        else:
            raise FileNotFoundError(
                f"No signal CSVs or combined_signals.csv found in {folder}"
            )

        # Load metadata
        metadata = {}
        meta_path = folder / "packet_metadata.json"
        if meta_path.exists():
            with open(meta_path) as fh:
                metadata = json.load(fh)

        # Build a minimal packet-like object
        return _MinimalPacket(
            signals=signals,
            combined=combined,
            metadata=metadata,
            source_type=metadata.get("source_type", "imported"),
            is_annotated=metadata.get("is_annotated", False),
            session_id=metadata.get("session_id", "unknown"),
            user_id=metadata.get("user_id", "unknown"),
        )

    # ── Annotation detection / stripping ──────────────────────────────────

    @staticmethod
    def _check_already_annotated(packet) -> bool:
        """Check if any DataFrame in the packet has annotation columns."""
        if hasattr(packet, "is_annotated") and packet.is_annotated:
            return True
        if packet.combined is not None:
            if "target_label" in packet.combined.columns:
                return True
        for df in packet.signals.values():
            if "target_label" in df.columns:
                return True
        return False

    @staticmethod
    def _strip_annotation_cols(packet) -> None:
        """Remove annotation columns from all DataFrames in the packet."""
        for name, df in packet.signals.items():
            cols_to_drop = [c for c in ANNOTATION_COLS if c in df.columns]
            if cols_to_drop:
                packet.signals[name] = df.drop(columns=cols_to_drop)

        if packet.combined is not None:
            cols_to_drop = [
                c for c in ANNOTATION_COLS if c in packet.combined.columns
            ]
            if cols_to_drop:
                packet.combined = packet.combined.drop(columns=cols_to_drop)

        packet.is_annotated = False

    @staticmethod
    def _detect_duration(packet) -> float:
        """Detect recording duration from the combined DataFrame."""
        if (packet.combined is not None and len(packet.combined) > 0
                and "timestamp_s" in packet.combined.columns):
            return float(packet.combined["timestamp_s"].max())

        # Fallback: max timestamp across all signals
        max_ts = 0.0
        for df in packet.signals.values():
            if "timestamp_s" in df.columns and len(df) > 0:
                max_ts = max(max_ts, float(df["timestamp_s"].max()))
        return max_ts


# ─────────────────────────────────────────────────────────────────────────────
# MINIMAL PACKET (fallback when PipelinePacket import is not available)
# ─────────────────────────────────────────────────────────────────────────────

class _MinimalPacket:
    """Lightweight packet-like object for standalone loading.

    Provides the same attribute interface as PipelinePacket for the
    annotator to work with.
    """

    def __init__(
        self,
        signals: dict,
        combined: pd.DataFrame,
        metadata: dict,
        source_type: str = "imported",
        is_annotated: bool = False,
        session_id: str = "unknown",
        user_id: str = "unknown",
    ):
        self.signals = signals
        self.combined = combined
        self.metadata = metadata
        self.source_type = source_type
        self.is_annotated = is_annotated
        self.session_id = session_id
        self.user_id = user_id

    @property
    def signal_names(self) -> list:
        return list(self.signals.keys())
