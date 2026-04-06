"""
=============================================================================
MODULE 2 – DATA ACQUISITION  |  mode_2_4_deployment.py
=============================================================================
Mode 2.4 — Non-Annotated Data Ingestion (Deployment Mode)

Accepts raw, unannotated physiological data and wraps it as a PipelinePacket
with is_annotated=False, routing it toward the deployment inference module
(Module 9) rather than the training pipeline (Modules 3-8).

This mode is used when:
  - The model is deployed and making real-time predictions on new children
  - Researchers collect data without ground-truth annotation
  - Data is ingested from third-party systems without label information

The packet produced is identical in structure to annotated packets but
has no target_label, event_id, or category columns. Downstream modules
check is_annotated before attempting to use label data.

=============================================================================
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from pipeline_packet import (
    PipelinePacket, build_packet_from_dataframes,
    SOURCE_DEPLOYMENT, ANNOTATION_COLS, REQUIRED_COL, _new_session_id,
)


# ─────────────────────────────────────────────────────────────────────────────
# DEPLOYMENT INGESTER
# ─────────────────────────────────────────────────────────────────────────────

class DeploymentIngester:
    """
    Ingest raw unannotated physiological data for deployment inference.

    Accepts the same folder layouts as Mode 2.1 (import) but strips
    any annotation columns and marks the packet as is_annotated=False.

    Parameters
    ----------
    source_path     : folder or CSV file containing signal data
    user_id         : participant identifier (can be anonymised)
    session_id      : optional session ID (auto-generated if None)
    strip_labels    : if True, actively remove any existing annotation columns
                      (for privacy / blinded evaluation)
    verbose         : print progress messages
    """

    SIGNAL_FILES = {
        "EDA": ("EDA.csv",  ["EDA_uS"]),
        "BVP": ("BVP.csv",  ["BVP_nT"]),
        "IBI": ("IBI.csv",  ["IBI_ms"]),
        "ST":  ("ST.csv",   ["ST_degC"]),
        "ACC": ("ACC.csv",  ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"]),
    }

    def __init__(
        self,
        source_path:  str | Path,
        user_id:      str           = "deployment_participant",
        session_id:   Optional[str] = None,
        strip_labels: bool          = True,
        verbose:      bool          = True,
    ):
        self.source_path  = Path(source_path)
        self.user_id      = user_id
        self.session_id   = session_id or _new_session_id("DEP")
        self.strip_labels = strip_labels
        self.verbose      = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> PipelinePacket:
        """Load data, strip annotations, return deployment PipelinePacket."""
        self._log(f"\n[Deployment 2.4] Ingesting data from: {self.source_path}")

        if not self.source_path.exists():
            raise FileNotFoundError(f"Source path not found: {self.source_path}")

        # Load signals (reuse import logic)
        signals, combined = self._load(self.source_path)

        # Strip annotation columns if present
        if self.strip_labels:
            signals  = {k: self._strip(df) for k, df in signals.items()}
            combined = self._strip(combined)
        else:
            # Just check — warn if labels found but we're in deployment mode
            if any("target_label" in df.columns for df in signals.values()):
                self._log(
                    "  WARNING: Annotation columns found in deployment data.\n"
                    "  Set strip_labels=True to remove them, or use Mode 2.1 "
                    "for training data."
                )

        # Load metadata
        extra = self._load_metadata()
        extra.update({
            "source_path":    str(self.source_path),
            "deployment_mode": True,
            "ingested_at":    datetime.now().isoformat(),
        })

        packet = build_packet_from_dataframes(
            signals      = signals,
            combined     = combined,
            source_type  = SOURCE_DEPLOYMENT,
            is_annotated = False,
            user_id      = self.user_id,
            session_id   = self.session_id,
            extra_meta   = extra,
        )

        self._log(f"[Deployment 2.4] Packet ready.  is_annotated=False.")
        self._log(packet.summary())
        return packet

    # ── Helpers ──────────────────────────────────────────────────────────

    def _load(self, source: Path):
        """Load signal files using the same logic as the importer."""
        # Import inline to avoid circular deps
        from mode_2_1_import import DataImporter
        importer = DataImporter(
            source_path = source,
            user_id     = self.user_id,
            session_id  = self.session_id,
            verbose     = False,
        )
        # Use importer's internal load methods directly
        if source.is_file():
            signals, combined = importer._load_single_csv(source)
        else:
            signals, combined = importer._load_folder(source)
        return signals, combined

    def _strip(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove annotation columns from a DataFrame."""
        cols_to_drop = [c for c in ANNOTATION_COLS if c in df.columns]
        if cols_to_drop:
            return df.drop(columns=cols_to_drop)
        return df

    def _load_metadata(self) -> dict:
        if self.source_path.is_dir():
            meta_path = self.source_path / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as fh:
                    meta = json.load(fh)
                # Remove any sensitive user info if anonymising
                meta.pop("user", None)
                return meta
        return {}

    def _log(self, msg):
        if self.verbose:
            print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def ingest_for_deployment(
    source_path:  str | Path,
    user_id:      str  = "deployment_participant",
    strip_labels: bool = True,
    verbose:      bool = True,
) -> PipelinePacket:
    """
    Shorthand: ingest unannotated data for the deployment inference path.

    Parameters
    ----------
    source_path  : folder or CSV file
    user_id      : participant identifier (can be anonymised)
    strip_labels : remove any existing annotation columns
    verbose      : print progress

    Returns
    -------
    PipelinePacket with is_annotated=False, routed to Module 9 (Deployment)
    """
    return DeploymentIngester(
        source_path  = source_path,
        user_id      = user_id,
        strip_labels = strip_labels,
        verbose      = verbose,
    ).run()
