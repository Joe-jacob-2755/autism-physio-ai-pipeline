"""
=============================================================================
MODULE 2 – DATA ACQUISITION  |  pipeline_packet.py
=============================================================================
PipelinePacket — the standardised data container passed between every
module in the Autism Physio-AI Pipeline.

Every acquisition mode (import, simulate, live, deployment) produces an
identical PipelinePacket, so all downstream modules (preprocessing,
feature extraction, model training, inference) operate identically
regardless of data source.

Schema
------
  signals       : {signal_name: pd.DataFrame}  — one DF per channel,
                  always containing columns: timestamp_s, <value_col(s)>
                  Annotated modes also include: target_label, event_id, category
  combined      : pd.DataFrame  — all channels at 64 Hz (interpolated)
  metadata      : dict  — source, session, user, timestamps, signal stats
  source_type   : 'imported' | 'simulated' | 'live' | 'deployment'
  is_annotated  : bool  — False for deployment mode (no target labels)
  session_id    : str   — unique session identifier
  user_id       : str   — participant identifier

=============================================================================
"""

from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE TYPE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_IMPORTED   = "imported"
SOURCE_SIMULATED  = "simulated"
SOURCE_LIVE       = "live"
SOURCE_DEPLOYMENT = "deployment"

VALID_SOURCES = {SOURCE_IMPORTED, SOURCE_SIMULATED, SOURCE_LIVE, SOURCE_DEPLOYMENT}

# Expected signal columns for each channel (value columns only, no metadata)
SIGNAL_VALUE_COLS: Dict[str, list] = {
    "EDA":   ["EDA_uS"],
    "BVP":   ["BVP_nT"],
    "IBI":   ["IBI_ms"],
    "ST":    ["ST_degC"],
    "ACC":   ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"],
}

ANNOTATION_COLS = ["target_label", "event_id", "category"]
REQUIRED_COL    = "timestamp_s"


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PACKET
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelinePacket:
    """
    Standardised data container passed between pipeline modules.

    Parameters
    ----------
    signals       : dict of {channel_name: DataFrame}
    combined      : single DataFrame with all channels at 64 Hz
    metadata      : dict of session / source / signal statistics
    source_type   : where the data came from
    is_annotated  : whether target_label columns are present
    session_id    : unique session string (auto-generated if not supplied)
    user_id       : participant identifier
    """
    signals:     Dict[str, pd.DataFrame]
    combined:    pd.DataFrame
    metadata:    dict
    source_type: str
    is_annotated: bool
    session_id:  str
    user_id:     str

    # ── Validation ─────────────────────────────────────────────────────────

    def __post_init__(self):
        if self.source_type not in VALID_SOURCES:
            raise ValueError(
                f"source_type must be one of {VALID_SOURCES}, "
                f"got '{self.source_type}'"
            )
        self._validate_signals()

    def _validate_signals(self):
        """Check required columns are present in each signal DataFrame."""
        for name, df in self.signals.items():
            if REQUIRED_COL not in df.columns:
                raise ValueError(
                    f"Signal '{name}' DataFrame is missing '{REQUIRED_COL}' column."
                )

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def signal_names(self) -> list:
        return list(self.signals.keys())

    @property
    def duration_s(self) -> float:
        """Recording duration derived from combined DataFrame."""
        if len(self.combined) > 0:
            return float(self.combined["timestamp_s"].max())
        return 0.0

    @property
    def n_events(self) -> int:
        if not self.is_annotated or "event_id" not in self.combined.columns:
            return 0
        return int(self.combined["event_id"].max())

    @property
    def unique_labels(self) -> list:
        if not self.is_annotated or "target_label" not in self.combined.columns:
            return []
        return sorted(self.combined["target_label"].unique().tolist())

    # ── Serialisation ───────────────────────────────────────────────────────

    def save(self, output_dir: str | Path):
        """
        Save the packet to disk.

        Creates:
          <output_dir>/
            EDA.csv, BVP.csv, IBI.csv, ST.csv, ACC.csv
            combined_signals.csv
            packet_metadata.json
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for name, df in self.signals.items():
            df.to_csv(out / f"{name}.csv", index=False, float_format="%.6f")

        self.combined.to_csv(
            out / "combined_signals.csv", index=False, float_format="%.6f"
        )

        meta = dict(self.metadata)
        meta.update({
            "session_id":   self.session_id,
            "user_id":      self.user_id,
            "source_type":  self.source_type,
            "is_annotated": self.is_annotated,
            "duration_s":   round(self.duration_s, 3),
            "n_events":     self.n_events,
            "unique_labels": self.unique_labels,
            "saved_at":     datetime.now().isoformat(),
        })
        with open(out / "packet_metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, input_dir: str | Path) -> "PipelinePacket":
        """
        Load a saved packet from disk.
        Expects the directory structure produced by save().
        """
        inp = Path(input_dir)

        signals = {}
        for name in ("EDA", "BVP", "IBI", "ST", "ACC"):
            p = inp / f"{name}.csv"
            if p.exists():
                signals[name] = pd.read_csv(p)

        combined = pd.read_csv(inp / "combined_signals.csv")

        with open(inp / "packet_metadata.json") as fh:
            meta = json.load(fh)

        return cls(
            signals      = signals,
            combined     = combined,
            metadata     = meta,
            source_type  = meta.get("source_type", SOURCE_IMPORTED),
            is_annotated = meta.get("is_annotated", True),
            session_id   = meta.get("session_id", _new_session_id()),
            user_id      = meta.get("user_id", "unknown"),
        )

    # ── Summary ──────────────────────────────────────────────────────────

    def summary(self) -> str:
        ann_str = "YES" if self.is_annotated else "NO (deployment mode)"
        labels  = ", ".join(self.unique_labels) if self.unique_labels else "none"
        lines   = [
            "=" * 60,
            "  PIPELINE PACKET",
            "=" * 60,
            f"  Session ID    : {self.session_id}",
            f"  User ID       : {self.user_id}",
            f"  Source        : {self.source_type}",
            f"  Annotated     : {ann_str}",
            f"  Duration      : {self.duration_s:.1f} s",
            f"  Events        : {self.n_events}",
            f"  Labels        : {labels}",
            "",
            "  Signals",
        ]
        for name, df in self.signals.items():
            lines.append(f"    {name:<6} → {len(df):>7,} rows  |  "
                         f"cols: {list(df.columns)}")
        lines += [
            "",
            f"  Combined      → {len(self.combined):>7,} rows  |  "
            f"cols: {list(self.combined.columns)}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _new_session_id(prefix: str = "SES") -> str:
    """Generate a short unique session ID e.g. SES-20250406-A3F7."""
    ts   = datetime.now().strftime("%Y%m%d-%H%M%S")
    uid  = hashlib.md5(ts.encode()).hexdigest()[:4].upper()
    return f"{prefix}-{ts}-{uid}"


def build_packet_from_dataframes(
    signals:     Dict[str, pd.DataFrame],
    combined:    pd.DataFrame,
    source_type: str,
    is_annotated: bool,
    user_id:     str       = "unknown",
    session_id:  str       = None,
    extra_meta:  dict      = None,
) -> PipelinePacket:
    """
    Convenience builder — construct a PipelinePacket from DataFrames.

    Parameters
    ----------
    signals     : {channel: DataFrame} — per-signal CSVs loaded as DataFrames
    combined    : combined signals DataFrame
    source_type : one of VALID_SOURCES
    is_annotated: whether annotation columns are present
    user_id     : participant ID
    session_id  : if None, one is auto-generated
    extra_meta  : any additional metadata fields
    """
    sid  = session_id or _new_session_id()
    meta = {"source_type": source_type, "user_id": user_id}
    if extra_meta:
        meta.update(extra_meta)

    # Add signal statistics
    meta["signal_stats"] = {}
    for name, df in signals.items():
        val_cols = [c for c in df.columns
                    if c not in (REQUIRED_COL,) + tuple(ANNOTATION_COLS)]
        for col in val_cols:
            meta["signal_stats"][f"{name}.{col}"] = {
                "n":    int(len(df)),
                "mean": round(float(df[col].mean()), 4),
                "std":  round(float(df[col].std()),  4),
                "min":  round(float(df[col].min()),  4),
                "max":  round(float(df[col].max()),  4),
            }

    return PipelinePacket(
        signals      = signals,
        combined     = combined,
        metadata     = meta,
        source_type  = source_type,
        is_annotated = is_annotated,
        session_id   = sid,
        user_id      = user_id,
    )
