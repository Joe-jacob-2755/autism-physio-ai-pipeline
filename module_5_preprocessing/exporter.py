"""
=============================================================================
MODULE 5 – DATA PREPROCESSING  |  exporter.py
=============================================================================
Exports cleaned and filtered signal CSVs.

After the Module 5/6 split, this exporter writes cleaned signals only.
Feature CSVs are now produced by Module 4 (Feature Engineering).

Folder structure:
  outputs/M3_v2.0.0_run_001/
    +-- cleaned_signals/
    |   +-- EDA.csv
    |   +-- BVP.csv
    |   +-- IBI.csv
    |   +-- ST.csv
    |   +-- ACC.csv
    +-- cleaning_report.csv
    +-- participant_demographics.csv   (copied from source if present)
    +-- preprocessing_metadata.json
=============================================================================
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from config import MODULE_VERSION, FLOAT_FMT


class PreprocessingExporter:
    """
    Write all Module 3 output files.

    Parameters
    ----------
    output_dir : base output directory for this run
    """

    def __init__(self, output_dir: str | Path):
        self.base = Path(output_dir)
        self.signals_dir = self.base / "cleaned_signals"
        self.signals_dir.mkdir(parents=True, exist_ok=True)

    # ── Master export ─────────────────────────────────────────────────────

    def export_all(
        self,
        signals_filtered: Dict[str, pd.DataFrame],
        cleaning_reports: list,
        metadata: dict,
    ) -> Dict[str, Path]:
        """Write all output files and return {label: path} dict."""
        saved = {}

        print("[Exporter M3] Writing cleaned signal CSVs ...")
        saved.update(self._write_signals(signals_filtered))

        print("[Exporter M3] Writing cleaning report ...")
        saved["cleaning_report"] = self._write_cleaning_report(cleaning_reports)

        print("[Exporter M3] Writing metadata ...")
        saved["metadata"] = self._write_metadata(metadata)

        print(f"\n[Exporter M3] Done — {len(saved)} files.")
        return saved

    # ── Signal CSVs ──────────────────────────────────────────────────────

    def _write_signals(
        self,
        signals: Dict[str, pd.DataFrame],
    ) -> Dict[str, Path]:
        """Write one CSV per cleaned/filtered signal."""
        saved = {}
        for sig_name, df in signals.items():
            fname = f"{sig_name}.csv"
            path = self._write_df(df, self.signals_dir / fname)
            saved[sig_name] = path
        return saved

    # ── Cleaning report ───────────────────────────────────────────────────

    def _write_cleaning_report(self, reports: list) -> Path:
        if not reports:
            return None
        rows = [r.to_dict() if hasattr(r, "to_dict") else r for r in reports]
        df = pd.DataFrame(rows)
        return self._write_df(df, self.base / "cleaning_report.csv")

    # ── Metadata JSON ─────────────────────────────────────────────────────

    def _write_metadata(self, metadata: dict) -> Path:
        meta = {
            "module": f"M3 v{MODULE_VERSION}",
            "preprocessed_at": datetime.now().isoformat(),
            **metadata,
        }
        path = self.base / "preprocessing_metadata.json"
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  [Exporter M3] Saved preprocessing_metadata.json")
        return path

    # ── Helper ────────────────────────────────────────────────────────────

    def _write_df(self, df: pd.DataFrame, path: Path) -> Path:
        if df is None or len(df) == 0:
            print(f"  [Exporter M3] SKIPPED {path.name} (empty)")
            return path
        df.to_csv(path, index=False, float_format=FLOAT_FMT)
        print(f"  [Exporter M3] Saved {path.name}  ({len(df):,} rows, "
              f"{df.shape[1]} cols)")
        return path
