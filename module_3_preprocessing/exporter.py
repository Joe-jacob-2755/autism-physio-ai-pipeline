"""
=============================================================================
MODULE 3 – DATA PREPROCESSING  |  exporter.py
=============================================================================
Exports four standardised CSV outputs:

  1. Non-normalised individual signal CSVs   → for unimodal models
  2. Non-normalised combined CSV             → for multimodal early fusion
  3. Normalised individual signal CSVs       → pre-scaled unimodal input
  4. Normalised combined CSV                 → pre-scaled multimodal input

Folder structure:
  outputs/M3_v1.0.0_run_001/
    ├── features_raw/
    │   ├── EDA_features.csv
    │   ├── BVP_features.csv
    │   ├── IBI_features.csv
    │   ├── ST_features.csv
    │   ├── ACC_features.csv
    │   └── combined_features.csv
    ├── features_normalised/
    │   ├── EDA_features_norm.csv
    │   ├── BVP_features_norm.csv
    │   ├── IBI_features_norm.csv
    │   ├── ST_features_norm.csv
    │   ├── ACC_features_norm.csv
    │   └── combined_features_norm.csv
    ├── cleaning_report.csv
    └── preprocessing_metadata.json
=============================================================================
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
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
        self.raw_dir = self.base / "features_raw"
        self.norm_dir = self.base / "features_normalised"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.norm_dir.mkdir(parents=True, exist_ok=True)

    # ── Master export ─────────────────────────────────────────────────────

    def export_all(
        self,
        raw_features: Dict[str, pd.DataFrame],
        norm_features: Dict[str, pd.DataFrame],
        raw_combined: pd.DataFrame,
        norm_combined: pd.DataFrame,
        cleaning_reports: list,
        metadata: dict,
    ) -> Dict[str, Path]:
        """Write all output files and return {label: path} dict."""
        saved = {}

        print("[Exporter M3] Writing non-normalised individual CSVs ...")
        saved.update(self._write_individual(raw_features, self.raw_dir, suffix=""))

        print("[Exporter M3] Writing non-normalised combined CSV ...")
        saved["raw_combined"] = self._write_df(
            raw_combined, self.raw_dir / "combined_features.csv")

        print("[Exporter M3] Writing normalised individual CSVs ...")
        saved.update(self._write_individual(norm_features, self.norm_dir, suffix="_norm"))

        print("[Exporter M3] Writing normalised combined CSV ...")
        saved["norm_combined"] = self._write_df(
            norm_combined, self.norm_dir / "combined_features_norm.csv")

        print("[Exporter M3] Writing cleaning report ...")
        saved["cleaning_report"] = self._write_cleaning_report(cleaning_reports)

        print("[Exporter M3] Writing metadata ...")
        saved["metadata"] = self._write_metadata(metadata)

        total = sum(len(df) for df in raw_features.values())
        print(f"\n[Exporter M3] Done — {len(saved)} files.  "
              f"Total feature rows: {total:,}")
        return saved

    # ── Individual signal CSVs ────────────────────────────────────────────

    def _write_individual(
        self,
        feature_dfs: Dict[str, pd.DataFrame],
        out_dir: Path,
        suffix: str,
    ) -> Dict[str, Path]:
        saved = {}
        for sig_name, df in feature_dfs.items():
            fname = f"{sig_name}_features{suffix}.csv"
            path = self._write_df(df, out_dir / fname)
            saved[f"{sig_name}{suffix}"] = path
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
