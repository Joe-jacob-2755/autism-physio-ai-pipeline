"""
=============================================================================
MODULE 3 – DATA SPLITTING  |  exporter.py
=============================================================================
Exports train/val/test split data as structured CSV folders with metadata.

Output folder structure (raw-signal split — before preprocessing):

  outputs/M3_v1.0.0_run_NNN/
    +-- train/
    |   +-- user_001/
    |   |   +-- EDA.csv, BVP.csv, IBI.csv, ST.csv, ACC.csv
    |   |   +-- combined_signals.csv
    |   +-- user_003/
    |       +-- ...
    +-- val/               (only if three_way split)
    |   +-- user_005/
    |       +-- ...
    +-- test/
    |   +-- user_002/
    |       +-- ...
    +-- split_metadata.json
    +-- user_assignments.csv

Test data is saved separately to enforce the discipline of not touching
it until final model evaluation.  The split_metadata.json records all
parameters needed to reproduce the exact same split.
=============================================================================
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from config import MODULE_VERSION, MODULE_LABEL, OUTPUT_ROOT, FLOAT_FMT


class SplitExporter:
    """
    Write all Module 3 output files.

    Parameters
    ----------
    output_root : base output directory (default: module_3/outputs/)
    """

    def __init__(self, output_root: str | Path = None):
        self.output_root = Path(output_root) if output_root else (
            Path(__file__).resolve().parent / OUTPUT_ROOT
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_dir = self._next_run_folder()
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ── Master export (raw signals, per-user) ────────────────────────────

    def export_raw_signals(
        self,
        split_result,
    ) -> Dict[str, Path]:
        """
        Write raw signal CSVs organized by split and user.

        Parameters
        ----------
        split_result : SplitResult with train_data / val_data / test_data
                       structured as {user_id: {signal_name: DataFrame}}

        Returns
        -------
        {label: Path} of all written files
        """
        saved = {}

        for split_name, user_dict in [
            ("train", split_result.train_data),
            ("val", split_result.val_data),
            ("test", split_result.test_data),
        ]:
            if not user_dict:
                continue
            print(f"[Exporter {MODULE_LABEL}] Writing {split_name} split "
                  f"({len(user_dict)} users) ...")
            for uid, signals in user_dict.items():
                user_dir = self.run_dir / split_name / uid
                user_dir.mkdir(parents=True, exist_ok=True)
                for sig_name, df in signals.items():
                    fname = f"{sig_name}.csv"
                    path = user_dir / fname
                    df.to_csv(path, index=False, float_format=FLOAT_FMT)
                    saved[f"{split_name}_{uid}_{sig_name}"] = path
                print(f"  {uid}: {len(signals)} signals")

        # ── User assignments CSV ──────────────────────────────────────
        print(f"[Exporter {MODULE_LABEL}] Writing user assignments ...")
        saved["user_assignments"] = self._write_user_assignments(split_result)

        # ── Split metadata JSON ───────────────────────────────────────
        print(f"[Exporter {MODULE_LABEL}] Writing metadata ...")
        saved["metadata"] = self._write_metadata(split_result)

        total_files = len(saved)
        print(f"\n[Exporter {MODULE_LABEL}] Done -- {total_files} files "
              f"written to {self.run_dir.name}/")
        return saved

    # ── Legacy: feature-based export (backward compat) ───────────────────

    def export_all(
        self,
        split_result,
        norm_feature_dfs: Dict[str, pd.DataFrame] = None,
        norm_combined_df: pd.DataFrame = None,
    ) -> Dict[str, Path]:
        """
        Write feature-based split data (legacy, for downstream compat).

        Parameters
        ----------
        split_result      : SplitResult from DataSplitter
        norm_feature_dfs  : {signal_name: DataFrame} normalised per-signal
        norm_combined_df  : normalised combined DataFrame (optional)

        Returns
        -------
        {label: Path} of all written files
        """
        saved = {}

        # ── Train split ───────────────────────────────────────────────
        print(f"[Exporter {MODULE_LABEL}] Writing train split ...")
        saved.update(self._write_feature_split(
            "train", split_result.train_data,
            norm_feature_dfs, norm_combined_df,
            split_result.train_users,
        ))

        # ── Validation split (three_way only) ─────────────────────────
        if split_result.val_users:
            print(f"[Exporter {MODULE_LABEL}] Writing validation split ...")
            saved.update(self._write_feature_split(
                "val", split_result.val_data,
                norm_feature_dfs, norm_combined_df,
                split_result.val_users,
            ))

        # ── Test split ────────────────────────────────────────────────
        print(f"[Exporter {MODULE_LABEL}] Writing test split ...")
        saved.update(self._write_feature_split(
            "test", split_result.test_data,
            norm_feature_dfs, norm_combined_df,
            split_result.test_users,
        ))

        # ── User assignments CSV ──────────────────────────────────────
        print(f"[Exporter {MODULE_LABEL}] Writing user assignments ...")
        saved["user_assignments"] = self._write_user_assignments(split_result)

        # ── Split metadata JSON ───────────────────────────────────────
        print(f"[Exporter {MODULE_LABEL}] Writing metadata ...")
        saved["metadata"] = self._write_metadata(split_result)

        total_files = len(saved)
        print(f"\n[Exporter {MODULE_LABEL}] Done -- {total_files} files "
              f"written to {self.run_dir.name}/")
        return saved

    # ── Write one feature-based split ────────────────────────────────────

    def _write_feature_split(
        self,
        split_name: str,
        raw_data: Dict[str, pd.DataFrame],
        norm_feature_dfs: Dict[str, pd.DataFrame],
        norm_combined_df: pd.DataFrame,
        user_ids: list,
    ) -> Dict[str, Path]:
        """Write feature CSVs for one split."""
        saved = {}
        split_dir = self.run_dir / split_name

        # ── Raw features ──────────────────────────────────────────────
        raw_dir = split_dir / "features_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        for sig_name, df in raw_data.items():
            if sig_name == "combined":
                fname = "combined_features.csv"
            else:
                fname = f"{sig_name}_features.csv"
            path = self._write_df(df, raw_dir / fname, split_name)
            saved[f"{split_name}_raw_{sig_name}"] = path

        # ── Normalised features (filter by user_ids) ──────────────────
        if norm_feature_dfs:
            norm_dir = split_dir / "features_normalised"
            norm_dir.mkdir(parents=True, exist_ok=True)

            for sig_name, df in norm_feature_dfs.items():
                if "user_id" in df.columns:
                    split_df = df[df["user_id"].isin(user_ids)].copy()
                else:
                    split_df = df
                fname = f"{sig_name}_features_norm.csv"
                path = self._write_df(split_df, norm_dir / fname, split_name)
                saved[f"{split_name}_norm_{sig_name}"] = path

            if norm_combined_df is not None and len(norm_combined_df) > 0:
                if "user_id" in norm_combined_df.columns:
                    split_comb = norm_combined_df[
                        norm_combined_df["user_id"].isin(user_ids)
                    ].copy()
                else:
                    split_comb = norm_combined_df
                path = self._write_df(
                    split_comb,
                    norm_dir / "combined_features_norm.csv",
                    split_name,
                )
                saved[f"{split_name}_norm_combined"] = path

        return saved

    # ── User assignments ──────────────────────────────────────────────────

    def _write_user_assignments(self, split_result) -> Path:
        """Write a CSV mapping each user to their assigned split."""
        rows = []
        for uid in split_result.train_users:
            rows.append({"user_id": uid, "split": "train"})
        for uid in split_result.val_users:
            rows.append({"user_id": uid, "split": "val"})
        for uid in split_result.test_users:
            rows.append({"user_id": uid, "split": "test"})

        df = pd.DataFrame(rows)
        path = self.run_dir / "user_assignments.csv"
        df.to_csv(path, index=False)
        print(f"  [Exporter {MODULE_LABEL}] Saved user_assignments.csv  "
              f"({len(df)} users)")
        return path

    # ── Metadata JSON ─────────────────────────────────────────────────────

    def _write_metadata(self, split_result) -> Path:
        """Write split configuration and summary as JSON."""
        summary = split_result.summary()

        # Handle both raw-signal and feature-based split result formats
        def _count_items(data):
            if not data:
                return {}
            first_val = next(iter(data.values()), None)
            if isinstance(first_val, dict):
                # Raw-signal format: {user_id: {signal: DataFrame}}
                return {uid: len(sigs) for uid, sigs in data.items()}
            elif isinstance(first_val, pd.DataFrame):
                # Feature format: {signal_name: DataFrame}
                return {k: len(v) for k, v in data.items()}
            return {}

        meta = {
            "module": f"{MODULE_LABEL} v{MODULE_VERSION}",
            "split_at": datetime.now().isoformat(),
            "run_folder": self.run_dir.name,
            **summary,
            "train_contents": _count_items(split_result.train_data),
            "val_contents": _count_items(split_result.val_data),
            "test_contents": _count_items(split_result.test_data),
            "train_demographics": split_result.train_demographics,
            "val_demographics": split_result.val_demographics,
            "test_demographics": split_result.test_demographics,
        }

        path = self.run_dir / "split_metadata.json"
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  [Exporter {MODULE_LABEL}] Saved split_metadata.json")
        return path

    # ── Auto-numbered run folder ──────────────────────────────────────────

    def _next_run_folder(self) -> Path:
        """Find next available run number: M3_v1.0.0_run_NNN."""
        prefix = f"{MODULE_LABEL}_v{MODULE_VERSION}_run_"
        pattern = re.compile(
            rf"^{re.escape(MODULE_LABEL)}_v{re.escape(MODULE_VERSION)}"
            r"_run_(\d+)$"
        )
        existing = [
            int(m.group(1))
            for d in self.output_root.iterdir()
            if d.is_dir() and (m := pattern.match(d.name))
        ]
        next_num = (max(existing) + 1) if existing else 1
        return self.output_root / f"{prefix}{next_num:03d}"

    # ── Helper ────────────────────────────────────────────────────────────

    def _write_df(self, df: pd.DataFrame, path: Path,
                  split_name: str = "") -> Path:
        """Write a DataFrame to CSV with progress logging."""
        if df is None or len(df) == 0:
            print(f"  [Exporter {MODULE_LABEL}] SKIPPED {path.name} (empty)")
            return path
        df.to_csv(path, index=False, float_format=FLOAT_FMT)
        print(f"  [Exporter {MODULE_LABEL}] {split_name}/{path.name}  "
              f"({len(df):,} rows, {df.shape[1]} cols)")
        return path
