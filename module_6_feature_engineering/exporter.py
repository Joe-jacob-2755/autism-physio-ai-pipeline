"""
=============================================================================
MODULE 6 – FEATURE ENGINEERING  |  exporter.py
=============================================================================
Exports all feature engineering outputs as structured CSVs.

Output folder structure:
  outputs/M4_v1.0.0_run_NNN/
    +-- features_raw/
    |   +-- EDA_features.csv, BVP_features.csv, ...
    |   +-- combined_features.csv
    +-- features_normalised/
    |   +-- EDA_features_norm.csv, ...
    |   +-- combined_features_norm.csv
    +-- features_encoded/
    |   +-- EDA_features_encoded.csv, ...
    |   +-- combined_features_encoded.csv
    +-- features_selected/
    |   +-- top_N_features.csv
    |   +-- top_N_features_encoded.csv
    |   +-- feature_importance.csv
    +-- dimensionality_reduction/
    |   +-- pca_components.csv
    |   +-- pca_transformed.csv
    |   +-- pca_explained_variance.csv
    |   +-- pls_vip_scores.csv
    |   +-- comparison_report.json
    +-- feature_engineering_metadata.json
=============================================================================
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from config import MODULE_VERSION, FLOAT_FMT
from selection_report import SelectionReportGenerator


class FeatureEngineeringExporter:
    """
    Write all Module 4 output files.

    Parameters
    ----------
    output_dir : base output directory for this run
    """

    def __init__(self, output_dir: str | Path):
        self.base = Path(output_dir)
        self.raw_dir = self.base / "features_raw"
        self.norm_dir = self.base / "features_normalised"
        self.encoded_dir = self.base / "features_encoded"
        self.selected_dir = self.base / "features_selected"
        self.dimred_dir = self.base / "dimensionality_reduction"

        for d in (self.raw_dir, self.norm_dir, self.encoded_dir,
                  self.selected_dir, self.dimred_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── Master export ─────────────────────────────────────────────────────

    def export_all(
        self,
        raw_features: Dict[str, pd.DataFrame],
        norm_features: Dict[str, pd.DataFrame],
        raw_combined: pd.DataFrame,
        norm_combined: pd.DataFrame,
        encoded_features: Dict[str, pd.DataFrame],
        encoded_combined: pd.DataFrame,
        importance_df: pd.DataFrame,
        selected_df: pd.DataFrame,
        selected_encoded_df: pd.DataFrame,
        dimred_results: dict,
        metadata: dict,
        generate_selection_report: bool = True,
    ) -> Dict[str, Path]:
        """Write all output files and return {label: path} dict."""
        saved = {}

        # ── Raw features ──────────────────────────────────────────────
        print("[Exporter M4] Writing raw feature CSVs ...")
        saved.update(self._write_individual(raw_features, self.raw_dir, ""))
        saved["raw_combined"] = self._write_df(
            raw_combined, self.raw_dir / "combined_features.csv")

        # ── Normalised features ───────────────────────────────────────
        print("[Exporter M4] Writing normalised feature CSVs ...")
        saved.update(self._write_individual(
            norm_features, self.norm_dir, "_norm"))
        saved["norm_combined"] = self._write_df(
            norm_combined, self.norm_dir / "combined_features_norm.csv")

        # ── One-hot encoded features ──────────────────────────────────
        print("[Exporter M4] Writing one-hot encoded feature CSVs ...")
        saved.update(self._write_individual(
            encoded_features, self.encoded_dir, "_encoded"))
        saved["encoded_combined"] = self._write_df(
            encoded_combined,
            self.encoded_dir / "combined_features_encoded.csv")

        # ── Selected features ─────────────────────────────────────────
        print("[Exporter M4] Writing feature selection results ...")
        if importance_df is not None and len(importance_df) > 0:
            saved["importance"] = self._write_df(
                importance_df, self.selected_dir / "feature_importance.csv")
        if selected_df is not None and len(selected_df) > 0:
            saved["selected"] = self._write_df(
                selected_df, self.selected_dir / "top_N_features.csv")
        if selected_encoded_df is not None and len(selected_encoded_df) > 0:
            saved["selected_encoded"] = self._write_df(
                selected_encoded_df,
                self.selected_dir / "top_N_features_encoded.csv")

        # ── Dimensionality reduction ──────────────────────────────────
        print("[Exporter M4] Writing dimensionality reduction results ...")
        saved.update(self._write_dimred(dimred_results))

        # ── Feature Selection Report (HTML) ──────────────────────────
        if generate_selection_report and importance_df is not None:
            print("[Exporter M4] Generating feature selection report ...")
            reporter = SelectionReportGenerator(self.base)
            saved["selection_report"] = reporter.generate(
                all_columns=list(raw_combined.columns),
                importance_df=importance_df,
                top_n=len(selected_df) if selected_df is not None else 20,
                selection_method=metadata.get("selection_method", "ensemble"),
                metadata=metadata,
            )

        # ── Metadata ──────────────────────────────────────────────────
        print("[Exporter M4] Writing metadata ...")
        saved["metadata"] = self._write_metadata(metadata)

        total_rows = sum(len(df) for df in raw_features.values())
        print(f"\n[Exporter M4] Done — {len(saved)} files.  "
              f"Total feature rows: {total_rows:,}")
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

    # ── Dimensionality reduction outputs ──────────────────────────────────

    def _write_dimred(self, dimred_results: dict) -> Dict[str, Path]:
        saved = {}
        if not dimred_results:
            return saved

        # PCA
        pca = dimred_results.get("pca_result", {})
        if pca and pca.get("transformed") is not None:
            saved["pca_transformed"] = self._write_df(
                pca["transformed"], self.dimred_dir / "pca_transformed.csv")
            saved["pca_components"] = self._write_df(
                pca["components"], self.dimred_dir / "pca_components.csv")
            ev = pca.get("explained_variance")
            if ev is not None:
                ev_df = pd.DataFrame({
                    "component": [f"PC{i+1}" for i in range(len(ev))],
                    "explained_variance_ratio": ev,
                    "cumulative_variance": np.cumsum(ev),
                })
                saved["pca_variance"] = self._write_df(
                    ev_df, self.dimred_dir / "pca_explained_variance.csv")

        # PLS-VIP
        vip = dimred_results.get("pls_vip_result")
        if vip and vip.get("vip_ranking") is not None:
            saved["pls_vip"] = self._write_df(
                vip["vip_ranking"], self.dimred_dir / "pls_vip_scores.csv")

        # SVD
        svd = dimred_results.get("svd_result", {})
        if svd and svd.get("transformed") is not None:
            saved["svd_transformed"] = self._write_df(
                svd["transformed"], self.dimred_dir / "svd_transformed.csv")

        # Comparison report
        comparison = dimred_results.get("comparison")
        if comparison:
            path = self.dimred_dir / "comparison_report.json"
            with open(path, "w") as f:
                json.dump(comparison, f, indent=2, default=str)
            print("  [Exporter M4] Saved comparison_report.json")
            saved["comparison"] = path

        return saved

    # ── Metadata JSON ─────────────────────────────────────────────────────

    def _write_metadata(self, metadata: dict) -> Path:
        meta = {
            "module": f"M4 v{MODULE_VERSION}",
            "processed_at": datetime.now().isoformat(),
            **metadata,
        }
        path = self.base / "feature_engineering_metadata.json"
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print("  [Exporter M4] Saved feature_engineering_metadata.json")
        return path

    # ── Helper ────────────────────────────────────────────────────────────

    def _write_df(self, df: pd.DataFrame, path: Path) -> Path:
        if df is None or len(df) == 0:
            print(f"  [Exporter M4] SKIPPED {path.name} (empty)")
            return path
        df.to_csv(path, index=False, float_format=FLOAT_FMT)
        print(f"  [Exporter M4] Saved {path.name}  ({len(df):,} rows, "
              f"{df.shape[1]} cols)")
        return path
