"""
=============================================================================
MODULE 4 - EXPLORATORY DATA ANALYSIS  |  analyser.py
=============================================================================
CombinedEDA — master orchestrator for the 10-phase EDA pipeline.

Loads combined multi-user data, runs all analyses, generates all charts,
and builds the thesis-quality Word report.
=============================================================================
"""
from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import MODULE_VERSION, MODULE_LABEL, OUTPUT_ROOT
from combined_loader import CombinedDataLoader, CombinedData
from data_quality import run_all_quality
from statistical_analyser import run_all_statistics
from correlation_analyser import run_all_correlations
from signal_analyser import SignalAnalyser, signal_pct_change_summary, time_to_peak_summary
from visualiser import EDAVisualiser
from report_builder import DocxReportBuilder

log = logging.getLogger(__name__)


@dataclass
class AnalysisResults:
    """Container for all EDA analysis outputs."""
    quality: Dict[str, pd.DataFrame] = field(default_factory=dict)
    statistics: Dict[str, pd.DataFrame] = field(default_factory=dict)
    correlations: Dict[str, Any] = field(default_factory=dict)
    temporal: Dict[str, Any] = field(default_factory=dict)
    plot_paths: Dict[str, Any] = field(default_factory=dict)
    report_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CombinedEDA:
    """
    Master orchestrator for Module 4 combined multi-user EDA.

    Usage
    -----
    eda = CombinedEDA(verbose=True)
    results = eda.run(
        source={"packets": packets_by_uid, "train_users": train_users},
        demographics=demographics_by_uid,
        output_dir="outputs/run_001/module_4_eda",
    )
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [{MODULE_LABEL}] {msg}")

    def run(
        self,
        source: Dict[str, Any],
        demographics: Optional[Dict[str, dict]] = None,
        output_dir: Optional[str | Path] = None,
    ) -> AnalysisResults:
        """
        Run the full 10-phase EDA pipeline.

        Parameters
        ----------
        source : dict with either:
            {"packets": {uid: PipelinePacket}, "train_users": [uid, ...]}
            or {"folders": {uid: Path}}
        demographics : {uid: {age, gender, ...}} or None
        output_dir : output directory (created if missing)

        Returns
        -------
        AnalysisResults with all analysis outputs
        """
        t0 = time.time()
        results = AnalysisResults()

        # ── Phase 0: Setup output directory ────────────────────────────────
        if output_dir is None:
            output_dir = self._auto_output_dir()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.output_dir = output_dir
        self._log(f"Output: {output_dir}")

        # ── Phase 1: Load and combine data ─────────────────────────────────
        self._log("Phase 1/10: Loading and combining data...")
        loader = CombinedDataLoader(verbose=self.verbose)
        if "packets" in source:
            combined = loader.load_from_packets(
                source["packets"], source["train_users"], demographics
            )
        elif "folders" in source:
            combined = loader.load_from_folders(source["folders"], demographics)
        else:
            raise ValueError("source must contain 'packets' or 'folders' key")

        self._log(f"  Combined: {combined.n_users} users, "
                  f"{len(combined.signals)} signals, "
                  f"{combined.total_duration_s / 3600:.1f} hours")

        # ── Phase 2: Data quality assessment ───────────────────────────────
        self._log("Phase 2/10: Data quality assessment...")
        results.quality = run_all_quality(combined.signals)

        # ── Phase 3: Statistical analysis ──────────────────────────────────
        self._log("Phase 3/10: Statistical analysis...")
        results.statistics = run_all_statistics(
            combined.signals, combined.demographics
        )

        # ── Phase 4: Correlational analysis ────────────────────────────────
        self._log("Phase 4/10: Correlational analysis...")
        results.correlations = run_all_correlations(combined.signals)

        # ── Phase 5: Temporal event dynamics ───────────────────────────────
        self._log("Phase 5/10: Temporal event dynamics...")
        analyser = SignalAnalyser(verbose=self.verbose)
        temporal_raw = analyser.analyse_all(combined.signals)
        dynamics_df = temporal_raw.get("event_dynamics", pd.DataFrame())

        # Add summary tables
        temporal_raw["pct_change_summary"] = signal_pct_change_summary(dynamics_df)
        temporal_raw["time_to_peak_summary"] = time_to_peak_summary(dynamics_df)
        results.temporal = temporal_raw

        # ── Phase 6: Export CSV tables ─────────────────────────────────────
        self._log("Phase 6/10: Exporting CSV tables...")
        csv_dir = output_dir / "summary_tables"
        csv_dir.mkdir(exist_ok=True)
        self._export_csvs(results, csv_dir)

        # ── Phase 7: Generate visualisations ───────────────────────────────
        self._log("Phase 7/10: Generating visualisations...")
        vis = EDAVisualiser(output_dir, dpi=200)
        plot_paths = self._generate_all_plots(
            vis, combined, results
        )
        results.plot_paths = plot_paths

        # ── Phase 8: Build Word report ─────────────────────────────────────
        self._log("Phase 8/10: Building Word report...")
        metadata = {
            "module": MODULE_LABEL,
            "version": MODULE_VERSION,
            "n_users": combined.n_users,
            "user_ids": combined.user_ids,
            "total_duration_s": combined.total_duration_s,
            "n_signals": len(combined.signals),
            "signal_names": list(combined.signals.keys()),
            "timestamp": datetime.now().isoformat(),
        }
        results.metadata = metadata

        try:
            builder = DocxReportBuilder(verbose=self.verbose)
            results.report_path = builder.build(
                output_dir, results.quality, results.statistics,
                results.correlations, results.temporal,
                plot_paths, metadata,
            )
        except Exception as e:
            self._log(f"WARNING: Report generation failed: {e}")
            results.report_path = None

        # ── Phase 9: Save metadata ─────────────────────────────────────────
        self._log("Phase 9/10: Saving metadata...")
        elapsed = time.time() - t0
        metadata["elapsed_s"] = round(elapsed, 1)
        meta_path = output_dir / "eda_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # ── Phase 10: Summary ──────────────────────────────────────────────
        self._log(f"Phase 10/10: Complete! ({elapsed:.1f}s)")
        n_plots = sum(1 for v in plot_paths.values()
                      if v is not None and (isinstance(v, Path) or isinstance(v, str)))
        # Count dict values too
        for v in plot_paths.values():
            if isinstance(v, dict):
                n_plots += sum(1 for p in v.values() if p is not None)
        self._log(f"  Charts: {n_plots}")
        self._log(f"  Report: {results.report_path}")

        return results

    # ── Plot generation ────────────────────────────────────────────────────

    def _generate_all_plots(
        self,
        vis: EDAVisualiser,
        combined: CombinedData,
        results: AnalysisResults,
    ) -> Dict[str, Any]:
        """Generate all 25 visualisations."""
        paths: Dict[str, Any] = {}
        q = results.quality
        s = results.statistics
        c = results.correlations
        t = results.temporal

        # V1-V4: Data Understanding & Quality
        paths["V01_target_distribution"] = vis.plot_target_distribution(
            q.get("target_distribution"))
        paths["V02_user_label_heatmap"] = vis.plot_user_label_heatmap(
            q.get("user_label_matrix"))
        paths["V03_missing_heatmap"] = vis.plot_missing_heatmap(
            q.get("missing_values"))
        paths["V04_outlier_boxplots"] = vis.plot_outlier_boxplots(combined.signals)

        # V5: Distribution
        paths["V05_histograms_kde"] = vis.plot_histograms_kde(combined.signals)

        # V6-V7: Violin + Box by target
        violin_paths = vis.plot_violin_by_target(combined.signals)
        box_paths = vis.plot_box_by_target(combined.signals)
        for sig, p in violin_paths.items():
            paths[f"V06_violin_{sig}"] = p
        for sig, p in box_paths.items():
            paths[f"V07_boxplot_{sig}"] = p

        # V8: Category bar
        paths["V08_category_bar"] = vis.plot_category_bar(
            s.get("descriptive_by_category"))

        # V9: Demographic box plots
        paths["V09_demographic_autism_severity"] = vis.plot_demographic_boxplots(
            combined.signals, combined.demographics, "autism_severity")

        # V10-V12: Bivariate
        paths["V10_correlation_heatmap"] = vis.plot_correlation_heatmap(
            c.get("spearman_matrix"))
        paths["V11_kw_significance"] = vis.plot_kw_significance(
            s.get("kruskal_wallis"))
        paths["V12_effect_size"] = vis.plot_effect_size_heatmap(
            s.get("kruskal_wallis"))

        # V13-V14: Correlational
        paths["V13_scatter_matrix"] = vis.plot_scatter_matrix(combined.signals)
        paths["V14_point_biserial_heatmap"] = vis.plot_point_biserial_heatmap(
            c.get("point_biserial_matrix"))

        # V15-V22: Temporal
        paths["V15_event_duration"] = vis.plot_event_duration(
            t.get("event_durations"))
        paths["V16_pct_change"] = vis.plot_pct_change(
            t.get("pct_change_summary"))
        paths["V17_time_to_peak"] = vis.plot_time_to_peak(
            t.get("time_to_peak_summary"))
        paths["V18_return_rate"] = vis.plot_return_rate(
            t.get("return_summary"))
        paths["V19_return_time_heatmap"] = vis.plot_return_time_heatmap(
            t.get("return_time_summary"))
        paths["V20_return_counts"] = vis.plot_return_counts(
            t.get("return_counts"))
        paths["V21_median_drift"] = vis.plot_median_drift(
            t.get("median_drift_summary"))
        threshold_paths = vis.plot_threshold_traces(
            combined.signals, t.get("threshold_events"))
        for sig, p in threshold_paths.items():
            paths[f"V22_threshold_{sig}"] = p

        # V23-V25: 3D
        paths["V23_3d_signals"] = vis.plot_3d_signals(combined.signals)
        paths["V24_pca_3d"] = vis.plot_pca_3d(combined.signals)
        paths["V25_demographic_3d"] = vis.plot_demographic_3d(
            combined.signals, combined.demographics)

        return paths

    # ── CSV export ─────────────────────────────────────────────────────────

    def _export_csvs(self, results: AnalysisResults, csv_dir: Path):
        """Export all analysis DataFrames as CSVs."""
        for name, df in results.quality.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(csv_dir / f"quality_{name}.csv", index=False)

        for name, df in results.statistics.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(csv_dir / f"stats_{name}.csv", index=False)

        for name, val in results.correlations.items():
            if isinstance(val, pd.DataFrame) and not val.empty:
                val.to_csv(csv_dir / f"corr_{name}.csv",
                           index=("matrix" in name))

        for name, val in results.temporal.items():
            if isinstance(val, pd.DataFrame) and not val.empty:
                val.to_csv(csv_dir / f"temporal_{name}.csv", index=False)

    # ── Auto output directory ──────────────────────────────────────────────

    def _auto_output_dir(self) -> Path:
        """Find the next auto-numbered output directory."""
        import re
        root = Path(OUTPUT_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        pattern = re.compile(rf"{MODULE_LABEL}_v{MODULE_VERSION}_run_(\d+)")
        existing = []
        if root.exists():
            for d in root.iterdir():
                m = pattern.match(d.name)
                if m:
                    existing.append(int(m.group(1)))
        next_num = (max(existing) + 1) if existing else 1
        return root / f"{MODULE_LABEL}_v{MODULE_VERSION}_run_{next_num:03d}"
