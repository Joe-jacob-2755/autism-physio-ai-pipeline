"""
=============================================================================
MODULE 4 – EXPLORATORY DATA ANALYSIS  |  analyser.py
=============================================================================
ExploratoryDataAnalyser — master orchestrator following IBM Data Science
Methodology for EDA.

This module runs on TRAINING DATA ONLY.  Test data is never seen.

IBM EDA Pipeline (7 Phases):
  Phase 1: Data Understanding — shape, types, summary statistics
  Phase 2: Data Quality Assessment — missing, outliers, consistency
  Phase 3: Distribution Analysis — skewness, normality, transforms
  Phase 4: Univariate Analysis — per-signal descriptive + boxplots
  Phase 5: Bivariate Analysis — correlations, group comparisons
  Phase 6: Temporal Analysis — signal dynamics, event response patterns
  Phase 7: Key Findings — comprehensive HTML report + recommendations

Accepts:
  - Direct dict of signal DataFrames  {name: DataFrame}
  - A folder path containing raw signal CSVs (EDA.csv, BVP.csv, etc.)

Produces:
  - Data quality report CSV
  - Distribution analysis CSV
  - Outlier detection CSV
  - Descriptive statistics CSV
  - Correlation analyses
  - Temporal dynamics analysis
  - Comprehensive HTML report
  - All visualisations (2D + 3D)
=============================================================================
"""
from __future__ import annotations
from config import MODULE_VERSION, MODULE_LABEL, OUTPUT_ROOT, META_COLS

import re
import sys
import json
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))


def _next_run_folder(custom_name: str = None) -> Path:
    output_root = MODULE_DIR / OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    vtag = f"{MODULE_LABEL}_v{MODULE_VERSION}"
    prefix = custom_name or f"{vtag}_run"
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    existing = [int(m.group(1)) for d in output_root.iterdir()
                if d.is_dir() and (m := pattern.match(d.name))]
    n = (max(existing) + 1) if existing else 1
    folder = output_root / f"{prefix}_{n:03d}"
    folder.mkdir(parents=True)
    return folder


class ExploratoryDataAnalyser:
    """
    Module 4 — IBM EDA master analysis pipeline.

    Runs on TRAINING data only.  Test data must never be passed here.

    Parameters
    ----------
    threshold_window_s   : adaptive threshold rolling window in seconds
    threshold_mad_factor : deviation trigger (N x MAD from running median)
    threshold_sustain_s  : minimum sustained duration to flag (seconds)
    out_name             : custom output folder name (auto-numbered if None)
    verbose              : print progress
    """

    def __init__(
        self,
        threshold_window_s: float = 300.0,
        threshold_mad_factor: float = 3.0,
        threshold_sustain_s: float = 30.0,
        out_name: str = None,
        verbose: bool = True,
    ):
        self.threshold_window_s = threshold_window_s
        self.threshold_mad_factor = threshold_mad_factor
        self.threshold_sustain_s = threshold_sustain_s
        self.out_name = out_name
        self.verbose = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        source: Union[str, Path, dict],
        session_id: str = "unknown",
        user_id: str = "unknown",
        output_dir: str = None,
    ) -> dict:
        """
        Run IBM EDA pipeline (7 phases) on training data.

        Parameters
        ----------
        source : folder path with raw signal CSVs, or dict with key
                 'signals' -> {name: DataFrame}
        """
        t0 = time.time()
        if output_dir is not None:
            run_folder = Path(output_dir)
            run_folder.mkdir(parents=True, exist_ok=True)
        else:
            run_folder = _next_run_folder(self.out_name)

        self._log("=" * 64)
        self._log(f"  MODULE {MODULE_LABEL}  --  Exploratory Data Analysis (IBM EDA)")
        self._log(f"  Version: v{MODULE_VERSION}")
        self._log(f"  Session: {session_id}  |  User: {user_id}")
        self._log(f"  NOTE: Training data only -- test data is held out")
        try:
            _rel = run_folder.relative_to(MODULE_DIR)
        except ValueError:
            try:
                _rel = run_folder.relative_to(MODULE_DIR.parent)
            except ValueError:
                _rel = run_folder
        self._log(f"  Output:  {_rel}")
        self._log("=" * 64)

        # ── Load data ─────────────────────────────────────────────────────
        signals, features, combined = self._load(source)

        self._log(f"\n  Signals loaded:  {list(signals.keys())}")
        if features:
            self._log(f"  Features loaded: {list(features.keys())}")
        self._log(f"  Combined shape:  {combined.shape}")

        # Create summary tables directory
        tables_dir = run_folder / "summary_tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 1: DATA UNDERSTANDING (IBM EDA Step 1)
        # ══════════════════════════════════════════════════════════════════
        self._log("\n[M4] Phase 1/7 -- Data Understanding ...")
        from data_quality import data_understanding
        understanding_df = data_understanding(signals)
        if not understanding_df.empty:
            understanding_df.to_csv(
                tables_dir / "data_understanding.csv", index=False)
            self._log(f"  {len(understanding_df)} signals characterised")
            for _, row in understanding_df.iterrows():
                self._log(f"    {row['signal']}: {row['n_rows']:,} samples, "
                          f"{row.get('sampling_rate_hz', '?')} Hz, "
                          f"{row.get('duration_s', '?')}s")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 2: DATA QUALITY ASSESSMENT (IBM EDA Step 2)
        # ══════════════════════════════════════════════════════════════════
        self._log("\n[M4] Phase 2/7 -- Data Quality Assessment ...")
        from data_quality import data_quality_report, outlier_detection
        quality_df = data_quality_report(signals)
        if not quality_df.empty:
            quality_df.to_csv(
                tables_dir / "data_quality_report.csv", index=False)
            for _, row in quality_df.iterrows():
                self._log(f"    {row['signal']}/{row['value_col']}: "
                          f"missing={row['pct_missing']:.1f}%, "
                          f"outliers_oor={row['pct_out_of_range']:.1f}%, "
                          f"grade={row['quality_grade']}")

        outlier_df = outlier_detection(signals, method="iqr")
        if not outlier_df.empty:
            outlier_df.to_csv(
                tables_dir / "outlier_detection.csv", index=False)
            self._log(f"  Outlier detection: {len(outlier_df)} channels assessed")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 3: DISTRIBUTION ANALYSIS (IBM EDA Step 3)
        # ══════════════════════════════════════════════════════════════════
        self._log("\n[M4] Phase 3/7 -- Distribution Analysis ...")
        from data_quality import distribution_analysis
        dist_df = distribution_analysis(signals)
        if not dist_df.empty:
            dist_df.to_csv(
                tables_dir / "distribution_analysis.csv", index=False)
            for _, row in dist_df.iterrows():
                self._log(f"    {row['signal']}/{row['value_col']}: "
                          f"skew={row['skewness']:.2f}, "
                          f"shape={row['distribution_shape']}, "
                          f"transform={row['recommended_transform']}")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 4: UNIVARIATE ANALYSIS (IBM EDA Step 4)
        # ══════════════════════════════════════════════════════════════════
        self._log("\n[M4] Phase 4/7 -- Univariate Analysis (descriptive stats) ...")
        from statistical_analyser import descriptive_analysis
        desc_df = descriptive_analysis(combined) if not combined.empty else pd.DataFrame()
        if not desc_df.empty:
            desc_df.to_csv(tables_dir / "descriptive_statistics.csv", index=False)
            self._log(f"  Descriptive: {len(desc_df)} rows")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 5: BIVARIATE & MULTIVARIATE ANALYSIS (IBM EDA Step 5)
        # ══════════════════════════════════════════════════════════════════
        self._log("\n[M4] Phase 5/7 -- Bivariate & Multivariate Analysis ...")
        from statistical_analyser import (
            kruskal_wallis_analysis, pairwise_mannwhitney,
            correlation_feature_target, correlation_feature_matrix,
            signal_pct_change_summary,
        )
        from data_quality import inter_signal_correlation

        # Inter-signal correlations (bivariate on raw signals)
        inter_corr_df = inter_signal_correlation(signals)
        if not inter_corr_df.empty:
            inter_corr_df.to_csv(
                tables_dir / "inter_signal_correlation.csv", index=False)
            self._log(f"  Inter-signal correlations: {len(inter_corr_df)} pairs")

        # Feature-level statistics (if features available)
        kw_df = kruskal_wallis_analysis(combined) if not combined.empty else pd.DataFrame()
        top_feats = kw_df[kw_df["significant"]].head(20)["feature"].tolist() \
            if not kw_df.empty and "significant" in kw_df.columns else None
        pair_df = pairwise_mannwhitney(combined, top_features=top_feats) \
            if not combined.empty else pd.DataFrame()
        corr_t_df = correlation_feature_target(combined) \
            if not combined.empty else pd.DataFrame()
        corr_m_df, _feats = correlation_feature_matrix(combined) \
            if not combined.empty else (pd.DataFrame(), [])

        if not kw_df.empty:
            kw_df.to_csv(tables_dir / "kruskal_wallis_results.csv", index=False)
            n_sig = int(kw_df['significant'].sum()) if 'significant' in kw_df.columns else 0
            self._log(f"  Kruskal-Wallis: {n_sig} significant features")
        if not pair_df.empty:
            pair_df.to_csv(tables_dir / "pairwise_mannwhitney.csv", index=False)
        if not corr_t_df.empty:
            corr_t_df.to_csv(tables_dir / "correlation_feature_target.csv", index=False)
            self._log(f"  Corr target: {len(corr_t_df)} features")
        if not corr_m_df.empty:
            corr_m_df.to_csv(tables_dir / "correlation_matrix.csv", index=False)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 6: TEMPORAL ANALYSIS (IBM EDA Step 6)
        # ══════════════════════════════════════════════════════════════════
        self._log("\n[M4] Phase 6/7 -- Temporal Analysis ...")
        from signal_analyser import SignalAnalyser
        sig_results = SignalAnalyser(
            threshold_window_s=self.threshold_window_s,
            threshold_mad_factor=self.threshold_mad_factor,
            threshold_sustain_s=self.threshold_sustain_s,
            verbose=self.verbose,
        ).analyse_all(signals)

        pct_df = signal_pct_change_summary(sig_results["event_dynamics"])

        # Save temporal tables
        if not sig_results["event_dynamics"].empty:
            sig_results["event_dynamics"].to_csv(
                tables_dir / "temporal_dynamics.csv", index=False)
        if not sig_results["return_summary"].empty:
            sig_results["return_summary"].to_csv(
                tables_dir / "return_to_median_summary.csv", index=False)
        if not sig_results["threshold_events"].empty:
            sig_results["threshold_events"].to_csv(
                tables_dir / "threshold_events.csv", index=False)
        if not pct_df.empty:
            pct_df.to_csv(tables_dir / "pct_change_summary.csv", index=False)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 7: VISUALISATIONS & REPORT (IBM EDA Step 7)
        # ══════════════════════════════════════════════════════════════════
        self._log("\n[M4] Phase 7/7 -- Generating visualisations & report ...")
        from visualiser import AnalysisVisualiser
        vis = AnalysisVisualiser(run_folder)
        plots = {}

        plots.update(vis.plot_descriptive_boxplots(features))
        p = vis.plot_pct_change(pct_df)
        if p:
            plots["pct_change"] = p
        p = vis.plot_temporal_dynamics(sig_results["event_dynamics"])
        if p:
            plots["temporal"] = p
        p = vis.plot_return_rate(sig_results["return_summary"])
        if p:
            plots["return_rate"] = p
        p = vis.plot_significant_features(kw_df)
        if p:
            plots["kruskal"] = p
        p = vis.plot_correlation_target(corr_t_df)
        if p:
            plots["corr_target"] = p
        p = vis.plot_correlation_matrix(corr_m_df)
        if p:
            plots["corr_matrix"] = p
        plots.update(vis.plot_threshold_overlay(
            sig_results["median_drift"],
            sig_results["threshold_events"],
            self.threshold_mad_factor,
        ))
        p3d = vis.plot_3d_signals(signals)
        if p3d:
            plots["3d_html"] = p3d
            png_3d = run_folder / "3d_signals.png"
            if png_3d.exists():
                plots["3d_png"] = png_3d
        p = vis.plot_median_drift(sig_results["event_dynamics"])
        if p:
            plots["median_drift"] = p

        self._log(f"  Plots generated: {len(plots)}")

        # ── HTML Report ────────────────────────────────────────────────────
        from reporter import AnalysisReporter
        meta = {
            "session_id": session_id,
            "user_id": user_id,
            "module_version": MODULE_VERSION,
            "module_label": MODULE_LABEL,
            "analysis_type": "IBM Exploratory Data Analysis",
            "data_scope": "TRAINING DATA ONLY",
            "window_s": self.threshold_window_s,
            "mad_factor": self.threshold_mad_factor,
            "sustain_s": self.threshold_sustain_s,
            "n_signals": len(signals),
            "n_feature_cols": combined.shape[1] if not combined.empty else 0,
            "n_windows": len(combined),
        }
        AnalysisReporter(run_folder).generate(
            session_id=session_id,
            user_id=user_id,
            metadata=meta,
            descriptive_df=desc_df,
            kruskal_df=kw_df,
            pairwise_df=pair_df,
            corr_target_df=corr_t_df,
            corr_matrix_df=corr_m_df,
            dynamics_df=sig_results["event_dynamics"],
            return_df=sig_results["return_summary"],
            pct_change_df=pct_df,
            threshold_df=sig_results["threshold_events"],
            plot_paths=plots,
        )

        # ── Metadata JSON ──────────────────────────────────────────────────
        meta["elapsed_s"] = round(time.time() - t0, 2)
        # Add IBM EDA phase summaries
        meta["phases"] = {
            "1_data_understanding": {
                "signals": len(understanding_df) if not understanding_df.empty else 0
            },
            "2_data_quality": {
                "channels_assessed": len(quality_df) if not quality_df.empty else 0,
                "outlier_channels": len(outlier_df) if not outlier_df.empty else 0,
            },
            "3_distribution": {
                "distributions_analysed": len(dist_df) if not dist_df.empty else 0
            },
            "4_univariate": {
                "descriptive_rows": len(desc_df)
            },
            "5_bivariate": {
                "inter_signal_pairs": len(inter_corr_df) if not inter_corr_df.empty else 0,
                "kruskal_significant": int(kw_df['significant'].sum()) if not kw_df.empty and 'significant' in kw_df.columns else 0,
            },
            "6_temporal": {
                "events_analysed": len(sig_results["event_dynamics"]),
                "threshold_events": len(sig_results["threshold_events"]),
            },
            "7_report": {
                "plots_generated": len(plots),
            },
        }
        with open(run_folder / "eda_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        elapsed = time.time() - t0
        self._log(f"\n[M4] EDA complete in {elapsed:.1f}s  ->  {run_folder}\n")

        return {
            "run_folder": run_folder,
            "understanding": understanding_df,
            "quality": quality_df,
            "distribution": dist_df,
            "outliers": outlier_df,
            "inter_signal_corr": inter_corr_df,
            "descriptive": desc_df,
            "kruskal": kw_df,
            "pairwise": pair_df,
            "corr_target": corr_t_df,
            "event_dynamics": sig_results["event_dynamics"],
            "return_summary": sig_results["return_summary"],
            "threshold_events": sig_results["threshold_events"],
            "pct_change": pct_df,
            "plots": plots,
        }

    # ── Data loader ────────────────────────────────────────────────────────

    def _load(self, source):
        """
        Load raw signals from folder path or dict.

        In the new pipeline (M3 split -> M4 EDA), this receives raw
        signal CSVs directly — no feature CSVs expected at this stage.

        Returns (signals_dict, features_dict, combined_df)
        """
        if isinstance(source, dict):
            signals = source.get("signals", {})
            features = source.get("features_raw", {})
            combined = source.get("combined", pd.DataFrame())
            return signals, features, combined

        # Folder path — load raw signal CSVs
        folder = Path(source)
        signals, features = {}, {}
        combined = pd.DataFrame()

        # Try multiple locations for signal CSVs
        cleaned_dir = folder / "cleaned_signals"
        for sig_name, fname in [("EDA", "EDA.csv"), ("BVP", "BVP.csv"),
                                ("IBI", "IBI.csv"), ("ST", "ST.csv"), ("ACC", "ACC.csv")]:
            for candidate in (folder / fname,
                              cleaned_dir / fname,
                              folder.parent / fname):
                if candidate.exists():
                    df = pd.read_csv(candidate)
                    if len(df) > 0:
                        signals[sig_name] = df
                        self._log(f"  Loaded {sig_name}: {len(df):,} rows "
                                  f"from {candidate.name}")
                    break

        # Load feature CSVs if available (from M5/M6 output)
        raw_dir = folder / "features_raw"
        if raw_dir.exists():
            for sig_name, fname in [("EDA", "EDA_features.csv"),
                                    ("BVP", "BVP_features.csv"),
                                    ("IBI", "IBI_features.csv"),
                                    ("ST", "ST_features.csv"),
                                    ("ACC", "ACC_features.csv")]:
                fpath = raw_dir / fname
                if fpath.exists():
                    df = pd.read_csv(fpath)
                    features[sig_name] = df
                    self._log(f"  Features {sig_name}: {df.shape}")

            combined_path = raw_dir / "combined_features.csv"
            if combined_path.exists():
                combined = pd.read_csv(combined_path)
                self._log(f"  Combined features: {combined.shape}")

        # Load combined_signals.csv as fallback
        combo_path = folder / "combined_signals.csv"
        if not combo_path.exists():
            combo_path = cleaned_dir / "combined_signals.csv"
        if combo_path.exists() and combined.empty:
            combined = pd.read_csv(combo_path)
            self._log(f"  Combined signals: {combined.shape}")

        return signals, features, combined

    def _log(self, msg):
        if self.verbose:
            print(msg)


# Backward compatibility alias
DataAnalyser = ExploratoryDataAnalyser
