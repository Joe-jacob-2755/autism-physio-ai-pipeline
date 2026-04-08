"""
=============================================================================
MODULE 4 – DATA ANALYSER  |  analyser.py
=============================================================================
DataAnalyser — master orchestrator.

Accepts:
  - A Module 3 output folder (reads cleaned signals + feature CSVs)
  - Or direct dicts of DataFrames

Produces:
  - Temporal signal dynamics analysis
  - Descriptive, statistical, correlation analyses
  - All visualisations (2D + 3D)
  - Comprehensive HTML report + CSV tables
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


def _proxy_col(sig: str) -> str:
    """Return the expected value column name for a signal."""
    return {"EDA": "EDA_uS", "BVP": "BVP_nT", "IBI": "IBI_ms",
            "ST": "ST_degC", "ACC": "ACC_X_g"}.get(sig, sig + "_val")


class DataAnalyser:
    """
    Module 4 master analysis pipeline.

    Parameters
    ----------
    threshold_window_s   : adaptive threshold rolling window in seconds
    threshold_mad_factor : deviation trigger (N × MAD from running median)
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
    ) -> dict:
        """
        Run full analysis pipeline.

        Parameters
        ----------
        source : Module 3 output folder path, or dict with keys:
                   'signals'      — {name: DataFrame} (raw cleaned signals)
                   'features_raw' — {name: DataFrame} (M3 feature CSVs)
        """
        t0 = time.time()
        run_folder = _next_run_folder(self.out_name)

        self._log("=" * 64)
        self._log(f"  MODULE 4  —  Data Analyser  |  v{MODULE_VERSION}")
        self._log(f"  Session: {session_id}  |  User: {user_id}")
        self._log(f"  Output:  {run_folder.relative_to(MODULE_DIR)}")
        self._log("=" * 64)

        # ── Load data ─────────────────────────────────────────────────────
        signals, features, combined = self._load(source)

        self._log(f"\n  Signals loaded:  {list(signals.keys())}")
        self._log(f"  Features loaded: {list(features.keys())}")
        self._log(f"  Combined shape:  {combined.shape}")

        # ── Step 1: Temporal signal dynamics ──────────────────────────────
        self._log("\n[M4] Step 1/5 — Temporal signal analysis ...")
        from signal_analyser import SignalAnalyser
        sig_results = SignalAnalyser(
            threshold_window_s=self.threshold_window_s,
            threshold_mad_factor=self.threshold_mad_factor,
            threshold_sustain_s=self.threshold_sustain_s,
            verbose=self.verbose,
        ).analyse_all(signals)

        # ── Step 2: Statistical analyses ──────────────────────────────────
        self._log("\n[M4] Step 2/5 — Statistical analysis ...")
        from statistical_analyser import (
            descriptive_analysis, kruskal_wallis_analysis,
            pairwise_mannwhitney, correlation_feature_target,
            correlation_feature_matrix, signal_pct_change_summary,
        )
        desc_df = descriptive_analysis(combined)
        kw_df = kruskal_wallis_analysis(combined)
        top_feats = kw_df[kw_df["significant"]].head(20)["feature"].tolist() \
            if not kw_df.empty and "significant" in kw_df.columns else None
        pair_df = pairwise_mannwhitney(combined, top_features=top_feats)
        corr_t_df = correlation_feature_target(combined)
        corr_m_df, _feats = correlation_feature_matrix(combined)
        pct_df = signal_pct_change_summary(sig_results["event_dynamics"])

        self._log(f"  Descriptive: {len(desc_df)} rows")
        self._log(
            f"  Kruskal-Wallis: {int(kw_df['significant'].sum()) if not kw_df.empty and 'significant' in kw_df.columns else 0} significant features")  # noqa: E501
        self._log(f"  Corr target: {len(corr_t_df)} features")

        # ── Step 3: Visualisations ─────────────────────────────────────────
        self._log("\n[M4] Step 3/5 — Generating visualisations ...")
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

        # ── Step 4: Report ──────────────────────────────────────────────────
        self._log("\n[M4] Step 4/5 — Generating report ...")
        from reporter import AnalysisReporter
        meta = {
            "session_id": session_id,
            "user_id": user_id,
            "module_version": MODULE_VERSION,
            "window_s": self.threshold_window_s,
            "mad_factor": self.threshold_mad_factor,
            "sustain_s": self.threshold_sustain_s,
            "n_signals": len(signals),
            "n_feature_cols": combined.shape[1],
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

        # ── Step 5: Metadata JSON ───────────────────────────────────────────
        self._log("\n[M4] Step 5/5 — Saving metadata ...")
        meta["elapsed_s"] = round(time.time() - t0, 2)
        with open(run_folder / "analysis_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        elapsed = time.time() - t0
        self._log(f"\n[M4] Done in {elapsed:.1f}s  →  {run_folder}\n")

        return {
            "run_folder": run_folder,
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
        Load signals and features from a Module 3 output folder or dict.

        Returns (signals_dict, features_dict, combined_df)
        """
        if isinstance(source, dict):
            signals = source.get("signals", {})
            features = source.get("features_raw", {})
            combined = source.get("combined", pd.DataFrame())
            return signals, features, combined

        # Folder path — detect M3 structure
        folder = Path(source)
        signals, features = {}, {}
        combined = pd.DataFrame()

        raw_dir = folder / "features_raw"
        sig_dir = folder  # M3 saves cleaned signal CSVs to the run folder too

        # Load raw signal files (from M3 run folder directly OR from M2A output)
        for sig_name, fname in [("EDA", "EDA.csv"), ("BVP", "BVP.csv"),
                                ("IBI", "IBI.csv"), ("ST", "ST.csv"), ("ACC", "ACC.csv")]:
            # Try M3 folder first, then look for signal file anywhere
            for candidate in (folder / fname, folder.parent / fname):
                if candidate.exists():
                    df = pd.read_csv(candidate)
                    if len(df) > 0:
                        signals[sig_name] = df
                        self._log(f"  Loaded {sig_name}: {len(df):,} rows from {candidate.name}")
                    break

        # Load feature CSVs
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

        # If no raw signals found, try reading from M2A/M1 output via
        # metadata.json, or reconstruct window-level signals from combined features
        if not signals:
            # Try locating original signal CSVs from M3 metadata
            meta_path = folder / "preprocessing_metadata.json"
            if meta_path.exists():
                try:
                    import json
                    meta = json.load(open(meta_path))
                    # M3 source was often the M2A or M1 output folder
                    # Try parent folders of M3 output directory
                    for candidate_root in [
                        folder.parent.parent / "module_2a_data_simulation" / "outputs",
                        folder.parent.parent / "module_1_data_acquisition" / "outputs",
                    ]:
                        if candidate_root.exists():
                            for run_dir in sorted(candidate_root.iterdir(), reverse=True):
                                if not run_dir.is_dir():
                                    continue
                                found_here = {}
                                for sig_name, fname in [("EDA", "EDA.csv"), ("BVP", "BVP.csv"),
                                                        ("IBI", "IBI.csv"), ("ST", "ST.csv"),
                                                        ("ACC", "ACC.csv")]:
                                    fpath = run_dir / fname
                                    if fpath.exists():
                                        df = pd.read_csv(fpath)
                                        if len(df) > 0:
                                            found_here[sig_name] = df
                                if len(found_here) >= 3:
                                    signals = found_here
                                    self._log(f"  Found raw signals in {run_dir.name}")
                                    break
                            if signals:
                                break
                except Exception:
                    pass

        # Final fallback: use feature mean values as window-level proxy signals
        # This gives temporal resolution at the feature window granularity
        if not signals and not combined.empty:
            self._log("  Reconstructing window-level proxy signals from feature means")
            # Map signal name → feature column used as proxy value
            proxy_map = {
                "EDA": "eda_mean",       # mean EDA conductance over window
                "BVP": "bvp_mean",       # mean BVP amplitude over window
                "ST": "st_mean",        # mean skin temperature over window
                "ACC": "acc_svm_mean",   # mean SVM magnitude over window
                "IBI": "ibi_mean",       # mean RR interval over window
            }
            ts_col = "window_start_s" if "window_start_s" in combined.columns else "timestamp_s"
            for sig, feat_col in proxy_map.items():
                if feat_col not in combined.columns:
                    continue
                keep = [ts_col, feat_col]
                if "target_label" in combined.columns:
                    keep.append("target_label")
                if "event_id" in combined.columns:
                    keep.append("event_id")
                keep = [c for c in keep if c in combined.columns]
                sub = combined[keep].copy()
                sub = sub.rename(columns={ts_col: "timestamp_s", feat_col: _proxy_col(sig)})
                signals[sig] = sub
            if signals:
                self._log(f"  Proxy signals built for: {list(signals.keys())} (window-level)")

        return signals, features, combined

    def _log(self, msg):
        if self.verbose:
            print(msg)
