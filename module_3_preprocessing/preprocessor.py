"""
=============================================================================
MODULE 3 – DATA PREPROCESSING  |  preprocessor.py
=============================================================================
DataPreprocessor — master orchestrator.

Ties together:
  SignalCleaner      → missing data, out-of-range removal, filling
  SignalFilterManager→ Hampel, Butterworth, Kalman filtering
  FeatureExtractor   → all 80 physiological features per signal
  DemographicEncoder → age, gender, ethnicity, severity, verbal, comorbidity
  FeatureNormaliser  → RobustScaler normalisation
  FeatureFuser       → per-signal + combined feature DataFrames
  PreprocessingVisualiser → processed + comparative plots
  PreprocessingExporter   → 4 CSV output sets

=============================================================================
"""
from __future__ import annotations
from exporter import PreprocessingExporter
from visualiser import PreprocessingVisualiser
from normaliser import FeatureNormaliser, DemographicEncoder, FeatureFuser
from feature_extractor import FeatureExtractor
from signal_filters import SignalFilterManager
from signal_cleaner import SignalCleaner
from config import (
    MODULE_VERSION, MODULE_LABEL, OUTPUT_ROOT,
    WINDOW_SIZE_S, WINDOW_OVERLAP, SCALER_TYPE,
)

import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Module directory on path
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FOLDER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def next_run_folder(custom_name: str = None) -> Path:
    output_root = MODULE_DIR / OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    version_tag = f"{MODULE_LABEL}_v{MODULE_VERSION}"

    if custom_name:
        base = output_root / custom_name
        if not base.exists():
            base.mkdir(parents=True)
            return base
        prefix = custom_name
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    else:
        prefix = f"{version_tag}_run"
        pattern = re.compile(rf"^{re.escape(version_tag)}_run_(\d+)$")

    existing = [
        int(m.group(1))
        for d in output_root.iterdir()
        if d.is_dir() and (m := pattern.match(d.name))
    ]
    next_num = (max(existing) + 1) if existing else 1
    folder = output_root / f"{prefix}_{next_num:03d}"
    folder.mkdir(parents=True)
    return folder


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PACKET ADAPTER
# — Convert Module 2 PipelinePacket to Module 3 input format
# ─────────────────────────────────────────────────────────────────────────────

def packet_to_signals(packet) -> Dict[str, pd.DataFrame]:
    """
    Convert a Module 2 PipelinePacket to a signals dict for the preprocessor.

    Handles both PipelinePacket objects and plain dicts.
    """
    if hasattr(packet, "signals"):
        signals = dict(packet.signals)
    elif isinstance(packet, dict):
        signals = packet
    else:
        raise TypeError(f"Cannot convert type {type(packet)} to signals dict")
    return signals


def packet_get_meta(packet) -> dict:
    """Extract session metadata from a PipelinePacket."""
    if hasattr(packet, "metadata"):
        return dict(packet.metadata)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class DataPreprocessor:
    """
    Module 3 master preprocessing pipeline.

    Pipeline steps
    --------------
    1. Clean signals  — remove/fill missing, flag out-of-range, flatlines
    2. Filter signals — Hampel outlier removal + Butterworth/Kalman smoothing
    3. Extract features — all 80 features in configurable windows
    4. Encode demographics — age, gender, severity, verbal, comorbidity
    5. Normalise features — RobustScaler per signal
    6. Fuse features — per-signal + combined DataFrames
    7. Visualise — processed signals + raw vs processed comparison
    8. Export — 4 CSV sets + cleaning report + metadata

    Parameters
    ----------
    filter_type      : 'butterworth' | 'kalman' | 'hampel_only' | 'none'
    apply_hampel     : apply Hampel pre-filter before main filter
    window_s         : feature extraction window in seconds
    overlap          : window overlap fraction
    scaler_type      : 'robust' | 'standard' | 'minmax'
    generate_plots   : save PNG visualisations
    out_name         : custom output folder name (auto-numbered if None)
    verbose          : print progress at each step
    """

    def __init__(
        self,
        filter_type: str = "butterworth",
        apply_hampel: bool = True,
        window_s: float = WINDOW_SIZE_S,
        overlap: float = WINDOW_OVERLAP,
        scaler_type: str = SCALER_TYPE,
        generate_plots: bool = True,
        out_name: str = None,
        verbose: bool = True,
    ):
        self.filter_type = filter_type
        self.apply_hampel = apply_hampel
        self.window_s = window_s
        self.overlap = overlap
        self.scaler_type = scaler_type
        self.generate_plots = generate_plots
        self.out_name = out_name
        self.verbose = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        signals_input,              # PipelinePacket, dict of DataFrames, or path
        demographics: dict = None,  # participant demographics (from UserProfile)
        session_id: str = "unknown",
        user_id: str = "unknown",
        output_dir: str = None,
    ) -> dict:
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        signals_input : one of:
          - Module 2 PipelinePacket
          - dict {signal_name: DataFrame}
          - Path to folder containing signal CSVs
        demographics  : dict with age, gender, ethnicity, autism_severity,
                        verbal_status, comorbidity
        session_id    : session identifier
        user_id       : participant identifier

        Returns
        -------
        result dict containing:
          raw_features, norm_features, raw_combined, norm_combined,
          run_folder, cleaning_reports
        """
        t0 = time.time()
        if output_dir is not None:
            run_folder = Path(output_dir)
            run_folder.mkdir(parents=True, exist_ok=True)
        else:
            run_folder = next_run_folder(self.out_name)
        try:
            rel = run_folder.relative_to(MODULE_DIR)
        except ValueError:
            rel = run_folder.relative_to(MODULE_DIR.parent) if MODULE_DIR.parent in run_folder.parents else run_folder

        self._log("=" * 64)
        self._log(f"  MODULE 3  —  Data Preprocessing  |  v{MODULE_VERSION}")
        self._log(f"  Session : {session_id}  |  User: {user_id}")
        self._log(f"  Filter  : {self.filter_type}"
                  f"{' + Hampel' if self.apply_hampel else ''}")
        self._log(f"  Window  : {self.window_s:.0f}s  "
                  f"overlap={self.overlap:.0%}  scaler={self.scaler_type}")
        self._log(f"  Output  : {rel}")
        self._log("=" * 64)

        # ── Resolve input ──────────────────────────────────────────────
        signals_raw, meta = self._resolve_input(signals_input)

        # ── Step 1: Clean ─────────────────────────────────────────────
        self._log("\n[M3] Step 1/6 — Cleaning signals ...")
        cleaner = SignalCleaner(verbose=self.verbose)
        signals_clean, reports = cleaner.clean_all(signals_raw)

        # ── Step 2: Filter ────────────────────────────────────────────
        self._log("\n[M3] Step 2/6 — Filtering signals ...")
        filt_mgr = SignalFilterManager(
            filter_type=self.filter_type,
            apply_hampel=self.apply_hampel,
            verbose=self.verbose,
        )
        signals_filtered = filt_mgr.filter_all(signals_clean)

        # ── Step 3: Feature extraction ────────────────────────────────
        self._log("\n[M3] Step 3/6 — Extracting features ...")
        extractor = FeatureExtractor(
            window_s=self.window_s,
            overlap=self.overlap,
            verbose=self.verbose,
        )
        raw_features = extractor.extract_all(
            signals_filtered, session_id=session_id, user_id=user_id
        )

        # ── Step 4: Demographics ──────────────────────────────────────
        self._log("\n[M3] Step 4/6 — Adding demographic features ...")
        if demographics:
            enc = DemographicEncoder.encode(demographics)
            for sig_name, df in raw_features.items():
                for k, v in enc.items():
                    raw_features[sig_name][k] = v
            self._log(f"  Demographics added: {list(enc.keys())}")
        else:
            self._log("  No demographics supplied — skipping.")

        # ── Step 5: Normalise ─────────────────────────────────────────
        self._log("\n[M3] Step 5/6 — Normalising features ...")
        normaliser = FeatureNormaliser(
            scaler_type=self.scaler_type, verbose=self.verbose
        )
        norm_features = normaliser.fit_transform(raw_features)

        # ── Step 5b: Fuse ─────────────────────────────────────────────
        fuser = FeatureFuser()
        _, raw_combined = fuser.fuse(raw_features, demographics)
        _, norm_combined = fuser.fuse(norm_features, demographics)
        self._log(f"  Raw combined:  {raw_combined.shape}")
        self._log(f"  Norm combined: {norm_combined.shape}")

        # ── Step 6: Visualise ─────────────────────────────────────────
        if self.generate_plots:
            self._log("\n[M3] Step 6/6 — Generating visualisations ...")
            viz = PreprocessingVisualiser(run_folder)
            viz.plot_processed_signals(
                signals_filtered, session_id=session_id, user_id=user_id
            )
            viz.plot_raw_vs_processed(
                signals_raw, signals_filtered,
                session_id=session_id, user_id=user_id
            )
        else:
            self._log("\n[M3] Step 6/6 — Skipping plots (generate_plots=False)")

        # ── Export ────────────────────────────────────────────────────
        self._log("\n[M3] Exporting output files ...")
        metadata = {
            "session_id": session_id,
            "user_id": user_id,
            "filter_type": self.filter_type,
            "apply_hampel": self.apply_hampel,
            "window_s": self.window_s,
            "overlap": self.overlap,
            "scaler_type": self.scaler_type,
            "n_signals_cleaned": len(signals_clean),
            "n_signals_discarded": len(signals_raw) - len(signals_clean),
            "demographics": demographics or {},
            "source_meta": meta,
            "elapsed_s": round(time.time() - t0, 2),
        }
        exporter = PreprocessingExporter(run_folder)
        exporter.export_all(
            raw_features=raw_features,
            norm_features=norm_features,
            raw_combined=raw_combined,
            norm_combined=norm_combined,
            cleaning_reports=reports,
            metadata=metadata,
        )

        elapsed = time.time() - t0
        self._log(f"\n[M3] Done in {elapsed:.1f}s.  Output: {run_folder}\n")

        return {
            "run_folder": run_folder,
            "raw_features": raw_features,
            "norm_features": norm_features,
            "raw_combined": raw_combined,
            "norm_combined": norm_combined,
            "cleaning_reports": reports,
            "signals_cleaned": signals_clean,
            "signals_filtered": signals_filtered,
            "metadata": metadata,
        }

    # ── Input resolver ────────────────────────────────────────────────────

    def _resolve_input(self, signals_input) -> tuple:
        """Convert any supported input type to {signal_name: DataFrame}."""
        meta = {}

        # Module 2 PipelinePacket
        if hasattr(signals_input, "signals"):
            sigs = dict(signals_input.signals)
            meta = packet_get_meta(signals_input)
            return sigs, meta

        # Plain dict
        if isinstance(signals_input, dict):
            return signals_input, meta

        # Path to folder
        if isinstance(signals_input, (str, Path)):
            folder = Path(signals_input)
            sigs = {}
            file_map = {
                "EDA": "EDA.csv", "BVP": "BVP.csv", "IBI": "IBI.csv",
                "ST": "ST.csv", "ACC": "ACC.csv",
            }
            for sig, fname in file_map.items():
                fpath = folder / fname
                if fpath.exists():
                    sigs[sig] = pd.read_csv(fpath)
                    self._log(f"  Loaded {fname}  ({len(sigs[sig]):,} rows)")
            meta_path = folder / "metadata.json"
            if meta_path.exists():
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
            return sigs, meta

        raise TypeError(
            f"Unsupported input type: {type(signals_input)}. "
            "Pass a PipelinePacket, dict of DataFrames, or folder path."
        )

    def _log(self, msg):
        if self.verbose:
            print(msg)
