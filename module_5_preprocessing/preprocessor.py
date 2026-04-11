"""
=============================================================================
MODULE 5 – DATA PREPROCESSING  |  preprocessor.py
=============================================================================
DataPreprocessor — master orchestrator.

Ties together:
  SignalCleaner      → missing data, out-of-range removal, filling
  SignalFilterManager→ Hampel, Butterworth, Kalman filtering
  PreprocessingVisualiser → processed + comparative plots
  PreprocessingExporter   → cleaned/filtered signal CSVs + reports

Feature extraction, normalisation, and encoding have been moved to
Module 4 (Feature Engineering).
=============================================================================
"""
from __future__ import annotations
from exporter import PreprocessingExporter
from visualiser import PreprocessingVisualiser
from signal_filters import SignalFilterManager
from signal_cleaner import SignalCleaner
from config import (
    MODULE_VERSION, MODULE_LABEL, OUTPUT_ROOT,
)

import re
import sys
import time
import json
from pathlib import Path
from typing import Dict

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
    3. Visualise — processed signals + raw vs processed comparison
    4. Export — cleaned/filtered signal CSVs + cleaning report + metadata

    Feature extraction, normalisation, demographic encoding, and feature
    fusion have been moved to Module 4 (Feature Engineering).

    Parameters
    ----------
    filter_type      : 'butterworth' | 'kalman' | 'hampel_only' | 'none'
    apply_hampel     : apply Hampel pre-filter before main filter
    generate_plots   : save PNG visualisations
    out_name         : custom output folder name (auto-numbered if None)
    verbose          : print progress at each step
    """

    def __init__(
        self,
        filter_type: str = "butterworth",
        apply_hampel: bool = True,
        generate_plots: bool = True,
        out_name: str = None,
        verbose: bool = True,
    ):
        self.filter_type = filter_type
        self.apply_hampel = apply_hampel
        self.generate_plots = generate_plots
        self.out_name = out_name
        self.verbose = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        signals_input,              # PipelinePacket, dict of DataFrames, or path
        session_id: str = "unknown",
        user_id: str = "unknown",
        output_dir: str = None,
    ) -> dict:
        """
        Run the preprocessing pipeline (clean + filter).

        Parameters
        ----------
        signals_input : one of:
          - Module 2 PipelinePacket
          - dict {signal_name: DataFrame}
          - Path to folder containing signal CSVs
        session_id    : session identifier
        user_id       : participant identifier

        Returns
        -------
        result dict containing:
          run_folder, signals_cleaned, signals_filtered,
          cleaning_reports, metadata
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
            rel = run_folder

        self._log("=" * 64)
        self._log(f"  MODULE 5  —  Data Preprocessing  |  v{MODULE_VERSION}")
        self._log(f"  Session : {session_id}  |  User: {user_id}")
        self._log(f"  Filter  : {self.filter_type}"
                  f"{'  + Hampel' if self.apply_hampel else ''}")
        self._log(f"  Output  : {rel}")
        self._log("=" * 64)

        # ── Resolve input ──────────────────────────────────────────────
        signals_raw, meta = self._resolve_input(signals_input)

        # ── Step 1: Clean ─────────────────────────────────────────────
        self._log("\n[M5] Step 1/4 — Cleaning signals ...")
        cleaner = SignalCleaner(verbose=self.verbose)
        signals_clean, reports = cleaner.clean_all(signals_raw)

        # ── Step 2: Filter ────────────────────────────────────────────
        self._log("\n[M5] Step 2/4 — Filtering signals ...")
        filt_mgr = SignalFilterManager(
            filter_type=self.filter_type,
            apply_hampel=self.apply_hampel,
            verbose=self.verbose,
        )
        signals_filtered = filt_mgr.filter_all(signals_clean)

        # ── Step 3: Visualise ─────────────────────────────────────────
        if self.generate_plots:
            self._log("\n[M5] Step 3/4 — Generating visualisations ...")
            viz = PreprocessingVisualiser(run_folder)
            viz.plot_processed_signals(
                signals_filtered, session_id=session_id, user_id=user_id
            )
            viz.plot_raw_vs_processed(
                signals_raw, signals_filtered,
                session_id=session_id, user_id=user_id
            )
        else:
            self._log("\n[M5] Step 3/4 — Skipping plots (generate_plots=False)")

        # ── Step 4: Export ────────────────────────────────────────────
        self._log("\n[M5] Step 4/4 — Exporting cleaned signals ...")
        metadata = {
            "session_id": session_id,
            "user_id": user_id,
            "filter_type": self.filter_type,
            "apply_hampel": self.apply_hampel,
            "n_signals_cleaned": len(signals_clean),
            "n_signals_discarded": len(signals_raw) - len(signals_clean),
            "source_meta": meta,
            "elapsed_s": round(time.time() - t0, 2),
        }
        exporter = PreprocessingExporter(run_folder)
        exporter.export_all(
            signals_filtered=signals_filtered,
            cleaning_reports=reports,
            metadata=metadata,
        )

        # Copy demographics CSV from source if present
        self._copy_demographics(signals_input, run_folder)

        elapsed = time.time() - t0
        self._log(f"\n[M5] Done in {elapsed:.1f}s.  Output: {run_folder}\n")

        return {
            "run_folder": run_folder,
            "signals_cleaned": signals_clean,
            "signals_filtered": signals_filtered,
            "cleaning_reports": reports,
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
                with open(meta_path) as f:
                    meta = json.load(f)
            return sigs, meta

        raise TypeError(
            f"Unsupported input type: {type(signals_input)}. "
            "Pass a PipelinePacket, dict of DataFrames, or folder path."
        )

    def _copy_demographics(self, signals_input, run_folder: Path):
        """Copy participant_demographics.csv from source folder to output."""
        import shutil
        source_folder = None
        if isinstance(signals_input, (str, Path)):
            source_folder = Path(signals_input)
        elif hasattr(signals_input, "metadata"):
            # PipelinePacket — check if source path is in metadata
            source_folder = None

        if source_folder is not None:
            demo_src = source_folder / "participant_demographics.csv"
            if demo_src.exists():
                shutil.copy2(demo_src, run_folder / "participant_demographics.csv")
                self._log("  Copied participant_demographics.csv to output")

    def _log(self, msg):
        if self.verbose:
            print(msg)
