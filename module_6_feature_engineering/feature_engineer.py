"""
=============================================================================
MODULE 6 – FEATURE ENGINEERING  |  feature_engineer.py
=============================================================================
FeatureEngineer — master orchestrator.

Pipeline steps:
  1. Extract features     — 80 physiological features per signal (windowed)
  2. Encode demographics  — ordinal encoding of participant demographics
  3. Normalise features   — RobustScaler standardisation
  4. Fuse features        — per-signal + combined DataFrames
  5. One-hot encode       — binary indicators for categorical demographics
  6. Select features      — rank importance, select top N
  7. Reduce dimensions    — PCA, PLS-VIP, SVD comparison
  8. Export               — all CSV variants + metadata

Input: cleaned/filtered signals from Module 5 (folder path or dict)
Output: auto-numbered folder with all feature CSVs

=============================================================================
"""
from __future__ import annotations

import re
import sys
import time
import json
from pathlib import Path
import pandas as pd

from config import (
    MODULE_VERSION, MODULE_LABEL, OUTPUT_ROOT,
    WINDOW_SIZE_S, WINDOW_OVERLAP, SCALER_TYPE,
    DEFAULT_TOP_N_FEATURES,
)
from feature_extractor import FeatureExtractor
from normaliser import FeatureNormaliser, DemographicEncoder, FeatureFuser
from encoder import OneHotEncoder
from feature_selector import FeatureSelector
from dimensionality_reducer import DimensionalityReducer
from exporter import FeatureEngineeringExporter

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
# FEATURE ENGINEER
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Module 4 master feature engineering pipeline.

    Parameters
    ----------
    window_s         : feature extraction window in seconds
    overlap          : window overlap fraction
    scaler_type      : 'robust' | 'standard' | 'minmax'
    top_n_features   : number of top features to select
    selection_method : feature selection method (default 'ensemble')
    pca_variance     : PCA variance retention threshold
    out_name         : custom output folder name (auto-numbered if None)
    verbose          : print progress at each step
    """

    def __init__(
        self,
        window_s: float = WINDOW_SIZE_S,
        overlap: float = WINDOW_OVERLAP,
        scaler_type: str = SCALER_TYPE,
        top_n_features: int = DEFAULT_TOP_N_FEATURES,
        selection_method: str = "ensemble",
        pca_variance: float = 0.95,
        out_name: str = None,
        verbose: bool = True,
    ):
        self.window_s = window_s
        self.overlap = overlap
        self.scaler_type = scaler_type
        self.top_n_features = top_n_features
        self.selection_method = selection_method
        self.pca_variance = pca_variance
        self.out_name = out_name
        self.verbose = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        signals_input,              # folder path or dict of DataFrames
        demographics: dict = None,  # participant demographics
        session_id: str = "unknown",
        user_id: str = "unknown",
        output_dir: str = None,
    ) -> dict:
        """
        Run the full feature engineering pipeline.

        Parameters
        ----------
        signals_input : one of:
          - Path to Module 5 output folder (cleaned_signals/)
          - dict {signal_name: DataFrame}
        demographics  : dict with age, gender, ethnicity, autism_severity,
                        verbal_status, comorbidity
        session_id    : session identifier
        user_id       : participant identifier

        Returns
        -------
        result dict with all feature DataFrames and metadata
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
        self._log(f"  MODULE 6  —  Feature Engineering  |  v{MODULE_VERSION}")
        self._log(f"  Session : {session_id}  |  User: {user_id}")
        self._log(f"  Window  : {self.window_s:.0f}s  "
                  f"overlap={self.overlap:.0%}  scaler={self.scaler_type}")
        self._log(f"  Top N   : {self.top_n_features}  "
                  f"method={self.selection_method}")
        self._log(f"  Output  : {rel}")
        self._log("=" * 64)

        # ── Resolve input ──────────────────────────────────────────────
        signals, meta, demographics = self._resolve_input(
            signals_input, demographics)

        # ── Step 1: Feature extraction ────────────────────────────────
        self._log("\n[M6] Step 1/7 — Extracting features ...")
        extractor = FeatureExtractor(
            window_s=self.window_s,
            overlap=self.overlap,
            verbose=self.verbose,
        )
        raw_features = extractor.extract_all(
            signals, session_id=session_id, user_id=user_id
        )

        # ── Step 2: Demographics ──────────────────────────────────────
        self._log("\n[M6] Step 2/7 — Checking demographic features ...")
        if demographics:
            enc = DemographicEncoder.encode(demographics)
            self._log(f"  Demographics available: {list(enc.keys())}")
        else:
            self._log("  No demographics supplied — skipping.")

        # ── Step 3: Normalise ─────────────────────────────────────────
        self._log("\n[M6] Step 3/7 — Normalising features ...")
        normaliser = FeatureNormaliser(
            scaler_type=self.scaler_type, verbose=self.verbose
        )
        norm_features = normaliser.fit_transform(raw_features)

        # Save fitted scaler for test/deployment use
        normaliser.save(run_folder / "fitted_scaler.joblib")
        self._log("  Saved fitted_scaler.joblib")

        # ── Step 4: Fuse ──────────────────────────────────────────────
        # Demographics are added here (single point of entry) to both
        # per-signal and combined DataFrames via FeatureFuser.
        self._log("\n[M6] Step 4/7 — Fusing per-signal features ...")
        fuser = FeatureFuser()
        raw_features, raw_combined = fuser.fuse(raw_features, demographics)
        norm_features, norm_combined = fuser.fuse(norm_features, demographics)
        self._log(f"  Raw combined:  {raw_combined.shape}")
        self._log(f"  Norm combined: {norm_combined.shape}")

        # ── Step 5: One-hot encode ────────────────────────────────────
        self._log("\n[M6] Step 5/7 — One-hot encoding demographics ...")
        ohe = OneHotEncoder(verbose=self.verbose)
        encoded_features = ohe.encode_all(norm_features)
        encoded_combined = ohe.encode_dataframe(norm_combined)

        # ── Step 6: Feature selection ─────────────────────────────────
        self._log(f"\n[M6] Step 6/7 — Selecting top {self.top_n_features} "
                  "features ...")
        selector = FeatureSelector(
            top_n=self.top_n_features,
            method=self.selection_method,
            verbose=self.verbose,
        )
        importance_df, selected_df, _ = selector.rank_and_select(norm_combined)
        # Also produce encoded version of selected features
        selected_encoded_df = ohe.encode_dataframe(selected_df)

        # ── Step 7: Dimensionality reduction ──────────────────────────
        self._log("\n[M6] Step 7/7 — Dimensionality reduction ...")
        reducer = DimensionalityReducer(
            pca_variance=self.pca_variance,
            verbose=self.verbose,
        )
        dimred_results = reducer.reduce_and_compare(norm_combined)

        # ── Export ────────────────────────────────────────────────────
        self._log("\n[M6] Exporting output files ...")
        metadata = {
            "session_id": session_id,
            "user_id": user_id,
            "window_s": self.window_s,
            "overlap": self.overlap,
            "scaler_type": self.scaler_type,
            "top_n_features": self.top_n_features,
            "selection_method": self.selection_method,
            "pca_variance_threshold": self.pca_variance,
            "n_signals": len(raw_features),
            "n_total_features": raw_combined.shape[1] if len(raw_combined) else 0,
            "demographics": demographics or {},
            "source_meta": meta,
            "elapsed_s": round(time.time() - t0, 2),
        }

        exporter = FeatureEngineeringExporter(run_folder)
        exporter.export_all(
            raw_features=raw_features,
            norm_features=norm_features,
            raw_combined=raw_combined,
            norm_combined=norm_combined,
            encoded_features=encoded_features,
            encoded_combined=encoded_combined,
            importance_df=importance_df,
            selected_df=selected_df,
            selected_encoded_df=selected_encoded_df,
            dimred_results=dimred_results,
            metadata=metadata,
        )

        elapsed = time.time() - t0
        self._log(f"\n[M6] Done in {elapsed:.1f}s.  Output: {run_folder}\n")

        return {
            "run_folder": run_folder,
            "raw_features": raw_features,
            "norm_features": norm_features,
            "raw_combined": raw_combined,
            "norm_combined": norm_combined,
            "encoded_features": encoded_features,
            "encoded_combined": encoded_combined,
            "importance_df": importance_df,
            "selected_df": selected_df,
            "selected_encoded_df": selected_encoded_df,
            "dimred_results": dimred_results,
            "metadata": metadata,
        }

    # ── Input resolver ────────────────────────────────────────────────────

    def _resolve_input(self, signals_input, demographics) -> tuple:
        """
        Convert any supported input to {signal_name: DataFrame}.

        Reads from:
          - Module 5 output folder (cleaned_signals/ subfolder)
          - Flat folder with signal CSVs
          - dict of DataFrames
        """
        meta = {}

        if isinstance(signals_input, dict):
            return signals_input, meta, demographics

        if isinstance(signals_input, (str, Path)):
            folder = Path(signals_input)
            sigs = {}

            # Check for Module 5 v2 output structure (cleaned_signals/)
            cleaned_dir = folder / "cleaned_signals"
            sig_dir = cleaned_dir if cleaned_dir.exists() else folder

            file_map = {
                "EDA": "EDA.csv", "BVP": "BVP.csv", "IBI": "IBI.csv",
                "ST": "ST.csv", "ACC": "ACC.csv",
            }
            for sig, fname in file_map.items():
                fpath = sig_dir / fname
                if fpath.exists():
                    sigs[sig] = pd.read_csv(fpath)
                    self._log(f"  Loaded {fname}  ({len(sigs[sig]):,} rows)")

            # Load metadata
            meta_path = folder / "preprocessing_metadata.json"
            if not meta_path.exists():
                meta_path = folder / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)

            # Auto-load demographics from participant_demographics.csv
            if not demographics:
                demo_path = folder / "participant_demographics.csv"
                if demo_path.exists():
                    demo_df = pd.read_csv(demo_path)
                    if len(demo_df) > 0:
                        row = demo_df.iloc[0].to_dict()
                        demo_keys = {"age", "gender", "ethnicity",
                                     "autism_severity", "verbal_status",
                                     "comorbidity"}
                        demographics = {k: v for k, v in row.items()
                                        if k in demo_keys}
                        self._log(f"  Loaded demographics from CSV "
                                  f"({len(demographics)} fields)")

            return sigs, meta, demographics

        raise TypeError(
            f"Unsupported input type: {type(signals_input)}. "
            "Pass a folder path or dict of DataFrames."
        )

    def _log(self, msg):
        if self.verbose:
            print(msg)
