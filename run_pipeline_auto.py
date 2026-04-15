"""
=============================================================================
AUTISM PHYSIO-AI PIPELINE  |  run_pipeline_auto.py
=============================================================================
Non-interactive full pipeline runner with sensible defaults.

Executes every available module in sequence using random/default values.
Automatically detects which modules are built and includes them.

Usage:
  python run_pipeline_auto.py
  python run_pipeline_auto.py --n_users 10 --seed 42
  python run_pipeline_auto.py --n_users 5 --duration 600 --no_plots

Pipeline order (new):
  M1+M2A  -> Simulate synthetic data
  M3      -> Split raw data (train/val/test) BEFORE any preprocessing
  M4      -> Exploratory Data Analysis (IBM EDA, TRAINING DATA ONLY)
  M5      -> Preprocess training data (clean + filter)
  [M5B]   -> Data augmentation (OPTIONAL, training only, if imbalanced)
  M6      -> Feature engineering (extract, select, reduce)
  M7      -> Model training (global + user-dependent models)

Test data is held out untouched until model evaluation.

After completion:
  - Opens the comprehensive HTML EDA report (if M4 ran)
  - Prints the selected features list (if M6 ran)
=============================================================================
"""
from __future__ import annotations

import argparse
import io
import sys
import time
import webbrowser
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

# Fix Windows console encoding for Unicode characters (arrows, etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
        line_buffering=True)
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace",
        line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from output_manager import (
    next_pipeline_run_folder, module_subdir, user_subdir,
    RunManifest, count_files,
)

# Module directories (new pipeline order)
M2A_DIR = REPO_ROOT / "module_2a_data_simulation"
M1_DIR = REPO_ROOT / "module_1_data_acquisition"
M3_DIR = REPO_ROOT / "module_3_data_splitting"
M4_DIR = REPO_ROOT / "module_4_eda"
M5_DIR = REPO_ROOT / "module_5_preprocessing"
M5B_DIR = REPO_ROOT / "module_5b_data_augmentation"
M6_DIR = REPO_ROOT / "module_6_feature_engineering"
M7_DIR = REPO_ROOT / "module_7_model_training"
M2B_DIR = REPO_ROOT / "module_2b_annotation"
M8_DIR = REPO_ROOT / "module_8_model_evaluation"

# Names of modules that conflict across pipeline modules
_CONFLICTING_MODULES = {
    "config", "signal_filters", "signal_cleaner",
    "annotator", "exporter", "visualiser", "visualizer",
    "simulator", "main", "normaliser",
    "analyser", "reporter", "signal_analyser", "statistical_analyser",
    "feature_extractor", "feature_engineer", "feature_selector",
    "dimensionality_reducer", "encoder", "splitter",
    "data_quality",
    "imbalance_detector", "augmentation_planner", "augmenter",
    "tgan_model", "tgan_trainer", "signal_generator", "sanity_checker",
    "selection_report",
    "trainer", "evaluator", "class_balancer", "data_loader",
    "models", "report_generator", "xai",
    "loader", "validator",
}

WIDTH = 66


# ─────────────────────────────────────────────────────────────────────────────
# PATH ISOLATION — prevents config.py collisions
# ─────────────────────────────────────────────────────────────────────────────

def _is_pipeline_module(name: str) -> bool:
    return name.split(".")[0] in _CONFLICTING_MODULES


@contextmanager
def _isolated_import(*module_dirs: Path):
    """Isolated import context -- same as pipeline_main.py."""
    saved_path = list(sys.path)
    saved_modules = {k: v for k, v in sys.modules.items()
                     if _is_pipeline_module(k)}
    for k in list(sys.modules.keys()):
        if _is_pipeline_module(k):
            del sys.modules[k]
    new_path = [str(d) for d in reversed(module_dirs)]
    sys.path[:] = new_path + [p for p in saved_path if p not in new_path]
    try:
        yield
    finally:
        for k in list(sys.modules.keys()):
            if _is_pipeline_module(k):
                del sys.modules[k]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


# ─────────────────────────────────────────────────────────────────────────────
# MODULE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_modules() -> dict:
    """Auto-detect which modules are available and ready to run."""
    modules = {}

    checks = [
        ("M2A", "Data Simulation", M2A_DIR, "simulator.py"),
        ("M1", "Data Acquisition", M1_DIR, "acquisition_module.py"),
        ("M2B", "Data Annotation (optional)", M2B_DIR, "annotator.py"),
        ("M3", "Data Splitting", M3_DIR, "splitter.py"),
        ("M4", "Exploratory Data Analysis", M4_DIR, "analyser.py"),
        ("M5", "Preprocessing", M5_DIR, "preprocessor.py"),
        ("M5B", "Data Augmentation (optional)", M5B_DIR, "augmenter.py"),
        ("M6", "Feature Engineering", M6_DIR, "feature_engineer.py"),
        ("M7", "Model Training", M7_DIR, "trainer.py"),
        ("M8", "Model Evaluation", M8_DIR, "main.py"),
    ]

    for label, name, directory, key_file in checks:
        if directory.exists() and (directory / key_file).exists():
            modules[label] = {
                "name": name,
                "dir": directory,
                "key_file": key_file,
            }

    return modules


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _banner():
    print()
    print("  " + "=" * (WIDTH - 2))
    print("  |    AUTISM PHYSIO-AI PIPELINE" + " " * (WIDTH - 38) + "|")
    print("  |    Automated Full Pipeline Runner" + " " * (WIDTH - 43) + "|")
    print("  " + "=" * (WIDTH - 2))
    print()


def _step(n, total, title):
    print()
    print("  " + "=" * (WIDTH - 4))
    print(f"  >>  Step {n}/{total}  --  {title}")
    print("  " + "=" * (WIDTH - 4))


def _ok(msg):
    print(f"  [OK]  {msg}")


def _skip(msg):
    print(f"  [SKIP]  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_m2_simulate(n_users, duration_s, seed, noise):
    """Run M2A + M1 simulation and return list of PipelinePackets."""
    with _isolated_import(M1_DIR, M2A_DIR):
        from acquisition_module import DataAcquisitionModule
        acq = DataAcquisitionModule(mode="2.2", save_packets=True, verbose=True)
        packets = acq.run_simulate(
            duration_s=duration_s,
            n_events="random",
            event_duration_s="random",
            emotions=None,       # random emotions
            noise_level=noise,
            seed=seed,
            n_users=n_users,
            shared_events=False,
            save_m1a_output=False,
        )
    return packets


def run_m2b_annotation(packet, output_dir, batch_csv=None):
    """Run Module 2B annotation (optional, for unannotated data)."""
    with _isolated_import(M2B_DIR):
        from main import run_annotation
        return run_annotation(
            packet=packet,
            output_dir=output_dir,
            interactive=False,
            batch_csv=batch_csv,
        )


def run_m3_split(packets, seed, output_dir=None):
    """Run Module 3 data splitting on raw signals from packets."""
    with _isolated_import(M3_DIR):
        from splitter import DataSplitter
        from exporter import SplitExporter

        splitter = DataSplitter(
            split_mode="three_way",
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            stratify=True,
            seed=seed,
            verbose=True,
        )

        # Load demographics and signals from packets
        signals_by_user, user_demo_df = splitter.load_from_packets(packets)

        if user_demo_df.empty:
            print("  [WARN] No user demographics found. Skipping M3.")
            return None

        # Assign users to train/val/test
        split_result = splitter.split_users(user_demo_df)

        # Partition raw signals by split
        split_result = splitter.split_raw_signals(
            split_result, signals_by_user)

        # Export raw signals organized by split
        exporter = SplitExporter(output_root=output_dir)
        exporter.export_raw_signals(split_result)

        return {
            "run_folder": exporter.run_dir,
            "split_result": split_result,
            "signals_by_user": signals_by_user,
        }


def run_m4_eda(packets_by_uid, train_users, demographics_by_uid=None,
               output_dir=None):
    """Run Module 4 Exploratory Data Analysis (combined multi-user)."""
    with _isolated_import(M4_DIR):
        from analyser import CombinedEDA
        eda = CombinedEDA(verbose=True)
        return eda.run(
            source={"packets": packets_by_uid, "train_users": train_users},
            demographics=demographics_by_uid,
            output_dir=output_dir,
        )


def run_m5_preprocess(source, demographics, session_id, user_id,
                      output_dir, gen_plots=True):
    """Run Module 5 preprocessing (clean + filter)."""
    with _isolated_import(M5_DIR):
        from preprocessor import DataPreprocessor
        pp = DataPreprocessor(
            filter_type="butterworth",
            apply_hampel=True,
            generate_plots=gen_plots,
            verbose=True,
        )
        return pp.run(
            signals_input=source,
            session_id=session_id,
            user_id=user_id,
            output_dir=output_dir,
        )


def run_m5b_augmentation(source_dir, output_dir, demographics=None,
                         device="cpu", seed=42):
    """Run Module 5B data augmentation (optional, training only)."""
    with _isolated_import(M5B_DIR):
        from augmenter import DataAugmenter
        augmenter = DataAugmenter(
            source_dir=source_dir,
            output_dir=output_dir,
            demographics=demographics,
            strategy="median",
            interactive=False,   # auto mode in pipeline
            device=device,
            seed=seed,
        )
        return augmenter.run()


def run_m6_feature_engineering(source, demographics, session_id, user_id,
                               output_dir=None):
    """Run Module 6 feature engineering."""
    with _isolated_import(M6_DIR):
        from feature_engineer import FeatureEngineer
        eng = FeatureEngineer(
            window_s=60.0,
            overlap=0.5,
            scaler_type="robust",
            top_n_features=20,
            selection_method="ensemble",
            pca_variance=0.95,
            verbose=True,
        )
        return eng.run(
            signals_input=source,
            demographics=demographics,
            session_id=session_id,
            user_id=user_id,
            output_dir=output_dir,
        )


# ─────────────────────────────────────────────────────────────────────────────
# POST-M6 AGGREGATION: combine per-user features into cross-user datasets
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_m6_across_users(
    m6_out: Path,
    m6_results: list,
    split_result=None,
    seed: int = 42,
) -> Path:
    """
    After per-user M6 runs, aggregate all user features into
    combined cross-user datasets with per-window train/val/test split.

    Creates:
      m6_out/combined_all_users/
        per_signal/
          EDA_features_all_users.csv
          BVP_features_all_users.csv
          ...
        combined_features_all_users.csv
        combined_features_normalised.csv
        feature_importance_global.csv
        combined_features_selected.csv
        feature_selection_report.html
        aggregation_metadata.json

    Returns
    -------
    Path to the combined_all_users/ folder
    """
    import json
    import numpy as np
    from datetime import datetime

    combined_dir = m6_out / "combined_all_users"
    signal_dir = combined_dir / "per_signal"
    signal_dir.mkdir(parents=True, exist_ok=True)

    signals = ["EDA", "BVP", "IBI", "ST", "ACC"]
    n_users = len(m6_results)

    print(f"\n  [Aggregator] Combining features across {n_users} users ...")

    # ── 1. Concat per-signal CSVs ─────────────────────────────────────────
    per_signal_dfs = {}
    for sig in signals:
        frames = []
        for res in m6_results:
            rf = Path(res["run_folder"]) / "features_raw" / f"{sig}_features.csv"
            if rf.exists():
                df = pd.read_csv(rf)
                frames.append(df)
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            out_path = signal_dir / f"{sig}_features_all_users.csv"
            merged.to_csv(out_path, index=False, float_format="%.6f")
            per_signal_dfs[sig] = merged
            print(f"    {sig}: {len(merged):,} rows from {len(frames)} users")

    # ── 2. Concat combined feature CSVs ──────────────────────────────────
    combined_frames = []
    for res in m6_results:
        cf = Path(res["run_folder"]) / "features_raw" / "combined_features.csv"
        if cf.exists():
            combined_frames.append(pd.read_csv(cf))

    if not combined_frames:
        print("  [Aggregator] No combined features found — skipping.")
        return combined_dir

    combined_all = pd.concat(combined_frames, ignore_index=True)
    combined_all.to_csv(
        combined_dir / "combined_features_all_users.csv",
        index=False, float_format="%.6f",
    )
    print(f"    Combined: {len(combined_all):,} rows, "
          f"{combined_all.shape[1]} cols")

    # ── 3. Per-window train/val/test split column ────────────────────────
    rng = np.random.default_rng(seed)
    n = len(combined_all)
    n_train = int(n * 0.60)
    n_val = int(n * 0.20)
    indices = rng.permutation(n)
    split_col = pd.Series("train", index=combined_all.index)
    split_col.iloc[indices[n_train:n_train + n_val]] = "val"
    split_col.iloc[indices[n_train + n_val:]] = "test"
    combined_all["split"] = split_col
    combined_all.to_csv(
        combined_dir / "combined_features_all_users_split.csv",
        index=False, float_format="%.6f",
    )
    print(f"    Window split: train={n_train}, "
          f"val={n_val}, test={n - n_train - n_val}")

    # ── 4. Re-normalise on the combined data ─────────────────────────────
    with _isolated_import(M6_DIR):
        from feature_selector import FeatureSelector
        from config import META_COLS
        from selection_report import SelectionReportGenerator

    from sklearn.preprocessing import RobustScaler

    # Identify feature columns
    feat_cols = [c for c in combined_all.columns
                 if c not in META_COLS and c != "split"]

    train_mask = combined_all["split"] == "train"
    scaler = RobustScaler()
    scaler.fit(combined_all.loc[train_mask, feat_cols])
    norm_all = combined_all.copy()
    norm_all[feat_cols] = scaler.transform(combined_all[feat_cols])
    norm_all.to_csv(
        combined_dir / "combined_features_normalised.csv",
        index=False, float_format="%.6f",
    )
    print("    Normalised (RobustScaler fitted on train windows)")

    # ── 5. Re-run feature selection on global data ───────────────────────
    selector = FeatureSelector(top_n=20, method="ensemble", seed=seed)
    # Use normalised combined (drop split col for selection)
    select_input = norm_all.drop(columns=["split"], errors="ignore")
    importance_df, selected_df, _ = selector.rank_and_select(select_input)

    if importance_df is not None and len(importance_df):
        importance_df.to_csv(
            combined_dir / "feature_importance_global.csv",
            index=False, float_format="%.6f",
        )
        print(f"    Global feature importance: {len(importance_df)} features ranked")

    if selected_df is not None and len(selected_df):
        # Add split column back
        selected_df["split"] = combined_all["split"].values
        selected_df.to_csv(
            combined_dir / "combined_features_selected.csv",
            index=False, float_format="%.6f",
        )
        print(f"    Selected: top {selected_df.shape[1] - 1} features, "
              f"{len(selected_df):,} rows")

    # ── 6. Generate global feature selection report ──────────────────────
    reporter = SelectionReportGenerator(combined_dir)
    reporter.generate(
        all_columns=list(combined_all.columns),
        importance_df=importance_df,
        top_n=20,
        selection_method="ensemble",
        metadata={
            "scope": "combined_all_users",
            "n_users": n_users,
            "n_windows": len(combined_all),
        },
    )

    # ── 7. Write aggregation metadata ────────────────────────────────────
    meta = {
        "aggregation": "combined_all_users",
        "n_users": n_users,
        "user_ids": [r["metadata"].get("user_id", "?") for r in m6_results],
        "n_windows_total": len(combined_all),
        "n_features": len(feat_cols),
        "split_method": "per_window_random",
        "split_seed": seed,
        "n_train_windows": int(train_mask.sum()),
        "n_val_windows": int((combined_all["split"] == "val").sum()),
        "n_test_windows": int((combined_all["split"] == "test").sum()),
        "created_at": datetime.now().isoformat(),
    }
    with open(combined_dir / "aggregation_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [Aggregator] Done — {combined_dir.relative_to(m6_out.parent)}")
    return combined_dir


def export_to_data_folder(run_folder: Path, m6_out: Path):
    """
    Copy the final curated dataset to data/ in the repo root
    for version control and GitHub fetch.

    Structure:
      data/
        per_user/
          user_001/
            per_signal/   (EDA, BVP, IBI, ST, ACC)
            combined_features.csv
            feature_importance.csv
          user_002/
          ...
        combined_all_users/
          per_signal/
          combined_features_all_users.csv
          combined_features_selected.csv
          feature_importance_global.csv
          feature_selection_report.html
        dataset_metadata.json
    """
    import json
    import shutil
    from datetime import datetime

    data_dir = REPO_ROOT / "data"
    per_user_dir = data_dir / "per_user"

    print("\n  [DataExport] Exporting final dataset to data/ ...")

    # ── Per-user exports ──────────────────────────────────────────────────
    user_dirs = sorted([d for d in m6_out.iterdir()
                        if d.is_dir() and d.name.startswith("user_")])

    for ud in user_dirs:
        uid = ud.name
        dest = per_user_dir / uid
        sig_dest = dest / "per_signal"
        sig_dest.mkdir(parents=True, exist_ok=True)

        # Copy per-signal raw features
        raw_dir = ud / "features_raw"
        if raw_dir.exists():
            for sig_csv in raw_dir.glob("*_features.csv"):
                if sig_csv.name != "combined_features.csv":
                    shutil.copy2(sig_csv, sig_dest / sig_csv.name)
            # Copy combined
            combined = raw_dir / "combined_features.csv"
            if combined.exists():
                shutil.copy2(combined, dest / "combined_features.csv")

        # Copy feature importance
        imp = ud / "features_selected" / "feature_importance.csv"
        if imp.exists():
            shutil.copy2(imp, dest / "feature_importance.csv")

        # Copy selection report
        report = ud / "feature_selection_report.html"
        if report.exists():
            shutil.copy2(report, dest / "feature_selection_report.html")

        print(f"    {uid}: exported")

    # ── Combined across users ────────────────────────────────────────────
    combined_src = m6_out / "combined_all_users"
    combined_dest = data_dir / "combined_all_users"
    if combined_src.exists():
        if combined_dest.exists():
            shutil.rmtree(combined_dest)
        shutil.copytree(combined_src, combined_dest)
        print("    combined_all_users: exported")

    # ── Dataset metadata ─────────────────────────────────────────────────
    meta = {
        "pipeline_run": run_folder.name,
        "exported_at": datetime.now().isoformat(),
        "n_users": len(user_dirs),
        "user_ids": [d.name for d in user_dirs],
        "contents": {
            "per_user": "Individual user feature sets with per-signal "
                        "and combined CSVs",
            "combined_all_users": "Cross-user aggregated features with "
                                  "per-window train/val/test split",
        },
        "fetch_url": (
            "https://github.com/Joe-jacob-2755/"
            "autism-physio-ai-pipeline/tree/main/data"
        ),
    }
    with open(data_dir / "dataset_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("  [DataExport] Done — data/ ready for git commit + push")
    return data_dir


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data_from_github(
    repo: str = "Joe-jacob-2755/autism-physio-ai-pipeline",
    branch: str = "main",
    data_path: str = "data",
    dest: Path = None,
) -> Path:
    """
    Download the data/ folder from a GitHub repository.

    Uses GitHub's API to download files without requiring git clone.
    Falls back to raw.githubusercontent.com for individual files.

    Parameters
    ----------
    repo     : GitHub repo in owner/repo format
    branch   : branch name (default: main)
    data_path: path within repo to the data folder
    dest     : local destination (default: REPO_ROOT / data)

    Returns
    -------
    Path to the downloaded data folder
    """
    import json
    import urllib.request
    import urllib.error

    dest = dest or REPO_ROOT / "data"
    dest.mkdir(parents=True, exist_ok=True)

    api_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    print(f"\n  [GitHubFetch] Fetching file list from {repo}@{branch} ...")

    try:
        req = urllib.request.Request(api_url)
        req.add_header("Accept", "application/vnd.github.v3+json")
        req.add_header("User-Agent", "autism-physio-ai-pipeline")

        with urllib.request.urlopen(req, timeout=30) as resp:
            tree = json.loads(resp.read().decode())

        # Filter files under data_path/
        data_files = [
            entry for entry in tree.get("tree", [])
            if entry["path"].startswith(f"{data_path}/")
            and entry["type"] == "blob"
        ]

        if not data_files:
            print(f"  [GitHubFetch] No files found under {data_path}/")
            return dest

        print(f"  [GitHubFetch] Found {len(data_files)} files to download")

        for entry in data_files:
            rel_path = entry["path"][len(data_path) + 1:]  # strip "data/"
            local_path = dest / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            raw_url = (f"https://raw.githubusercontent.com/"
                       f"{repo}/{branch}/{entry['path']}")
            try:
                urllib.request.urlretrieve(raw_url, local_path)
            except urllib.error.URLError as e:
                print(f"    [WARN] Failed to download {rel_path}: {e}")
                continue

        print(f"  [GitHubFetch] Downloaded {len(data_files)} files → {dest}")
        return dest

    except urllib.error.URLError as e:
        print(f"  [GitHubFetch] API request failed: {e}")
        print("  Falling back to local data/ folder if available.")
        return dest


# ─────────────────────────────────────────────────────────────────────────────
# M7 — MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def run_m7_model_training(source, output_dir, seed=42,
                          skip_temporal=False, skip_unsupervised=False,
                          per_user_dir=None):
    """Run Module 7 model training on combined features from M6."""
    with _isolated_import(M7_DIR):
        from main import run_module_7

        return run_module_7(
            source=source,
            output_dir=Path(output_dir),
            seed=seed,
            verbose=True,
            skip_temporal=skip_temporal,
            skip_unsupervised=skip_unsupervised,
            user_dependent=per_user_dir is not None,
            per_user_dir=per_user_dir,
        )


# ─────────────────────────────────────────────────────────────────────────────
# M8 — MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def run_m8_model_evaluation(m7_dir, features_csv, output_dir, seed=42):
    """Run Module 8 model evaluation on the held-out test set."""
    with _isolated_import(M8_DIR):
        from main import run_module_8

        return run_module_8(
            m7_dir=str(m7_dir),
            features_csv=str(features_csv),
            output_dir=str(output_dir),
            seed=seed,
            verbose=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def show_selected_features(run_folder: Path):
    """Find and display the selected features from M6 output."""
    patterns = [
        "**/features_selected/feature_importance.csv",
        "**/features_selected/top_*_features.csv",
        "**/feature_importance.csv",
    ]
    for pattern in patterns:
        matches = list(run_folder.rglob(pattern.replace("**", "*")))
        if not matches:
            matches = list(run_folder.rglob(
                pattern.split("/")[-1]))
        if matches:
            for match in matches:
                try:
                    df = pd.read_csv(match)
                    print(f"\n  {'=' * (WIDTH - 4)}")
                    print(f"  SELECTED FEATURES  ({match.name})")
                    print(f"  {'=' * (WIDTH - 4)}")
                    if "feature" in df.columns and "importance" in df.columns:
                        print(f"\n  {'Rank':<6}{'Feature':<40}{'Importance':<12}")
                        print(f"  {'-'*6}{'-'*40}{'-'*12}")
                        for idx, row in df.head(20).iterrows():
                            print(f"  {idx+1:<6}{row['feature']:<40}"
                                  f"{row['importance']:.4f}")
                    else:
                        print(f"\n  Columns: {list(df.columns)}")
                        print(f"  Rows: {len(df)}")
                        print(df.head(20).to_string(index=False))
                    print()
                    return match
                except Exception as e:
                    print(f"  [WARN] Could not read {match}: {e}")

    print("\n  [INFO] No feature importance file found.")
    return None


def open_eda_report(run_folder: Path):
    """Find and open the comprehensive HTML EDA report."""
    matches = list(run_folder.rglob("analysis_report.html"))
    if not matches:
        matches = list(run_folder.rglob("*.html"))

    if matches:
        report_path = matches[0]
        print(f"\n  Opening EDA report: {report_path.name}")
        try:
            webbrowser.open(report_path.as_uri())
            return report_path
        except Exception as e:
            print(f"  [WARN] Could not open browser: {e}")
            print(f"  Report saved at: {report_path}")
            return report_path
    else:
        print("\n  [INFO] No HTML EDA report found.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_auto_pipeline(
    n_users: int = 10,
    duration_s: float = 300.0,
    seed: int = 42,
    noise: str = "medium",
    gen_plots: bool = True,
    open_report: bool = True,
    skip_m5b: bool = False,
    skip_m7: bool = False,
    stop_after: str = None,
):
    """
    Run the full pipeline non-interactively with defaults.

    New pipeline order:
      M1+M2A (simulate) -> M3 (split raw) -> M4 (EDA, train only)
      -> M5 (preprocess, train+val) -> [M5B augment if needed]
      -> M6 (feature eng, train+val)

    Test data is held out until model evaluation.
    """
    _banner()

    # ── Detect available modules ──────────────────────────────────────────
    available = _detect_modules()
    module_list = list(available.keys())

    print("  Detected modules:")
    for label, info in available.items():
        print(f"    [{label}]  {info['name']}")
    print()

    # Count total steps
    total_steps = 1  # M2A+M1 always
    if "M3" in available:
        total_steps += 1
    if "M4" in available:
        total_steps += 1
    if "M5" in available:
        total_steps += 1
    if "M5B" in available and not skip_m5b:
        total_steps += 1  # optional step
    if "M6" in available:
        total_steps += 1
        total_steps += 1  # cross-user aggregation + data export
    if "M7" in available:
        total_steps += 1

    print(f"  Pipeline: {' -> '.join(module_list)}")
    print(f"  Users: {n_users}  |  Duration: {duration_s:.0f}s  |  "
          f"Seed: {seed}  |  Noise: {noise}")
    print(f"  Steps: {total_steps}")
    print()

    t_start = time.time()
    run_folder = next_pipeline_run_folder()
    manifest = RunManifest(run_folder)
    rel_run = run_folder.relative_to(REPO_ROOT)
    print(f"  Output folder: {rel_run}\n")

    step = 0
    split_result = None
    train_users = []
    m4_results = []
    m5_results = []
    m6_results = []

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: M2A + M1 -- Data Simulation & Acquisition
    # ══════════════════════════════════════════════════════════════════════
    step += 1
    _step(step, total_steps, "Data Simulation + Acquisition  (M2A + M1)")

    print(f"\n  Simulating {n_users} users x {duration_s:.0f}s  "
          f"(seed={seed}, noise={noise})")

    packets = run_m2_simulate(n_users, duration_s, seed, noise)
    _ok(f"Module 1 complete -- {len(packets)} packet(s) acquired")

    m1_out = module_subdir(run_folder, "module_1_acquisition")
    manifest.record_module("M1", m1_out, {
        "mode": "simulate", "n_packets": len(packets)})
    manifest.save()

    if not packets:
        print("\n  [ERROR] No data packets returned. Aborting.")
        return

    # Build packets lookup and demographics
    packets_by_uid = {}
    demographics_by_uid = {}
    for i, pkt in enumerate(packets):
        uid = getattr(pkt, "user_id", f"user_{i+1:03d}")
        packets_by_uid[uid] = pkt
        if hasattr(pkt, "metadata") and pkt.metadata:
            # Try both "user_profile" (M2A simulated) and "user" (imported)
            meta_user = (pkt.metadata.get("user_profile")
                         or pkt.metadata.get("user", {}))
            if meta_user:
                demographics_by_uid[uid] = {
                    k: meta_user[k]
                    for k in ("age", "gender", "ethnicity",
                              "autism_severity", "verbal_status",
                              "comorbidity")
                    if k in meta_user and meta_user[k] is not None
                }
        manifest.record_user(uid, demographics_by_uid.get(uid))

    # ══════════════════════════════════════════════════════════════════════
    # OPTIONAL: M2B -- Data Annotation (only if packets are unannotated)
    # ══════════════════════════════════════════════════════════════════════
    if "M2B" in available:
        unannotated = [
            (uid, pkt) for uid, pkt in packets_by_uid.items()
            if not getattr(pkt, "is_annotated", True)
        ]
        if unannotated:
            step += 1
            _step(step, total_steps,
                  "Data Annotation  (M2B) -- OPTIONAL, unannotated data")
            m2b_out = module_subdir(run_folder, "module_2b_annotation")
            n_annotated = 0
            for uid, pkt in unannotated:
                try:
                    annotated_pkt = run_m2b_annotation(
                        pkt, user_subdir(m2b_out, uid))
                    if getattr(annotated_pkt, "is_annotated", False):
                        packets_by_uid[uid] = annotated_pkt
                        n_annotated += 1
                except Exception as e:
                    print(f"  [WARN] M2B annotation failed for {uid}: {e}")
            if n_annotated:
                manifest.record_module("M2B", m2b_out, {
                    "n_annotated": n_annotated})
                _ok(f"Module 2B complete -- {n_annotated} packet(s) annotated")
            else:
                _skip("M2B: No packets annotated (batch CSV required "
                      "for non-interactive mode)")
            # Rebuild packets list from updated dict
            packets = list(packets_by_uid.values())
        else:
            _skip("M2B: All packets already annotated")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: M3 -- Data Splitting (BEFORE preprocessing)
    # ══════════════════════════════════════════════════════════════════════
    if "M3" in available:
        step += 1
        _step(step, total_steps,
              "Data Splitting  (M3) -- split BEFORE preprocessing")

        if len(packets) >= 2:
            m3_split_dir = module_subdir(run_folder, "module_3_splitting")
            m3_result = run_m3_split(packets, seed,
                                     output_dir=str(m3_split_dir))
            if m3_result:
                split_result = m3_result["split_result"]
                train_users = split_result.train_users
                val_users = split_result.val_users
                test_users = split_result.test_users

                m3_out = m3_result["run_folder"]
                manifest.record_module("M3", m3_out, {
                    "n_train": len(train_users),
                    "n_val": len(val_users),
                    "n_test": len(test_users),
                })
                manifest.save()

                print(f"\n  Train users ({len(train_users)}): {train_users}")
                if val_users:
                    print(f"  Val users   ({len(val_users)}): {val_users}")
                print(f"  Test users  ({len(test_users)}): {test_users}")
                print("  ** Test data held out -- NOT processed until "
                      "model evaluation **")
                _ok(f"Module 3 complete -- data split into "
                    f"train/val/test at {m3_out.name}")
            else:
                # Fallback: all users are training users
                train_users = list(packets_by_uid.keys())
                _skip("M3 splitting failed. Using all users for training.")
        else:
            train_users = list(packets_by_uid.keys())
            _skip("M3 requires at least 2 users. Using all for training.")
    else:
        train_users = list(packets_by_uid.keys())

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: M4 -- Exploratory Data Analysis (TRAINING DATA ONLY)
    # ══════════════════════════════════════════════════════════════════════
    if "M4" in available:
        step += 1
        _step(step, total_steps,
              "Exploratory Data Analysis  (M4) -- COMBINED TRAINING DATA")

        m4_out = module_subdir(run_folder, "module_4_eda")

        try:
            m4_result = run_m4_eda(
                packets_by_uid=packets_by_uid,
                train_users=train_users,
                demographics_by_uid=demographics_by_uid,
                output_dir=str(m4_out),
            )
            m4_results.append(m4_result)
            manifest.record_module("M4", m4_out, {
                "n_users": len(train_users),
                "data_scope": "combined_training",
            })
            manifest.save()
            _ok(f"Module 4 complete -- {len(train_users)} users analysed "
                f"(combined training data)")
        except Exception as e:
            print(f"  [WARN] M4 EDA failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Early stop check ────────────────────────────────────────────────
    _STOP_ORDER = ["M1", "M2B", "M3", "M4", "M5", "M5B", "M6", "M7", "M8"]
    if stop_after and stop_after in _STOP_ORDER:
        stop_idx = _STOP_ORDER.index(stop_after)
        if stop_idx <= 2:  # M1, M3, or M4
            elapsed = time.time() - t_start
            _ok(f"Pipeline stopped after {stop_after} "
                f"(--stop_after). Total time: {elapsed:.1f}s")
            manifest.save()
            return run_folder

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: M5 -- Preprocessing  (train + val users only)
    # ══════════════════════════════════════════════════════════════════════
    if "M5" in available:
        step += 1
        _step(step, total_steps,
              "Preprocessing  (M5) -- train + val users")

        m5_out = module_subdir(run_folder, "module_5_preprocessing")

        # Process train + val users (NOT test)
        process_users = list(train_users)
        if split_result and split_result.val_users:
            process_users += list(split_result.val_users)

        for uid in process_users:
            if uid not in packets_by_uid:
                continue
            pkt = packets_by_uid[uid]
            sid = getattr(pkt, "session_id", "session_001")
            user_m5_dir = user_subdir(m5_out, uid)
            dem = demographics_by_uid.get(uid)

            split_label = "train"
            if split_result and uid in split_result.val_users:
                split_label = "val"
            print(f"\n  Preprocessing [{uid}] ({split_label}) ...")

            try:
                result = run_m5_preprocess(
                    source=pkt, demographics=dem,
                    session_id=sid, user_id=uid,
                    output_dir=str(user_m5_dir),
                    gen_plots=gen_plots,
                )
                m5_results.append(result)
            except Exception as e:
                print(f"  [WARN] M5 failed for {uid}: {e}")

        if m5_results:
            manifest.record_module("M5", m5_out, {
                "n_users": len(m5_results)})
            manifest.save()
            _ok(f"Module 5 complete -- {len(m5_results)} users preprocessed")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5 (optional): M5B -- Data Augmentation  (training only)
    # ══════════════════════════════════════════════════════════════════════
    m5b_result = None
    if "M5B" in available and m5_results and not skip_m5b:
        step += 1
        _step(step, total_steps,
              "Data Augmentation  (M5B) -- OPTIONAL, training only")

        m5b_out = module_subdir(run_folder, "module_5b_augmentation")
        m5_out_path = module_subdir(run_folder, "module_5_preprocessing")

        # Build training demographics
        train_demographics = {
            uid: demographics_by_uid.get(uid, {})
            for uid in train_users
        }

        try:
            m5b_result = run_m5b_augmentation(
                source_dir=m5_out_path,
                output_dir=m5b_out,
                demographics=train_demographics,
                device="cpu",
                seed=seed,
            )

            if m5b_result.skipped:
                _skip(f"M5B skipped: {m5b_result.skip_reason}")
            else:
                n_synth = sum(
                    len(segs)
                    for by_emo in m5b_result.accepted_segments.values()
                    for segs in by_emo.values()
                )
                manifest.record_module("M5B", m5b_out, {
                    "n_synthetic_segments": n_synth,
                    "imbalance_ratio": m5b_result.imbalance_report.imbalance_ratio,
                    "severity": m5b_result.imbalance_report.severity,
                })
                manifest.save()
                _ok(f"Module 5B complete -- {n_synth} synthetic segments "
                    f"accepted")
        except Exception as e:
            print(f"  [WARN] M5B augmentation failed: {e}")
            _skip("M5B failed, continuing without augmentation")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: M6 -- Feature Engineering  (train + val users only)
    # ══════════════════════════════════════════════════════════════════════
    if "M6" in available:
        step += 1
        _step(step, total_steps,
              "Feature Engineering  (M6) -- train + val users")

        m6_out = module_subdir(run_folder, "module_6_feature_engineering")

        for i, result in enumerate(m5_results):
            uid = result["metadata"].get("user_id", "user_001")
            sid = result["metadata"].get("session_id", "session_001")
            user_m6_dir = user_subdir(m6_out, uid)
            dem = demographics_by_uid.get(uid)

            split_label = "train"
            if split_result and uid in split_result.val_users:
                split_label = "val"
            print(f"\n  Feature engineering [{uid}] ({split_label}) ...")

            try:
                m6_result = run_m6_feature_engineering(
                    source=result["run_folder"],
                    demographics=dem,
                    session_id=sid,
                    user_id=uid,
                    output_dir=str(user_m6_dir),
                )
                m6_results.append(m6_result)
            except Exception as e:
                print(f"  [WARN] M6 failed for {uid}: {e}")

        if m6_results:
            manifest.record_module("M6", m6_out, {
                "n_users": len(m6_results)})
            manifest.save()
            _ok(f"Module 6 complete -- {len(m6_results)} users processed")

            # ── Post-M6: Aggregate across users ──────────────────────────
            step += 1
            _step(step, total_steps,
                  "Cross-User Aggregation + Data Export")

            try:
                combined_dir = aggregate_m6_across_users(
                    m6_out=m6_out,
                    m6_results=m6_results,
                    split_result=split_result,
                    seed=seed,
                )
                manifest.record_module("M6_AGG", combined_dir, {
                    "scope": "combined_all_users"})
                manifest.save()

                # Export to data/ folder for GitHub
                export_to_data_folder(run_folder, m6_out)
                _ok("Cross-user aggregation + data/ export complete")
            except Exception as e:
                print(f"  [WARN] Aggregation failed: {e}")
                import traceback
                traceback.print_exc()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: M7 -- Model Training (global models on combined data)
    # ══════════════════════════════════════════════════════════════════════
    m7_out = None
    if "M7" in available and m6_results and not skip_m7:
        step += 1
        _step(step, total_steps, "Model Training  (M7) -- global models")

        m7_out_dir = module_subdir(run_folder, "module_7_model_training")

        # Find the combined features CSV from M6 aggregation
        combined_csv = None
        combined_dir = run_folder / "module_6_feature_engineering" / "combined_all_users"
        if combined_dir.exists():
            candidates = [
                combined_dir / "combined_features_all_users_split.csv",
                combined_dir / "combined_features_all_users.csv",
            ]
            for c in candidates:
                if c.exists():
                    combined_csv = c
                    break
        if combined_csv is None:
            # Search broader
            csvs = list(run_folder.rglob("combined_features_all_users*.csv"))
            if csvs:
                combined_csv = csvs[0]

        if combined_csv:
            print(f"\n  Training on: {combined_csv.name}")
            try:
                # Find per-user data for user-dependent models
                per_user_m6 = run_folder / "module_6_feature_engineering"
                per_user_dirs = [
                    d for d in per_user_m6.iterdir()
                    if d.is_dir() and d.name.startswith("user_")]
                per_user = per_user_m6 if per_user_dirs else None

                m7_out = run_m7_model_training(
                    source=str(combined_csv),
                    output_dir=str(m7_out_dir),
                    seed=seed,
                    skip_temporal=False,
                    skip_unsupervised=False,
                    per_user_dir=per_user,
                )
                manifest.record_module("M7", m7_out_dir, {
                    "source": str(combined_csv),
                })
                manifest.save()
                _ok("Module 7 complete -- models trained and compared")
            except Exception as e:
                print(f"  [WARN] M7 failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("  [WARN] No combined features CSV found for M7")
            _skip("M7 skipped -- no combined features")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 8: M8 -- Model Evaluation (held-out test set)
    # ══════════════════════════════════════════════════════════════════════
    m8_out = None
    if "M8" in available and m7_out is not None:
        step += 1
        _step(step, total_steps, "Model Evaluation  (M8) -- held-out test set")

        m8_out_dir = module_subdir(run_folder, "module_8_model_evaluation")

        # Locate M7 output dir and combined features CSV
        m7_out_dir = run_folder / "module_7_model_training"

        combined_csv = None
        combined_dir = run_folder / "module_6_feature_engineering" / "combined_all_users"
        if combined_dir.exists():
            candidates = [
                combined_dir / "combined_features_all_users_split.csv",
                combined_dir / "combined_features_all_users.csv",
            ]
            for c in candidates:
                if c.exists():
                    combined_csv = c
                    break
        if combined_csv is None:
            csvs = list(run_folder.rglob("combined_features_all_users*.csv"))
            if csvs:
                combined_csv = csvs[0]

        if combined_csv and m7_out_dir.exists():
            print(f"\n  Evaluating models on test set...")
            print(f"  M7 dir:     {m7_out_dir}")
            print(f"  Features:   {combined_csv.name}")
            try:
                m8_out = run_m8_model_evaluation(
                    m7_dir=m7_out_dir,
                    features_csv=combined_csv,
                    output_dir=m8_out_dir,
                    seed=seed,
                )
                manifest.record_module("M8", m8_out_dir, {
                    "best_model": m8_out.get("best_model"),
                    "best_test_f1": m8_out.get("best_test_f1"),
                })
                manifest.save()
                _ok(f"Module 8 complete -- best model: "
                    f"{m8_out.get('best_model', '?')} "
                    f"(F1={m8_out.get('best_test_f1', 0):.4f})")
            except Exception as e:
                print(f"  [WARN] M8 failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            if not combined_csv:
                print("  [WARN] No combined features CSV found for M8")
            if not m7_out_dir.exists():
                print("  [WARN] M7 output directory not found for M8")
            _skip("M8 skipped -- missing M7 output or features CSV")

    # ══════════════════════════════════════════════════════════════════════
    # FINALISE
    # ══════════════════════════════════════════════════════════════════════
    manifest.complete()
    elapsed = time.time() - t_start
    total = count_files(run_folder)

    print()
    print("  " + "=" * (WIDTH - 4))
    print(f"  Pipeline Complete  ({elapsed:.1f}s)")
    print("  " + "=" * (WIDTH - 4))

    # Build status summary
    test_status = ""
    if split_result:
        test_status = (f"\n  Test users held out: {split_result.test_users}"
                       f"\n  ** Test data untouched -- ready for model eval **")

    print(f"""
  Output: {rel_run}
    {total['csv']} CSV files  |  {total['png']} PNG plots
    {total['html']} HTML reports  |  {total['json']} JSON metadata
    {total['total']} total files

  Modules executed: {' -> '.join(module_list)}
  Users: {n_users}  |  Seed: {seed}{test_status}
""")

    # ── Show selected features (M6) ──────────────────────────────────────
    if m6_results:
        show_selected_features(run_folder)

    # ── Open EDA report (M4) ─────────────────────────────────────────────
    if m4_results and open_report:
        open_eda_report(run_folder)

    return run_folder


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated full pipeline runner with defaults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline_auto.py                         # 10 users, defaults
  python run_pipeline_auto.py --n_users 5 --seed 99
  python run_pipeline_auto.py --n_users 20 --noise high --no_plots
  python run_pipeline_auto.py --no_report             # skip opening browser
""",
    )
    parser.add_argument("--n_users", type=int, default=10,
                        help="Number of simulated users (default: 10)")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="Simulation duration per user in seconds "
                             "(default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed (default: 42)")
    parser.add_argument("--noise", default="medium",
                        choices=["low", "medium", "high"],
                        help="Noise level (default: medium)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip PNG plot generation")
    parser.add_argument("--no_report", action="store_true",
                        help="Do not open EDA report in browser")
    parser.add_argument("--skip_m5b", action="store_true",
                        help="Skip M5B data augmentation (slow on CPU)")
    parser.add_argument("--skip_m7", action="store_true",
                        help="Skip M7 model training")
    parser.add_argument("--stop_after", type=str, default=None,
                        choices=["M1", "M3", "M4", "M5", "M5B", "M6", "M7", "M8"],
                        help="Stop pipeline after this module "
                             "(e.g. --stop_after M4 for EDA only)")
    parser.add_argument("--fetch_github", action="store_true",
                        help="Fetch existing data from GitHub data/ folder "
                             "instead of simulating")
    parser.add_argument("--source_url", type=str, default=None,
                        help="GitHub repo to fetch data from "
                             "(default: Joe-jacob-2755/"
                             "autism-physio-ai-pipeline)")

    args = parser.parse_args()

    if args.fetch_github:
        repo = (args.source_url
                or "Joe-jacob-2755/autism-physio-ai-pipeline")
        print(f"\n  Fetching data from GitHub: {repo}")
        fetch_data_from_github(repo=repo)
        print("  Data available at: data/")
    else:
        run_auto_pipeline(
            n_users=args.n_users,
            duration_s=args.duration,
            seed=args.seed,
            noise=args.noise,
            gen_plots=not args.no_plots,
            open_report=not args.no_report,
            skip_m5b=args.skip_m5b,
            skip_m7=args.skip_m7,
            stop_after=args.stop_after,
        )
