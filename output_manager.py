"""
=============================================================================
AUTISM PHYSIO-AI PIPELINE  |  output_manager.py
=============================================================================
Central output folder management.

Every pipeline run writes ALL outputs to one numbered folder:

    outputs/
        run_001/
            run_manifest.json
            module_1_acquisition/user_001/
                EDA.csv  BVP.csv  IBI.csv  ST.csv  ACC.csv
                combined_signals.csv  packet_metadata.json
            module_2a_simulation/user_001/       (if simulation used)
                EDA.csv  BVP.csv  ...  metadata.json
                annotations_events.csv  signal_EDA.png ...
            module_3_preprocessing/user_001/
                features_raw/
                    EDA_features.csv ... combined_features.csv
                features_normalised/
                    EDA_features_norm.csv ... combined_features_norm.csv
                processed_EDA.png  comparison_all_signals.png
                cleaning_report.csv  preprocessing_metadata.json
            module_4_analysis/user_001/
                analysis_report.html  3d_signals.html
                summary_tables/
                    descriptive_statistics.csv
                    kruskal_wallis_results.csv
                    pairwise_mannwhitney.csv
                    correlation_feature_target.csv
                    correlation_matrix.csv
                    temporal_dynamics.csv
                    return_to_median_summary.csv
                    pct_change_summary.csv
                    threshold_events.csv
                pct_change_from_baseline.png  temporal_dynamics.png
                significant_features_kruskal.png  ...  (17 PNGs)

Standalone runs use timestamped folders:
    outputs/standalone_M3_20250608_143022/
=============================================================================
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = REPO_ROOT / "outputs"


# ── Run folder creation ────────────────────────────────────────────────────

def next_pipeline_run_folder() -> Path:
    """Create outputs/run_NNN/ and return its path."""
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^run_(\d+)$")
    existing = [
        int(m.group(1))
        for d in OUTPUTS_ROOT.iterdir()
        if d.is_dir() and (m := pattern.match(d.name))
    ]
    n = (max(existing) + 1) if existing else 1
    folder = OUTPUTS_ROOT / f"run_{n:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def standalone_run_folder(module_label: str) -> Path:
    """Create outputs/standalone_MX_YYYYMMDD_HHMMSS/ and return its path."""
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = OUTPUTS_ROOT / f"standalone_{module_label}_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def module_subdir(run_folder: Path, module_name: str) -> Path:
    """Create and return run_folder/module_name/."""
    sub = run_folder / module_name
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def user_subdir(module_dir: Path, user_id: str) -> Path:
    """Create and return module_dir/user_id/."""
    sub = module_dir / user_id
    sub.mkdir(parents=True, exist_ok=True)
    return sub


# ── File counter ──────────────────────────────────────────────────────────

def count_files(folder: Path) -> dict:
    """Count files by extension under folder."""
    counts: dict = {"csv": 0, "png": 0, "html": 0, "json": 0,
                    "other": 0, "total": 0}
    if not folder.exists():
        return counts
    for f in folder.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower().lstrip(".")
        if ext in counts:
            counts[ext] += 1
        else:
            counts["other"] += 1
        counts["total"] += 1
    return counts


# ── Run manifest ──────────────────────────────────────────────────────────

class RunManifest:
    """Tracks and writes outputs/run_NNN/run_manifest.json."""

    def __init__(self, run_folder: Path):
        self.run_folder = run_folder
        self._data: dict = {
            "run_folder": str(run_folder),
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "modules_run": [],
            "users": [],
            "file_counts": {},
        }

    def record_module(self, label: str, folder: Path,
                      extra: Optional[dict] = None):
        counts = count_files(folder)
        entry = {
            "module": label,
            "folder": str(folder.relative_to(self.run_folder)),
            **counts,
        }
        if extra:
            entry.update(extra)
        self._data["modules_run"].append(entry)
        self._data["file_counts"][label] = counts["total"]

    def record_user(self, user_id: str, demographics: Optional[dict] = None):
        if not any(u["user_id"] == user_id for u in self._data["users"]):
            entry = {"user_id": user_id}
            if demographics:
                entry.update({
                    k: v for k, v in demographics.items()
                    if k in ("age", "gender", "autism_severity",
                             "verbal_status", "comorbidity", "ethnicity")
                })
            self._data["users"].append(entry)

    def save(self) -> Path:
        return self._write()

    def complete(self) -> Path:
        self._data["completed_at"] = datetime.now().isoformat()
        self._data["total_files"] = count_files(self.run_folder)
        index = {}
        for sub in sorted(self.run_folder.iterdir()):
            if sub.is_dir():
                index[sub.name] = sorted(
                    str(f.relative_to(self.run_folder))
                    for f in sub.rglob("*") if f.is_file()
                )
        self._data["file_index"] = index
        return self._write()

    def _write(self) -> Path:
        path = self.run_folder / "run_manifest.json"
        with open(path, "w") as fh:
            json.dump(self._data, fh, indent=2, default=str)
        return path
