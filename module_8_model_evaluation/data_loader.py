"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  data_loader.py
=============================================================================
Load test features from M6 combined CSV and M7 training metadata.
=============================================================================
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    TARGET_COL, SPLIT_COL,
    ALL_NON_FEATURE_COLS, DEMOGRAPHIC_RAW, DEMOGRAPHIC_ENC,
    KNOWN_NAN_COLS, AGE_BINS, AGE_LABELS,
)

log = logging.getLogger(__name__)


@dataclass
class TestData:
    """Container for test set data ready for evaluation."""
    X: np.ndarray                     # (n_test, n_features)
    y: np.ndarray                     # (n_test,)  encoded labels
    y_str: np.ndarray                 # (n_test,)  string labels
    feature_names: List[str]
    class_names: List[str]
    n_classes: int
    demographics: Optional[pd.DataFrame]  # demographic cols for test rows
    user_ids: Optional[np.ndarray]        # user_id per test row
    n_samples: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.n_samples = len(self.y)


@dataclass
class M7Metadata:
    """Training metadata loaded from M7."""
    best_model: str
    class_names: List[str]
    n_classes: int
    seed: int
    balance_strategy: str
    primary_metric: str
    class_distribution: Dict[str, int]
    n_models_trained: int
    total_time_s: float
    raw: Dict[str, Any] = field(default_factory=dict)


class M8DataLoader:
    """Load test data from M6 CSV and metadata from M7."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [M8-Data] {msg}")

    def load_test_data(self, features_csv: Path) -> TestData:
        """Load test split from combined features CSV."""
        features_csv = Path(features_csv)
        if not features_csv.exists():
            raise FileNotFoundError(f"Features CSV not found: {features_csv}")

        self._log(f"Loading: {features_csv.name}")
        df = pd.read_csv(features_csv)
        self._log(f"  Total rows: {len(df):,}")

        # Filter to test split
        if SPLIT_COL not in df.columns:
            raise ValueError(f"CSV missing '{SPLIT_COL}' column -- cannot identify test set")

        test_df = df[df[SPLIT_COL] == "test"].copy()
        if test_df.empty:
            raise ValueError("No test rows found (split=='test')")
        self._log(f"  Test rows: {len(test_df):,}")

        # Extract target labels
        if TARGET_COL not in test_df.columns:
            raise ValueError(f"CSV missing '{TARGET_COL}' column")

        y_str = test_df[TARGET_COL].values
        # Use ALL classes from the full dataset (not just test) so label
        # encoding matches M7 training.  Models may predict classes absent
        # from the test split.
        all_classes = sorted(df[TARGET_COL].unique().tolist())
        class_names = all_classes
        label_map = {name: i for i, name in enumerate(class_names)}
        y = np.array([label_map[lbl] for lbl in y_str])

        class_counts = dict(test_df[TARGET_COL].value_counts())
        self._log(f"  Classes: {len(class_names)} -- {class_counts}")

        # Extract features -- must match M7 feature selection exactly:
        # exclude non-feature cols, demographic encoded cols, known NaN cols
        exclude = ALL_NON_FEATURE_COLS | KNOWN_NAN_COLS
        present_exclude = exclude & set(test_df.columns)
        feature_cols = [c for c in test_df.columns if c not in present_exclude]
        # Also exclude any remaining non-numeric columns
        feature_cols = [c for c in feature_cols
                        if pd.api.types.is_numeric_dtype(test_df[c])]
        # Drop all-NaN columns (same as M7)
        nan_cols = [c for c in feature_cols if test_df[c].isnull().all()]
        if nan_cols:
            feature_cols = [c for c in feature_cols if c not in nan_cols]
            self._log(f"  Dropped {len(nan_cols)} all-NaN cols: {nan_cols}")

        X = test_df[feature_cols].values.astype(np.float32)
        # Fill NaN with 0 for model inference
        X = np.nan_to_num(X, nan=0.0)
        self._log(f"  Features: {X.shape[1]}")

        # Demographics
        demo_cols = list((DEMOGRAPHIC_RAW | DEMOGRAPHIC_ENC) & set(test_df.columns))
        demographics = test_df[demo_cols].copy() if demo_cols else None

        # Add age group
        if demographics is not None and "age" in demographics.columns:
            demographics["age_group"] = pd.cut(
                demographics["age"], bins=AGE_BINS, labels=AGE_LABELS,
                include_lowest=True,
            )

        # User IDs
        user_ids = test_df["user_id"].values if "user_id" in test_df.columns else None

        return TestData(
            X=X, y=y, y_str=y_str,
            feature_names=feature_cols,
            class_names=class_names,
            n_classes=len(class_names),
            demographics=demographics,
            user_ids=user_ids,
            class_counts=class_counts,
        )

    def load_train_val_summary(self, features_csv: Path) -> Dict[str, Dict[str, int]]:
        """Load train/val class distributions for drift comparison."""
        df = pd.read_csv(features_csv, usecols=[TARGET_COL, SPLIT_COL])
        summary = {}
        for split in ("train", "val", "test"):
            subset = df[df[SPLIT_COL] == split]
            summary[split] = dict(subset[TARGET_COL].value_counts())
        return summary

    def load_m7_metadata(self, m7_dir: Path) -> M7Metadata:
        """Load training metadata from M7 output."""
        m7_dir = Path(m7_dir)
        meta_path = m7_dir / "training_metadata.json"
        if not meta_path.exists():
            # Try searching subdirectories
            candidates = list(m7_dir.rglob("training_metadata.json"))
            if candidates:
                meta_path = candidates[0]
            else:
                raise FileNotFoundError(
                    f"training_metadata.json not found in {m7_dir}")

        with open(meta_path) as f:
            raw = json.load(f)

        self._log(f"M7 metadata: best_model={raw.get('best_model', '?')}, "
                  f"n_models={raw.get('n_models_trained', '?')}")

        return M7Metadata(
            best_model=raw.get("best_model", "unknown"),
            class_names=raw.get("class_names", []),
            n_classes=raw.get("n_classes", 0),
            seed=raw.get("seed", 42),
            balance_strategy=raw.get("balance_strategy", "none"),
            primary_metric=raw.get("primary_metric", "f1_weighted"),
            class_distribution=raw.get("class_distribution", {}),
            n_models_trained=raw.get("n_models_trained", 0),
            total_time_s=raw.get("total_time_s", 0),
            raw=raw,
        )

    def load_per_model_val_metrics(self, m7_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Load validation metrics.json from each M7 model folder."""
        m7_dir = Path(m7_dir)
        val_metrics = {}

        for model_type in ("supervised", "unsupervised"):
            type_dir = m7_dir / model_type
            if not type_dir.exists():
                continue
            for model_dir in sorted(type_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                metrics_path = model_dir / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        val_metrics[model_dir.name] = json.load(f)

        self._log(f"Loaded val metrics for {len(val_metrics)} models")
        return val_metrics
