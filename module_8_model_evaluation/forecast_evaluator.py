"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  forecast_evaluator.py
=============================================================================
Evaluate the BiLSTM Forecaster on held-out test data.

The forecaster predicts emotional states 1 minute ahead using 3 minutes
of physiological history. This module reconstructs the forecast sequences
from the M6 CSV, loads the trained forecaster, and computes metrics.

Key difference from test_evaluator.py: the forecaster needs 3D sequence
input (lookback x n_features) and its labels come from FUTURE windows,
not current windows. Standard tabular evaluation cannot be used.
=============================================================================
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, confusion_matrix, classification_report,
)

from config import (
    TARGET_COL, SPLIT_COL, ALL_NON_FEATURE_COLS, KNOWN_NAN_COLS,
    DEMOGRAPHIC_ENC, PRIMARY_METRIC, RANDOM_SEED,
)

log = logging.getLogger(__name__)

# Forecast parameters (must match M7 config)
FORECAST_LOOKBACK = 6   # 6 windows = 3 minutes
FORECAST_HORIZON = 2    # 2 windows = 1 minute ahead
FORECAST_STRIDE = 1     # 1 window = 30 seconds


class ForecastEvaluator:
    """Evaluate forecaster models on held-out test data."""

    def __init__(self, verbose: bool = True, seed: int = RANDOM_SEED):
        self.verbose = verbose
        self.seed = seed

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [M8-Forecast] {msg}")

    def evaluate_all(
        self,
        forecast_models: Dict[str, Any],
        m7_dir: Path,
        features_csv: Path,
    ) -> list:
        """
        Evaluate all forecaster models.

        Parameters
        ----------
        forecast_models : {name: LoadedModel} with data_format="forecast"
        m7_dir : M7 output directory (contains forecast_config.json)
        features_csv : M6 combined features CSV

        Returns
        -------
        List of EvalResult-compatible objects for forecaster models
        """
        from test_evaluator import EvalResult

        # Build forecast sequences from CSV
        forecast_data = self._build_forecast_sequences(features_csv, m7_dir)
        if forecast_data is None:
            self._log("Could not build forecast sequences -- skipping")
            return []

        X_test = forecast_data["X_test"]
        y_test = forecast_data["y_test"]
        class_names = forecast_data["class_names"]

        self._log(f"Forecast test set: {len(y_test)} sequences, "
                  f"{len(class_names)} classes")
        self._log(f"  Lookback: {forecast_data['lookback']} windows "
                  f"({forecast_data['lookback'] * 30}s)")
        self._log(f"  Horizon:  {forecast_data['horizon']} windows "
                  f"({forecast_data['horizon'] * 30}s ahead)")

        results = []
        for name, model in forecast_models.items():
            self._log(f"Evaluating forecaster: {name}")
            result = self._evaluate_one(
                model, X_test, y_test, class_names,
                forecast_data,
            )
            results.append(result)

        return results

    def _evaluate_one(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: List[str],
        forecast_data: dict,
    ):
        """Evaluate a single forecaster model."""
        from test_evaluator import EvalResult

        result = EvalResult(
            model_name=model.name,
            model_type="supervised",
            framework=model.framework,
            val_metrics=model.val_metrics,
        )

        try:
            # Predict -- model expects (N, lookback, n_features) 3D input
            t0 = time.time()
            y_pred = model.predict(X_test)
            result.inference_time_s = time.time() - t0

            # Probabilities
            y_prob = model.predict_proba(X_test)

            result.y_pred = y_pred
            result.y_prob = y_prob

            # Compute metrics
            result.test_metrics = self._compute_metrics(
                y_test, y_pred, y_prob, class_names
            )

            # Add forecast-specific info
            result.test_metrics["forecast_lookback_s"] = forecast_data["lookback"] * 30
            result.test_metrics["forecast_horizon_s"] = forecast_data["horizon"] * 30
            result.test_metrics["forecast_description"] = (
                f"{forecast_data['lookback'] * 30}s history -> "
                f"predict {forecast_data['horizon'] * 30}s ahead"
            )
            result.test_metrics["is_forecaster"] = True

            f1 = result.test_metrics.get("f1_weighted", 0)
            acc = result.test_metrics.get("accuracy", 0)
            self._log(f"  {model.name}: F1w={f1:.4f}, Acc={acc:.4f} "
                      f"({result.inference_time_s:.3f}s)")

        except Exception as e:
            result.skipped = True
            result.skip_reason = str(e)
            self._log(f"  {model.name}: ERROR -- {e}")

        return result

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        class_names: List[str],
    ) -> Dict[str, Any]:
        """Compute full metric suite on forecast predictions."""
        metrics: Dict[str, Any] = {}

        all_labels = list(range(len(class_names)))

        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1_weighted"] = float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0))
        metrics["f1_macro"] = float(f1_score(
            y_true, y_pred, average="macro", zero_division=0))
        metrics["precision_weighted"] = float(precision_score(
            y_true, y_pred, average="weighted", zero_division=0))
        metrics["recall_weighted"] = float(recall_score(
            y_true, y_pred, average="weighted", zero_division=0))
        metrics["cohens_kappa"] = float(cohen_kappa_score(y_true, y_pred))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["confusion_labels"] = class_names

        # Per-class metrics
        per_class = {}
        report = classification_report(
            y_true, y_pred, labels=all_labels,
            target_names=class_names,
            output_dict=True, zero_division=0,
        )
        for cls_name in class_names:
            if cls_name in report:
                cls_report = report[cls_name]
                per_class[cls_name] = {
                    "precision": cls_report["precision"],
                    "recall": cls_report["recall"],
                    "f1": cls_report["f1-score"],
                    "support": int(cls_report["support"]),
                }
        metrics["per_class"] = per_class

        return metrics

    def _build_forecast_sequences(
        self,
        features_csv: Path,
        m7_dir: Path,
    ) -> Optional[dict]:
        """
        Build forecast test sequences from the M6 CSV.

        Mirrors M7 data_loader.get_forecast_data() logic but only
        returns test sequences.
        """
        features_csv = Path(features_csv)
        m7_dir = Path(m7_dir)

        # Try to load forecast config from M7
        lookback = FORECAST_LOOKBACK
        horizon = FORECAST_HORIZON
        stride = FORECAST_STRIDE

        # Check all model dirs for forecast_config.json
        for fc_json in m7_dir.rglob("forecast_config.json"):
            with open(fc_json) as f:
                fc = json.load(f)
            lookback = fc.get("lookback_windows", lookback)
            horizon = fc.get("horizon_windows", horizon)
            self._log(f"  Loaded forecast config: lookback={lookback}, "
                      f"horizon={horizon}")
            break

        # Load CSV
        df = pd.read_csv(features_csv)
        if SPLIT_COL not in df.columns or TARGET_COL not in df.columns:
            self._log("  Missing split or target column")
            return None

        # Determine feature columns (match M7/M8 logic)
        nan_cols_present = [c for c in KNOWN_NAN_COLS if c in df.columns]
        if nan_cols_present:
            df = df.drop(columns=nan_cols_present)

        exclude = ALL_NON_FEATURE_COLS | KNOWN_NAN_COLS
        feature_cols = [c for c in df.columns if c not in exclude
                        and c not in DEMOGRAPHIC_ENC
                        and pd.api.types.is_numeric_dtype(df[c])]
        # Drop all-NaN columns
        nan_cols = [c for c in feature_cols if df[c].isnull().all()]
        feature_cols = [c for c in feature_cols if c not in nan_cols]

        # Encode labels
        all_classes = sorted(df[TARGET_COL].unique().tolist())
        label_map = {name: i for i, name in enumerate(all_classes)}

        # Build sequences per user/session, only keep test sequences
        sequences = []
        labels = []

        group_cols = [c for c in ["user_id", "session_id"]
                      if c in df.columns]
        groups = df.groupby(group_cols) if group_cols else [("all", df)]

        for _, group_df in groups:
            group_df = group_df.sort_values("window_start_s").reset_index(
                drop=True)
            X_group = group_df[feature_cols].values.astype(np.float32)
            y_group = np.array([label_map[lbl]
                                for lbl in group_df[TARGET_COL].values])
            split_group = group_df[SPLIT_COL].values

            total_needed = lookback + horizon
            if len(X_group) < total_needed:
                continue

            for i in range(0, len(X_group) - total_needed + 1, stride):
                # Use split of history end
                history_end_idx = i + lookback - 1
                if split_group[history_end_idx] != "test":
                    continue

                seq = X_group[i:i + lookback]
                future_idx = i + lookback - 1 + horizon
                lbl = y_group[future_idx]

                # Fill NaN
                seq = np.nan_to_num(seq, nan=0.0)
                sequences.append(seq)
                labels.append(lbl)

        if len(sequences) == 0:
            self._log("  No test forecast sequences found")
            return None

        X_test = np.array(sequences, dtype=np.float32)
        y_test = np.array(labels, dtype=np.int64)

        self._log(f"  Built {len(y_test)} test forecast sequences")

        return {
            "X_test": X_test,
            "y_test": y_test,
            "class_names": all_classes,
            "lookback": lookback,
            "horizon": horizon,
            "feature_names": feature_cols,
        }
