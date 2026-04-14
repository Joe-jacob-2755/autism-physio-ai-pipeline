"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  trainer.py
=============================================================================
Master orchestrator — trains all models (supervised + unsupervised),
compares results, selects best global model, then trains user-dependent
models using the winning architecture.
=============================================================================
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    MODULE_VERSION, MODULE_LABEL, OUTPUT_ROOT, RANDOM_SEED,
    BALANCE_STRATEGY, PRIMARY_METRIC,
)
from data_loader import DataLoader, DataBundle, TabularData, SequenceData, TFTData, TPPData, ForecastData
from class_balancer import ClassBalancer
from evaluator import ModelEvaluator
from models import TrainResult, get_supervised_models, get_unsupervised_models


class ModelTrainer:
    """
    Orchestrate training of all model families on global data,
    compare performance, and optionally train user-dependent models.
    """

    def __init__(self, source: str | Path, seed: int = RANDOM_SEED,
                 verbose: bool = True, output_dir: Optional[Path] = None):
        self.source = Path(source)
        self.seed = seed
        self.verbose = verbose
        self._output_dir = output_dir  # if None, auto-numbered

        self.data_loader = DataLoader(source, seed=seed, verbose=verbose)
        self.bundle: Optional[DataBundle] = None
        self.results: List[TrainResult] = []
        self.best_model_name: Optional[str] = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Trainer] {msg}")

    # ── Output directory ─────────────────────────────────────────────────

    def _make_output_dir(self) -> Path:
        """Create auto-numbered output directory."""
        if self._output_dir:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            return self._output_dir

        import re
        root = Path(OUTPUT_ROOT)
        root.mkdir(exist_ok=True)
        prefix = f"{MODULE_LABEL}_v{MODULE_VERSION}_run_"
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        existing = [int(m.group(1)) for d in root.iterdir()
                    if d.is_dir() and (m := pattern.match(d.name))]
        next_num = (max(existing) + 1) if existing else 1
        out = root / f"{prefix}{next_num:03d}"
        out.mkdir(parents=True)
        return out

    # ── Main entry point ─────────────────────────────────────────────────

    def run(self, skip_temporal: bool = False,
            skip_unsupervised: bool = False,
            user_dependent: bool = False,
            per_user_dir: Optional[Path] = None) -> Path:
        """
        Full training pipeline.

        Parameters
        ----------
        skip_temporal     : skip BiLSTM, CNN, TPP, TFT
        skip_unsupervised : skip KMeans, GMM, IsolationForest, Autoencoder
        user_dependent    : train user-dependent models after global selection
        per_user_dir      : directory with per-user CSVs for user-dependent models

        Returns
        -------
        output_dir : Path to output directory
        """
        t_start = time.time()
        output_dir = self._make_output_dir()
        self._log(f"Output: {output_dir}")

        # ── 1. Load data ────────────────────────────────────────────────
        self._log("=" * 60)
        self._log("PHASE 1: Loading and preparing data")
        self._log("=" * 60)
        self.bundle = self.data_loader.load()

        # Get data formats
        tabular = self.data_loader.get_tabular_data(self.bundle)

        seq_data = None
        tft_data = None
        tpp_data = None
        forecast_data = None
        if not skip_temporal:
            seq_data = self.data_loader.get_sequence_data(self.bundle)
            tft_raw = self.data_loader.get_tft_data(self.bundle)
            # Adapt TFTData field names to what TFTTrainer expects
            tft_data = _adapt_tft_data(tft_raw)
            tpp_data = self.data_loader.get_tpp_data(self.bundle)
            # Adapt TPPData field names
            tpp_data = _adapt_tpp_data(tpp_data)
            # Forecast data: 3-min lookback -> 1-min ahead
            forecast_data = self.data_loader.get_forecast_data(self.bundle)

        unsup_data = None
        if not skip_unsupervised:
            unsup_data = self.data_loader.get_unsupervised_data(self.bundle)

        # ── 2. Balance training data ────────────────────────────────────
        self._log("\n" + "=" * 60)
        self._log("PHASE 2: Balancing training data")
        self._log("=" * 60)
        balancer = ClassBalancer(
            strategy=BALANCE_STRATEGY, seed=self.seed, verbose=self.verbose)
        X_bal, y_bal = balancer.balance_tabular(
            tabular.X_train, tabular.y_train)
        # Create balanced tabular data
        tabular_balanced = TabularData(
            X_train=X_bal, X_val=tabular.X_val, X_test=tabular.X_test,
            y_train=y_bal, y_val=tabular.y_val, y_test=tabular.y_test,
            feature_names=tabular.feature_names,
            class_names=tabular.class_names,
            n_classes=tabular.n_classes,
            demographics_train=tabular.demographics_train,
            demographics_val=tabular.demographics_val,
            demographics_test=tabular.demographics_test,
        )

        # ── 3. Train supervised models ──────────────────────────────────
        self._log("\n" + "=" * 60)
        self._log("PHASE 3: Training supervised models")
        self._log("=" * 60)
        supervised = get_supervised_models()
        tabular_models = {"decision_tree", "random_forest", "svm", "xgboost"}
        temporal_models = {"bilstm", "cnn_1d"}
        special_models = {"tpp", "tft"}
        forecast_models = {"bilstm_forecaster"}

        for name, TrainerCls in supervised.items():
            if skip_temporal and name in (temporal_models | special_models | forecast_models):
                self._log(f"\nSkipping {name} (temporal models disabled)")
                continue

            self._log(f"\n--- {name.upper()} ---")
            model_dir = output_dir / "supervised" / name
            trainer = TrainerCls(seed=self.seed, verbose=self.verbose)

            try:
                if name in tabular_models:
                    result = trainer.train(
                        tabular_balanced, model_dir)
                elif name in temporal_models:
                    if seq_data is None or len(seq_data.X_train) == 0:
                        self._log(f"No sequence data for {name}")
                        result = TrainResult(
                            model_name=name, model_type="supervised",
                            framework="pytorch", metrics={},
                            skipped=True, skip_reason="No sequence data")
                    else:
                        result = trainer.train(seq_data, model_dir)
                elif name == "tpp":
                    if tpp_data is None:
                        result = TrainResult(
                            model_name=name, model_type="supervised",
                            framework="pytorch", metrics={},
                            skipped=True, skip_reason="No TPP data")
                    else:
                        result = trainer.train(tpp_data, model_dir)
                elif name == "tft":
                    if tft_data is None:
                        result = TrainResult(
                            model_name=name, model_type="supervised",
                            framework="pytorch", metrics={},
                            skipped=True, skip_reason="No TFT data")
                    else:
                        result = trainer.train(tft_data, model_dir)
                elif name in forecast_models:
                    if forecast_data is None or len(forecast_data.y_train) == 0:
                        result = TrainResult(
                            model_name=name, model_type="supervised",
                            framework="pytorch", metrics={},
                            skipped=True, skip_reason="No forecast data")
                    else:
                        result = trainer.train(forecast_data, model_dir)
                else:
                    result = trainer.train(
                        tabular_balanced, model_dir)
            except Exception as e:
                self._log(f"ERROR training {name}: {e}")
                result = TrainResult(
                    model_name=name, model_type="supervised",
                    framework=TrainerCls.framework, metrics={},
                    skipped=True, skip_reason=f"Error: {e}")

            self.results.append(result)
            self._log_result(result)

        # ── 4. Train unsupervised models ────────────────────────────────
        if not skip_unsupervised and unsup_data is not None:
            self._log("\n" + "=" * 60)
            self._log("PHASE 4: Training unsupervised models")
            self._log("=" * 60)
            unsupervised = get_unsupervised_models()
            for name, TrainerCls in unsupervised.items():
                self._log(f"\n--- {name.upper()} ---")
                model_dir = output_dir / "unsupervised" / name
                trainer = TrainerCls(seed=self.seed, verbose=self.verbose)
                try:
                    result = trainer.train(
                        unsup_data, model_dir,
                        class_names=self.bundle.class_names)
                except Exception as e:
                    self._log(f"ERROR training {name}: {e}")
                    result = TrainResult(
                        model_name=name, model_type="unsupervised",
                        framework=TrainerCls.framework, metrics={},
                        skipped=True, skip_reason=f"Error: {e}")
                self.results.append(result)
                self._log_result(result)

        # ── 5. Compare models ───────────────────────────────────────────
        self._log("\n" + "=" * 60)
        self._log("PHASE 5: Model comparison and selection")
        self._log("=" * 60)
        comparison_dir = output_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        ev = ModelEvaluator(self.bundle.class_names, verbose=self.verbose)
        comparison_df = ev.compare_models(self.results, comparison_dir)

        # Demographic breakdown for best supervised model
        best_result = self._get_best_supervised()
        if best_result and tabular.demographics_test is not None:
            self.best_model_name = best_result.model_name
            # Use tabular predictions if available and sizes match;
            # temporal models have different test sizes (sequences)
            y_pred_demo = (
                best_result.predictions
                if best_result.predictions is not None
                and len(best_result.predictions) == len(tabular.y_test)
                else np.zeros(len(tabular.y_test), dtype=int)
            )
            demo_df = ev.demographic_breakdown(
                tabular.y_test, y_pred_demo,
                tabular.demographics_test)
            if len(demo_df) > 0:
                demo_df.to_csv(
                    comparison_dir / "demographic_breakdown.csv",
                    index=False)
                self._log(f"Demographic breakdown saved "
                           f"({len(demo_df)} groups)")

        # ── 6. Export metrics ───────────────────────────────────────────
        self._save_all_metrics(output_dir)

        # ── 7. User-dependent models ────────────────────────────────────
        if user_dependent and per_user_dir:
            self._log("\n" + "=" * 60)
            self._log("PHASE 6: Training user-dependent models")
            self._log("=" * 60)
            self._train_user_dependent(
                per_user_dir, output_dir / "user_dependent")

        # ── Done ────────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        self._log(f"\n{'=' * 60}")
        self._log(f"TRAINING COMPLETE in {elapsed:.1f}s")
        self._log(f"Results: {output_dir}")
        if self.best_model_name:
            self._log(f"Best model: {self.best_model_name}")
        self._log(f"{'=' * 60}")

        # Save metadata
        self._save_metadata(output_dir, elapsed)

        return output_dir

    # ── User-dependent training ──────────────────────────────────────────

    def _train_user_dependent(self, per_user_dir: Path, output_dir: Path):
        """Train the best global architecture on per-user data."""
        best = self._get_best_supervised()
        if best is None:
            self._log("No best model found — skipping user-dependent")
            return

        model_name = best.model_name
        supervised = get_supervised_models()
        if model_name not in supervised:
            self._log(f"Best model '{model_name}' not in registry")
            return

        TrainerCls = supervised[model_name]
        user_csvs = sorted(per_user_dir.glob("**/combined_features*.csv"))
        if not user_csvs:
            # Try per_user subdirectories
            user_dirs = [d for d in per_user_dir.iterdir() if d.is_dir()]
            for ud in user_dirs:
                csvs = list(ud.glob("combined_features*.csv"))
                user_csvs.extend(csvs)

        self._log(f"Found {len(user_csvs)} user CSV(s) for {model_name}")

        for csv_path in user_csvs:
            user_id = csv_path.parent.name
            self._log(f"\n  Training {model_name} for user: {user_id}")
            user_out = output_dir / user_id
            user_out.mkdir(parents=True, exist_ok=True)

            try:
                loader = DataLoader(csv_path, seed=self.seed,
                                    verbose=False)
                bundle = loader.load()

                # Determine correct data format
                tabular_names = {"decision_tree", "random_forest",
                                 "svm", "xgboost"}
                if model_name in tabular_names:
                    data = loader.get_tabular_data(bundle)
                    # Balance
                    bal = ClassBalancer(
                        strategy=BALANCE_STRATEGY, seed=self.seed,
                        verbose=False)
                    X_b, y_b = bal.balance_tabular(
                        data.X_train, data.y_train)
                    data = TabularData(
                        X_train=X_b, X_val=data.X_val, X_test=data.X_test,
                        y_train=y_b, y_val=data.y_val, y_test=data.y_test,
                        feature_names=data.feature_names,
                        class_names=data.class_names,
                        n_classes=data.n_classes)
                else:
                    data = loader.get_sequence_data(bundle)

                trainer = TrainerCls(seed=self.seed, verbose=False)
                result = trainer.train(data, user_out)

                if not result.skipped:
                    self._log(f"    {model_name} user={user_id}: "
                               f"f1={result.metrics.get('f1_weighted', 0):.4f}")
                else:
                    self._log(f"    SKIPPED: {result.skip_reason}")

                # Save per-user metrics
                metrics_path = user_out / "metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(result.metrics, f, indent=2, default=str)

            except Exception as e:
                self._log(f"    ERROR for user {user_id}: {e}")

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_best_supervised(self) -> Optional[TrainResult]:
        """Return the best non-skipped supervised result."""
        supervised = [r for r in self.results
                      if r.model_type == "supervised" and not r.skipped
                      and PRIMARY_METRIC in r.metrics]
        if not supervised:
            return None
        return max(supervised, key=lambda r: r.metrics[PRIMARY_METRIC])

    def _log_result(self, result: TrainResult):
        if result.skipped:
            self._log(f"  -> SKIPPED: {result.skip_reason}")
        else:
            f1 = result.metrics.get("f1_weighted", "N/A")
            t = result.training_time_s
            self._log(f"  -> f1_weighted={f1:.4f} ({t:.1f}s)"
                       if isinstance(f1, float)
                       else f"  -> {f1} ({t:.1f}s)")

    def _save_all_metrics(self, output_dir: Path):
        """Save per-model metrics.json files."""
        for r in self.results:
            if r.skipped:
                continue
            model_dir = output_dir / r.model_type / r.model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = model_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(r.metrics, f, indent=2, default=str)

    def _save_metadata(self, output_dir: Path, elapsed: float):
        """Save training run metadata."""
        # Class distribution from data bundle
        class_counts = {}
        split_sizes = {}
        n_classes = 0
        class_names = []
        if self.bundle:
            class_counts = {k: int(v) for k, v in
                           self.bundle.class_counts.items()}
            split_sizes = {k: int(v) for k, v in
                          self.bundle.split_counts.items()}
            n_classes = self.bundle.n_classes
            class_names = self.bundle.class_names

        meta = {
            "module": MODULE_LABEL,
            "version": MODULE_VERSION,
            "source": str(self.source),
            "seed": self.seed,
            "balance_strategy": BALANCE_STRATEGY,
            "primary_metric": PRIMARY_METRIC,
            "best_model": self.best_model_name,
            "n_classes": n_classes,
            "class_names": class_names,
            "class_distribution": class_counts,
            "split_sizes": split_sizes,
            "n_models_trained": sum(
                1 for r in self.results if not r.skipped),
            "n_models_skipped": sum(
                1 for r in self.results if r.skipped),
            "total_time_s": round(elapsed, 1),
            "results_summary": [
                {
                    "model": r.model_name,
                    "type": r.model_type,
                    "skipped": r.skipped,
                    "f1_weighted": r.metrics.get("f1_weighted"),
                    "training_time_s": round(r.training_time_s, 1),
                    "hyperparams": r.hyperparams,
                }
                for r in self.results
            ],
        }
        with open(output_dir / "training_metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)


# ── Data adapters ────────────────────────────────────────────────────────

class _TFTDataAdapter:
    """Adapt TFTData field names to what TFTTrainer expects."""

    def __init__(self, tft_data: TFTData):
        self.X_static_train = tft_data.static_train
        self.X_static_val = tft_data.static_val
        self.X_static_test = tft_data.static_test
        self.X_temporal_train = tft_data.temporal_train
        self.X_temporal_val = tft_data.temporal_val
        self.X_temporal_test = tft_data.temporal_test
        self.y_train = tft_data.y_train
        self.y_val = tft_data.y_val
        self.y_test = tft_data.y_test
        self.class_names = tft_data.class_names
        self.n_classes = tft_data.n_classes
        self.seq_len = tft_data.seq_len
        self.static_feature_names = tft_data.static_feature_names
        self.temporal_feature_names = tft_data.temporal_feature_names


def _adapt_tft_data(tft_data: TFTData):
    return _TFTDataAdapter(tft_data)


class _TPPDataAdapter:
    """Adapt TPPData field names to what TPPTrainer expects."""

    def __init__(self, tpp_data: TPPData):
        self.event_times = tpp_data.inter_event_times  # TPP uses inter-event
        self.event_types = tpp_data.event_types
        self.n_types = tpp_data.n_event_types
        self.event_type_names = tpp_data.event_type_names
        self.n_sequences = tpp_data.n_sequences
        self.sufficient = tpp_data.sufficient


def _adapt_tpp_data(tpp_data: TPPData):
    return _TPPDataAdapter(tpp_data)
