"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  models/svm.py
=============================================================================
SVM with RBF kernel -- strong in small-sample, high-dimensional settings.
Compact model, suitable for mobile deployment.
=============================================================================
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import SVM_PARAM_GRID, CV_FOLDS, CV_SCORING, RANDOM_SEED
from models import BaseModelTrainer, TrainResult


class SVMTrainer(BaseModelTrainer):
    name = "svm"
    model_type = "supervised"
    framework = "sklearn"

    def __init__(self, seed: int = RANDOM_SEED, verbose: bool = True):
        super().__init__(seed, verbose)
        self._pipeline = None

    def can_run(self, data) -> tuple:
        if len(np.unique(data.y_train)) < 2:
            return False, "Only 1 class in training set"
        return True, ""

    def train(self, data, output_dir: Path, **kwargs) -> TrainResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        ok, reason = self.can_run(data)
        if not ok:
            self._log(f"SKIP: {reason}")
            return TrainResult(
                model_name=self.name, model_type=self.model_type,
                framework=self.framework, metrics={},
                skipped=True, skip_reason=reason)

        self._log("Training SVM (RBF) with GridSearchCV ...")
        t0 = time.time()

        n_folds = min(CV_FOLDS, min(np.bincount(data.y_train)))
        n_folds = max(2, n_folds)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=self.seed)

        # SVM benefits from feature scaling -- wrap in pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=self.seed)),
        ])
        # Prefix param grid keys with "svm__"
        param_grid = {f"svm__{k}": v for k, v in SVM_PARAM_GRID.items()}

        grid = GridSearchCV(
            pipe, param_grid, cv=cv, scoring=CV_SCORING,
            n_jobs=-1, refit=True, error_score=0.0,
        )
        grid.fit(data.X_train, data.y_train)
        self._pipeline = grid.best_estimator_
        self.model = self._pipeline
        elapsed = time.time() - t0

        self._log(f"Best params: {grid.best_params_}")
        self._log(f"CV score: {grid.best_score_:.4f} ({elapsed:.1f}s)")

        y_pred = self.predict(data.X_test)
        y_prob = self.predict_proba(data.X_test)

        model_path = output_dir / "model.pkl"
        self.save(model_path)

        latency = self.measure_inference_latency(data.X_test)
        size_kb = self.measure_model_size(model_path)

        from evaluator import ModelEvaluator
        ev = ModelEvaluator(data.class_names, verbose=False)
        metrics = ev.evaluate_supervised(data.y_test, y_pred, y_prob)
        ev.plot_confusion_matrix(
            data.y_test, y_pred,
            output_dir / "confusion_matrix.png",
            title="SVM (RBF) - Confusion Matrix")
        if y_prob is not None:
            ev.plot_roc_curves(
                data.y_test, y_prob,
                output_dir / "roc_curves.png",
                title="SVM (RBF) - ROC Curves")

        return TrainResult(
            model_name=self.name, model_type=self.model_type,
            framework=self.framework, metrics=metrics,
            model_path=model_path, training_time_s=elapsed,
            hyperparams=grid.best_params_, predictions=y_pred,
            probabilities=y_prob, model_size_kb=size_kb,
            inference_latency_ms=latency)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)
        self._pipeline = self.model
