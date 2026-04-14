"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  models/unsupervised.py
=============================================================================
Unsupervised clustering and anomaly detection models:
  - KMeansTrainer     : cluster discovery with elbow/silhouette selection
  - GMMTrainer        : Gaussian mixture with BIC model selection
  - IsolationForestTrainer : lightweight anomaly detection
=============================================================================
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from config import KMEANS, GMM, ISOLATION_FOREST, RANDOM_SEED
from models import BaseModelTrainer, TrainResult


# ── K-Means ──────────────────────────────────────────────────────────────

class KMeansTrainer(BaseModelTrainer):
    name = "kmeans"
    model_type = "unsupervised"
    framework = "sklearn"

    def can_run(self, data) -> tuple:
        n = data[0].shape[0] if isinstance(data, tuple) else len(data)
        if n < 10:
            return False, f"Only {n} samples -- need >= 10"
        return True, ""

    def train(self, data, output_dir: Path, **kwargs) -> TrainResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        X, y_true, feature_names = data
        ok, reason = self.can_run(data)
        if not ok:
            self._log(f"SKIP: {reason}")
            return TrainResult(
                model_name=self.name, model_type=self.model_type,
                framework=self.framework, metrics={},
                skipped=True, skip_reason=reason)

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        self._log("Training K-Means (testing multiple k) ...")
        t0 = time.time()

        best_score, best_k, best_model = -1, 0, None
        scores = {}
        inertias = {}

        for k in KMEANS["n_clusters_range"]:
            if k >= len(X):
                continue
            km = KMeans(n_clusters=k, n_init=KMEANS["n_init"],
                        max_iter=KMEANS["max_iter"],
                        random_state=self.seed)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            scores[k] = sil
            inertias[k] = km.inertia_
            if sil > best_score:
                best_score, best_k, best_model = sil, k, km

        self.model = best_model
        elapsed = time.time() - t0
        self._log(f"Best k={best_k}, silhouette={best_score:.4f} "
                   f"({elapsed:.1f}s)")

        labels = self.model.predict(X)
        model_path = output_dir / "model.pkl"
        self.save(model_path)

        from evaluator import ModelEvaluator
        ev = ModelEvaluator(kwargs.get("class_names", []), verbose=False)
        metrics = ev.evaluate_unsupervised(X, labels, y_true)
        metrics["best_k"] = best_k
        metrics["silhouette_by_k"] = scores
        metrics["inertia_by_k"] = {k: float(v)
                                    for k, v in inertias.items()}

        ev.plot_cluster_visualization(
            X, labels, output_dir / "clusters.png",
            true_labels=y_true, title=f"K-Means (k={best_k})")

        # Elbow plot
        self._plot_elbow(inertias, scores, output_dir / "elbow.png")

        return TrainResult(
            model_name=self.name, model_type=self.model_type,
            framework=self.framework, metrics=metrics,
            model_path=model_path, training_time_s=elapsed,
            hyperparams={"n_clusters": best_k},
            predictions=labels,
            model_size_kb=self.measure_model_size(model_path))

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)

    def _plot_elbow(self, inertias, scores, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from config import PLOT_DPI

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ks = sorted(inertias.keys())
        ax1.plot(ks, [inertias[k] for k in ks], "b-o", label="Inertia")
        ax1.set_xlabel("k")
        ax1.set_ylabel("Inertia", color="b")

        ax2 = ax1.twinx()
        ax2.plot(ks, [scores[k] for k in ks], "r-s", label="Silhouette")
        ax2.set_ylabel("Silhouette Score", color="r")

        fig.suptitle("K-Means: Elbow + Silhouette")
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)


# ── GMM ──────────────────────────────────────────────────────────────────

class GMMTrainer(BaseModelTrainer):
    name = "gmm"
    model_type = "unsupervised"
    framework = "sklearn"

    def can_run(self, data) -> tuple:
        n = data[0].shape[0] if isinstance(data, tuple) else len(data)
        if n < 20:
            return False, f"Only {n} samples -- need >= 20"
        return True, ""

    def train(self, data, output_dir: Path, **kwargs) -> TrainResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        X, y_true, feature_names = data
        ok, reason = self.can_run(data)
        if not ok:
            self._log(f"SKIP: {reason}")
            return TrainResult(
                model_name=self.name, model_type=self.model_type,
                framework=self.framework, metrics={},
                skipped=True, skip_reason=reason)

        from sklearn.mixture import GaussianMixture

        self._log("Training GMM (testing components x covariance) ...")
        t0 = time.time()

        best_bic, best_model, best_params = float("inf"), None, {}

        for n_comp in GMM["n_components_range"]:
            if n_comp >= len(X):
                continue
            for cov in GMM["covariance_type"]:
                gmm = GaussianMixture(
                    n_components=n_comp, covariance_type=cov,
                    random_state=self.seed, max_iter=200,
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_model = gmm
                    best_params = {"n_components": n_comp,
                                   "covariance_type": cov}

        self.model = best_model
        elapsed = time.time() - t0
        self._log(f"Best: {best_params}, BIC={best_bic:.1f} ({elapsed:.1f}s)")

        labels = self.model.predict(X)
        model_path = output_dir / "model.pkl"
        self.save(model_path)

        from evaluator import ModelEvaluator
        ev = ModelEvaluator(kwargs.get("class_names", []), verbose=False)
        metrics = ev.evaluate_unsupervised(X, labels, y_true)
        metrics["bic"] = float(best_bic)
        metrics["aic"] = float(self.model.aic(X))

        ev.plot_cluster_visualization(
            X, labels, output_dir / "clusters.png",
            true_labels=y_true,
            title=f"GMM (n={best_params['n_components']})")

        return TrainResult(
            model_name=self.name, model_type=self.model_type,
            framework=self.framework, metrics=metrics,
            model_path=model_path, training_time_s=elapsed,
            hyperparams=best_params, predictions=labels,
            model_size_kb=self.measure_model_size(model_path))

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)


# ── Isolation Forest ─────────────────────────────────────────────────────

class IsolationForestTrainer(BaseModelTrainer):
    name = "isolation_forest"
    model_type = "unsupervised"
    framework = "sklearn"

    def can_run(self, data) -> tuple:
        n = data[0].shape[0] if isinstance(data, tuple) else len(data)
        if n < 20:
            return False, f"Only {n} samples -- need >= 20"
        return True, ""

    def train(self, data, output_dir: Path, **kwargs) -> TrainResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        X, y_true, feature_names = data
        ok, reason = self.can_run(data)
        if not ok:
            self._log(f"SKIP: {reason}")
            return TrainResult(
                model_name=self.name, model_type=self.model_type,
                framework=self.framework, metrics={},
                skipped=True, skip_reason=reason)

        from sklearn.ensemble import IsolationForest
        import pandas as pd

        self._log("Training Isolation Forest ...")
        t0 = time.time()

        # Try multiple contamination levels, pick best
        best_model, best_contam = None, None
        best_score = -float("inf")

        for contam in ISOLATION_FOREST["contamination_range"]:
            iso = IsolationForest(
                n_estimators=ISOLATION_FOREST["n_estimators"],
                contamination=contam,
                random_state=self.seed, n_jobs=-1,
            )
            iso.fit(X)
            # Use average anomaly score as quality measure
            scores = iso.decision_function(X)
            # Higher average score for inliers is better separation
            score = np.std(scores)  # More spread = better separation
            if score > best_score:
                best_score = score
                best_model = iso
                best_contam = contam

        self.model = best_model
        elapsed = time.time() - t0
        self._log(f"Best contamination={best_contam} ({elapsed:.1f}s)")

        # Get anomaly scores and predictions
        raw_scores = self.model.decision_function(X)
        raw_preds = self.model.predict(X)
        # Convert: -1 = anomaly -> 1, 1 = normal -> 0
        anomaly_preds = (raw_preds == -1).astype(int)

        model_path = output_dir / "model.pkl"
        self.save(model_path)

        from evaluator import ModelEvaluator
        ev = ModelEvaluator(kwargs.get("class_names", []), verbose=False)
        metrics = ev.evaluate_anomaly(raw_scores, anomaly_preds, y_true)
        metrics["contamination"] = best_contam

        # Save anomaly scores
        scores_df = pd.DataFrame({
            "anomaly_score": raw_scores,
            "prediction": anomaly_preds,
        })
        if y_true is not None:
            scores_df["true_label"] = y_true
        scores_df.to_csv(output_dir / "anomaly_scores.csv", index=False)

        return TrainResult(
            model_name=self.name, model_type=self.model_type,
            framework=self.framework, metrics=metrics,
            model_path=model_path, training_time_s=elapsed,
            hyperparams={"contamination": best_contam},
            predictions=anomaly_preds,
            model_size_kb=self.measure_model_size(model_path))

    def predict(self, X) -> np.ndarray:
        raw = self.model.predict(X)
        return (raw == -1).astype(int)

    def save(self, path: Path):
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)
