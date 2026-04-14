"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  evaluator.py
=============================================================================
Unified evaluation for all model types: supervised metrics, unsupervised
metrics, confusion matrices, ROC curves, per-demographic breakdown,
and model comparison reporting.
=============================================================================
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, confusion_matrix, classification_report,
    roc_auc_score, silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score,
    normalized_mutual_info_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    HIGH_PRIORITY_STATES, PLOT_DPI, PLOT_FIGSIZE,
    PRIMARY_METRIC,
)


class ModelEvaluator:
    """Compute and visualize metrics for supervised and unsupervised models."""

    def __init__(self, class_names: List[str], verbose: bool = True):
        self.class_names = class_names
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Evaluator] {msg}")

    # ── Supervised metrics ────────────────────────────────────────────────

    def evaluate_supervised(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict:
        """Full supervised metric suite. Returns metric dict."""
        present_labels = sorted(set(y_true) | set(y_pred))
        present_names = [self.class_names[i] for i in present_labels
                         if i < len(self.class_names)]

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(
                y_true, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(
                y_true, y_pred, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(
                y_true, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(
                y_true, y_pred, average="weighted", zero_division=0)),
            "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)),
        }

        # Per-class metrics
        per_class = {}
        for i, name in enumerate(self.class_names):
            if i not in present_labels:
                continue
            y_bin_true = (y_true == i).astype(int)
            y_bin_pred = (y_pred == i).astype(int)
            tp = ((y_bin_true == 1) & (y_bin_pred == 1)).sum()
            fn = ((y_bin_true == 1) & (y_bin_pred == 0)).sum()
            fp = ((y_bin_true == 0) & (y_bin_pred == 1)).sum()
            tn = ((y_bin_true == 0) & (y_bin_pred == 0)).sum()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1_cls = f1_score(y_bin_true, y_bin_pred, zero_division=0)
            per_class[name] = {
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "f1": float(f1_cls),
                "support": int((y_true == i).sum()),
            }

        metrics["per_class"] = per_class

        # False negative rate for high-priority states
        fnr_priority = {}
        for state in HIGH_PRIORITY_STATES:
            if state in self.class_names:
                idx = self.class_names.index(state)
                if idx in present_labels and (y_true == idx).sum() > 0:
                    fn = ((y_true == idx) & (y_pred != idx)).sum()
                    total = (y_true == idx).sum()
                    fnr_priority[state] = float(fn / total)
        metrics["fnr_high_priority"] = fnr_priority

        # AUC-ROC per class (if probabilities available)
        if y_prob is not None and y_prob.ndim == 2:
            auc_per_class = {}
            for i, name in enumerate(self.class_names):
                if i >= y_prob.shape[1]:
                    continue
                if (y_true == i).sum() > 0 and (y_true != i).sum() > 0:
                    try:
                        auc = roc_auc_score(
                            (y_true == i).astype(int), y_prob[:, i])
                        auc_per_class[name] = float(auc)
                    except ValueError:
                        pass
            metrics["auc_roc_per_class"] = auc_per_class
            if auc_per_class:
                metrics["auc_roc_macro"] = float(
                    np.mean(list(auc_per_class.values())))

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(
            y_true, y_pred, labels=present_labels).tolist()
        metrics["confusion_labels"] = present_names

        return metrics

    # ── Unsupervised metrics ──────────────────────────────────────────────

    def evaluate_unsupervised(
        self,
        X: np.ndarray,
        cluster_labels: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
    ) -> Dict:
        """Evaluate clustering quality."""
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters[unique_clusters >= 0])

        metrics = {"n_clusters": n_clusters}

        if n_clusters >= 2 and n_clusters < len(X):
            # Filter out noise points (label=-1) for silhouette
            valid = cluster_labels >= 0
            if valid.sum() >= 2:
                metrics["silhouette"] = float(silhouette_score(
                    X[valid], cluster_labels[valid]))
                metrics["calinski_harabasz"] = float(
                    calinski_harabasz_score(X[valid], cluster_labels[valid]))
                metrics["davies_bouldin"] = float(
                    davies_bouldin_score(X[valid], cluster_labels[valid]))

        if true_labels is not None and n_clusters >= 2:
            valid = cluster_labels >= 0
            if valid.sum() >= 2:
                metrics["adjusted_rand"] = float(
                    adjusted_rand_score(true_labels[valid],
                                        cluster_labels[valid]))
                metrics["nmi"] = float(
                    normalized_mutual_info_score(
                        true_labels[valid], cluster_labels[valid]))
                metrics["cluster_purity"] = float(
                    self._cluster_purity(true_labels[valid],
                                          cluster_labels[valid]))

        n_noise = (cluster_labels < 0).sum()
        metrics["n_noise_points"] = int(n_noise)

        return metrics

    # ── Anomaly detection metrics ─────────────────────────────────────────

    def evaluate_anomaly(
        self,
        scores: np.ndarray,
        predictions: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
    ) -> Dict:
        """Evaluate anomaly detection (IsolationForest, Autoencoder)."""
        metrics = {
            "n_anomalies": int((predictions == 1).sum()),
            "n_normal": int((predictions == 0).sum()),
            "anomaly_rate": float((predictions == 1).mean()),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
        }

        if true_labels is not None:
            # Non-baseline = anomaly for evaluation purposes
            true_anomaly = (true_labels != 0).astype(int)
            if true_anomaly.sum() > 0:
                metrics["anomaly_f1"] = float(f1_score(
                    true_anomaly, predictions, zero_division=0))
                metrics["anomaly_precision"] = float(precision_score(
                    true_anomaly, predictions, zero_division=0))
                metrics["anomaly_recall"] = float(recall_score(
                    true_anomaly, predictions, zero_division=0))

        return metrics

    # ── Plotting ──────────────────────────────────────────────────────────

    def plot_confusion_matrix(
        self, y_true, y_pred, path: Path, title: str = "Confusion Matrix",
    ):
        """Save confusion matrix heatmap."""
        present = sorted(set(y_true) | set(y_pred))
        names = [self.class_names[i] for i in present
                 if i < len(self.class_names)]
        cm = confusion_matrix(y_true, y_pred, labels=present)

        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)

    def plot_roc_curves(
        self, y_true, y_prob, path: Path, title: str = "ROC Curves",
    ):
        """Save one-vs-rest ROC curves."""
        if y_prob is None or y_prob.ndim != 2:
            return
        from sklearn.metrics import roc_curve

        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        for i, name in enumerate(self.class_names):
            if i >= y_prob.shape[1]:
                continue
            if (y_true == i).sum() == 0 or (y_true != i).sum() == 0:
                continue
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
            try:
                auc = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
            except ValueError:
                continue
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)

    def plot_training_curves(
        self, history: Dict[str, list], path: Path,
        title: str = "Training Curves",
    ):
        """Save loss/F1 training curves for deep models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if "train_loss" in history:
            axes[0].plot(history["train_loss"], label="Train")
            if "val_loss" in history:
                axes[0].plot(history["val_loss"], label="Val")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Loss")
            axes[0].legend()

        if "train_f1" in history:
            axes[1].plot(history["train_f1"], label="Train")
            if "val_f1" in history:
                axes[1].plot(history["val_f1"], label="Val")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("F1 Weighted")
            axes[1].set_title("F1 Score")
            axes[1].legend()

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)

    def plot_cluster_visualization(
        self, X: np.ndarray, labels: np.ndarray, path: Path,
        true_labels: Optional[np.ndarray] = None,
        title: str = "Cluster Visualization",
    ):
        """PCA 2D projection of clusters."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        X2d = pca.fit_transform(X)

        n_plots = 2 if true_labels is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]

        scatter = axes[0].scatter(X2d[:, 0], X2d[:, 1], c=labels,
                                  cmap="tab10", s=20, alpha=0.7)
        axes[0].set_title("Predicted Clusters")
        fig.colorbar(scatter, ax=axes[0])

        if true_labels is not None:
            scatter2 = axes[1].scatter(X2d[:, 0], X2d[:, 1], c=true_labels,
                                       cmap="tab10", s=20, alpha=0.7)
            axes[1].set_title("True Labels")
            fig.colorbar(scatter2, ax=axes[1])

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)

    def plot_feature_importance(
        self, importance: np.ndarray, feature_names: List[str], path: Path,
        title: str = "Feature Importance", top_n: int = 20,
    ):
        """Horizontal bar chart of top-N features."""
        idx = np.argsort(importance)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax.barh(range(len(idx)), importance[idx][::-1])
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i] for i in idx][::-1], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)

    # ── Demographic breakdown ─────────────────────────────────────────────

    def demographic_breakdown(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        demographics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Per-demographic-group metrics."""
        rows = []
        demo_groups = {
            "autism_severity": ["autism_severity", "autism_severity_enc"],
            "verbal_status": ["verbal_status", "verbal_status_enc"],
        }
        # Add age bins if age column present
        if "age" in demographics.columns:
            demographics = demographics.copy()
            demographics["age_group"] = pd.cut(
                demographics["age"], bins=[4, 8, 12, 16],
                labels=["5-8", "9-12", "13-15"],
            )
            demo_groups["age_group"] = ["age_group"]

        for group_name, possible_cols in demo_groups.items():
            col = next((c for c in possible_cols
                        if c in demographics.columns), None)
            if col is None:
                continue
            for val in sorted(demographics[col].dropna().unique()):
                mask = demographics[col] == val
                if mask.sum() < 2:
                    continue
                yt, yp = y_true[mask], y_pred[mask]
                rows.append({
                    "group": group_name,
                    "value": str(val),
                    "n_samples": int(mask.sum()),
                    "accuracy": float(accuracy_score(yt, yp)),
                    "f1_weighted": float(f1_score(
                        yt, yp, average="weighted", zero_division=0)),
                    "f1_macro": float(f1_score(
                        yt, yp, average="macro", zero_division=0)),
                })

        return pd.DataFrame(rows)

    # ── Model comparison ──────────────────────────────────────────────────

    def compare_models(
        self, results: list, output_dir: Path,
    ) -> pd.DataFrame:
        """
        Build comparison table from list of TrainResult objects.
        Sorted by primary metric (f1_weighted).
        """
        rows = []
        for r in results:
            if r.skipped:
                rows.append({
                    "model": r.model_name,
                    "type": r.model_type,
                    "framework": r.framework,
                    "status": f"SKIPPED: {r.skip_reason}",
                    PRIMARY_METRIC: None,
                })
                continue

            row = {
                "model": r.model_name,
                "type": r.model_type,
                "framework": r.framework,
                "status": "OK",
                "training_time_s": r.training_time_s,
                "model_size_kb": r.model_size_kb,
                "inference_latency_ms": r.inference_latency_ms,
            }
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)):
                    row[k] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        if PRIMARY_METRIC in df.columns:
            df = df.sort_values(PRIMARY_METRIC, ascending=False,
                                na_position="last")

        # Save CSV
        csv_path = output_dir / "model_comparison.csv"
        df.to_csv(csv_path, index=False)
        self._log(f"Saved {csv_path.name}")

        # Save deployment recommendation
        supervised_df = df[
            (df["type"] == "supervised") & (df["status"] == "OK")]
        if len(supervised_df) > 0:
            best = supervised_df.iloc[0]
            rec = {
                "best_model": best["model"],
                "primary_metric": PRIMARY_METRIC,
                "primary_score": best.get(PRIMARY_METRIC),
                "model_size_kb": best.get("model_size_kb"),
                "inference_latency_ms": best.get("inference_latency_ms"),
                "rationale": (
                    f"Selected {best['model']} as best global model based on "
                    f"{PRIMARY_METRIC}={best.get(PRIMARY_METRIC, 'N/A'):.4f}. "
                    f"Framework: {best['framework']}."
                ),
            }
            rec_path = output_dir / "deployment_recommendation.json"
            with open(rec_path, "w") as f:
                json.dump(rec, f, indent=2, default=str)
            self._log(f"Best model: {best['model']} "
                       f"({PRIMARY_METRIC}={best.get(PRIMARY_METRIC):.4f})")

        return df

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _cluster_purity(true_labels, cluster_labels):
        """Compute cluster purity."""
        correct = 0
        for cluster in np.unique(cluster_labels):
            mask = cluster_labels == cluster
            if mask.sum() == 0:
                continue
            most_common = np.bincount(true_labels[mask]).argmax()
            correct += (true_labels[mask] == most_common).sum()
        return correct / len(true_labels)
