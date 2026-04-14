"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  models/__init__.py
=============================================================================
BaseModelTrainer ABC, TrainResult dataclass, and model registry.
=============================================================================
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Result dataclass ─────────────────────────────────────────────────────

@dataclass
class TrainResult:
    """Standard result from any model trainer."""
    model_name: str
    model_type: str            # "supervised" | "unsupervised"
    framework: str             # "sklearn" | "pytorch"
    metrics: Dict[str, Any]
    model_path: Optional[Path] = None
    training_time_s: float = 0.0
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, list] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    model_size_kb: float = 0.0
    inference_latency_ms: float = 0.0
    skipped: bool = False
    skip_reason: str = ""


# ── Base trainer ABC ─────────────────────────────────────────────────────

class BaseModelTrainer(ABC):
    """Abstract base class for all model trainers in Module 7."""

    name: str = "base"
    model_type: str = "supervised"   # "supervised" | "unsupervised"
    framework: str = "sklearn"       # "sklearn" | "pytorch"

    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        self.model = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [{self.name}] {msg}")

    @abstractmethod
    def train(self, data, output_dir: Path, **kwargs) -> TrainResult:
        """Train the model. Return TrainResult."""
        ...

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        ...

    def predict_proba(self, X) -> Optional[np.ndarray]:
        """Predict class probabilities. Override if supported."""
        return None

    @abstractmethod
    def save(self, path: Path):
        """Save model to disk."""
        ...

    @abstractmethod
    def load(self, path: Path):
        """Load model from disk."""
        ...

    def can_run(self, data) -> Tuple[bool, str]:
        """
        Check if there is enough data to train this model.
        Returns (can_run: bool, reason: str).
        """
        return True, ""

    def measure_inference_latency(self, X_sample: np.ndarray,
                                  n_runs: int = 100) -> float:
        """Measure average inference latency in milliseconds."""
        if self.model is None:
            return 0.0
        # Warm up
        try:
            self.predict(X_sample[:1])
        except Exception:
            return 0.0
        t0 = time.perf_counter()
        for _ in range(n_runs):
            self.predict(X_sample[:1])
        elapsed = (time.perf_counter() - t0) / n_runs * 1000
        return round(elapsed, 3)

    def measure_model_size(self, path: Path) -> float:
        """Return model file size in KB."""
        if path and path.exists():
            return round(path.stat().st_size / 1024, 1)
        return 0.0


# ── Model registry ───────────────────────────────────────────────────────

def get_supervised_models():
    """Import and return supervised model trainers (lazy import)."""
    from .decision_tree import DecisionTreeTrainer
    from .random_forest import RandomForestTrainer
    from .svm import SVMTrainer
    from .xgboost_model import XGBoostTrainer
    from .bilstm import BiLSTMTrainer
    from .cnn_1d import CNN1DTrainer
    from .temporal_point_process import TPPTrainer
    from .temporal_fusion_transformer import TFTTrainer
    from .bilstm_forecaster import BiLSTMForecasterTrainer

    return {
        "decision_tree": DecisionTreeTrainer,
        "random_forest": RandomForestTrainer,
        "svm": SVMTrainer,
        "xgboost": XGBoostTrainer,
        "bilstm": BiLSTMTrainer,
        "cnn_1d": CNN1DTrainer,
        "tpp": TPPTrainer,
        "tft": TFTTrainer,
        "bilstm_forecaster": BiLSTMForecasterTrainer,
    }


def get_unsupervised_models():
    """Import and return unsupervised model trainers (lazy import)."""
    from .unsupervised import (
        KMeansTrainer, GMMTrainer, IsolationForestTrainer,
    )
    from .autoencoder import AutoencoderTrainer

    return {
        "kmeans": KMeansTrainer,
        "gmm": GMMTrainer,
        "isolation_forest": IsolationForestTrainer,
        "autoencoder": AutoencoderTrainer,
    }


def get_all_models():
    """Return all model trainers."""
    return {**get_supervised_models(), **get_unsupervised_models()}
