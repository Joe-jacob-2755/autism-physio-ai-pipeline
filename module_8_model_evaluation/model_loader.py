"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  model_loader.py
=============================================================================
Load trained models from M7 output directory. Handles sklearn (.pkl via
joblib) and PyTorch (.pt) models. Returns model objects ready for prediction.
=============================================================================
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """A trained model loaded from M7 output."""
    name: str
    model_type: str         # "supervised" | "unsupervised"
    framework: str          # "sklearn" | "pytorch"
    data_format: str        # "tabular" | "sequence" | "tft" | "tpp"
    model: Any              # The model object (sklearn estimator or PyTorch module)
    model_path: Path
    val_metrics: Dict[str, Any] = field(default_factory=dict)
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    model_size_kb: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels. X shape: (n_samples, n_features)."""
        if self.framework == "sklearn":
            return self.model.predict(X)
        elif self.framework == "pytorch":
            return self._pytorch_predict(X)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict class probabilities. Returns None if unsupported."""
        try:
            if self.framework == "sklearn":
                if hasattr(self.model, "predict_proba"):
                    return self.model.predict_proba(X)
                elif hasattr(self.model, "decision_function"):
                    # SVM: convert decision function to pseudo-probabilities
                    from scipy.special import softmax
                    dec = self.model.decision_function(X)
                    if dec.ndim == 1:
                        dec = np.column_stack([-dec, dec])
                    return softmax(dec, axis=1)
            elif self.framework == "pytorch":
                return self._pytorch_predict_proba(X)
        except Exception:
            return None
        return None

    def _pytorch_predict(self, X: np.ndarray) -> np.ndarray:
        """PyTorch inference."""
        import torch
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X)
            # Only unsqueeze for tabular data; sequence/forecast data is already 3D
            if tensor.ndim == 2 and self.data_format not in ("forecast", "sequence"):
                tensor = tensor.unsqueeze(1)  # (N, 1, F) for sequence models
            output = self.model(tensor)
            if isinstance(output, tuple):
                output = output[0]
            return output.argmax(dim=-1).cpu().numpy()

    def _pytorch_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """PyTorch class probabilities."""
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X)
            if tensor.ndim == 2 and self.data_format not in ("forecast", "sequence"):
                tensor = tensor.unsqueeze(1)
            output = self.model(tensor)
            if isinstance(output, tuple):
                output = output[0]
            return F.softmax(output, dim=-1).cpu().numpy()


class ModelLoader:
    """Load trained models from M7 output directory."""

    # Model name -> (framework, data_format)
    MODEL_INFO = {
        "decision_tree":      ("sklearn", "tabular"),
        "random_forest":      ("sklearn", "tabular"),
        "svm":                ("sklearn", "tabular"),
        "xgboost":            ("sklearn", "tabular"),
        "bilstm":             ("pytorch", "sequence"),
        "cnn_1d":             ("pytorch", "sequence"),
        "tpp":                ("pytorch", "tpp"),
        "tft":                ("pytorch", "tft"),
        "bilstm_forecaster":  ("pytorch", "forecast"),
        "kmeans":             ("sklearn", "tabular"),
        "gmm":                ("sklearn", "tabular"),
        "isolation_forest":   ("sklearn", "tabular"),
        "autoencoder":        ("pytorch", "tabular"),
    }

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [M8-Models] {msg}")

    def load_all(
        self,
        m7_dir: Path,
        supervised_only: bool = True,
    ) -> Dict[str, LoadedModel]:
        """
        Load all trained models from M7 output directory.

        Parameters
        ----------
        m7_dir : Path to M7 output (contains supervised/ and unsupervised/)
        supervised_only : If True, skip unsupervised models

        Returns
        -------
        Dict[model_name, LoadedModel]
        """
        m7_dir = Path(m7_dir)
        models = {}

        model_types = ["supervised"]
        if not supervised_only:
            model_types.append("unsupervised")

        for model_type in model_types:
            type_dir = m7_dir / model_type
            if not type_dir.exists():
                self._log(f"  {model_type}/ not found, skipping")
                continue

            for model_dir in sorted(type_dir.iterdir()):
                if not model_dir.is_dir():
                    continue

                name = model_dir.name
                loaded = self._load_one(model_dir, name, model_type)
                if loaded is not None:
                    models[name] = loaded

        self._log(f"Loaded {len(models)} models: {list(models.keys())}")
        return models

    def _load_one(
        self, model_dir: Path, name: str, model_type: str
    ) -> Optional[LoadedModel]:
        """Load a single model from its directory."""
        info = self.MODEL_INFO.get(name, ("sklearn", "tabular"))
        framework, data_format = info

        # Find model file
        pkl_path = model_dir / "model.pkl"
        pt_path = model_dir / "model.pt"

        if pkl_path.exists():
            model_path = pkl_path
        elif pt_path.exists():
            model_path = pt_path
        else:
            self._log(f"  {name}: no model file found, skipping")
            return None

        # Load model
        try:
            if model_path.suffix == ".pkl":
                model = self._load_sklearn(model_path)
                framework = "sklearn"
            elif model_path.suffix == ".pt":
                model = self._load_pytorch(model_path, model_dir, name)
                framework = "pytorch"
                if model is None:
                    self._log(f"  {name}: PyTorch load failed, skipping")
                    return None
            else:
                self._log(f"  {name}: unknown format {model_path.suffix}")
                return None
        except Exception as e:
            self._log(f"  {name}: load error -- {e}")
            return None

        # Load val metrics
        val_metrics = {}
        metrics_path = model_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                val_metrics = json.load(f)

        # Model size
        model_size_kb = model_path.stat().st_size / 1024

        self._log(f"  {name}: loaded ({framework}, {model_size_kb:.0f} KB)")

        return LoadedModel(
            name=name,
            model_type=model_type,
            framework=framework,
            data_format=data_format,
            model=model,
            model_path=model_path,
            val_metrics=val_metrics,
            model_size_kb=model_size_kb,
        )

    def _load_sklearn(self, path: Path):
        """Load sklearn model via joblib."""
        import joblib
        return joblib.load(path)

    def _load_pytorch(
        self, model_path: Path, model_dir: Path, name: str
    ) -> Any:
        """
        Load PyTorch model. Attempts to load full model first, then
        falls back to state_dict with architecture reconstruction.
        """
        try:
            import torch

            # Try loading as full model first
            try:
                model = torch.load(model_path, map_location="cpu",
                                   weights_only=False)
                if hasattr(model, "eval"):
                    model.eval()
                    return model
            except Exception:
                pass

            # Try loading state_dict -- need architecture info
            hyperparams_path = model_dir / "hyperparams.json"
            if hyperparams_path.exists():
                with open(hyperparams_path) as f:
                    hp = json.load(f)
            else:
                hp = {}

            state_dict = torch.load(model_path, map_location="cpu",
                                    weights_only=True)

            # Reconstruct architecture from state_dict shape
            model = self._reconstruct_pytorch_model(name, state_dict, hp)
            if model is not None:
                model.load_state_dict(state_dict)
                model.eval()
                return model

        except ImportError:
            self._log(f"  {name}: torch not available")
        except Exception as e:
            self._log(f"  {name}: PyTorch load error -- {e}")

        return None

    def _reconstruct_pytorch_model(
        self, name: str, state_dict: dict, hp: dict
    ):
        """Attempt to reconstruct PyTorch model architecture from state_dict."""
        import torch
        import torch.nn as nn

        try:
            if name in ("bilstm", "bilstm_forecaster"):
                return self._reconstruct_bilstm(state_dict, name)
        except Exception as e:
            self._log(f"  {name}: reconstruction failed -- {e}")

        return None

    def _reconstruct_bilstm(self, state_dict: dict, name: str):
        """Reconstruct BiLSTM(Forecast)Model from state_dict shapes."""
        import torch
        import torch.nn as nn

        # Infer architecture from weight shapes
        # lstm.weight_ih_l0: (4*hidden_size, input_size)
        # fc.weight: (n_classes, hidden_size*2) for bidirectional
        ih_key = "lstm.weight_ih_l0"
        fc_key = "fc.weight"

        if ih_key not in state_dict or fc_key not in state_dict:
            return None

        ih_shape = state_dict[ih_key].shape
        fc_shape = state_dict[fc_key].shape

        hidden_size = ih_shape[0] // 4  # LSTM gates = 4 * hidden_size
        input_size = ih_shape[1]
        n_classes = fc_shape[0]

        # Count layers by looking for weight_ih_lN keys
        num_layers = 0
        for key in state_dict:
            if key.startswith("lstm.weight_ih_l") and "_reverse" not in key:
                num_layers += 1

        # Check for bidirectional (reverse keys present)
        bidir = any("_reverse" in k for k in state_dict)

        # Build model -- same architecture for both bilstm and bilstm_forecaster
        class _BiLSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers,
                         n_classes, dropout=0.3):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers,
                    batch_first=True, bidirectional=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size * 2, n_classes)

            def forward(self, x):
                lstm_out, (h_n, _) = self.lstm(x)
                h_forward = h_n[-2]
                h_backward = h_n[-1]
                h_cat = torch.cat([h_forward, h_backward], dim=1)
                h_cat = self.dropout(h_cat)
                return self.fc(h_cat)

        model = _BiLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            n_classes=n_classes,
            dropout=0.3,
        )

        self._log(f"  Reconstructed {name}: input={input_size}, "
                  f"hidden={hidden_size}, layers={num_layers}, "
                  f"classes={n_classes}")
        return model
