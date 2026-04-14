"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  models/bilstm.py
=============================================================================
Bidirectional LSTM -- captures onset + recovery patterns in both temporal
directions. Suitable for physiological signal classification where
events have asymmetric rise/decay dynamics.
=============================================================================
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from config import BILSTM, MIN_SEQUENCES_FOR_TEMPORAL, RANDOM_SEED
from models import BaseModelTrainer, TrainResult


# ── Model ────────────────────────────────────────────────────────────────

class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM for sequence classification.

    Input:  (batch, seq_len, n_features)
    Output: (batch, n_classes)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)  # *2 for bidir

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, _) = self.lstm(x)
        # Concatenate final hidden from both directions
        h_forward = h_n[-2]   # Last layer, forward
        h_backward = h_n[-1]  # Last layer, backward
        h_cat = torch.cat([h_forward, h_backward], dim=1)
        h_cat = self.dropout(h_cat)
        return self.fc(h_cat)


# ── Trainer ──────────────────────────────────────────────────────────────

class BiLSTMTrainer(BaseModelTrainer):
    name = "bilstm"
    model_type = "supervised"
    framework = "pytorch"

    def __init__(self, seed: int = RANDOM_SEED, verbose: bool = True):
        super().__init__(seed, verbose)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def can_run(self, data) -> tuple:
        n_train = len(data.y_train)
        if n_train < MIN_SEQUENCES_FOR_TEMPORAL:
            return False, (f"Only {n_train} train sequences "
                           f"(need >= {MIN_SEQUENCES_FOR_TEMPORAL})")
        if len(np.unique(data.y_train)) < 2:
            return False, "Only 1 class in training sequences"
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

        cfg = BILSTM
        self._log(f"Training BiLSTM on {self.device} ...")
        t0 = time.time()

        # Set seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Create data loaders
        train_loader = self._make_loader(
            data.X_train, data.y_train, cfg["batch_size"], shuffle=True)
        val_loader = self._make_loader(
            data.X_val, data.y_val, cfg["batch_size"], shuffle=False)

        # Model
        n_features = data.X_train.shape[2]
        self.model = BiLSTMModel(
            input_size=n_features,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            n_classes=data.n_classes,
            dropout=cfg["dropout"],
        ).to(self.device)

        # Class weights for loss
        from class_balancer import ClassBalancer
        balancer = ClassBalancer(strategy="class_weight", verbose=False)
        weights = balancer.compute_class_weights(data.y_train)
        weight_tensor = torch.zeros(data.n_classes, device=self.device)
        for k, v in weights.items():
            if k < data.n_classes:
                weight_tensor[k] = v

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg["lr_factor"],
            patience=cfg["lr_patience"])

        # Training loop
        history = {"train_loss": [], "val_loss": [],
                    "train_f1": [], "val_f1": []}
        best_val_f1 = 0.0
        no_improve = 0
        best_state = None

        for epoch in range(cfg["epochs"]):
            # Train
            self.model.train()
            train_loss, train_preds, train_targets = 0, [], []
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_b)
                loss = criterion(logits, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg["grad_clip"])
                optimizer.step()
                train_loss += loss.item() * len(y_b)
                train_preds.extend(logits.argmax(1).cpu().numpy())
                train_targets.extend(y_b.cpu().numpy())

            train_loss /= len(data.y_train)
            train_f1 = self._f1(train_targets, train_preds)

            # Validate
            val_loss, val_preds, val_targets = self._evaluate_epoch(
                val_loader, criterion)
            val_f1 = self._f1(val_targets, val_preds)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_f1"].append(train_f1)
            history["val_f1"].append(val_f1)

            scheduler.step(val_loss)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    self._log(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                self._log(f"  Epoch {epoch+1}: loss={train_loss:.4f} "
                           f"val_f1={val_f1:.4f}")

        # Restore best
        if best_state:
            self.model.load_state_dict(best_state)
        self.model.to(self.device)
        elapsed = time.time() - t0

        # Test evaluation
        test_loader = self._make_loader(
            data.X_test, data.y_test, cfg["batch_size"], shuffle=False)
        _, y_pred_list, y_true_list = self._evaluate_epoch(
            test_loader, criterion)
        y_pred = np.array(y_pred_list)
        y_prob = self._predict_proba_loader(test_loader)

        # Save
        model_path = output_dir / "model.pt"
        self.save(model_path)
        latency = self._measure_torch_latency(data.X_test[:1])
        size_kb = self.measure_model_size(model_path)

        from evaluator import ModelEvaluator
        ev = ModelEvaluator(data.class_names, verbose=False)
        metrics = ev.evaluate_supervised(data.y_test, y_pred, y_prob)
        ev.plot_confusion_matrix(
            data.y_test, y_pred,
            output_dir / "confusion_matrix.png",
            title="BiLSTM - Confusion Matrix")
        ev.plot_training_curves(
            history, output_dir / "training_curves.png",
            title="BiLSTM - Training Curves")
        if y_prob is not None:
            ev.plot_roc_curves(
                data.y_test, y_prob,
                output_dir / "roc_curves.png",
                title="BiLSTM - ROC Curves")

        return TrainResult(
            model_name=self.name, model_type=self.model_type,
            framework=self.framework, metrics=metrics,
            model_path=model_path, training_time_s=elapsed,
            hyperparams=cfg, history=history,
            predictions=y_pred, probabilities=y_prob,
            model_size_kb=size_kb, inference_latency_ms=latency)

    def predict(self, X) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self.device)
            return self.model(t).argmax(1).cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self.device)
            return torch.softmax(self.model(t), dim=1).cpu().numpy()

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True))

    # ── Helpers ──────────────────────────────────────────────────────────

    def _make_loader(self, X, y, batch_size, shuffle):
        ds = TensorDataset(
            torch.FloatTensor(X), torch.LongTensor(y))
        return TorchDataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _evaluate_epoch(self, loader, criterion):
        self.model.eval()
        total_loss, preds, targets = 0, [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                logits = self.model(X_b)
                total_loss += criterion(logits, y_b).item() * len(y_b)
                preds.extend(logits.argmax(1).cpu().numpy())
                targets.extend(y_b.cpu().numpy())
        total_loss /= len(targets) if targets else 1
        return total_loss, preds, targets

    def _predict_proba_loader(self, loader):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for X_b, _ in loader:
                X_b = X_b.to(self.device)
                probs.append(
                    torch.softmax(self.model(X_b), dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0) if probs else None

    def _measure_torch_latency(self, X_sample, n_runs=100):
        self.model.eval()
        t = torch.FloatTensor(X_sample).to(self.device)
        with torch.no_grad():
            self.model(t)  # Warm up
            t0 = time.perf_counter()
            for _ in range(n_runs):
                self.model(t)
            return round((time.perf_counter() - t0) / n_runs * 1000, 3)

    @staticmethod
    def _f1(y_true, y_pred):
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)
