"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  models/temporal_fusion_transformer.py
=============================================================================
Temporal Fusion Transformer -- attention-based architecture that handles
both static (demographics) and temporal (signal features) inputs.
Provides interpretable variable importance via Variable Selection Networks.
=============================================================================
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from config import TFT, MIN_SEQUENCES_FOR_TEMPORAL, RANDOM_SEED
from models import BaseModelTrainer, TrainResult


# ── Sub-modules ──────────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network — core building block of TFT."""

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, dropout: float = 0.1,
                 context_size: int = 0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(output_size)

        if context_size > 0:
            self.context_proj = nn.Linear(context_size, hidden_size,
                                          bias=False)
        else:
            self.context_proj = None

        self.skip = (nn.Linear(input_size, output_size)
                     if input_size != output_size else None)

    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip else x
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = self.elu(h)
        a = self.sigmoid(self.gate(h))
        h = self.dropout(self.fc2(h))
        return self.layer_norm(residual + a * h)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network — learns which inputs matter."""

    def __init__(self, n_vars: int, input_size: int,
                 hidden_size: int, dropout: float = 0.1,
                 context_size: int = 0):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_size = hidden_size

        # Per-variable GRNs
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size,
                                 dropout)
            for _ in range(n_vars)
        ])

        # Softmax gate over variables
        gate_input = n_vars * hidden_size
        if context_size > 0:
            gate_input += context_size
        self.gate_grn = GatedResidualNetwork(
            gate_input, hidden_size, n_vars, dropout)

    def forward(self, inputs, context=None):
        # inputs: (batch, [seq_len,] n_vars, input_size)
        var_outputs = []
        for i in range(self.n_vars):
            var_outputs.append(self.var_grns[i](inputs[..., i, :]))
        var_outputs = torch.stack(var_outputs, dim=-2)  # (..., n_vars, H)

        # Flatten for gate
        flat = var_outputs.reshape(
            *var_outputs.shape[:-2], -1)  # (..., n_vars*H)
        if context is not None:
            flat = torch.cat([flat, context], dim=-1)
        weights = torch.softmax(
            self.gate_grn(flat), dim=-1)  # (..., n_vars)

        # Weighted sum
        combined = (var_outputs * weights.unsqueeze(-1)).sum(dim=-2)
        return combined, weights


# ── Main TFT Model ──────────────────────────────────────────────────────

class TFTModel(nn.Module):
    """
    Simplified Temporal Fusion Transformer for classification.

    Inputs:
        static_input:   (batch, n_static_features)
        temporal_input:  (batch, seq_len, n_temporal_features)
    Output:
        logits:          (batch, n_classes)
        var_importance:  dict of attention weights
    """

    def __init__(self, n_static: int, n_temporal: int, hidden_size: int,
                 n_classes: int, attention_heads: int, dropout: float,
                 num_encoder_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_static = n_static
        self.n_temporal = n_temporal

        # Static variable processing
        if n_static > 0:
            self.static_vsn = VariableSelectionNetwork(
                n_vars=n_static, input_size=1,
                hidden_size=hidden_size, dropout=dropout)
            self.static_grn = GatedResidualNetwork(
                hidden_size, hidden_size, hidden_size, dropout)
        else:
            self.static_vsn = None
            self.static_grn = None

        # Temporal variable processing
        self.temporal_proj = nn.Linear(n_temporal, hidden_size)

        # LSTM encoder
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_encoder_layers,
            batch_first=True, dropout=dropout if num_encoder_layers > 1
            else 0.0)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_size, attention_heads, dropout=dropout,
            batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)

        # Output GRN + classifier
        context_size = hidden_size if n_static > 0 else 0
        self.output_grn = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout,
            context_size=context_size)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, static_input, temporal_input):
        var_importance = {}

        # Static context
        if self.static_vsn is not None and static_input is not None:
            # Reshape static: (B, n_static) → (B, n_static, 1)
            static_vars = static_input.unsqueeze(-1)
            static_enc, static_weights = self.static_vsn(static_vars)
            static_context = self.static_grn(static_enc)
            var_importance["static"] = static_weights.detach()
        else:
            static_context = None

        # Temporal encoding
        temp_enc = self.temporal_proj(temporal_input)  # (B, T, H)

        # LSTM
        lstm_out, _ = self.lstm(temp_enc)  # (B, T, H)

        # Self-attention
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out)
        attn_out = self.attention_norm(lstm_out + attn_out)
        var_importance["temporal_attention"] = attn_weights.detach()

        # Pool: take last timestep
        final_hidden = attn_out[:, -1, :]  # (B, H)

        # Output with static enrichment
        output = self.output_grn(final_hidden, static_context)
        logits = self.classifier(output)

        return logits, var_importance


# ── Trainer ──────────────────────────────────────────────────────────────

class TFTTrainer(BaseModelTrainer):
    name = "tft"
    model_type = "supervised"
    framework = "pytorch"

    def __init__(self, seed: int = RANDOM_SEED, verbose: bool = True):
        super().__init__(seed, verbose)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def can_run(self, data) -> tuple:
        if not hasattr(data, "X_temporal_train"):
            return False, "No TFT-format data available"
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

        cfg = TFT
        self._log(f"Training TFT on {self.device} ...")
        t0 = time.time()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        n_static = data.X_static_train.shape[1] if (
            data.X_static_train is not None) else 0
        n_temporal = data.X_temporal_train.shape[2]

        # Data loaders
        train_loader = self._make_tft_loader(
            data.X_static_train, data.X_temporal_train,
            data.y_train, cfg["batch_size"], shuffle=True)
        val_loader = self._make_tft_loader(
            data.X_static_val, data.X_temporal_val,
            data.y_val, cfg["batch_size"], shuffle=False)

        # Model
        self.model = TFTModel(
            n_static=n_static,
            n_temporal=n_temporal,
            hidden_size=cfg["hidden_size"],
            n_classes=data.n_classes,
            attention_heads=cfg["attention_heads"],
            dropout=cfg["dropout"],
            num_encoder_layers=cfg["num_encoder_layers"],
        ).to(self.device)

        # Class weights
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
        last_var_importance = None

        for epoch in range(cfg["epochs"]):
            # Train
            self.model.train()
            train_loss, train_preds, train_targets = 0, [], []
            for static_b, temp_b, y_b in train_loader:
                static_b = static_b.to(self.device) if n_static > 0 else None
                temp_b = temp_b.to(self.device)
                y_b = y_b.to(self.device)

                optimizer.zero_grad()
                logits, var_imp = self.model(static_b, temp_b)
                loss = criterion(logits, y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg["grad_clip"])
                optimizer.step()

                train_loss += loss.item() * len(y_b)
                train_preds.extend(logits.argmax(1).cpu().numpy())
                train_targets.extend(y_b.cpu().numpy())
                last_var_importance = var_imp

            train_loss /= len(data.y_train)
            train_f1 = self._f1(train_targets, train_preds)

            # Validate
            val_loss, val_preds, val_targets = self._evaluate_epoch(
                val_loader, criterion, n_static)
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
        test_loader = self._make_tft_loader(
            data.X_static_test, data.X_temporal_test,
            data.y_test, cfg["batch_size"], shuffle=False)
        _, y_pred_list, y_true_list = self._evaluate_epoch(
            test_loader, criterion, n_static)
        y_pred = np.array(y_pred_list)
        y_prob = self._predict_proba_loader(test_loader, n_static)

        # Save
        model_path = output_dir / "model.pt"
        self.save(model_path)
        latency = self._measure_tft_latency(data, n_static)
        size_kb = self.measure_model_size(model_path)

        from evaluator import ModelEvaluator
        ev = ModelEvaluator(data.class_names, verbose=False)
        metrics = ev.evaluate_supervised(data.y_test, y_pred, y_prob)
        ev.plot_confusion_matrix(
            data.y_test, y_pred,
            output_dir / "confusion_matrix.png",
            title="TFT - Confusion Matrix")
        ev.plot_training_curves(
            history, output_dir / "training_curves.png",
            title="TFT - Training Curves")
        if y_prob is not None:
            ev.plot_roc_curves(
                data.y_test, y_prob,
                output_dir / "roc_curves.png",
                title="TFT - ROC Curves")

        # Plot variable importance
        if last_var_importance and "static" in last_var_importance:
            self._plot_variable_importance(
                last_var_importance, data,
                output_dir / "variable_importance.png")

        return TrainResult(
            model_name=self.name, model_type=self.model_type,
            framework=self.framework, metrics=metrics,
            model_path=model_path, training_time_s=elapsed,
            hyperparams=cfg, history=history,
            predictions=y_pred, probabilities=y_prob,
            model_size_kb=size_kb, inference_latency_ms=latency)

    def predict(self, X_static, X_temporal) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            static = (torch.FloatTensor(X_static).to(self.device)
                       if X_static is not None else None)
            temporal = torch.FloatTensor(X_temporal).to(self.device)
            logits, _ = self.model(static, temporal)
            return logits.argmax(1).cpu().numpy()

    def predict_proba(self, X_static, X_temporal):
        self.model.eval()
        with torch.no_grad():
            static = (torch.FloatTensor(X_static).to(self.device)
                       if X_static is not None else None)
            temporal = torch.FloatTensor(X_temporal).to(self.device)
            logits, _ = self.model(static, temporal)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True))

    # ── Helpers ──────────────────────────────────────────────────────────

    def _make_tft_loader(self, X_static, X_temporal, y,
                          batch_size, shuffle):
        tensors = []
        if X_static is not None:
            tensors.append(torch.FloatTensor(X_static))
        else:
            # Placeholder zero tensor
            tensors.append(torch.zeros(len(y), 0))
        tensors.append(torch.FloatTensor(X_temporal))
        tensors.append(torch.LongTensor(y))
        ds = TensorDataset(*tensors)
        return TorchDataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _evaluate_epoch(self, loader, criterion, n_static):
        self.model.eval()
        total_loss, preds, targets = 0, [], []
        with torch.no_grad():
            for batch in loader:
                static_b = batch[0].to(self.device) if n_static > 0 else None
                temp_b = batch[1].to(self.device)
                y_b = batch[2].to(self.device)

                logits, _ = self.model(static_b, temp_b)
                total_loss += criterion(logits, y_b).item() * len(y_b)
                preds.extend(logits.argmax(1).cpu().numpy())
                targets.extend(y_b.cpu().numpy())
        total_loss /= len(targets) if targets else 1
        return total_loss, preds, targets

    def _predict_proba_loader(self, loader, n_static):
        self.model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                static_b = batch[0].to(self.device) if n_static > 0 else None
                temp_b = batch[1].to(self.device)
                logits, _ = self.model(static_b, temp_b)
                probs.append(
                    torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probs, axis=0) if probs else None

    def _measure_tft_latency(self, data, n_static, n_runs=100):
        self.model.eval()
        static = None
        if n_static > 0 and data.X_static_test is not None:
            static = torch.FloatTensor(
                data.X_static_test[:1]).to(self.device)
        temporal = torch.FloatTensor(
            data.X_temporal_test[:1]).to(self.device)
        with torch.no_grad():
            self.model(static, temporal)  # Warm up
            t0 = time.perf_counter()
            for _ in range(n_runs):
                self.model(static, temporal)
            return round((time.perf_counter() - t0) / n_runs * 1000, 3)

    def _plot_variable_importance(self, var_importance, data, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from config import PLOT_DPI

        fig, ax = plt.subplots(figsize=(10, 6))

        if "static" in var_importance:
            weights = var_importance["static"].cpu().numpy()
            # Average across batch
            avg_weights = weights.mean(axis=0)
            n_vars = len(avg_weights)
            names = (data.static_feature_names[:n_vars]
                     if hasattr(data, "static_feature_names")
                     else [f"Static_{i}" for i in range(n_vars)])
            sorted_idx = np.argsort(avg_weights)[::-1]
            ax.barh(range(n_vars),
                    avg_weights[sorted_idx], color="steelblue")
            ax.set_yticks(range(n_vars))
            ax.set_yticklabels([names[i] for i in sorted_idx])
            ax.set_xlabel("Variable Selection Weight")
            ax.set_title("TFT - Static Variable Importance")

        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)

    @staticmethod
    def _f1(y_true, y_pred):
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="weighted", zero_division=0)
