"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  models/temporal_point_process.py
=============================================================================
Neural Temporal Point Process -- models the timing and type of behavioural
events. Uses a GRU-based intensity function with softplus activation.
Research-oriented; captures inter-event dynamics for onset prediction.
=============================================================================
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from config import TPP, RANDOM_SEED
from models import BaseModelTrainer, TrainResult


# ── Model ────────────────────────────────────────────────────────────────

class TPPModel(nn.Module):
    """
    Neural Temporal Point Process for event sequence modelling.

    Inputs:
        event_times:  (batch, max_seq_len)  -- inter-event times
        event_types:  (batch, max_seq_len)  -- event type indices
        mask:         (batch, max_seq_len)  -- 1 for valid, 0 for padding

    Outputs:
        log_intensity: (batch, max_seq_len, n_types)  -- log conditional intensity
        type_logits:   (batch, max_seq_len, n_types)  -- type prediction logits
    """

    def __init__(self, n_types: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.n_types = n_types
        self.type_embed = nn.Embedding(n_types + 1, hidden_size,
                                        padding_idx=0)
        self.time_linear = nn.Linear(1, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers,
                          batch_first=True)
        # Intensity function (softplus ensures positivity)
        self.intensity_linear = nn.Linear(hidden_size, n_types)
        # Type prediction head
        self.type_linear = nn.Linear(hidden_size, n_types)

    def forward(self, event_times, event_types, mask):
        # Encode inputs
        type_emb = self.type_embed(event_types)  # (B, T, H)
        time_enc = self.time_linear(
            event_times.unsqueeze(-1))            # (B, T, H)
        combined = torch.cat([type_emb, time_enc], dim=-1)  # (B, T, 2H)

        # GRU
        h_out, _ = self.gru(combined)  # (B, T, H)

        # Outputs
        log_intensity = torch.log(
            nn.functional.softplus(self.intensity_linear(h_out)) + 1e-8)
        type_logits = self.type_linear(h_out)

        return log_intensity, type_logits


# ── Trainer ──────────────────────────────────────────────────────────────

class TPPTrainer(BaseModelTrainer):
    name = "tpp"
    model_type = "supervised"
    framework = "pytorch"

    def __init__(self, seed: int = RANDOM_SEED, verbose: bool = True):
        super().__init__(seed, verbose)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def can_run(self, data) -> tuple:
        """Check if TPP data has enough events."""
        if not hasattr(data, "event_times") or data.event_times is None:
            return False, "No TPP event data available"
        total_events = sum(len(seq) for seq in data.event_times)
        if total_events < TPP["min_events"]:
            return False, (f"Only {total_events} events "
                           f"(need >= {TPP['min_events']})")
        if data.n_types < 2:
            return False, "Only 1 event type"
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

        cfg = TPP
        self._log(f"Training Temporal Point Process on {self.device} ...")
        t0 = time.time()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Prepare data: event_times (list of arrays), event_types (list of arrays)
        train_loader = self._make_tpp_loader(
            data.event_times, data.event_types,
            cfg["batch_size"], shuffle=True)

        # Model
        self.model = TPPModel(
            n_types=data.n_types,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        # Training loop
        history = {"train_loss": [], "type_accuracy": []}
        best_loss = float("inf")
        no_improve = 0
        best_state = None

        for epoch in range(cfg["epochs"]):
            self.model.train()
            epoch_loss, correct, total = 0, 0, 0

            for times_b, types_b, mask_b in train_loader:
                times_b = times_b.to(self.device)
                types_b = types_b.to(self.device)
                mask_b = mask_b.to(self.device)

                optimizer.zero_grad()
                log_intensity, type_logits = self.model(
                    times_b, types_b, mask_b)

                # NLL loss for temporal component
                # Shift: predict next event from current hidden state
                # Use type_logits[:, :-1] to predict types_b[:, 1:]
                if types_b.shape[1] < 2:
                    continue

                target_types = types_b[:, 1:]  # Next event types
                pred_logits = type_logits[:, :-1]  # Predictions from prev
                target_mask = mask_b[:, 1:]

                # Cross-entropy for type prediction
                loss_type = nn.functional.cross_entropy(
                    pred_logits.reshape(-1, data.n_types),
                    target_types.reshape(-1),
                    reduction="none")
                loss_type = (loss_type * target_mask.reshape(-1)).sum()
                n_valid = target_mask.sum().clamp(min=1)
                loss = loss_type / n_valid

                # Intensity NLL: -log_intensity + integral approximation
                target_times = times_b[:, 1:]
                log_int = log_intensity[:, :-1]
                # Gather the intensity for the correct type
                type_idx = target_types.unsqueeze(-1)  # (B, T-1, 1)
                log_int_correct = log_int.gather(2, type_idx).squeeze(-1)
                nll_time = -(log_int_correct * target_mask).sum() / n_valid
                loss = loss + nll_time

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * n_valid.item()
                preds = pred_logits.argmax(-1)
                correct += ((preds == target_types) * target_mask).sum().item()
                total += n_valid.item()

            avg_loss = epoch_loss / max(total, 1)
            accuracy = correct / max(total, 1)
            history["train_loss"].append(avg_loss)
            history["type_accuracy"].append(accuracy)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    self._log(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                self._log(f"  Epoch {epoch+1}: loss={avg_loss:.4f} "
                           f"type_acc={accuracy:.4f}")

        # Restore best
        if best_state:
            self.model.load_state_dict(best_state)
        self.model.to(self.device)
        elapsed = time.time() - t0

        # Metrics
        metrics = {
            "final_loss": best_loss,
            "final_type_accuracy": history["type_accuracy"][-1]
                if history["type_accuracy"] else 0.0,
            "total_events_trained": sum(
                len(s) for s in data.event_times),
            "n_event_types": data.n_types,
        }

        # Save
        model_path = output_dir / "model.pt"
        self.save(model_path)

        # Plot training curves
        self._plot_training(history, output_dir / "training_curves.png")
        self._plot_intensity(data, output_dir / "intensity_plot.png")

        return TrainResult(
            model_name=self.name, model_type=self.model_type,
            framework=self.framework, metrics=metrics,
            model_path=model_path, training_time_s=elapsed,
            hyperparams=cfg, history=history,
            model_size_kb=self.measure_model_size(model_path))

    def predict(self, X) -> np.ndarray:
        """Not applicable for TPP in the standard sense."""
        return np.array([])

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True))

    # ── Helpers ──────────────────────────────────────────────────────────

    def _make_tpp_loader(self, event_times_list, event_types_list,
                         batch_size, shuffle):
        """Create a DataLoader from variable-length event sequences."""
        # Pad sequences
        max_len = max(len(s) for s in event_times_list)
        n = len(event_times_list)

        times_padded = np.zeros((n, max_len), dtype=np.float32)
        types_padded = np.zeros((n, max_len), dtype=np.int64)
        masks = np.zeros((n, max_len), dtype=np.float32)

        for i, (t, ty) in enumerate(
                zip(event_times_list, event_types_list)):
            L = len(t)
            times_padded[i, :L] = t
            types_padded[i, :L] = ty
            masks[i, :L] = 1.0

        ds = TensorDataset(
            torch.FloatTensor(times_padded),
            torch.LongTensor(types_padded),
            torch.FloatTensor(masks))
        return TorchDataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _plot_training(self, history, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from config import PLOT_DPI

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history["train_loss"])
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("TPP - Training Loss")

        ax2.plot(history["type_accuracy"])
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Type Accuracy")
        ax2.set_title("TPP - Event Type Prediction Accuracy")

        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)

    def _plot_intensity(self, data, path):
        """Plot estimated intensity over time for first sequence."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from config import PLOT_DPI

        if not data.event_times or len(data.event_times[0]) < 2:
            return

        self.model.eval()
        # Take first sequence
        times = torch.FloatTensor(
            data.event_times[0]).unsqueeze(0).to(self.device)
        types = torch.LongTensor(
            data.event_types[0]).unsqueeze(0).to(self.device)
        mask = torch.ones_like(times)

        with torch.no_grad():
            log_int, _ = self.model(times, types, mask)
            intensity = torch.exp(log_int).cpu().numpy()[0]

        fig, ax = plt.subplots(figsize=(12, 5))
        cum_times = np.cumsum(data.event_times[0])
        for t_idx in range(min(intensity.shape[1], 5)):
            ax.plot(cum_times, intensity[:, t_idx],
                    alpha=0.7, label=f"Type {t_idx}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Intensity")
        ax.set_title("TPP - Estimated Event Intensity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)
