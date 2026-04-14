"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  models/autoencoder.py
=============================================================================
Autoencoder for anomaly detection via reconstruction error.
Architecture: n_features → 64 → 32 → 16 → 8 → 16 → 32 → 64 → n_features
Anomaly = reconstruction error exceeds 95th percentile of training errors.
=============================================================================
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from config import AUTOENCODER, RANDOM_SEED
from models import BaseModelTrainer, TrainResult


# ── Model ────────────────────────────────────────────────────────────────

class AutoencoderModel(nn.Module):
    """
    Symmetric autoencoder for anomaly detection.

    Input:  (batch, n_features)
    Output: (batch, n_features)  — reconstruction
    """

    def __init__(self, n_features: int, encoder_dims: list,
                 latent_dim: int, dropout: float = 0.2):
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = n_features
        for dim in encoder_dims:
            enc_layers.extend([
                nn.Linear(in_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = dim
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror of encoder)
        dec_layers = []
        in_dim = latent_dim
        for dim in reversed(encoder_dims):
            dec_layers.extend([
                nn.Linear(in_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = dim
        dec_layers.append(nn.Linear(in_dim, n_features))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


# ── Trainer ──────────────────────────────────────────────────────────────

class AutoencoderTrainer(BaseModelTrainer):
    name = "autoencoder"
    model_type = "unsupervised"
    framework = "pytorch"

    def __init__(self, seed: int = RANDOM_SEED, verbose: bool = True):
        super().__init__(seed, verbose)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = None  # anomaly threshold (set after training)

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

        cfg = AUTOENCODER
        self._log(f"Training Autoencoder on {self.device} ...")
        t0 = time.time()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        n_features = X.shape[1]

        # Split X into train/val (80/20) for early stopping
        n_val = max(1, int(len(X) * 0.2))
        indices = np.random.permutation(len(X))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        train_loader = self._make_loader(
            X[train_idx], cfg["batch_size"], shuffle=True)
        val_loader = self._make_loader(
            X[val_idx], cfg["batch_size"], shuffle=False)

        # Model
        self.model = AutoencoderModel(
            n_features=n_features,
            encoder_dims=cfg["encoder_dims"],
            latent_dim=cfg["latent_dim"],
            dropout=cfg["dropout"],
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        # Training loop
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        no_improve = 0
        best_state = None

        for epoch in range(cfg["epochs"]):
            # Train
            self.model.train()
            train_loss = 0
            for (X_b,) in train_loader:
                X_b = X_b.to(self.device)
                optimizer.zero_grad()
                recon = self.model(X_b)
                loss = criterion(recon, X_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(X_b)
            train_loss /= len(train_idx)

            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for (X_b,) in val_loader:
                    X_b = X_b.to(self.device)
                    recon = self.model(X_b)
                    val_loss += criterion(recon, X_b).item() * len(X_b)
            val_loss /= max(len(val_idx), 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    self._log(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 20 == 0 or epoch == 0:
                self._log(f"  Epoch {epoch+1}: train_loss={train_loss:.6f} "
                           f"val_loss={val_loss:.6f}")

        # Restore best
        if best_state:
            self.model.load_state_dict(best_state)
        self.model.to(self.device)
        elapsed = time.time() - t0

        # Compute reconstruction errors on full data
        errors = self._reconstruction_errors(X)
        self.threshold = float(np.percentile(
            errors, cfg["anomaly_percentile"]))
        self._log(f"Anomaly threshold (p{cfg['anomaly_percentile']}): "
                   f"{self.threshold:.6f}")

        anomaly_preds = (errors > self.threshold).astype(int)

        # Save model
        model_path = output_dir / "model.pt"
        self.save(model_path)

        # Compute latent representations for visualisation
        latent = self._encode(X)

        # Evaluate
        from evaluator import ModelEvaluator
        ev = ModelEvaluator(kwargs.get("class_names", []), verbose=False)
        metrics = ev.evaluate_anomaly(errors, anomaly_preds, y_true)
        metrics["threshold"] = self.threshold
        metrics["mean_recon_error"] = float(np.mean(errors))
        metrics["std_recon_error"] = float(np.std(errors))

        # Save scores
        scores_df = pd.DataFrame({
            "reconstruction_error": errors,
            "anomaly_prediction": anomaly_preds,
        })
        if y_true is not None:
            scores_df["true_label"] = y_true
        scores_df.to_csv(output_dir / "anomaly_scores.csv", index=False)

        # Plot error distribution
        self._plot_error_distribution(
            errors, anomaly_preds, y_true,
            output_dir / "error_distribution.png")

        # Plot latent space
        self._plot_latent_space(
            latent, y_true, output_dir / "latent_space.png")

        # Plot training curves
        self._plot_training_curves(
            history, output_dir / "training_curves.png")

        return TrainResult(
            model_name=self.name, model_type=self.model_type,
            framework=self.framework, metrics=metrics,
            model_path=model_path, training_time_s=elapsed,
            hyperparams=cfg, history=history,
            predictions=anomaly_preds,
            model_size_kb=self.measure_model_size(model_path))

    def predict(self, X) -> np.ndarray:
        errors = self._reconstruction_errors(X)
        return (errors > self.threshold).astype(int)

    def save(self, path: Path):
        torch.save({
            "state_dict": self.model.state_dict(),
            "threshold": self.threshold,
        }, path)

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.threshold = ckpt["threshold"]

    # ── Helpers ──────────────────────────────────────────────────────────

    def _make_loader(self, X, batch_size, shuffle):
        ds = TensorDataset(torch.FloatTensor(X))
        return TorchDataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _reconstruction_errors(self, X):
        self.model.eval()
        errors = []
        loader = self._make_loader(X, 64, shuffle=False)
        with torch.no_grad():
            for (X_b,) in loader:
                X_b = X_b.to(self.device)
                recon = self.model(X_b)
                err = ((X_b - recon) ** 2).mean(dim=1).cpu().numpy()
                errors.extend(err)
        return np.array(errors)

    def _encode(self, X):
        self.model.eval()
        latents = []
        loader = self._make_loader(X, 64, shuffle=False)
        with torch.no_grad():
            for (X_b,) in loader:
                X_b = X_b.to(self.device)
                z = self.model.encode(X_b).cpu().numpy()
                latents.append(z)
        return np.concatenate(latents, axis=0)

    def _plot_error_distribution(self, errors, preds, y_true, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from config import PLOT_DPI

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors[preds == 0], bins=50, alpha=0.6, label="Normal",
                color="steelblue")
        ax.hist(errors[preds == 1], bins=50, alpha=0.6, label="Anomaly",
                color="tomato")
        ax.axvline(self.threshold, color="red", linestyle="--",
                   label=f"Threshold ({self.threshold:.4f})")
        ax.set_xlabel("Reconstruction Error (MSE)")
        ax.set_ylabel("Count")
        ax.set_title("Autoencoder - Error Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)

    def _plot_latent_space(self, latent, y_true, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from config import PLOT_DPI

        if latent.shape[1] < 2:
            return

        fig, ax = plt.subplots(figsize=(10, 8))
        # Use first 2 latent dims (or PCA if > 2)
        if latent.shape[1] > 2:
            from sklearn.decomposition import PCA
            coords = PCA(n_components=2).fit_transform(latent)
        else:
            coords = latent

        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                             c=y_true if y_true is not None else "steelblue",
                             cmap="tab10", alpha=0.6, s=15)
        if y_true is not None:
            fig.colorbar(scatter, ax=ax, label="True Label")
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        ax.set_title("Autoencoder - Latent Space")
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)

    def _plot_training_curves(self, history, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from config import PLOT_DPI

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history["train_loss"], label="Train Loss")
        ax.plot(history["val_loss"], label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Autoencoder - Training Curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)
