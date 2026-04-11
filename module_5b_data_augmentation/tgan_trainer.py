"""
=============================================================================
MODULE 5B - DATA AUGMENTATION  |  tgan_trainer.py
=============================================================================
Three-phase training loop for Conditional Transformer TimeGAN.

Phase 1: Autoencoder (Embedder E + Recovery R)
  - Minimise reconstruction loss: ||x - R(E(x))||
  - Teaches the model to compress and reconstruct real signals

Phase 2: Supervisor (S with frozen E)
  - Minimise supervised loss: ||E(x)[t+1] - S(E(x))[t]||
  - Teaches temporal dynamics in latent space

Phase 3: Joint adversarial (G + D + E + R + S together)
  - Generator tries to fool Discriminator
  - Discriminator tries to distinguish real from fake latent
  - Reconstruction + supervised losses keep E, R, S grounded

References:
  Yoon et al. (2019) — TimeGAN training procedure
=============================================================================
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import MODULE_LABEL, TGAN
from tgan_model import CTTimeGAN

log = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Training loss history across all three phases."""
    phase1_recon: List[float] = field(default_factory=list)
    phase2_supervised: List[float] = field(default_factory=list)
    phase3_discriminator: List[float] = field(default_factory=list)
    phase3_generator: List[float] = field(default_factory=list)
    phase3_reconstruction: List[float] = field(default_factory=list)
    phase3_supervised: List[float] = field(default_factory=list)
    best_epoch: int = 0
    total_time_s: float = 0.0


class TGANTrainer:
    """
    Three-phase trainer for CTTimeGAN.

    Parameters
    ----------
    model : CTTimeGAN
        The CT-TimeGAN model instance to train.
    device : str
        "cuda" or "cpu".
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model: CTTimeGAN,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.model = model.to(device)
        self.device = device
        self.seed = seed

        # Training hyperparameters from config
        self.batch_size = TGAN["batch_size"]
        self.lr = TGAN["learning_rate"]
        self.beta1 = TGAN["beta1"]
        self.beta2 = TGAN["beta2"]
        self.weight_decay = TGAN["weight_decay"]
        self.patience = TGAN["patience"]
        self.min_delta = TGAN["min_delta"]

        # Loss weights
        self.lambda_recon = TGAN["lambda_reconstruction"]
        self.lambda_sup = TGAN["lambda_supervised"]
        self.lambda_gen = TGAN["lambda_generator"]

        # Phase epochs
        self.p1_epochs = TGAN["phase1_epochs"]
        self.p2_epochs = TGAN["phase2_epochs"]
        self.p3_epochs = TGAN["phase3_epochs"]

        self.history = TrainingHistory()

    # ═════════════════════════════════════════════════════════════════════
    # Public API
    # ═════════════════════════════════════════════════════════════════════

    def train(
        self,
        X: np.ndarray,
        conditions: Dict[str, np.ndarray],
    ) -> TrainingHistory:
        """
        Run full 3-phase training.

        Parameters
        ----------
        X : ndarray of shape (n_samples, seq_len, n_features)
            Real signal segments (already normalised to [0, 1]).
        conditions : dict with keys "emotion", "severity", "verbal"
            Each value is an ndarray of shape (n_samples,) with integer indices.

        Returns
        -------
        TrainingHistory
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        t_start = time.time()

        # Prepare data
        X_t = torch.FloatTensor(X).to(self.device)
        emo_t = torch.LongTensor(conditions["emotion"]).to(self.device)
        sev_t = torch.LongTensor(conditions["severity"]).to(self.device)
        vrb_t = torch.LongTensor(conditions["verbal"]).to(self.device)

        dataset = TensorDataset(X_t, emo_t, sev_t, vrb_t)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=True, generator=torch.Generator().manual_seed(self.seed),
        )

        n_samples = len(X)
        seq_len = X.shape[1]
        log.info(f"[{MODULE_LABEL}] Training CT-TimeGAN: "
                 f"{n_samples} segments, seq_len={seq_len}, "
                 f"features={X.shape[2]}")

        # Phase 1: Autoencoder
        log.info(f"[{MODULE_LABEL}] Phase 1: Autoencoder (E+R) — "
                 f"{self.p1_epochs} epochs")
        self._train_phase1(loader)

        # Phase 2: Supervisor
        log.info(f"[{MODULE_LABEL}] Phase 2: Supervisor (S) — "
                 f"{self.p2_epochs} epochs")
        self._train_phase2(loader)

        # Phase 3: Joint adversarial
        log.info(f"[{MODULE_LABEL}] Phase 3: Joint adversarial — "
                 f"{self.p3_epochs} epochs")
        self._train_phase3(loader, seq_len)

        self.history.total_time_s = time.time() - t_start
        log.info(f"[{MODULE_LABEL}] Training complete in "
                 f"{self.history.total_time_s:.1f}s")

        return self.history

    def save_checkpoint(self, path: Path) -> None:
        """Save model state dict + training history."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "history": {
                "phase1_recon": self.history.phase1_recon,
                "phase2_supervised": self.history.phase2_supervised,
                "phase3_discriminator": self.history.phase3_discriminator,
                "phase3_generator": self.history.phase3_generator,
                "best_epoch": self.history.best_epoch,
                "total_time_s": self.history.total_time_s,
            },
            "config": TGAN,
            "n_features": self.model.n_features,
            "seed": self.seed,
        }, path)
        log.info(f"[{MODULE_LABEL}] Checkpoint saved: {path}")

    @classmethod
    def load_checkpoint(
        cls, path: Path, device: str = "cpu"
    ) -> Tuple["TGANTrainer", CTTimeGAN]:
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = CTTimeGAN(n_features=ckpt["n_features"])
        model.load_state_dict(ckpt["model_state_dict"])
        trainer = cls(model, device=device, seed=ckpt.get("seed", 42))
        return trainer, model

    # ═════════════════════════════════════════════════════════════════════
    # Phase 1: Autoencoder (E + R)
    # ═════════════════════════════════════════════════════════════════════

    def _train_phase1(self, loader: DataLoader) -> None:
        """Train Embedder + Recovery to minimise reconstruction loss."""
        params = (
            list(self.model.embedder.parameters())
            + list(self.model.recovery.parameters())
        )
        optimiser = torch.optim.Adam(
            params, lr=self.lr, betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.p1_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, _, _, _ in loader:
                h = self.model.embed(x_batch)
                x_hat = self.model.recover(h)

                loss = criterion(x_hat, x_batch)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.history.phase1_recon.append(avg_loss)

            if avg_loss < best_loss - self.min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                log.info(f"[{MODULE_LABEL}]   Phase 1 early stop at epoch "
                         f"{epoch + 1} (loss={avg_loss:.6f})")
                break

            if (epoch + 1) % 20 == 0:
                log.info(f"[{MODULE_LABEL}]   P1 epoch {epoch + 1:4d}: "
                         f"recon_loss={avg_loss:.6f}")

    # ═════════════════════════════════════════════════════════════════════
    # Phase 2: Supervisor (S, with E frozen)
    # ═════════════════════════════════════════════════════════════════════

    def _train_phase2(self, loader: DataLoader) -> None:
        """Train Supervisor to predict next-step latent (E frozen)."""
        # Freeze embedder
        for p in self.model.embedder.parameters():
            p.requires_grad = False

        optimiser = torch.optim.Adam(
            self.model.supervisor.parameters(),
            lr=self.lr, betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.p2_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, _, _, _ in loader:
                with torch.no_grad():
                    h_real = self.model.embed(x_batch)

                h_supervised = self.model.supervise(h_real)

                # Predict next step: S(h[:-1]) should match h[1:]
                loss = criterion(h_supervised[:, :-1, :], h_real[:, 1:, :])

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.history.phase2_supervised.append(avg_loss)

            if avg_loss < best_loss - self.min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                log.info(f"[{MODULE_LABEL}]   Phase 2 early stop at epoch "
                         f"{epoch + 1} (loss={avg_loss:.6f})")
                break

            if (epoch + 1) % 20 == 0:
                log.info(f"[{MODULE_LABEL}]   P2 epoch {epoch + 1:4d}: "
                         f"sup_loss={avg_loss:.6f}")

        # Unfreeze embedder for Phase 3
        for p in self.model.embedder.parameters():
            p.requires_grad = True

    # ═════════════════════════════════════════════════════════════════════
    # Phase 3: Joint Adversarial (G + D + E + R + S)
    # ═════════════════════════════════════════════════════════════════════

    def _train_phase3(self, loader: DataLoader, seq_len: int) -> None:
        """
        Joint adversarial training.

        Generator loss = adversarial + lambda_sup * supervised + lambda_gen * moment
        Discriminator loss = binary cross-entropy (real vs fake)
        Autoencoder loss = lambda_recon * reconstruction
        """
        # Separate optimisers for G, D, and E+R
        opt_g = torch.optim.Adam(
            list(self.model.generator.parameters())
            + list(self.model.supervisor.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )
        opt_d = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.lr, betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )
        opt_er = torch.optim.Adam(
            list(self.model.embedder.parameters())
            + list(self.model.recovery.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )

        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()

        best_g_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.p3_epochs):
            d_losses, g_losses = [], []
            r_losses, s_losses = [], []

            for x_batch, emo, sev, vrb in loader:
                batch_size = x_batch.size(0)
                cond = self.model.encode_condition(emo, sev, vrb)

                # ─── Embed real data ─────────────────────────────
                h_real = self.model.embed(x_batch)
                h_sup_real = self.model.supervise(h_real)

                # ─── Generate fake data ──────────────────────────
                z = torch.randn(
                    batch_size, seq_len, self.model.noise_dim,
                    device=self.device,
                )
                h_fake = self.model.generate(z, cond)
                h_sup_fake = self.model.supervise(h_fake)

                # ═══════════════════════════════════════════════
                # Train Discriminator
                # ═══════════════════════════════════════════════
                # Detach cond for D — D shouldn't update condition encoder
                cond_d = cond.detach()
                d_real = self.model.discriminate(h_real.detach(), cond_d)
                d_fake = self.model.discriminate(h_fake.detach(), cond_d)

                ones = torch.ones_like(d_real)
                zeros = torch.zeros_like(d_fake)

                d_loss = bce(d_real, ones) + bce(d_fake, zeros)

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                # ═══════════════════════════════════════════════
                # Train Generator + Supervisor
                # ═══════════════════════════════════════════════
                # Re-generate (fresh forward pass through G for grad flow)
                h_fake = self.model.generate(z, cond)
                h_sup_fake = self.model.supervise(h_fake)

                # Adversarial: fool D
                d_fake_for_g = self.model.discriminate(h_fake, cond)
                g_adv_loss = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))

                # Supervised: S(h_fake) matches temporal dynamics
                g_sup_loss = mse(
                    h_sup_fake[:, :-1, :], h_fake[:, 1:, :]
                )

                # Moment matching: mean + variance of real vs fake latent
                # Detach h_real — we only want gradients through h_fake
                g_moment_loss = (
                    torch.abs(h_real.detach().mean(dim=0) - h_fake.mean(dim=0)).mean()
                    + torch.abs(
                        h_real.detach().var(dim=0) - h_fake.var(dim=0)
                    ).mean()
                )

                g_loss = (
                    g_adv_loss
                    + self.lambda_sup * g_sup_loss
                    + self.lambda_gen * g_moment_loss
                )

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                # ═══════════════════════════════════════════════
                # Train Autoencoder (E + R)
                # ═══════════════════════════════════════════════
                h_real = self.model.embed(x_batch)
                x_hat = self.model.recover(h_real)
                er_loss = self.lambda_recon * mse(x_hat, x_batch)

                opt_er.zero_grad()
                er_loss.backward()
                opt_er.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                r_losses.append(er_loss.item())
                s_losses.append(g_sup_loss.item())

            avg_d = np.mean(d_losses) if d_losses else 0
            avg_g = np.mean(g_losses) if g_losses else 0
            avg_r = np.mean(r_losses) if r_losses else 0
            avg_s = np.mean(s_losses) if s_losses else 0

            self.history.phase3_discriminator.append(avg_d)
            self.history.phase3_generator.append(avg_g)
            self.history.phase3_reconstruction.append(avg_r)
            self.history.phase3_supervised.append(avg_s)

            # Early stopping on generator loss
            if avg_g < best_g_loss - self.min_delta:
                best_g_loss = avg_g
                patience_counter = 0
                self.history.best_epoch = epoch + 1
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                log.info(f"[{MODULE_LABEL}]   Phase 3 early stop at epoch "
                         f"{epoch + 1} (g_loss={avg_g:.6f})")
                break

            if (epoch + 1) % 20 == 0:
                log.info(
                    f"[{MODULE_LABEL}]   P3 epoch {epoch + 1:4d}: "
                    f"D={avg_d:.4f}  G={avg_g:.4f}  "
                    f"R={avg_r:.4f}  S={avg_s:.4f}"
                )
