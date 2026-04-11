"""
=============================================================================
MODULE 5B - DATA AUGMENTATION  |  tgan_model.py
=============================================================================
Conditional Transformer TimeGAN (CT-TimeGAN) architecture.

Based on:
  Yoon et al. (2019) — "Time-series Generative Adversarial Networks"
  Vaswani et al. (2017) — "Attention Is All You Need"

Architecture:
  5 networks with Transformer encoder backbone:
    1. Embedder E:      raw signal -> latent space
    2. Recovery R:       latent space -> reconstructed signal
    3. Generator G:      noise + condition -> fake latent
    4. Discriminator D:  latent + condition -> real/fake score
    5. Supervisor S:     latent(t) -> latent(t+1) (temporal dynamics)

  Conditioning vector:
    emotion_label (10 classes) + autism_severity (3) + verbal_status (3)
    Each one-hot encoded and projected to a shared condition_dim embedding.

  Per-signal:
    Separate CT-TimeGAN instance per signal type (EDA, BVP, IBI, ST, ACC)
    because each has different sampling rates, dynamics, and dimensionality.
=============================================================================
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from config import TGAN


# ═════════════════════════════════════════════════════════════════════════════
# Positional Encoding (sinusoidal, Vaswani et al.)
# ═════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ═════════════════════════════════════════════════════════════════════════════
# Condition Encoder
# ═════════════════════════════════════════════════════════════════════════════

class ConditionEncoder(nn.Module):
    """
    Encode categorical conditions into a dense vector.

    Inputs: (emotion_idx, severity_idx, verbal_idx) as integer tensors.
    Output: condition vector of size condition_dim, broadcast across time steps.
    """

    def __init__(
        self,
        n_emotions: int = TGAN["n_emotion_classes"],
        n_severity: int = TGAN["n_severity_levels"],
        n_verbal: int = TGAN["n_verbal_levels"],
        condition_dim: int = TGAN["condition_dim"],
    ):
        super().__init__()
        embed_per = condition_dim // 3
        remainder = condition_dim - 3 * embed_per

        self.emotion_embed = nn.Embedding(n_emotions, embed_per + remainder)
        self.severity_embed = nn.Embedding(n_severity, embed_per)
        self.verbal_embed = nn.Embedding(n_verbal, embed_per)

        total_in = (embed_per + remainder) + embed_per + embed_per
        self.project = nn.Sequential(
            nn.Linear(total_in, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(
        self,
        emotion: torch.Tensor,    # (batch,) int
        severity: torch.Tensor,   # (batch,) int
        verbal: torch.Tensor,     # (batch,) int
    ) -> torch.Tensor:
        """Returns (batch, condition_dim)."""
        e = self.emotion_embed(emotion)
        s = self.severity_embed(severity)
        v = self.verbal_embed(verbal)
        cat = torch.cat([e, s, v], dim=-1)
        return self.project(cat)


# ═════════════════════════════════════════════════════════════════════════════
# Transformer Block (shared backbone for all 5 networks)
# ═════════════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Transformer encoder stack used as backbone for E, R, G, D, S.

    Parameters
    ----------
    input_dim : int
        Input feature dimension per time step.
    hidden_dim : int
        Internal / output hidden dimension.
    n_layers : int
        Number of Transformer encoder layers.
    n_heads : int
        Number of attention heads.
    ff_dim : int
        Feedforward hidden dimension inside each Transformer layer.
    dropout : float
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int = TGAN["n_transformer_layers"],
        n_heads: int = TGAN["n_attention_heads"],
        ff_dim: int = TGAN["feedforward_dim"],
        dropout: float = TGAN["dropout"],
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)

        Returns
        -------
        (batch, seq_len, hidden_dim)
        """
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.transformer(h)
        return self.output_proj(h)


# ═════════════════════════════════════════════════════════════════════════════
# 5 CT-TimeGAN Networks
# ═════════════════════════════════════════════════════════════════════════════

class Embedder(nn.Module):
    """E: raw signal (batch, seq, n_features) -> latent (batch, seq, hidden)."""

    def __init__(self, n_features: int, hidden_dim: int = TGAN["hidden_dim"]):
        super().__init__()
        self.net = TransformerBlock(n_features, hidden_dim)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x))


class Recovery(nn.Module):
    """R: latent (batch, seq, hidden) -> reconstructed (batch, seq, n_features)."""

    def __init__(self, n_features: int, hidden_dim: int = TGAN["hidden_dim"]):
        super().__init__()
        self.net = TransformerBlock(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.output(self.net(h))


class Generator(nn.Module):
    """
    G: noise + condition -> fake latent.

    Input: (batch, seq, noise_dim) concatenated with condition broadcast.
    Output: (batch, seq, hidden_dim) — fake latent to fool D and match S.
    """

    def __init__(
        self,
        noise_dim: int = TGAN["noise_dim"],
        condition_dim: int = TGAN["condition_dim"],
        hidden_dim: int = TGAN["hidden_dim"],
    ):
        super().__init__()
        self.net = TransformerBlock(noise_dim + condition_dim, hidden_dim)
        self.act = nn.Sigmoid()

    def forward(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (batch, seq, noise_dim)
        cond : (batch, condition_dim) — broadcast across time steps

        Returns
        -------
        (batch, seq, hidden_dim)
        """
        seq_len = z.size(1)
        cond_expanded = cond.unsqueeze(1).expand(-1, seq_len, -1)
        inp = torch.cat([z, cond_expanded], dim=-1)
        return self.act(self.net(inp))


class Discriminator(nn.Module):
    """
    D: latent + condition -> real/fake score per time step.

    Output: (batch, seq, 1) — logits (no sigmoid, use BCEWithLogitsLoss).
    """

    def __init__(
        self,
        hidden_dim: int = TGAN["hidden_dim"],
        condition_dim: int = TGAN["condition_dim"],
    ):
        super().__init__()
        self.net = TransformerBlock(hidden_dim + condition_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self, h: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (batch, seq, hidden_dim) — real or fake latent
        cond : (batch, condition_dim)

        Returns
        -------
        (batch, seq, 1) — logits
        """
        seq_len = h.size(1)
        cond_expanded = cond.unsqueeze(1).expand(-1, seq_len, -1)
        inp = torch.cat([h, cond_expanded], dim=-1)
        return self.output(self.net(inp))


class Supervisor(nn.Module):
    """
    S: latent(t) -> latent(t+1).

    Learns temporal dynamics in latent space.
    Input: (batch, seq, hidden_dim) — latent representations
    Output: (batch, seq, hidden_dim) — predicted next-step latent
    """

    def __init__(self, hidden_dim: int = TGAN["hidden_dim"]):
        super().__init__()
        self.net = TransformerBlock(hidden_dim, hidden_dim)
        self.act = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(h))


# ═════════════════════════════════════════════════════════════════════════════
# CT-TimeGAN (assembled model)
# ═════════════════════════════════════════════════════════════════════════════

class CTTimeGAN(nn.Module):
    """
    Conditional Transformer TimeGAN — all 5 networks + condition encoder.

    One CTTimeGAN instance per signal type. Each operates on segments of
    fixed length (SEGMENT_DURATION_S * sampling_rate samples).

    Parameters
    ----------
    n_features : int
        Number of signal value columns (e.g. 1 for EDA, 3 for ACC).
    hidden_dim : int
        Latent space dimensionality.
    noise_dim : int
        Random noise vector dimensionality for Generator.
    condition_dim : int
        Condition embedding dimensionality.
    n_emotions : int
    n_severity : int
    n_verbal : int
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = TGAN["hidden_dim"],
        noise_dim: int = TGAN["noise_dim"],
        condition_dim: int = TGAN["condition_dim"],
        n_emotions: int = TGAN["n_emotion_classes"],
        n_severity: int = TGAN["n_severity_levels"],
        n_verbal: int = TGAN["n_verbal_levels"],
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim

        # Condition encoder (shared across networks)
        self.condition_encoder = ConditionEncoder(
            n_emotions, n_severity, n_verbal, condition_dim
        )

        # 5 core networks
        self.embedder = Embedder(n_features, hidden_dim)
        self.recovery = Recovery(n_features, hidden_dim)
        self.generator = Generator(noise_dim, condition_dim, hidden_dim)
        self.discriminator = Discriminator(hidden_dim, condition_dim)
        self.supervisor = Supervisor(hidden_dim)

    def encode_condition(
        self,
        emotion: torch.Tensor,
        severity: torch.Tensor,
        verbal: torch.Tensor,
    ) -> torch.Tensor:
        """Encode condition variables. Returns (batch, condition_dim)."""
        return self.condition_encoder(emotion, severity, verbal)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed raw signal into latent space. (batch, seq, feat) -> (batch, seq, hidden)."""
        return self.embedder(x)

    def recover(self, h: torch.Tensor) -> torch.Tensor:
        """Recover signal from latent. (batch, seq, hidden) -> (batch, seq, feat)."""
        return self.recovery(h)

    def supervise(self, h: torch.Tensor) -> torch.Tensor:
        """Predict next-step latent. (batch, seq, hidden) -> (batch, seq, hidden)."""
        return self.supervisor(h)

    def generate(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Generate fake latent from noise + condition."""
        return self.generator(z, cond)

    def discriminate(
        self, h: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Score latent as real/fake. Returns logits (batch, seq, 1)."""
        return self.discriminator(h, cond)

    def generate_signal(
        self,
        z: torch.Tensor,
        emotion: torch.Tensor,
        severity: torch.Tensor,
        verbal: torch.Tensor,
    ) -> torch.Tensor:
        """
        End-to-end generation: noise -> fake latent -> fake signal.

        Parameters
        ----------
        z : (batch, seq, noise_dim)
        emotion, severity, verbal : (batch,) int tensors

        Returns
        -------
        (batch, seq, n_features) — synthetic signal values
        """
        cond = self.encode_condition(emotion, severity, verbal)
        fake_h = self.generate(z, cond)
        supervised_h = self.supervise(fake_h)
        return self.recover(supervised_h)

    def count_parameters(self) -> dict:
        """Count trainable parameters per sub-network."""
        counts = {}
        for name in ["embedder", "recovery", "generator",
                      "discriminator", "supervisor", "condition_encoder"]:
            module = getattr(self, name)
            counts[name] = sum(p.numel() for p in module.parameters()
                               if p.requires_grad)
        counts["total"] = sum(counts.values())
        return counts
