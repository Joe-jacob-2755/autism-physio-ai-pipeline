"""
=============================================================================
MODULE 5B - DATA AUGMENTATION  |  config.py
=============================================================================
Configuration for class imbalance detection and Conditional Transformer
TimeGAN (CT-TimeGAN) based data augmentation.

This module is OPTIONAL and runs ONLY on TRAINING DATA.
It sits between M5 (Preprocessing) and M6 (Feature Engineering).

Pipeline position:
  M5 (clean+filter) -> [M5B augment if imbalanced] -> M6 (feature eng)

Key design decisions:
  1. Imbalance detection uses event-count distribution across training users
  2. Augmentation generates synthetic EVENT SEGMENTS (not full sessions)
  3. Each signal type has its own TGAN (different sampling rates + dynamics)
  4. Generated signals are sanity-checked before merging
  5. Human-in-the-loop checkpoints at: detection, planning, acceptance
=============================================================================
"""
from __future__ import annotations

MODULE_VERSION = "1.0.0"
MODULE_LABEL = "M5B"
OUTPUT_ROOT = "outputs"

# ═════════════════════════════════════════════════════════════════════════════
# IMBALANCE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

# Imbalance ratio = N_majority / N_minority
# Classification follows standard ML convention (He & Garcia, 2009)
IMBALANCE_THRESHOLDS = {
    "balanced":  1.5,    # IR < 1.5  -> no augmentation needed
    "mild":      3.0,    # 1.5 <= IR < 3.0  -> optional augmentation
    "moderate": 10.0,    # 3.0 <= IR < 10.0 -> recommended augmentation
    # IR >= 10.0  -> severe, augmentation strongly recommended
}

# Minimum samples per class below which augmentation is recommended
# regardless of imbalance ratio (absolute floor)
MIN_SAMPLES_PER_CLASS = 5

# Minimum total training events needed before augmentation is attempted
# (TGAN needs enough data to learn meaningful patterns)
MIN_TOTAL_EVENTS_FOR_TGAN = 20

# ═════════════════════════════════════════════════════════════════════════════
# AUGMENTATION PLANNING
# ═════════════════════════════════════════════════════════════════════════════

# Target strategy: how to decide how many synthetic samples to generate
# Options:
#   "equalize"       -> bring all classes to the majority count
#   "median"         -> bring all below-median classes to the median
#   "minimum_ratio"  -> bring all classes to within MAX_RATIO of majority
#   "custom"         -> user specifies per-class targets
DEFAULT_STRATEGY = "median"

# For "minimum_ratio" strategy: maximum acceptable IR after augmentation
MAX_RATIO_AFTER_AUGMENTATION = 2.0

# Maximum synthetic-to-real ratio per class
# Guards against generating too much synthetic data relative to real data
MAX_SYNTHETIC_RATIO = 3.0

# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL SPECIFICATIONS (must match M5 output)
# ═════════════════════════════════════════════════════════════════════════════

SIGNAL_SPECS = {
    "EDA": {
        "sampling_rate": 4,
        "value_cols": ["EDA_uS"],
        "range": (0.01, 30.0),
        "segment_samples": None,  # computed from event duration
    },
    "BVP": {
        "sampling_rate": 64,
        "value_cols": ["BVP_nT"],
        "range": (-300.0, 300.0),
        "segment_samples": None,
    },
    "IBI": {
        "sampling_rate": None,    # event-based
        "value_cols": ["IBI_ms"],
        "range": (300.0, 1500.0),
        "segment_samples": None,
    },
    "ST": {
        "sampling_rate": 4,
        "value_cols": ["ST_degC"],
        "range": (25.0, 40.0),
        "segment_samples": None,
    },
    "ACC": {
        "sampling_rate": 32,
        "value_cols": ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"],
        "range": (-4.0, 4.0),
        "segment_samples": None,
    },
}

# Fixed segment length for TGAN training (seconds)
# Event segments are resampled/padded to this length
SEGMENT_DURATION_S = 30.0

# Pre-event baseline context included in each segment (seconds)
BASELINE_CONTEXT_S = 5.0

# ═════════════════════════════════════════════════════════════════════════════
# CT-TimeGAN ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════
#
# Architecture: Conditional TimeGAN with Transformer backbone
# Based on: Yoon et al. (2019) "Time-series Generative Adversarial Networks"
#           with Transformer replacing GRU (Vaswani et al., 2017)
#
# 5 Networks:
#   1. Embedder E:     raw signal -> latent space
#   2. Recovery R:     latent space -> reconstructed signal
#   3. Generator G:    noise + condition -> fake latent
#   4. Discriminator D: latent + condition -> real/fake
#   5. Supervisor S:   latent -> next-step latent (temporal dynamics)
#
# Conditioning: emotion label + autism_severity + verbal_status
# Per-signal: separate TGAN per signal type (different dynamics)
# ═════════════════════════════════════════════════════════════════════════════

TGAN = {
    # Latent / hidden dimensions
    "hidden_dim": 64,
    "noise_dim": 32,

    # Condition embedding
    "n_emotion_classes": 10,      # 6 affective + 4 physiological needs
    "n_severity_levels": 3,       # Low / Medium / Severe
    "n_verbal_levels": 3,         # Verbal / Minimally verbal / Non-verbal
    "condition_dim": 32,          # total condition embedding dim

    # Transformer architecture
    "n_transformer_layers": 2,
    "n_attention_heads": 4,
    "feedforward_dim": 128,
    "dropout": 0.1,

    # Training
    "batch_size": 16,
    "learning_rate": 1e-4,
    "beta1": 0.5,                 # Adam beta1 (GAN standard)
    "beta2": 0.999,               # Adam beta2
    "weight_decay": 1e-5,

    # Training schedule (epochs per phase)
    "phase1_epochs": 100,         # autoencoder (E+R)
    "phase2_epochs": 100,         # supervisor (S)
    "phase3_epochs": 200,         # joint adversarial (G+D+E+R+S)

    # Loss weights (Phase 3 joint training)
    "lambda_reconstruction": 10.0,
    "lambda_supervised": 1.0,
    "lambda_generator": 1.0,

    # Early stopping
    "patience": 20,               # epochs without improvement
    "min_delta": 0.001,           # minimum improvement threshold
}

# ═════════════════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ═════════════════════════════════════════════════════════════════════════════

SANITY = {
    # Physiological range check: reject if > X% of samples out of range
    "max_oor_pct": 5.0,

    # Statistical similarity: KS test p-value threshold
    # p > threshold means distributions are similar (good)
    "ks_test_alpha": 0.05,

    # Maximum Mean Discrepancy (MMD) threshold
    # Lower is better; reject if MMD > threshold
    "mmd_threshold": 0.5,

    # Temporal autocorrelation check: lag-1 ACF difference
    # reject if |ACF_real - ACF_synthetic| > threshold
    "acf_diff_threshold": 0.3,

    # Discriminative score: if a classifier can distinguish real vs fake
    # with accuracy > threshold, the synthetic data is too distinguishable
    "discriminative_accuracy_threshold": 0.85,

    # Minimum acceptance rate: fraction of generated segments passing
    # all sanity checks. If below this, warn the user.
    "min_acceptance_rate": 0.7,
}

# ═════════════════════════════════════════════════════════════════════════════
# HUMAN-IN-THE-LOOP
# ═════════════════════════════════════════════════════════════════════════════

# In auto-pipeline mode, these thresholds determine automatic decisions.
# In interactive mode, the user is always prompted.
AUTO_AUGMENT_THRESHOLD = "moderate"   # auto-augment if IR >= moderate
AUTO_ACCEPT_SANITY_RATE = 0.8        # auto-accept if >= 80% pass sanity

# ═════════════════════════════════════════════════════════════════════════════
# EMOTION AND DEMOGRAPHIC ENCODING
# ═════════════════════════════════════════════════════════════════════════════

EMOTION_TO_IDX = {
    "Happy": 0, "Anger": 1, "Fear": 2, "Disgust": 3, "Sad": 4,
    "Surprise": 5, "Hunger": 6, "Thirst": 7, "Toilet": 8, "Tired": 9,
}
IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}

SEVERITY_TO_IDX = {"Low": 0, "Medium": 1, "Severe": 2}
VERBAL_TO_IDX = {"Verbal": 0, "Minimally verbal": 1, "Non-verbal": 2}

# ═════════════════════════════════════════════════════════════════════════════
# SMOTE (feature-level augmentation — alternative to CT-TimeGAN)
# ═════════════════════════════════════════════════════════════════════════════
#
# SMOTE operates on extracted feature vectors (post M6) rather than raw
# signals. It is faster and deterministic but cannot preserve temporal
# signal morphology. Use as complement to or fallback for CT-TimeGAN.
#
# Methods:
#   "smote"             -> standard SMOTE (Chawla et al., 2002)
#   "borderline_smote"  -> focus on decision boundary samples (Han et al., 2005)
#   "adasyn"            -> adaptive synthetic (He et al., 2008)
#   "random_oversample" -> duplicate minority samples (no synthesis)
# ═════════════════════════════════════════════════════════════════════════════

SMOTE = {
    "default_method": "smote",
    "default_k_neighbors": 5,
    "min_k_neighbors": 1,
    "min_samples_for_smote": 2,       # Need >= k+1 samples per class
    "min_samples_for_adasyn": 6,      # ADASYN needs more boundary diversity
    "min_samples_for_borderline": 6,  # BorderlineSMOTE needs boundary samples
}

# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

FLOAT_FMT = "%.6f"
PLOT_DPI = 150
