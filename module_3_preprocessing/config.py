"""
=============================================================================
MODULE 3 – DATA PREPROCESSING  |  config.py
=============================================================================
Centralised configuration for all preprocessing parameters.

Covers:
  - Signal channel specifications (aligned with Module 1A / 2)
  - Cleaning thresholds
  - Filter default parameters
  - Feature extraction settings
  - Output paths

=============================================================================
"""
from __future__ import annotations
from pathlib import Path

MODULE_VERSION = "1.0.0"
MODULE_LABEL   = "M3"
OUTPUT_ROOT    = "outputs"

# ── Signal specifications (must match Module 1A) ──────────────────────────
SAMPLING_RATES: dict = {
    "EDA":   4,
    "BVP":   64,
    "IBI":   None,    # event-based
    "ST":    4,
    "ACC_X": 32,
    "ACC_Y": 32,
    "ACC_Z": 32,
    "ACC":   32,      # alias used in combined ACC file
}

# Column name → signal key mapping (from Module 1A / 2 CSV headers)
SIGNAL_VALUE_COLUMNS: dict = {
    "EDA":   ["EDA_uS"],
    "BVP":   ["BVP_nT"],
    "IBI":   ["IBI_ms"],
    "ST":    ["ST_degC"],
    "ACC":   ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"],
}

# ── Cleaning thresholds ───────────────────────────────────────────────────
MISSING_DISCARD_THRESHOLD: float = 0.70  # Discard signal if >70% missing
MISSING_FILL_METHOD: str         = "linear"   # interpolation method for <70%
FLAT_SIGNAL_STD_THRESHOLD: float = 1e-5  # Flag as flatlined

# ── Filter defaults ───────────────────────────────────────────────────────
FILTER_DEFAULTS: dict = {
    "EDA": {
        "type":        "lowpass",
        "cutoff_hz":   1.0,      # Removes noise above 1 Hz from 4 Hz EDA
        "order":       4,
        "hampel_window": 5,
        "hampel_sigma":  3.0,
    },
    "BVP": {
        "type":        "bandpass",
        "lowcut_hz":   0.5,      # Remove baseline wander
        "highcut_hz":  8.0,      # Keep cardiac harmonics, remove HF noise
        "order":       4,
        "hampel_window": 11,
        "hampel_sigma":  3.0,
    },
    "IBI": {
        "type":        "none",   # IBI filtered via ectopic removal
        "hampel_window": 5,
        "hampel_sigma":  3.0,
    },
    "ST": {
        "type":        "lowpass",
        "cutoff_hz":   0.1,      # ST changes very slowly
        "order":       2,
        "hampel_window": 7,
        "hampel_sigma":  3.0,
    },
    "ACC": {
        "type":        "bandpass",
        "lowcut_hz":   0.1,      # Remove gravity DC component
        "highcut_hz":  15.0,     # Keep up to tremor range
        "order":       4,
        "hampel_window": 9,
        "hampel_sigma":  3.0,
    },
}

# Kalman filter defaults (per-signal noise covariances)
KALMAN_DEFAULTS: dict = {
    "EDA":   {"observation_noise": 0.10,  "process_noise": 0.01},
    "BVP":   {"observation_noise": 5.00,  "process_noise": 0.50},
    "IBI":   {"observation_noise": 10.00, "process_noise": 1.00},
    "ST":    {"observation_noise": 0.02,  "process_noise": 0.001},
    "ACC":   {"observation_noise": 0.05,  "process_noise": 0.01},
}

# ── Feature extraction windows ────────────────────────────────────────────
WINDOW_SIZE_S: float    = 60.0    # Feature extraction window (seconds)
WINDOW_OVERLAP: float   = 0.50    # 50% overlap between windows
MIN_WINDOW_SAMPLES: int = 10      # Minimum samples to compute features

# Frequency band definitions for spectral features
FREQ_BANDS: dict = {
    "EDA": {
        "vlf":  (0.000, 0.050),
        "scr":  (0.050, 0.250),
    },
    "IBI": {
        "vlf":  (0.000,  0.04),
        "lf":   (0.040,  0.15),
        "hf":   (0.150,  0.40),
    },
    "ACC": {
        "sway":      (0.0,  1.0),
        "voluntary": (1.0,  3.0),
        "rapid":     (3.0,  8.0),
        "tremor":    (8.0, 16.0),
    },
    "ST": {
        "ultra_slow": (0.000, 0.003),
        "slow":       (0.003, 0.040),
    },
    "BVP": {
        "cardiac":    (0.50,  3.50),
        "respiratory":(0.15,  0.40),
    },
}

# ── Standardisation ───────────────────────────────────────────────────────
# RobustScaler is used instead of StandardScaler or MinMaxScaler because:
#   1. Physiological signals contain frequent outliers (artefacts, SCR peaks,
#      motion bursts) that inflate the mean and inflate std in StandardScaler.
#   2. RobustScaler uses the median (Q2) and IQR (Q1–Q3), which are
#      resistant to outliers — critical for EDA, BVP, and ACC signals.
#   3. Children's physiological data shows high inter-subject variability;
#      RobustScaler preserves relative within-subject structure while
#      normalising across users.
#   4. It does not collapse the distribution to [0,1], so outlier
#      information is preserved (unlike MinMaxScaler) — useful for models
#      that need to detect anomalous events.
SCALER_TYPE: str = "robust"   # Options: "robust" | "standard" | "minmax"

# ── Demographic feature encoding ──────────────────────────────────────────
DEMOGRAPHIC_ENCODINGS: dict = {
    "gender": {
        "Male": 0, "Female": 1, "Non-binary": 2
    },
    "autism_severity": {
        "Low": 1, "Medium": 2, "Severe": 3
    },
    "verbal_status": {
        "Verbal": 0, "Minimally verbal": 1, "Non-verbal": 2
    },
    "comorbidity": {
        "Yes": 1, "No": 0
    },
    "ethnicity": {
        "White British": 0,
        "Asian / Asian British": 1,
        "Black / African / Caribbean": 2,
        "Mixed / Multiple": 3,
        "Other": 4,
    },
}

# ── Output configuration ──────────────────────────────────────────────────
PLOT_DPI: int    = 150
PLOT_STYLE: str  = "seaborn-v0_8-whitegrid"
FLOAT_FMT: str   = "%.6f"
