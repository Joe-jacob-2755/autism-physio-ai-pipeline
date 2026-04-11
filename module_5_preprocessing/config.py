"""
=============================================================================
MODULE 5 – DATA PREPROCESSING  |  config.py
=============================================================================
Centralised configuration for signal cleaning and filtering.

Feature extraction, normalisation, and encoding config has been moved to
Module 6 (Feature Engineering) config.py.

=============================================================================
"""
from __future__ import annotations

MODULE_VERSION = "2.0.0"
MODULE_LABEL = "M5"
OUTPUT_ROOT = "outputs"

# ── Signal specifications (must match Module 2A) ────────────────────────
SAMPLING_RATES: dict = {
    "EDA": 4,
    "BVP": 64,
    "IBI": None,    # event-based
    "ST": 4,
    "ACC_X": 32,
    "ACC_Y": 32,
    "ACC_Z": 32,
    "ACC": 32,      # alias used in combined ACC file
}

# Column name -> signal key mapping (from Module 2A CSV headers)
SIGNAL_VALUE_COLUMNS: dict = {
    "EDA": ["EDA_uS"],
    "BVP": ["BVP_nT"],
    "IBI": ["IBI_ms"],
    "ST": ["ST_degC"],
    "ACC": ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"],
}

# ── Cleaning thresholds ───────────────────────────────────────────────────
MISSING_DISCARD_THRESHOLD: float = 0.70  # Discard signal if >70% missing
MISSING_FILL_METHOD: str = "linear"   # interpolation method for <70%
FLAT_SIGNAL_STD_THRESHOLD: float = 1e-5  # Flag as flatlined

# ── Filter defaults ───────────────────────────────────────────────────────
FILTER_DEFAULTS: dict = {
    "EDA": {
        "type": "lowpass",
        "cutoff_hz": 1.0,
        "order": 4,
        "hampel_window": 5,
        "hampel_sigma": 3.0,
    },
    "BVP": {
        "type": "bandpass",
        "lowcut_hz": 0.5,
        "highcut_hz": 8.0,
        "order": 4,
        "hampel_window": 11,
        "hampel_sigma": 3.0,
    },
    "IBI": {
        "type": "none",
        "hampel_window": 5,
        "hampel_sigma": 3.0,
    },
    "ST": {
        "type": "lowpass",
        "cutoff_hz": 0.1,
        "order": 2,
        "hampel_window": 7,
        "hampel_sigma": 3.0,
    },
    "ACC": {
        "type": "bandpass",
        "lowcut_hz": 0.1,
        "highcut_hz": 15.0,
        "order": 4,
        "hampel_window": 9,
        "hampel_sigma": 3.0,
    },
}

# Kalman filter defaults (per-signal noise covariances)
KALMAN_DEFAULTS: dict = {
    "EDA": {"observation_noise": 0.10, "process_noise": 0.01},
    "BVP": {"observation_noise": 5.00, "process_noise": 0.50},
    "IBI": {"observation_noise": 10.00, "process_noise": 1.00},
    "ST": {"observation_noise": 0.02, "process_noise": 0.001},
    "ACC": {"observation_noise": 0.05, "process_noise": 0.01},
}

# ── Output configuration ─────────────────────────────────────────────────
PLOT_DPI: int = 100
PLOT_STYLE: str = "seaborn-v0_8-whitegrid"
FLOAT_FMT: str = "%.6f"
