"""
=============================================================================
MODULE 6 – FEATURE ENGINEERING  |  config.py
=============================================================================
Centralised configuration for feature extraction, normalisation,
encoding, feature selection, and dimensionality reduction.

Moved from Module 5 (preprocessing) during the pipeline restructuring.
=============================================================================
"""
from __future__ import annotations

MODULE_VERSION = "1.0.0"
MODULE_LABEL = "M6"
OUTPUT_ROOT = "outputs"

# ── Signal specifications (must match Module 2A / Module 3) ──────────────
SAMPLING_RATES: dict = {
    "EDA": 4,
    "BVP": 64,
    "IBI": None,    # event-based
    "ST": 4,
    "ACC_X": 32,
    "ACC_Y": 32,
    "ACC_Z": 32,
    "ACC": 32,
}

# Column name -> signal key mapping
SIGNAL_VALUE_COLUMNS: dict = {
    "EDA": ["EDA_uS"],
    "BVP": ["BVP_nT"],
    "IBI": ["IBI_ms"],
    "ST": ["ST_degC"],
    "ACC": ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"],
}

# ── Feature extraction windows ────────────────────────────────────────────
WINDOW_SIZE_S: float = 60.0    # Feature extraction window (seconds)
WINDOW_OVERLAP: float = 0.50   # 50% overlap between windows
MIN_WINDOW_SAMPLES: int = 10   # Minimum samples to compute features

# Frequency band definitions for spectral features
FREQ_BANDS: dict = {
    "EDA": {
        "vlf": (0.000, 0.050),
        "scr": (0.050, 0.250),
    },
    "IBI": {
        "vlf": (0.000, 0.04),
        "lf": (0.040, 0.15),
        "hf": (0.150, 0.40),
    },
    "ACC": {
        "sway": (0.0, 1.0),
        "voluntary": (1.0, 3.0),
        "rapid": (3.0, 8.0),
        "tremor": (8.0, 16.0),
    },
    "ST": {
        "ultra_slow": (0.000, 0.003),
        "slow": (0.003, 0.040),
    },
    "BVP": {
        "cardiac": (0.50, 3.50),
        "respiratory": (0.15, 0.40),
    },
}

# ── Standardisation ──────────────────────────────────────────────────────
# RobustScaler: median + IQR — resistant to physiological outliers
# See normaliser.py docstring for full rationale.
SCALER_TYPE: str = "robust"   # Options: "robust" | "standard" | "minmax"

# ── Demographic feature encoding ────────────────────────────────────────
# Ordinal encoding — clinically meaningful ordering
DEMOGRAPHIC_ENCODINGS: dict = {
    "gender": {
        "Male": 0, "Female": 1, "Non-binary": 2,
    },
    "autism_severity": {
        "Low": 1, "Medium": 2, "Severe": 3,
    },
    "verbal_status": {
        "Verbal": 0, "Minimally verbal": 1, "Non-verbal": 2,
    },
    "comorbidity": {
        "Yes": 1, "No": 0,
    },
    "ethnicity": {
        "White British": 0,
        "Asian / Asian British": 1,
        "Black / African / Caribbean": 2,
        "Mixed / Multiple": 3,
        "Other": 4,
    },
}

# Columns that should never be scaled or selected as features
META_COLS: set = {
    "session_id", "user_id", "window_start_s", "window_end_s",
    "target_label", "event_id", "category",
    "age", "gender", "gender_enc", "ethnicity", "ethnicity_enc",
    "autism_severity", "autism_severity_enc",
    "verbal_status", "verbal_status_enc",
    "comorbidity", "comorbidity_enc",
}

# Categorical demographic columns that need one-hot encoding
DEMOGRAPHIC_CATEGORICAL_COLS: list = [
    "gender", "ethnicity", "autism_severity", "verbal_status", "comorbidity",
]

# ── Feature selection defaults ───────────────────────────────────────────
DEFAULT_TOP_N_FEATURES: int = 20   # Default number of top features to select
FEATURE_SELECTION_METHODS: list = [
    "mutual_information",
    "random_forest",
    "f_statistic",
    "variance_threshold",
]

# ── Dimensionality reduction defaults ────────────────────────────────────
DEFAULT_PCA_VARIANCE: float = 0.95   # Retain 95% explained variance
DEFAULT_N_PLS_COMPONENTS: int = 10   # PLS components for VIP calculation

# ── Output configuration ─────────────────────────────────────────────────
PLOT_DPI: int = 100
FLOAT_FMT: str = "%.6f"
