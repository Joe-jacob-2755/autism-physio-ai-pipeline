"""
=============================================================================
MODULE 3 – DATA SPLITTING  |  config.py
=============================================================================
Centralised configuration for train/validation/test data splitting.

Split is performed at the USER level (not window level) to prevent
data leakage from participant-specific physiological patterns.

This module sits immediately after data acquisition (M1 + M2A).
Raw signals are split BEFORE any preprocessing or feature engineering.
Test data is kept untouched until final model evaluation.

All fitting operations in downstream modules (scalers, PCA, feature
selectors) must be performed on training data ONLY.

=============================================================================
"""
from __future__ import annotations

MODULE_VERSION = "1.0.0"
MODULE_LABEL = "M3"
OUTPUT_ROOT = "outputs"

# ── Default split ratios ─────────────────────────────────────────────────
# User chooses between 2-way (train/test) or 3-way (train/val/test).
# Ratios must sum to 1.0.
DEFAULT_SPLIT_MODE = "three_way"   # "two_way" | "three_way"
DEFAULT_TRAIN_RATIO = 0.60
DEFAULT_VAL_RATIO = 0.20
DEFAULT_TEST_RATIO = 0.20

# 2-way defaults (when validation is not needed)
DEFAULT_TRAIN_RATIO_2WAY = 0.80
DEFAULT_TEST_RATIO_2WAY = 0.20

# ── Stratification ───────────────────────────────────────────────────────
# Split is stratified by these demographic columns to ensure each split
# contains representative participants across severity and verbal status.
# Falls back to unstratified if too few users per stratum.
STRATIFY_COLUMNS = ["autism_severity", "verbal_status"]

# Minimum users per stratum for stratification to be applied.
# If any stratum has fewer users, fall back to unstratified random split.
MIN_USERS_PER_STRATUM = 2

# ── Reproducibility ──────────────────────────────────────────────────────
DEFAULT_SEED = 42

# ── Signal file mappings ────────────────────────────────────────────────
# Raw signal files from M1/M2A (split BEFORE preprocessing)
SIGNAL_FILES = ["EDA.csv", "BVP.csv", "IBI.csv", "ST.csv", "ACC.csv"]
COMBINED_FILE = "combined_signals.csv"

# Legacy: feature subdirs from old pipeline (kept for backward compat)
FEATURE_SUBDIR = "features_raw"
FEATURE_NORM_SUBDIR = "features_normalised"

# ── Columns used for identification and splitting ────────────────────────
USER_COL = "user_id"
SESSION_COL = "session_id"
TARGET_COL = "target_label"

# Demographic columns that carry stratification info
DEMOGRAPHIC_COLS = [
    "age", "gender", "ethnicity",
    "autism_severity", "verbal_status", "comorbidity",
]

# ── Output configuration ────────────────────────────────────────────────
FLOAT_FMT = "%.6f"
