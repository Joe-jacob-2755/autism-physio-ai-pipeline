"""
=============================================================================
MODULE 4 – EXPLORATORY DATA ANALYSIS  |  config.py
=============================================================================
Configuration for EDA following IBM Data Science Methodology.

This module runs on TRAINING DATA ONLY (after M3 data splitting).
Test data is never seen during exploratory analysis.

IBM EDA Phases implemented:
  Phase 1: Data Understanding — shape, types, summary statistics
  Phase 2: Data Quality Assessment — missing, outliers, consistency
  Phase 3: Distribution Analysis — skewness, normality, transforms
  Phase 4: Univariate Analysis — per-signal boxplots, histograms
  Phase 5: Bivariate Analysis — correlations, group comparisons
  Phase 6: Temporal Analysis — signal dynamics, event response patterns
  Phase 7: Key Findings & Recommendations — preprocessing guidance
=============================================================================
"""
from __future__ import annotations
from pathlib import Path

MODULE_VERSION = "1.0.0"
MODULE_LABEL = "M4"
OUTPUT_ROOT = "outputs"

# ── Signal metadata ───────────────────────────────────────────────────────────
SIGNAL_UNITS = {
    "EDA": "uS", "BVP": "nT", "IBI": "ms",
    "ST": "degC", "ACC": "g",
}
SIGNAL_COLORS = {
    "EDA": "#2ECC71", "BVP": "#E74C3C", "IBI": "#9B59B6",
    "ST": "#F39C12", "ACC": "#3498DB",
}
VALUE_COLS = {
    "EDA": ["EDA_uS"],
    "BVP": ["BVP_nT"],
    "IBI": ["IBI_ms"],
    "ST": ["ST_degC"],
    "ACC": ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"],
}

# ── Target category colours ───────────────────────────────────────────────────
CATEGORY_COLORS = {
    "affective": "#3498DB",
    "physiological_need": "#27AE60",
    "behavioural": "#C0392B",
    "baseline": "#95A5A6",
}
TARGET_COLORS = {
    "Happy": "#FFD700", "Anger": "#DC143C", "Fear": "#8B008B",
    "Disgust": "#6B8E23", "Sad": "#4169E1", "Surprise": "#FF8C00",
    "Hunger": "#FF6347", "Thirst": "#00BFFF", "Toilet": "#A0522D",
    "Tired": "#708090", "SIB": "#8B0000", "ATO": "#FF4500",
    "GAB": "#FF6347", "baseline": "#AAAAAA",
}

# ── Statistical analysis ──────────────────────────────────────────────────────
ALPHA = 0.05        # significance level
MIN_SAMPLES_PER_GROUP = 3         # minimum windows per target for stats
EFFECT_SIZE_THRESHOLDS = {        # Cohen's d / eta^2 interpretation
    "small": 0.2, "medium": 0.5, "large": 0.8,
}

# ── Temporal dynamics ─────────────────────────────────────────────────────────
BASELINE_WINDOW_S = 30.0    # seconds of pre-event baseline to compare against
RETURN_TOLERANCE = 0.10    # signal "returned" if within 10% of pre-event median
SUBSIDE_THRESHOLD = 0.50    # "subside" = returned to 50% of peak rise

# ── Adaptive threshold ────────────────────────────────────────────────────────
# Default: flag if signal deviates > N x MAD from running median for > T seconds
DEFAULT_THRESHOLD_WINDOW_S = 300.0   # 5-minute running window (user-adjustable)
DEFAULT_THRESHOLD_MAD_FACTOR = 3.0    # N x MAD deviation trigger
DEFAULT_THRESHOLD_SUSTAIN_S = 30.0   # must sustain for this long to fire

# ── Output formatting ──────────────────────────────────────────────────────────
PLOT_DPI = 150
FLOAT_FMT = "%.4f"

# ── Feature columns to exclude from analysis (metadata) ──────────────────────
META_COLS = {
    "session_id", "user_id", "window_start_s", "window_end_s",
    "target_label", "event_id", "category",
    "gender", "ethnicity", "autism_severity",
    "verbal_status", "comorbidity",
    "gender_enc", "ethnicity_enc", "autism_severity_enc",
    "verbal_status_enc", "comorbidity_enc",
}
