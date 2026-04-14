"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  config.py
=============================================================================
Central configuration: thresholds, clinical cost matrix, robustness
parameters, report styling. Mirrors M7 config constants where needed.
=============================================================================
"""
from __future__ import annotations

# ── Module metadata ─────────────────────────────────────────────────────
MODULE_VERSION = "1.0.0"
MODULE_LABEL = "M8"
OUTPUT_ROOT = "outputs"
RANDOM_SEED = 42

# ── Column definitions (must match M7/M6) ──────────────────────────────
META_COLS = {"session_id", "user_id", "window_start_s", "window_end_s"}
TARGET_COL = "target_label"
SPLIT_COL = "split"
DEMOGRAPHIC_RAW = {
    "age", "gender", "ethnicity",
    "autism_severity", "verbal_status", "comorbidity",
}
DEMOGRAPHIC_ENC = {
    "gender_enc", "autism_severity_enc", "verbal_status_enc",
    "comorbidity_enc", "ethnicity_enc",
}
ALL_NON_FEATURE_COLS = META_COLS | {TARGET_COL, SPLIT_COL} | DEMOGRAPHIC_RAW | DEMOGRAPHIC_ENC

# Known all-NaN columns in simulated data (must match M7 config)
KNOWN_NAN_COLS = {"bvp_resp_mod", "st_asymmetry", "st_pow_ultra_slow", "st_pow_slow"}

# ── Primary metric (matches M7) ────────────────────────────────────────
PRIMARY_METRIC = "f1_weighted"

# ── High-priority states (clinical) ────────────────────────────────────
HIGH_PRIORITY_STATES = {"Fear", "SIB", "Toilet", "Anger"}

# ── Generalization gap thresholds ───────────────────────────────────────
OVERFIT_THRESHOLD = 0.05          # >5% train-test gap flags overfitting
SEVERE_OVERFIT_THRESHOLD = 0.15   # >15% is severe overfitting

# ── Calibration parameters ──────────────────────────────────────────────
ECE_BINS = 15
CALIBRATION_GOOD_THRESHOLD = 0.05  # ECE < 5% is well-calibrated

# ── Clinical cost matrix ────────────────────────────────────────────────
# Higher cost for missing high-priority states (false negatives)
CLINICAL_COST_WEIGHTS = {
    "Fear":    {"fn_cost": 10.0, "fp_cost": 1.0},
    "SIB":     {"fn_cost": 10.0, "fp_cost": 1.0},
    "Toilet":  {"fn_cost": 5.0,  "fp_cost": 1.0},
    "Anger":   {"fn_cost": 8.0,  "fp_cost": 1.0},
    "default": {"fn_cost": 2.0,  "fp_cost": 1.0},
}

# NNS (Number Needed to Screen)
NNS_PREVALENCE_ASSUMED = 0.05  # 5% event prevalence for NNS calc

# ── Robustness testing ──────────────────────────────────────────────────
NOISE_LEVELS = [0.01, 0.05, 0.10, 0.20]
MISSING_FRACTIONS = [0.05, 0.10, 0.20, 0.30]
CHANNEL_DROPOUT_CONFIGS = [
    ["EDA"], ["BVP"], ["ST"], ["ACC"], ["IBI"],
    ["EDA", "BVP"], ["EDA", "BVP", "ST"],
]
ROBUSTNESS_TOP_N = 3  # Only test top-N models for robustness

# Feature-to-signal mapping for channel dropout
SIGNAL_FEATURE_PREFIXES = {
    "EDA": ["eda_"],
    "BVP": ["bvp_"],
    "IBI": ["ibi_", "hrv_"],
    "ST":  ["st_"],
    "ACC": ["acc_"],
}

# ── Equity testing ──────────────────────────────────────────────────────
EQUITY_ALPHA = 0.05
MIN_SUBGROUP_SIZE = 10

# ── Inference profiling ─────────────────────────────────────────────────
PROFILE_N_RUNS = 100
PROFILE_BATCH_SIZES = [1, 16, 64]

# ── Age groups for demographic breakdown ────────────────────────────────
AGE_BINS = [4, 8, 11, 16]
AGE_LABELS = ["5-8", "9-11", "12-15"]

# ── Plotting ────────────────────────────────────────────────────────────
PLOT_DPI = 150
PLOT_FIGSIZE = (10, 8)
PLOT_FIGSIZE_WIDE = (14, 6)

# ── Report styling (matches M4 pattern) ─────────────────────────────────
REPORT = {
    "font": "Calibri",
    "size": 11,
    "heading_rgb": (0x1A, 0x36, 0x5D),
    "subtitle_rgb": (0x4A, 0x55, 0x68),
    "meta_rgb": (0x71, 0x80, 0x96),
    "warning_rgb": (0x9B, 0x21, 0x21),
    "table_style": "Light Grid Accent 1",
    "title_pt": 26,
    "subtitle_pt": 16,
    "table_pt": 9,
    "caption_pt": 9,
    "interp_pt": 10,
    "img_sm": 4.5,
    "img_md": 5.5,
    "img_lg": 6.0,
}

# ── Metric display names ────────────────────────────────────────────────
METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "f1_weighted": "F1 (weighted)",
    "f1_macro": "F1 (macro)",
    "precision_weighted": "Precision (weighted)",
    "recall_weighted": "Recall (weighted)",
    "cohens_kappa": "Cohen's Kappa",
    "auc_roc_macro": "AUC-ROC (macro)",
}

COMPARISON_METRICS = [
    "accuracy", "f1_weighted", "f1_macro",
    "precision_weighted", "recall_weighted",
    "cohens_kappa", "auc_roc_macro",
]
