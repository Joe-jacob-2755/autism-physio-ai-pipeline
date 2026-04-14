"""
=============================================================================
MODULE 4 - EXPLORATORY DATA ANALYSIS  |  config.py
=============================================================================
Central configuration: signal metadata, target colours, statistical
parameters, report styling, and the Analysis Logic Table.
=============================================================================
"""
from __future__ import annotations

# ── Module metadata ─────────────────────────────────────────────────────
MODULE_VERSION = "2.0.0"
MODULE_LABEL = "M4"
OUTPUT_ROOT = "outputs"

# ── Signal metadata ─────────────────────────────────────────────────────

SIGNAL_UNITS = {
    "EDA": "uS", "BVP": "nT", "IBI": "ms",
    "ST": "degC", "ACC": "g",
}

SIGNAL_SAMPLING_RATES = {
    "EDA": 4, "BVP": 64, "IBI": None, "ST": 4, "ACC": 32,
}

VALUE_COLS = {
    "EDA": ["EDA_uS"],
    "BVP": ["BVP_nT"],
    "IBI": ["IBI_ms"],
    "ST":  ["ST_degC"],
    "ACC": ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"],
}

# Flat list of all value columns
ALL_VALUE_COLS = [c for cols in VALUE_COLS.values() for c in cols]

SIGNAL_RANGES = {
    "EDA_uS":  (0.01, 30.0),
    "BVP_nT":  (-300.0, 300.0),
    "IBI_ms":  (300.0, 1500.0),
    "ST_degC": (25.0, 40.0),
    "ACC_X_g": (-4.0, 4.0),
    "ACC_Y_g": (-4.0, 4.0),
    "ACC_Z_g": (-4.0, 4.0),
}

SIGNAL_COLORS = {
    "EDA": "#2ECC71", "BVP": "#E67E22", "IBI": "#9B59B6",
    "ST":  "#3498DB", "ACC": "#E74C3C",
}

# ── Target labels & colours ─────────────────────────────────────────────

TARGET_COLORS = {
    "baseline":  "#3498DB",
    "Happy":     "#F1C40F",
    "Anger":     "#E74C3C",
    "Fear":      "#8E44AD",
    "Disgust":   "#2ECC71",
    "Sad":       "#34495E",
    "Surprise":  "#E67E22",
    "Hunger":    "#1ABC9C",
    "Thirst":    "#2980B9",
    "Toilet":    "#D35400",
    "Tired":     "#7F8C8D",
    "SIB":       "#C0392B",
    "ATO":       "#16A085",
    "GAB":       "#8E44AD",
}

CATEGORY_COLORS = {
    "affective":          "#3498DB",
    "physiological_need": "#2ECC71",
    "behavioural":        "#E74C3C",
    "baseline":           "#95A5A6",
}

LABEL_TO_CATEGORY = {
    "baseline": "baseline",
    "Happy": "affective", "Anger": "affective", "Fear": "affective",
    "Disgust": "affective", "Sad": "affective", "Surprise": "affective",
    "Hunger": "physiological_need", "Thirst": "physiological_need",
    "Toilet": "physiological_need", "Tired": "physiological_need",
    "SIB": "behavioural", "ATO": "behavioural", "GAB": "behavioural",
}

# ── Metadata columns (excluded from analysis) ──────────────────────────

META_COLS = {
    "timestamp_s", "session_id", "user_id", "window_start_s",
    "window_end_s", "target_label", "event_id", "category",
    "age", "gender", "ethnicity", "autism_severity",
    "verbal_status", "comorbidity",
}

DEMOGRAPHIC_FIELDS = [
    "age", "gender", "ethnicity", "autism_severity",
    "verbal_status", "comorbidity",
]

# ── Statistical parameters ──────────────────────────────────────────────

ALPHA = 0.05
MIN_SAMPLES_PER_GROUP = 3
EFFECT_SIZE_THRESHOLDS = {"small": 0.2, "medium": 0.5, "large": 0.8}

# ── Temporal dynamics parameters ────────────────────────────────────────

BASELINE_WINDOW_S = 30.0
RETURN_TOLERANCE = 0.10
SUBSIDE_THRESHOLD = 0.50
POST_EVENT_WINDOW_S = 120.0

DEFAULT_THRESHOLD_WINDOW_S = 300.0
DEFAULT_THRESHOLD_MAD_FACTOR = 3.0
DEFAULT_THRESHOLD_SUSTAIN_S = 30.0

# ── Visualisation parameters ───────────────────────────────────────────

PLOT_DPI = 200
FLOAT_FMT = "%.4f"
MAX_SCATTER_POINTS = 50_000

# ── Report styling (matches M7 pattern) ─────────────────────────────────

REPORT = {
    "font": "Calibri",
    "size": 11,
    "heading_rgb": (0x1A, 0x36, 0x5D),
    "subtitle_rgb": (0x4A, 0x55, 0x68),
    "meta_rgb": (0x71, 0x80, 0x96),
    "warning_rgb": (0x9B, 0x21, 0x21),
    "footer_rgb": (0x9B, 0x9B, 0x9B),
    "table_style": "Light Grid Accent 1",
    "title_pt": 26,
    "subtitle_pt": 16,
    "table_pt": 9,
    "caption_pt": 9,
    "interp_pt": 10,
    "img_sm": 4.5,
    "img_md": 5.0,
    "img_lg": 5.5,
}

# ── Analysis Logic Table ────────────────────────────────────────────────

ANALYSIS_LOGIC_TABLE = [
    {"id": 1, "test": "Descriptive Statistics",
     "measures": "Central tendency and spread per signal per target",
     "why": "Fundamental characterisation; required before any inferential test",
     "assumptions": "None (descriptive)", "output": "Tables + violin/box plots"},
    {"id": 2, "test": "Shapiro-Wilk Normality Test",
     "measures": "Whether signal distributions are Gaussian",
     "why": "Determines parametric vs non-parametric test selection",
     "assumptions": "N >= 3; conservative for large N",
     "output": "W statistic, p-value"},
    {"id": 3, "test": "D'Agostino-Pearson Omnibus",
     "measures": "Normality via combined skewness and kurtosis",
     "why": "More robust than Shapiro-Wilk for N > 5000",
     "assumptions": "N >= 20", "output": "K2 statistic, p-value"},
    {"id": 4, "test": "Skewness and Kurtosis",
     "measures": "Distribution asymmetry and tail heaviness",
     "why": "Guides transformation selection (e.g., log for right-skewed EDA)",
     "assumptions": "None", "output": "Numeric values, shape classification"},
    {"id": 5, "test": "IQR Outlier Detection",
     "measures": "Data points beyond 1.5 x IQR fences",
     "why": "Non-parametric, robust to non-normal distributions",
     "assumptions": "None", "output": "Counts, percentages, fence values"},
    {"id": 6, "test": "Kruskal-Wallis H Test",
     "measures": "Signal distribution differences across target labels",
     "why": "Non-parametric ANOVA; normality violated for physio signals",
     "assumptions": "Independent observations; 3+ groups with >= 3 samples",
     "output": "H, p-value, eta-squared"},
    {"id": 7, "test": "Bonferroni Correction",
     "measures": "Adjusted significance for multiple comparisons",
     "why": "Controls family-wise error rate across multiple tests",
     "assumptions": "Conservative if tests correlated",
     "output": "Adjusted p-values"},
    {"id": 8, "test": "Mann-Whitney U (pairwise)",
     "measures": "Two-group differences on a signal",
     "why": "Non-parametric post-hoc following significant Kruskal-Wallis",
     "assumptions": "Independent samples",
     "output": "U, p (Bonferroni), rank-biserial r"},
    {"id": 9, "test": "Spearman Rank Correlation",
     "measures": "Monotonic association between variables",
     "why": "No normality/linearity assumption; appropriate for physio data",
     "assumptions": "Monotonic relationship",
     "output": "rho, p-value"},
    {"id": 10, "test": "Point-Biserial Correlation",
     "measures": "Signal vs binary target indicator",
     "why": "Quantifies per-target signal tracking (one-vs-rest)",
     "assumptions": "Robust with large N",
     "output": "r_pb, p-value per signal-target pair"},
    {"id": 11, "test": "Event Duration Analysis",
     "measures": "Length of labelled events in seconds",
     "why": "Critical for window-size selection in feature engineering",
     "assumptions": "Events from contiguous target_label transitions",
     "output": "Descriptive stats per target"},
    {"id": 12, "test": "Signal % Change from Baseline",
     "measures": "Peak deviation relative to pre-event median",
     "why": "Normalises response magnitude across signal scales",
     "assumptions": "30s pre-event baseline; baseline = resting state",
     "output": "Percentage change per event per signal"},
    {"id": 13, "test": "Time to Peak",
     "measures": "Onset-to-peak latency in seconds",
     "why": "Characterises autonomic response dynamics",
     "assumptions": "Peak within event window",
     "output": "Seconds per event per signal"},
    {"id": 14, "test": "Return-to-Median Rate",
     "measures": "Proportion of events where signal returns to baseline",
     "why": "Sustained vs transient activation; recovery characterisation",
     "assumptions": "Returned = within 10% of pre-event median",
     "output": "Proportion, mean return time"},
    {"id": 15, "test": "Average Return Time",
     "measures": "Mean seconds for signal to recover",
     "why": "Autonomic recovery time; may correlate with severity",
     "assumptions": "Computed for events that return only",
     "output": "Seconds (mean, std, median)"},
    {"id": 16, "test": "Median Drift",
     "measures": "Post-event shift in running median",
     "why": "Detects lasting physiological shifts (allostatic load)",
     "assumptions": ">= 60s post-event data available",
     "output": "Drift value per event per signal"},
    {"id": 17, "test": "Adaptive Threshold Detection",
     "measures": "Sustained deviations > N x MAD from running median",
     "why": "Episode detection independent of labels",
     "assumptions": "Approximately stationary within rolling window",
     "output": "Crossing events with duration and magnitude"},
    {"id": 18, "test": "PCA 3D Projection",
     "measures": "Low-dimensional structure in multi-signal space",
     "why": "Reveals cluster separability of physiological states",
     "assumptions": "Linear structure (PCA); variance-maximising",
     "output": "3D scatter plot coloured by target"},
]
