"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  config.py
=============================================================================
Central configuration for all model training: hyperparameters, column
definitions, validation metrics, and deployment constraints.
=============================================================================
"""
from __future__ import annotations

# ── Module metadata ──────────────────────────────────────────────────────
MODULE_VERSION = "1.0.0"
MODULE_LABEL = "M7"
OUTPUT_ROOT = "outputs"
RANDOM_SEED = 42
FLOAT_FMT = "%.6f"

# ── Column definitions (must match M6 output) ────────────────────────────
META_COLS = {"session_id", "user_id", "window_start_s", "window_end_s"}
TARGET_COL = "target_label"
SPLIT_COL = "split"

DEMOGRAPHIC_RAW = {
    "age", "gender", "ethnicity", "autism_severity",
    "verbal_status", "comorbidity",
}
DEMOGRAPHIC_ENC = {
    "gender_enc", "autism_severity_enc", "verbal_status_enc",
    "comorbidity_enc", "ethnicity_enc",
}
ALL_NON_FEATURE_COLS = META_COLS | {TARGET_COL, SPLIT_COL} | DEMOGRAPHIC_RAW

# Known all-NaN columns in current simulated data -- dropped during loading
KNOWN_NAN_COLS = {"bvp_resp_mod", "st_asymmetry", "st_pow_ultra_slow", "st_pow_slow"}

# ── Target classes ────────────────────────────────────────────────────────
TARGET_CLASSES = [
    "baseline", "Happy", "Anger", "Fear", "Disgust", "Sad",
    "Surprise", "Hunger", "Thirst", "Toilet", "Tired", "SIB",
]
# High-priority states: missing these is clinically dangerous
HIGH_PRIORITY_STATES = {"Fear", "SIB", "Toilet", "Anger"}

# ── Class balancing ──────────────────────────────────────────────────────
BALANCE_STRATEGY = "smote"  # "smote" | "class_weight" | "oversample" | "none"
SMOTE_K_NEIGHBORS = 3
MIN_SAMPLES_FOR_SMOTE = 6  # Need >= k_neighbors + 1 per class

# ── Sequence formation (BiLSTM, CNN, TFT) ────────────────────────────────
SEQUENCE_LENGTH = 10       # Number of consecutive windows per sequence
SEQUENCE_STRIDE = 1        # Sliding window stride
MIN_SEQUENCES_FOR_TEMPORAL = 20  # Skip temporal models below this

# ── Cross-validation ─────────────────────────────────────────────────────
CV_FOLDS = 5               # StratifiedKFold folds
CV_SCORING = "f1_weighted"  # Primary CV metric

# ── Decision Tree ─────────────────────────────────────────────────────────
DT_PARAM_GRID = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 5],
    "criterion": ["gini", "entropy"],
    "class_weight": ["balanced", None],
}

# ── Random Forest ─────────────────────────────────────────────────────────
RF_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 3],
    "class_weight": ["balanced", "balanced_subsample"],
}

# ── SVM ───────────────────────────────────────────────────────────────────
SVM_PARAM_GRID = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", "auto", 0.01, 0.001],
    "kernel": ["rbf"],
    "class_weight": ["balanced", None],
}

# ── XGBoost ───────────────────────────────────────────────────────────────
XGB_PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# ── BiLSTM ────────────────────────────────────────────────────────────────
BILSTM = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 16,
    "patience": 15,
    "lr_factor": 0.5,
    "lr_patience": 5,
    "grad_clip": 1.0,
}

# ── 1D CNN ────────────────────────────────────────────────────────────────
CNN_1D = {
    "channels": [32, 64, 128],
    "kernel_sizes": [3, 3, 3],
    "pool_size": 2,
    "dropout": 0.3,
    "fc_hidden": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 16,
    "patience": 15,
    "lr_factor": 0.5,
    "lr_patience": 5,
    "grad_clip": 1.0,
}

# ── Temporal Point Process ────────────────────────────────────────────────
TPP = {
    "hidden_size": 32,
    "num_layers": 1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 16,
    "patience": 15,
    "min_events": 10,  # Minimum non-baseline events to train
}

# ── Temporal Fusion Transformer ───────────────────────────────────────────
TFT = {
    "hidden_size": 32,
    "attention_heads": 4,
    "dropout": 0.1,
    "num_encoder_layers": 2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 16,
    "patience": 15,
    "lr_factor": 0.5,
    "lr_patience": 5,
    "grad_clip": 1.0,
}

# ── BiLSTM Forecaster (time series prediction) ──────────────────────────
# Predict emotional state N minutes ahead using M minutes of history
# Feature windows are 60s with 50% overlap = 30s stride
FORECAST_LOOKBACK = 6      # 6 windows x 30s stride = 3 minutes of history
FORECAST_HORIZON = 2       # 2 windows x 30s stride = 1 minute ahead
FORECAST_STRIDE = 1        # 1 window = 30s effective (closest to 15s target)
MIN_FORECAST_SEQUENCES = 20  # Minimum sequences to train forecaster

BILSTM_FORECAST = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 16,
    "patience": 15,
    "lr_factor": 0.5,
    "lr_patience": 5,
    "grad_clip": 1.0,
}

# ── Unsupervised ──────────────────────────────────────────────────────────
KMEANS = {"n_clusters_range": [3, 5, 7, 10], "n_init": 10, "max_iter": 300}
GMM = {
    "n_components_range": [3, 5, 7, 10],
    "covariance_type": ["full", "diag"],
}
ISOLATION_FOREST = {
    "n_estimators": 100,
    "contamination_range": [0.01, 0.05, 0.1],
}

# ── Autoencoder ───────────────────────────────────────────────────────────
AUTOENCODER = {
    "encoder_dims": [64, 32, 16],
    "latent_dim": 8,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 150,
    "batch_size": 16,
    "patience": 20,
    "anomaly_percentile": 95,
}

# ── Validation metrics ────────────────────────────────────────────────────
PRIMARY_METRIC = "f1_weighted"
METRICS_LIST = [
    "accuracy",
    "f1_weighted",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "cohens_kappa",
]
# Per-class metrics always computed: sensitivity, specificity, F1, AUC-ROC

# ── Deployment constraints ────────────────────────────────────────────────
MAX_INFERENCE_LATENCY_MS = 1000   # Must classify within 1 second
MAX_MODEL_SIZE_MB = 50            # Must fit on mobile device

# ── Plotting ──────────────────────────────────────────────────────────────
PLOT_DPI = 150
PLOT_FIGSIZE = (10, 8)
