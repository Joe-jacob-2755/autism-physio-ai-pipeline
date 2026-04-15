"""
=============================================================================
MODULE 2B -- DATA ANNOTATION  |  config.py
=============================================================================
Centralised configuration for the manual annotation module.

Constants are self-contained (copied from M2A where needed) to avoid
config.py namespace collisions when modules are imported via
_isolated_import().

Author  : Autism Physio-AI Pipeline
Version : 1.0.0
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODULE IDENTITY
# ─────────────────────────────────────────────────────────────────────────────

MODULE_VERSION: str = "1.0.0"
MODULE_LABEL: str = "M2B"
OUTPUT_ROOT: str = "outputs"

# ─────────────────────────────────────────────────────────────────────────────
# WEARABLE SIGNAL SPECIFICATIONS (mirrored from M2A config)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLING_RATES: dict = {
    "EDA": 4,       # Hz
    "BVP": 64,      # Hz
    "IBI": None,    # event-based
    "ST": 4,        # Hz
    "ACC_X": 32,    # Hz
    "ACC_Y": 32,
    "ACC_Z": 32,
}

COMBINED_RATE_HZ: int = 64  # reference rate for sample-level labels

SIGNAL_COLUMNS: dict = {
    "EDA": "EDA_uS",
    "BVP": "BVP_nT",
    "IBI": "IBI_ms",
    "ST": "ST_degC",
    "ACC": ["ACC_X_g", "ACC_Y_g", "ACC_Z_g"],
}

ANNOTATION_COLS: list = ["target_label", "event_id", "category"]

# ─────────────────────────────────────────────────────────────────────────────
# EMOTION / BEHAVIOUR METADATA
# Extracted from M2A EMOTION_PROFILES -- only annotation-relevant fields.
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_METADATA: dict = {
    # ── Affective Emotions ────────────────────────────────────────────────
    "Happy": {
        "category": "affective",
        "color": "#FFD700",
        "valence": "positive",
        "arousal": "moderate",
        "description": "Positive valence, moderate-high arousal",
        "expect_EDA_dir": "up",
        "expect_HR_dir": "up",
        "expect_ST_dir": "up",
        "expect_ACC_dir": "moderate",
    },
    "Anger": {
        "category": "affective",
        "color": "#DC143C",
        "valence": "negative",
        "arousal": "very_high",
        "description": "Negative valence, very high arousal, sympathetic surge",
        "expect_EDA_dir": "up",
        "expect_HR_dir": "up",
        "expect_ST_dir": "up",
        "expect_ACC_dir": "high",
    },
    "Fear": {
        "category": "affective",
        "color": "#8B008B",
        "valence": "negative",
        "arousal": "very_high",
        "description": "Negative valence, very high arousal, freeze/flight",
        "expect_EDA_dir": "up",
        "expect_HR_dir": "up",
        "expect_ST_dir": "down",
        "expect_ACC_dir": "variable",
    },
    "Disgust": {
        "category": "affective",
        "color": "#6B8E23",
        "valence": "negative",
        "arousal": "moderate",
        "description": "Negative valence, moderate arousal, aversive response",
        "expect_EDA_dir": "up",
        "expect_HR_dir": "up",
        "expect_ST_dir": "down",
        "expect_ACC_dir": "low",
    },
    "Sad": {
        "category": "affective",
        "color": "#4169E1",
        "valence": "negative",
        "arousal": "low",
        "description": "Negative valence, low arousal, parasympathetic dominance",
        "expect_EDA_dir": "down",
        "expect_HR_dir": "down",
        "expect_ST_dir": "down",
        "expect_ACC_dir": "low",
    },
    "Surprise": {
        "category": "affective",
        "color": "#FF8C00",
        "valence": "neutral",
        "arousal": "high_brief",
        "description": "Brief high arousal orienting response",
        "expect_EDA_dir": "up",
        "expect_HR_dir": "up",
        "expect_ST_dir": "neutral",
        "expect_ACC_dir": "high",
    },

    # ── Physiological Need States ─────────────────────────────────────────
    "Hunger": {
        "category": "physiological_need",
        "color": "#FF6347",
        "valence": "negative_internal",
        "arousal": "low_moderate",
        "description": "Internal state -- food deprivation",
        "expect_EDA_dir": "up",
        "expect_HR_dir": "up",
        "expect_ST_dir": "down",
        "expect_ACC_dir": "moderate",
    },
    "Thirst": {
        "category": "physiological_need",
        "color": "#00BFFF",
        "valence": "negative_internal",
        "arousal": "low_moderate",
        "description": "Internal state -- fluid deprivation / dehydration",
        "expect_EDA_dir": "up",
        "expect_HR_dir": "up",
        "expect_ST_dir": "up",
        "expect_ACC_dir": "low",
    },
    "Toilet": {
        "category": "physiological_need",
        "color": "#A0522D",
        "valence": "negative_internal",
        "arousal": "moderate",
        "description": "Internal state -- elimination need (bladder/bowel)",
        "expect_EDA_dir": "up",
        "expect_HR_dir": "up",
        "expect_ST_dir": "neutral",
        "expect_ACC_dir": "moderate",
    },
    "Tired": {
        "category": "physiological_need",
        "color": "#708090",
        "valence": "negative_internal",
        "arousal": "very_low",
        "description": "Fatigue / drowsiness -- parasympathetic dominance",
        "expect_EDA_dir": "down",
        "expect_HR_dir": "down",
        "expect_ST_dir": "down",
        "expect_ACC_dir": "low",
    },

    # ── Behavioural Targets ───────────────────────────────────────────────
    "SIB": {
        "category": "behavioural",
        "color": "#8B0000",
        "valence": "negative",
        "arousal": "very_high",
        "description": "Self-injurious behaviour; repetitive motor, extreme arousal",
        "expect_EDA_dir": "up",
        "expect_HR_dir": "up",
        "expect_ST_dir": "up",
        "expect_ACC_dir": "very_high",
    },
    "ATO": {
        "category": "behavioural",
        "color": "#FF4500",
        "valence": "negative",
        "arousal": "very_high",
        "description": "Aggression to others; explosive motor burst, fastest onset",
        "expect_EDA_dir": "up",
        "expect_HR_dir": "up",
        "expect_ST_dir": "up",
        "expect_ACC_dir": "very_high",
    },
    "GAB": {
        "category": "behavioural",
        "color": "#FF6347",
        "valence": "negative",
        "arousal": "very_high",
        "description": "General aggressive behaviour; tantrums, property destruction",
        "expect_EDA_dir": "up",
        "expect_HR_dir": "up",
        "expect_ST_dir": "up",
        "expect_ACC_dir": "very_high",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE LISTS
# ─────────────────────────────────────────────────────────────────────────────

EMOTIONS_ALL: list = list(EMOTION_METADATA.keys())

CATEGORY_MAP: dict = {
    em: meta["category"] for em, meta in EMOTION_METADATA.items()
}

EMOTIONS_AFFECTIVE: list = [
    e for e, m in EMOTION_METADATA.items() if m["category"] == "affective"
]
EMOTIONS_NEEDS: list = [
    e for e, m in EMOTION_METADATA.items() if m["category"] == "physiological_need"
]
EMOTIONS_BEHAVIOURAL: list = [
    e for e, m in EMOTION_METADATA.items() if m["category"] == "behavioural"
]

# ─────────────────────────────────────────────────────────────────────────────
# INTENSITY LEVELS
# Maps observer's intensity rating to the existing severity reactivity
# modifiers used in the simulation model.
# ─────────────────────────────────────────────────────────────────────────────

INTENSITY_LEVELS: list = ["Low", "Medium", "High"]

INTENSITY_REACTIVITY: dict = {
    "Low": 1.0,
    "Medium": 1.3,
    "High": 1.6,
}

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

MIN_EVENT_DURATION_S: float = 3.0       # minimum valid event duration
MIN_ANNOTATION_DENSITY: float = 0.05    # warn if < 5% of recording is labelled
MIN_RECORDING_DURATION_S: float = 30.0  # warn if recording is very short
BOUNDARY_MARGIN_S: float = 2.0          # warn if event is within 2s of boundary
SHORT_EVENT_WARN_S: float = 10.0        # warn if event < 10s (suboptimal)

# ─────────────────────────────────────────────────────────────────────────────
# EXCEL / BATCH IMPORT FORMAT
# ─────────────────────────────────────────────────────────────────────────────

# Common time formats the observer might use (auto-detected)
CLOCK_TIME_FORMATS: list = [
    "%H:%M:%S",      # 14:32:15
    "%H:%M",         # 14:32
    "%I:%M:%S %p",   # 2:32:15 PM
    "%I:%M %p",      # 2:32 PM
]

BATCH_CSV_REQUIRED_COLS: list = ["start_s", "emotion"]
BATCH_CSV_OPTIONAL_COLS: list = ["end_s", "duration_s"]

# Excel observer sheet: flexible column name mapping
# The system auto-detects these column names (case-insensitive)
EXCEL_COL_ALIASES: dict = {
    "start": ["start_s", "start_time", "start", "time", "onset", "begin"],
    "end": ["end_s", "end_time", "end", "offset", "stop"],
    "duration": ["duration_s", "duration", "dur", "length"],
    "emotion": ["emotion", "behaviour", "behavior", "target", "label",
                "event", "targeted_behaviour", "targeted_behavior",
                "target_behaviour", "target_behavior"],
    "intensity": ["intensity", "severity", "level", "magnitude"],
}
