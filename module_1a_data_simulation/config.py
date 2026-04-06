"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  config.py
=============================================================================
Centralised configuration for all physiological signal parameters and
emotion/behaviour profiles.

Signal specifications are aligned with Empatica E4 / equivalent research
wristband devices commonly used in autism research.

Author  : Autism Physio-AI Pipeline
Version : 1.0.0
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# WEARABLE SIGNAL SPECIFICATIONS
# Based on Empatica E4 and similar research-grade wearables
# ─────────────────────────────────────────────────────────────────────────────

SAMPLING_RATES: dict = {
    "EDA":   4,     # Hz  – Electrodermal Activity (Skin Conductance)
    "BVP":   64,    # Hz  – Blood Volume Pulse (PPG)
    "IBI":   None,  # n/a – Inter-Beat Interval (event-based ms timestamps)
    "ST":    4,     # Hz  – Skin Temperature
    "ACC_X": 32,    # Hz  – Accelerometer X-axis
    "ACC_Y": 32,    # Hz  – Accelerometer Y-axis
    "ACC_Z": 32,    # Hz  – Accelerometer Z-axis
}

SIGNAL_UNITS: dict = {
    "EDA":   "µS",   # microsiemens
    "BVP":   "nT",   # normalised units (device-specific)
    "IBI":   "ms",   # milliseconds
    "ST":    "°C",   # Celsius
    "ACC_X": "g",
    "ACC_Y": "g",
    "ACC_Z": "g",
}

SIGNAL_RANGES: dict = {
    "EDA":   (0.01, 30.0),
    "BVP":   (-300.0, 300.0),
    "IBI":   (300.0, 1500.0),   # 40 – 200 bpm
    "ST":    (25.0,  40.0),
    "ACC_X": (-4.0,  4.0),
    "ACC_Y": (-4.0,  4.0),
    "ACC_Z": (-4.0,  4.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE SIGNAL PARAMETERS  (resting / neutral state)
# ─────────────────────────────────────────────────────────────────────────────

BASELINE: dict = {
    "EDA": {
        "tonic_mean":   3.0,    # µS  – average SCL at rest
        "tonic_std":    0.4,    # µS  – subject-to-subject variability
        "drift_amp":    0.6,    # µS  – slow ultradian tonic drift amplitude
        "drift_freq":   0.004,  # Hz  – drift cycle (very slow)
        "scr_rate":     0.05,   # SCR/s – spontaneous non-event SCRs
        "scr_amp_mean": 0.15,   # µS  – small spontaneous SCR amplitude
    },
    "BVP": {
        "hr_bpm":       75.0,   # bpm – resting heart rate
        "hr_std":        3.0,   # bpm – baseline HR variability
        "amplitude":   100.0,   # nT  – peak BVP amplitude
        "noise_std":     4.0,   # nT  – sensor noise floor
    },
    "IBI": {
        "mean_ms":     800.0,   # ms  – resting RR interval (~75 bpm)
        "hrv_std_ms":   28.0,   # ms  – normal SDNN (HRV spread)
    },
    "ST": {
        "mean_celsius":   33.0,
        "slow_drift_amp":  0.12, # °C – circadian / vascular drift
        "slow_drift_freq": 0.002,# Hz
        "noise_std":       0.02, # °C – sensor noise
    },
    "ACC": {
        "x_mean":        0.02,  # g  – slight forward tilt
        "y_mean":        0.00,
        "z_mean":        0.98,  # g  – gravity dominant on wrist
        "noise_std":     0.03,  # g
        "breathing_amp": 0.04,  # g  – respiratory artifact
        "breathing_freq":0.25,  # Hz – ~15 breaths/min
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# NOISE PROFILES
# ─────────────────────────────────────────────────────────────────────────────

NOISE_PROFILES: dict = {
    "low": {
        "EDA":   {"gaussian_std": 0.03, "powerline_amp": 0.00, "drift_amp": 0.05},
        "BVP":   {"gaussian_std": 3.0,  "powerline_amp": 0.00, "motion_amp": 0.00},
        "ST":    {"gaussian_std": 0.01},
        "ACC":   {"gaussian_std": 0.02},
    },
    "medium": {
        "EDA":   {"gaussian_std": 0.10, "powerline_amp": 0.02, "drift_amp": 0.15},
        "BVP":   {"gaussian_std": 8.0,  "powerline_amp": 1.5,  "motion_amp": 5.0},
        "ST":    {"gaussian_std": 0.03},
        "ACC":   {"gaussian_std": 0.06},
    },
    "high": {
        "EDA":   {"gaussian_std": 0.25, "powerline_amp": 0.08, "drift_amp": 0.40},
        "BVP":   {"gaussian_std": 18.0, "powerline_amp": 4.0,  "motion_amp": 15.0},
        "ST":    {"gaussian_std": 0.08},
        "ACC":   {"gaussian_std": 0.15},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# EMOTION / BEHAVIOUR PROFILES
# Each profile defines how each physiological signal deviates from baseline
# during that state. Grounded in published autism + affective physiology lit.
#
# Key references:
#   Kreibig (2010) – Autonomic NS reactions to emotions (review)
#   Stephens et al. (2022) – EDA in autism
#   Kushki et al. (2013) – Physiological needs in ASD
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_PROFILES: dict = {

    # ── AFFECTIVE EMOTIONS ────────────────────────────────────────────────

    "Happy": {
        "category":    "affective",
        "description": "Positive valence, moderate-high arousal",
        "color":       "#FFD700",
        "valence":     "positive",
        "arousal":     "moderate",

        "EDA": {
            "tonic_delta":        1.5,    # µS added to tonic level
            "scr_amplitude_mean": 0.8,    # µS per phasic SCR peak
            "scr_amplitude_std":  0.25,
            "scr_rate_hz":        0.45,   # SCRs/s during event
            "rise_time_s":        2.0,    # onset ramp constant
            "recovery_factor":    0.65,   # fractional recovery at event end
        },
        "BVP": {
            "hr_delta_bpm":      8.0,     # positive = faster HR
            "hrv_factor":        1.10,    # multiplier on HRV std (>1 = more HRV)
            "amplitude_factor":  1.08,    # multiplier on peak BVP amplitude
        },
        "IBI": {
            "delta_mean_ms":    -60.0,    # negative = shorter IBI = faster HR
            "delta_std_ms":      5.0,
        },
        "ST": {
            "delta_celsius":     0.20,    # warming
            "change_rate":       0.04,    # °C/s approach speed
        },
        "ACC": {
            "activity_amp":      0.35,    # g – movement amplitude
            "dominant_freq_hz":  1.5,     # Hz – characteristic movement freq
            "jitter":            0.08,    # g – irregular component
        },
    },

    "Anger": {
        "category":    "affective",
        "description": "Negative valence, very high arousal, sympathetic surge",
        "color":       "#DC143C",
        "valence":     "negative",
        "arousal":     "very_high",

        "EDA": {
            "tonic_delta":        4.5,
            "scr_amplitude_mean": 2.8,
            "scr_amplitude_std":  0.60,
            "scr_rate_hz":        1.6,
            "rise_time_s":        0.7,
            "recovery_factor":    0.25,
        },
        "BVP": {
            "hr_delta_bpm":      28.0,
            "hrv_factor":        0.52,    # reduced HRV during anger
            "amplitude_factor":  1.30,
        },
        "IBI": {
            "delta_mean_ms":   -165.0,
            "delta_std_ms":      22.0,
        },
        "ST": {
            "delta_celsius":     0.60,    # cutaneous vasodilation
            "change_rate":       0.10,
        },
        "ACC": {
            "activity_amp":      1.80,
            "dominant_freq_hz":  3.5,     # agitated motion
            "jitter":            0.45,
        },
    },

    "Fear": {
        "category":    "affective",
        "description": "Negative valence, very high arousal, freeze/flight",
        "color":       "#8B008B",
        "valence":     "negative",
        "arousal":     "very_high",

        "EDA": {
            "tonic_delta":        5.2,
            "scr_amplitude_mean": 3.4,
            "scr_amplitude_std":  0.70,
            "scr_rate_hz":        2.0,
            "rise_time_s":        0.45,
            "recovery_factor":    0.18,
        },
        "BVP": {
            "hr_delta_bpm":      36.0,
            "hrv_factor":        0.42,
            "amplitude_factor":  1.20,
        },
        "IBI": {
            "delta_mean_ms":   -205.0,
            "delta_std_ms":      30.0,
        },
        "ST": {
            "delta_celsius":    -0.65,    # peripheral vasoconstriction (cold extremities)
            "change_rate":       0.12,
        },
        "ACC": {
            "activity_amp":      0.90,
            "dominant_freq_hz":  9.0,     # high-frequency tremor
            "jitter":            0.25,
        },
    },

    "Disgust": {
        "category":    "affective",
        "description": "Negative valence, moderate arousal, aversive response",
        "color":       "#6B8E23",
        "valence":     "negative",
        "arousal":     "moderate",

        "EDA": {
            "tonic_delta":        2.4,
            "scr_amplitude_mean": 1.5,
            "scr_amplitude_std":  0.35,
            "scr_rate_hz":        0.70,
            "rise_time_s":        1.2,
            "recovery_factor":    0.50,
        },
        "BVP": {
            "hr_delta_bpm":      10.0,
            "hrv_factor":        0.80,
            "amplitude_factor":  1.05,
        },
        "IBI": {
            "delta_mean_ms":    -70.0,
            "delta_std_ms":      12.0,
        },
        "ST": {
            "delta_celsius":    -0.20,
            "change_rate":       0.04,
        },
        "ACC": {
            "activity_amp":      0.28,
            "dominant_freq_hz":  1.0,
            "jitter":            0.10,
        },
    },

    "Sad": {
        "category":    "affective",
        "description": "Negative valence, low arousal, parasympathetic dominance",
        "color":       "#4169E1",
        "valence":     "negative",
        "arousal":     "low",

        "EDA": {
            "tonic_delta":       -0.80,   # EDA drops in sadness
            "scr_amplitude_mean": 0.28,
            "scr_amplitude_std":  0.10,
            "scr_rate_hz":        0.12,
            "rise_time_s":        3.5,
            "recovery_factor":    1.00,
        },
        "BVP": {
            "hr_delta_bpm":     -10.0,
            "hrv_factor":        1.38,
            "amplitude_factor":  0.88,
        },
        "IBI": {
            "delta_mean_ms":    92.0,
            "delta_std_ms":     18.0,
        },
        "ST": {
            "delta_celsius":   -0.30,
            "change_rate":      0.025,
        },
        "ACC": {
            "activity_amp":     0.08,
            "dominant_freq_hz": 0.35,     # near-stillness
            "jitter":           0.025,
        },
    },

    "Surprise": {
        "category":    "affective",
        "description": "Brief high arousal orienting response",
        "color":       "#FF8C00",
        "valence":     "neutral",
        "arousal":     "high_brief",

        "EDA": {
            "tonic_delta":        3.8,
            "scr_amplitude_mean": 3.0,
            "scr_amplitude_std":  0.65,
            "scr_rate_hz":        2.4,
            "rise_time_s":        0.30,
            "recovery_factor":    0.38,
        },
        "BVP": {
            "hr_delta_bpm":      22.0,
            "hrv_factor":        0.60,
            "amplitude_factor":  1.28,
        },
        "IBI": {
            "delta_mean_ms":   -130.0,
            "delta_std_ms":      22.0,
        },
        "ST": {
            "delta_celsius":     0.10,
            "change_rate":       0.20,
        },
        "ACC": {
            "activity_amp":      1.30,
            "dominant_freq_hz":  2.5,     # startle/jerk response
            "jitter":            0.38,
        },
    },

    # ── PHYSIOLOGICAL NEED STATES ─────────────────────────────────────────

    "Hunger": {
        "category":    "physiological_need",
        "description": "Internal state – food deprivation",
        "color":       "#FF6347",
        "valence":     "negative_internal",
        "arousal":     "low_moderate",

        "EDA": {
            "tonic_delta":        0.80,
            "scr_amplitude_mean": 0.50,
            "scr_amplitude_std":  0.18,
            "scr_rate_hz":        0.25,
            "rise_time_s":        4.0,
            "recovery_factor":    0.85,
        },
        "BVP": {
            "hr_delta_bpm":       5.0,
            "hrv_factor":         0.95,
            "amplitude_factor":   1.00,
        },
        "IBI": {
            "delta_mean_ms":    -32.0,
            "delta_std_ms":       8.0,
        },
        "ST": {
            "delta_celsius":    -0.10,    # mild peripheral cooling
            "change_rate":       0.015,
        },
        "ACC": {
            "activity_amp":      0.30,
            "dominant_freq_hz":  1.2,     # restless fidgeting
            "jitter":            0.12,
        },
    },

    "Thirst": {
        "category":    "physiological_need",
        "description": "Internal state – fluid deprivation / dehydration",
        "color":       "#00BFFF",
        "valence":     "negative_internal",
        "arousal":     "low_moderate",

        "EDA": {
            "tonic_delta":        1.20,
            "scr_amplitude_mean": 0.62,
            "scr_amplitude_std":  0.20,
            "scr_rate_hz":        0.28,
            "rise_time_s":        3.5,
            "recovery_factor":    0.80,
        },
        "BVP": {
            "hr_delta_bpm":       7.0,
            "hrv_factor":         0.92,
            "amplitude_factor":   1.02,
        },
        "IBI": {
            "delta_mean_ms":    -42.0,
            "delta_std_ms":       9.0,
        },
        "ST": {
            "delta_celsius":     0.40,    # skin warming in dehydration
            "change_rate":       0.020,
        },
        "ACC": {
            "activity_amp":      0.25,
            "dominant_freq_hz":  1.0,
            "jitter":            0.10,
        },
    },

    "Toilet": {
        "category":    "physiological_need",
        "description": "Internal state – elimination need (bladder/bowel)",
        "color":       "#A0522D",
        "valence":     "negative_internal",
        "arousal":     "moderate",

        "EDA": {
            "tonic_delta":        2.00,
            "scr_amplitude_mean": 1.20,
            "scr_amplitude_std":  0.30,
            "scr_rate_hz":        0.58,
            "rise_time_s":        2.5,
            "recovery_factor":    0.60,
        },
        "BVP": {
            "hr_delta_bpm":      12.0,
            "hrv_factor":         0.85,
            "amplitude_factor":   1.10,
        },
        "IBI": {
            "delta_mean_ms":    -72.0,
            "delta_std_ms":      14.0,
        },
        "ST": {
            "delta_celsius":     0.05,
            "change_rate":       0.020,
        },
        "ACC": {
            "activity_amp":      0.65,
            "dominant_freq_hz":  2.2,     # squirming / shifting weight
            "jitter":            0.22,
        },
    },

    "Tired": {
        "category":    "physiological_need",
        "description": "Fatigue / drowsiness – parasympathetic dominance",
        "color":       "#708090",
        "valence":     "negative_internal",
        "arousal":     "very_low",

        "EDA": {
            "tonic_delta":       -1.20,   # EDA suppression in drowsiness
            "scr_amplitude_mean": 0.18,
            "scr_amplitude_std":  0.06,
            "scr_rate_hz":        0.07,
            "rise_time_s":        5.5,
            "recovery_factor":    1.00,
        },
        "BVP": {
            "hr_delta_bpm":     -14.0,
            "hrv_factor":         1.48,
            "amplitude_factor":   0.82,
        },
        "IBI": {
            "delta_mean_ms":    115.0,
            "delta_std_ms":      24.0,
        },
        "ST": {
            "delta_celsius":    -0.50,    # mild hypothermia in fatigue
            "change_rate":       0.018,
        },
        "ACC": {
            "activity_amp":      0.04,
            "dominant_freq_hz":  0.20,    # near-stillness / head-drooping
            "jitter":            0.012,
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# QUICK REFERENCE LISTS
# ─────────────────────────────────────────────────────────────────────────────

EMOTIONS_ALL: list      = list(EMOTION_PROFILES.keys())
EMOTIONS_AFFECTIVE: list = [
    k for k, v in EMOTION_PROFILES.items() if v["category"] == "affective"
]
EMOTIONS_NEEDS: list    = [
    k for k, v in EMOTION_PROFILES.items() if v["category"] == "physiological_need"
]

SIGNAL_NAMES: list = ["EDA", "BVP", "IBI", "ST", "ACC_X", "ACC_Y", "ACC_Z"]
ACC_AXES: list     = ["ACC_X", "ACC_Y", "ACC_Z"]

# Default simulation parameters
DEFAULT_DURATION_S:  int   = 300   # 5 minutes
DEFAULT_N_EVENTS:    int   = 5
DEFAULT_EVENT_DUR_S: float = 30.0
DEFAULT_NOISE_LEVEL: str   = "medium"
DEFAULT_SEED:        int   = 42
POWERLINE_FREQ_HZ:   float = 50.0  # EU/UK standard (use 60.0 for NA)

# ─────────────────────────────────────────────────────────────────────────────
# VERSION & OUTPUT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

MODULE_VERSION: str  = "1.1.0"       # bump when module code changes
MODULE_LABEL:   str  = "M1A"         # short prefix used in output folder names
# Output folders are auto-named:  outputs/M1A_v1.0.0_run_001/
OUTPUT_ROOT:    str  = "outputs"     # root output folder, relative to module dir

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-USER SIMULATION — POPULATION PARAMETERS
# Based on paediatric physiology (age 5-18) and autism research literature
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_N_USERS: int = 1

# Population-level baseline distributions for child participants.
# Each simulated user draws their personal physiology from these distributions,
# creating realistic inter-subject variability.
POPULATION: dict = {

    "EDA": {
        "tonic_mean_mu":    3.0,    # µS  — population mean resting SCL
        "tonic_mean_sigma": 1.8,    # µS  — between-subject SD
        "tonic_mean_min":   0.3,
        "tonic_mean_max":  18.0,
        # Reactivity multiplier on event EDA response magnitude.
        # Autism shows wide range from hypo- to hyper-reactive.
        "reactivity_mu":    1.0,
        "reactivity_sigma": 0.45,
        "reactivity_min":   0.20,
        "reactivity_max":   2.80,
    },

    "BVP": {
        "hr_bpm_mu":        82.0,   # bpm — children have higher HR than adults
        "hr_bpm_sigma":     11.0,
        "hr_bpm_min":       58.0,
        "hr_bpm_max":      115.0,
        "amplitude_mu":    100.0,
        "amplitude_sigma":  18.0,
        "reactivity_mu":    1.0,
        "reactivity_sigma": 0.30,
        "reactivity_min":   0.30,
        "reactivity_max":   2.00,
    },

    "IBI": {
        # Autism often shows reduced HRV — captured by lower hrv_sdnn values
        "hrv_sdnn_mu":     38.0,    # ms — higher in children than adults
        "hrv_sdnn_sigma":  14.0,
        "hrv_sdnn_min":     8.0,
        "hrv_sdnn_max":    85.0,
    },

    "ST": {
        "mean_celsius_mu":   32.5,  # Children wrist temp slightly lower than adults
        "mean_celsius_sigma": 1.4,
        "mean_celsius_min":  28.5,
        "mean_celsius_max":  36.5,
    },

    "ACC": {
        # Movement varies widely — some autistic children are very still,
        # others are hyperactive
        "movement_scale_mu":    1.0,
        "movement_scale_sigma": 0.40,
        "movement_scale_min":   0.15,
        "movement_scale_max":   2.50,
    },

    "NOISE": {
        # Skin contact quality varies per person / session
        "eda_contact_mu":    1.0,
        "eda_contact_sigma": 0.25,
        "eda_contact_min":   0.40,
        "eda_contact_max":   1.80,
    },
}
