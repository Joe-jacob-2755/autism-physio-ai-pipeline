# Emotion & Behaviour Profiles — Module 1A: Data Simulation

Complete reference for all 10 target states simulated in Module 1A.

Profiles are divided into two categories:
- **Affective emotions** — discrete emotional states with clear valence and arousal characteristics
- **Physiological need states** — internal homeostatic states that autistic children may have difficulty communicating verbally

---

## Emotion Profile Parameters Explained

Each emotion profile contains the following structure:

```python
"EmotionName": {
    "category":    str,     # "affective" or "physiological_need"
    "description": str,     # plain-language description
    "color":       str,     # hex colour for visualisation
    "valence":     str,     # "positive", "negative", "neutral"
    "arousal":     str,     # descriptive arousal level

    "EDA": {
        "tonic_delta":        float,  # µS – change from resting SCL
        "scr_amplitude_mean": float,  # µS – mean SCR peak height
        "scr_amplitude_std":  float,  # µS – SD of SCR peaks
        "scr_rate_hz":        float,  # SCRs/second during event
        "rise_time_s":        float,  # seconds – tonic onset time constant
        "recovery_factor":    float,  # 0–1 – fraction of tonic that recovers
    },
    "BVP": {
        "hr_delta_bpm":      float,  # change in HR from baseline (neg = slower)
        "hrv_factor":        float,  # multiplier on HRV std (>1 = more HRV)
        "amplitude_factor":  float,  # multiplier on beat amplitude
    },
    "IBI": {
        "delta_mean_ms":     float,  # ms change in mean IBI (neg = faster HR)
        "delta_std_ms":      float,  # ms change in IBI variability
    },
    "ST": {
        "delta_celsius":     float,  # °C change from baseline skin temperature
        "change_rate":       float,  # °C/s – thermal approach speed
    },
    "ACC": {
        "activity_amp":      float,  # g – movement amplitude
        "dominant_freq_hz":  float,  # Hz – characteristic motion frequency
        "jitter":            float,  # g – aperiodic/random movement component
    },
}
```

---

## Affective Emotions

### 😊 Happy

```
Category : Affective
Valence  : Positive
Arousal  : Moderate
Colour   : #FFD700
```

**Physiological rationale:** Positive affect with moderate arousal produces a mild sympathetic activation. HR increases slightly and EDA rises moderately. Skin warms due to increased peripheral blood flow. Movement is light and rhythmic (e.g. rocking, hand-flapping — common positive self-stimulatory behaviours in autism).

| Signal | Parameter | Value |
|--------|-----------|-------|
| EDA | Tonic Δ | +1.5 µS |
| EDA | SCR rate | 0.45 /s |
| EDA | SCR amplitude | 0.80 ± 0.25 µS |
| EDA | Rise time | 2.0 s |
| EDA | Recovery | 65% |
| BVP | HR Δ | +8.0 bpm |
| BVP | HRV factor | 1.10× (slightly more variable) |
| ST | ΔT | +0.20°C |
| ACC | Amplitude | 0.35 g |
| ACC | Frequency | 1.5 Hz |

---

### 😡 Anger

```
Category : Affective
Valence  : Negative
Arousal  : Very High
Colour   : #DC143C
```

**Physiological rationale:** Anger is one of the highest-arousal negative states. Strong sympathetic surge produces large EDA responses, pronounced cardiac acceleration, and cutaneous vasodilation (hot, flushed face/wrists). Movement is energetic and at medium frequency — striking, throwing, pacing.

| Signal | Parameter | Value |
|--------|-----------|-------|
| EDA | Tonic Δ | +4.5 µS |
| EDA | SCR rate | 1.60 /s |
| EDA | SCR amplitude | 2.80 ± 0.60 µS |
| EDA | Rise time | 0.7 s (fast onset) |
| EDA | Recovery | 25% (slow recovery) |
| BVP | HR Δ | +28.0 bpm |
| BVP | HRV factor | 0.52× (strongly reduced) |
| ST | ΔT | +0.60°C (vasodilation) |
| ACC | Amplitude | 1.80 g |
| ACC | Frequency | 3.5 Hz |

---

### 😨 Fear

```
Category : Affective
Valence  : Negative
Arousal  : Very High
Colour   : #8B008B
```

**Physiological rationale:** Fear produces the most extreme physiological response in the dataset. The freeze/flight response drives maximal sympathetic activation. Uniquely, ST decreases (peripheral vasoconstriction → cold extremities — the "cold sweat" phenomenon). ACC shows high-frequency tremor (8–12 Hz), characteristic of physiological tremor from adrenaline.

| Signal | Parameter | Value |
|--------|-----------|-------|
| EDA | Tonic Δ | +5.2 µS (highest) |
| EDA | SCR rate | 2.00 /s (highest) |
| EDA | SCR amplitude | 3.40 ± 0.70 µS (highest) |
| EDA | Rise time | 0.45 s (fastest) |
| EDA | Recovery | 18% (slowest recovery) |
| BVP | HR Δ | +36.0 bpm (highest) |
| BVP | HRV factor | 0.42× (most suppressed) |
| ST | ΔT | **−0.65°C (vasoconstriction)** |
| ACC | Amplitude | 0.90 g |
| ACC | Frequency | **9.0 Hz (tremor)** |

---

### 🤢 Disgust

```
Category : Affective
Valence  : Negative
Arousal  : Moderate
Colour   : #6B8E23
```

**Physiological rationale:** Disgust is a moderate-arousal aversive state. Produces moderate EDA and HR increase. Associated with mild withdrawal/avoidance movement. Slight peripheral cooling from mild vasoconstriction.

| Signal | Parameter | Value |
|--------|-----------|-------|
| EDA | Tonic Δ | +2.4 µS |
| EDA | SCR rate | 0.70 /s |
| EDA | SCR amplitude | 1.50 ± 0.35 µS |
| BVP | HR Δ | +10.0 bpm |
| ST | ΔT | −0.20°C |
| ACC | Amplitude | 0.28 g |
| ACC | Frequency | 1.0 Hz |

---

### 😢 Sad

```
Category : Affective
Valence  : Negative
Arousal  : Low
Colour   : #4169E1
```

**Physiological rationale:** Sadness is a low-arousal negative state with parasympathetic dominance. EDA decreases below baseline (reduced sympathetic tone), HR slows, and HRV increases. Movement approaches stillness. Skin cools slightly. This profile presents a clear contrast to anger/fear, making it important for arousal classification.

| Signal | Parameter | Value |
|--------|-----------|-------|
| EDA | Tonic Δ | **−0.80 µS (decreases)** |
| EDA | SCR rate | 0.12 /s (very low) |
| BVP | HR Δ | **−10.0 bpm (slows)** |
| BVP | HRV factor | **1.38× (increases)** |
| ST | ΔT | −0.30°C |
| ACC | Amplitude | **0.08 g (near-stillness)** |
| ACC | Frequency | 0.35 Hz |

---

### 😲 Surprise

```
Category : Affective
Valence  : Neutral
Arousal  : High (brief)
Colour   : #FF8C00
```

**Physiological rationale:** Surprise is an orienting response — a brief, intense sympathetic burst regardless of whether the surprise is positive or negative. It has the fastest EDA onset (0.30 s rise time) and the highest spontaneous SCR rate (2.4/s). The ACC shows a startle/jerk pattern characteristic of the orienting reflex.

| Signal | Parameter | Value |
|--------|-----------|-------|
| EDA | Tonic Δ | +3.8 µS |
| EDA | SCR rate | **2.40 /s (highest)** |
| EDA | SCR amplitude | 3.00 ± 0.65 µS |
| EDA | Rise time | **0.30 s (fastest)** |
| BVP | HR Δ | +22.0 bpm |
| ST | ΔT | +0.10°C |
| ACC | Amplitude | 1.30 g |
| ACC | Frequency | 2.5 Hz (jerk/startle) |

---

## Physiological Need States

These states represent internal homeostatic drives that are often underrepresented in standard affective computing datasets but are critically important in autism research. Non-verbal autistic children may be unable to communicate these needs, making physiological detection potentially high-value.

---

### 🍽️ Hunger

```
Category : Physiological Need
Arousal  : Low–Moderate
Colour   : #FF6347
```

**Physiological rationale:** Food deprivation produces gradual mild sympathetic activation via hypothalamic-pituitary signalling. HR increases slightly, EDA rises modestly. Movement is characterised by restless fidgeting rather than directed activity.

| Signal | Parameter | Value |
|--------|-----------|-------|
| EDA | Tonic Δ | +0.80 µS |
| EDA | SCR rate | 0.25 /s |
| BVP | HR Δ | +5.0 bpm |
| ST | ΔT | −0.10°C |
| ACC | Amplitude | 0.30 g (fidgeting) |
| ACC | Frequency | 1.2 Hz |

---

### 💧 Thirst

```
Category : Physiological Need
Arousal  : Low–Moderate
Colour   : #00BFFF
```

**Physiological rationale:** Dehydration produces unique thermal effects: elevated skin temperature due to reduced sweating capacity and peripheral blood pooling. EDA rises due to increased sympathetic tone from osmotic stress. HR increases to compensate for reduced blood volume.

| Signal | Parameter | Value |
|--------|-----------|-------|
| EDA | Tonic Δ | +1.20 µS |
| EDA | SCR rate | 0.28 /s |
| BVP | HR Δ | +7.0 bpm |
| ST | ΔT | **+0.40°C (warming — dehydration effect)** |
| ACC | Amplitude | 0.25 g |
| ACC | Frequency | 1.0 Hz |

---

### 🚽 Toilet

```
Category : Physiological Need
Arousal  : Moderate
Colour   : #A0522D
```

**Physiological rationale:** Bladder/bowel urgency produces a distinct pattern of increased sympathetic activation combined with characteristic movement: weight-shifting, squirming, and postural adjustment at ~2 Hz. EDA and HR are clearly elevated above baseline but below anger/fear levels.

| Signal | Parameter | Value |
|--------|-----------|-------|
| EDA | Tonic Δ | +2.00 µS |
| EDA | SCR rate | 0.58 /s |
| BVP | HR Δ | +12.0 bpm |
| ST | ΔT | +0.05°C |
| ACC | Amplitude | 0.65 g |
| ACC | Frequency | **2.2 Hz (squirming)** |

---

### 😴 Tired

```
Category : Physiological Need
Arousal  : Very Low
Colour   : #708090
```

**Physiological rationale:** Fatigue/drowsiness shows the strongest parasympathetic dominance in the dataset. EDA falls significantly below baseline. HR is the slowest of all states. HRV is at its maximum. Temperature decreases (mild hypothermic tendency in fatigue). ACC approaches complete stillness. This state is the clearest low-arousal signal and is physiologically well-separated from all other states.

| Signal | Parameter | Value |
|--------|-----------|-------|
| EDA | Tonic Δ | **−1.20 µS (lowest)** |
| EDA | SCR rate | **0.07 /s (lowest)** |
| BVP | HR Δ | **−14.0 bpm (slowest)** |
| BVP | HRV factor | **1.48× (highest)** |
| ST | ΔT | **−0.50°C** |
| ACC | Amplitude | **0.04 g (near-stillness)** |
| ACC | Frequency | **0.20 Hz** |

---

## State Comparison Summary

### Arousal Ordering (EDA tonic Δ, descending)

```
Fear        +5.2 µS  ████████████████████████████████
Anger       +4.5 µS  ████████████████████████████
Surprise    +3.8 µS  █████████████████████████
Disgust     +2.4 µS  ████████████████
Toilet      +2.0 µS  █████████████
Thirst      +1.2 µS  ████████
Happy       +1.5 µS  ██████████
Hunger      +0.8 µS  █████
Baseline     0.0 µS  ▏
Sad         −0.8 µS  ▏(below baseline)
Tired       −1.2 µS  ▏▏(below baseline)
```

### HR Change Summary

```
Fear       +36 bpm  ↑↑↑↑↑↑↑↑↑
Anger      +28 bpm  ↑↑↑↑↑↑↑
Surprise   +22 bpm  ↑↑↑↑↑
Toilet     +12 bpm  ↑↑↑
Disgust    +10 bpm  ↑↑
Happy       +8 bpm  ↑↑
Thirst      +7 bpm  ↑
Hunger      +5 bpm  ↑
Baseline     0 bpm
Sad        −10 bpm  ↓↓
Tired      −14 bpm  ↓↓↓
```

### Classification Difficulty Assessment

| State | Similar To | Key Discriminator |
|-------|-----------|-------------------|
| Fear | Anger | ST (−0.65 vs +0.60°C), ACC frequency (9 Hz vs 3.5 Hz) |
| Surprise | Fear/Anger | Very brief duration, near-instantaneous EDA onset |
| Tired | Sad | HR delta magnitude (−14 vs −10), EDA depth (−1.2 vs −0.8) |
| Hunger | Thirst | ST direction (−0.1 vs +0.4°C), ACC pattern |
| Toilet | Disgust | ACC squirming frequency (2.2 vs 1.0 Hz) |
| Happy | Hunger | EDA elevation magnitude, ACC amplitude |
