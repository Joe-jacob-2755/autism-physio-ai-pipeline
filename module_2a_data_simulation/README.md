# 🧠 Autism Physio-AI Pipeline — Module 1A: Data Simulation

<p align="center">
  <img src="assets/combined_signals_preview.png" alt="Combined Physiological Signal Preview" width="860"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
  <img src="https://img.shields.io/badge/Pipeline-Module%201A-orange" />
  <img src="https://img.shields.io/badge/Domain-Autism%20%7C%20Affective%20Computing-purple" />
  <img src="https://img.shields.io/badge/Signals-EDA%20%7C%20BVP%20%7C%20IBI%20%7C%20ST%20%7C%20ACC-teal" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## Overview

**Module 1A** is the first of nine independent modules in the **Autism Physio-AI Pipeline** — an end-to-end system for predicting emotions and behaviours in autistic children using physiological signals acquired from wearable devices.

This module generates **synthetic, physiologically realistic, multi-channel wearable sensor data** for:

- ✅ Initial model architecture development before real device data is available
- ✅ Pipeline code validation and integration testing across all 9 modules
- ✅ Controlled ablation experiments on specific emotion/behaviour classes
- ✅ Reproducible benchmarks and dataset augmentation

Synthetic data faithfully simulates the output of research-grade wristband devices (Empatica E4 and equivalent) across five physiological modalities, for **10 target emotional and physiological need states** observed in autistic children.

---

## Table of Contents

- [Pipeline Context](#pipeline-context)
- [Features](#features)
- [Target States](#target-states)
- [Signal Specifications](#signal-specifications)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Output Files](#output-files)
- [Module Architecture](#module-architecture)
- [Physiological Realism](#physiological-realism)
- [Noise Model](#noise-model)
- [Annotation System](#annotation-system)
- [Configuration](#configuration)
- [Extending the Module](#extending-the-module)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

---

## Pipeline Context

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AUTISM PHYSIO-AI PIPELINE                           │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Module   │ Module   │ Module   │ Module   │ Module   │                 │
│ 1A ◄─── │ 1B       │ 2        │ 3        │ 4        │  Modules 5–9    │
│ Data     │ Live     │ Pre-     │ Feature  │ Model    │  (Inference,    │
│ Simulation│ Ingestion│processing│Extraction│ Training │   App, Deploy)  │
│ [YOU ARE │          │          │          │          │                 │
│  HERE]   │          │          │          │          │                 │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────────────┘
```

Module 1A produces data that is **format-identical** to Module 1B (Live Data Ingestion), ensuring that models trained on synthetic data in early development can be immediately retrained on real device data without changes to any downstream module.

---

## Features

| Feature | Description |
|---------|-------------|
| **10 Target States** | 6 affective emotions + 4 physiological needs |
| **5 Signal Modalities** | EDA, BVP, IBI, Skin Temperature, 3-Axis Accelerometer |
| **Flexible Scheduling** | Fixed or random event count, duration, and emotion assignment |
| **3 Noise Tiers** | Low / Medium / High — realistic sensor, electrode, and motion artefacts |
| **Auto-Annotation** | Event labels, baseline windows, Signal Quality Index (SQI), sample-level labels |
| **Rich Output** | 9 PNG plots + 5 per-signal CSVs + 1 combined CSV + 4 annotation CSVs + JSON metadata |
| **Fully Reproducible** | Integer seed controls every random process end-to-end |
| **CLI + Library API** | Use from the terminal or import into other pipeline modules |

---

## Target States

### Affective Emotions

| Emotion | Valence | Arousal | Primary ANS Signature |
|---------|---------|---------|----------------------|
| 😊 **Happy** | Positive | Moderate | Mild EDA↑, HR↑ +8 bpm, Warm skin |
| 😡 **Anger** | Negative | Very High | Strong EDA↑↑, HR↑ +28 bpm, Increased movement |
| 😨 **Fear** | Negative | Very High | Strongest EDA↑↑, HR↑ +36 bpm, Cold extremities, Tremor |
| 🤢 **Disgust** | Negative | Moderate | EDA↑, HR↑ +10 bpm, Slight cooling |
| 😢 **Sad** | Negative | Low | EDA↓, HR↓ −10 bpm, Near-stillness |
| 😲 **Surprise** | Neutral | High (brief) | Fast EDA↑↑, HR↑ +22 bpm, Startle motion |

### Physiological Need States

| State | Arousal | Primary ANS Signature |
|-------|---------|----------------------|
| 🍽️ **Hunger** | Low–Moderate | Mild EDA↑, HR↑ +5 bpm, Restless fidgeting |
| 💧 **Thirst** | Low–Moderate | EDA↑, HR↑ +7 bpm, Skin warming (dehydration) |
| 🚽 **Toilet** | Moderate | EDA↑, HR↑ +12 bpm, Squirming/weight-shifting |
| 😴 **Tired** | Very Low | EDA↓↓, HR↓ −14 bpm, Near-stillness |

> **Physiological basis:** All profiles are parameterised from peer-reviewed autonomic nervous system literature. See [Physiological Realism](#physiological-realism) and [References](#references).

---

## Signal Specifications

Specifications aligned with **Empatica E4** and equivalent research-grade wearables.

| Signal | Full Name | Sample Rate | Unit | Physiological Range | Model |
|--------|-----------|------------|------|-------------------|-------|
| **EDA** | Electrodermal Activity (Skin Conductance) | 4 Hz | µS | 0.01 – 30.0 µS | SCL tonic + SCR phasic kernels |
| **BVP** | Blood Volume Pulse (PPG) | 64 Hz | nT | −300 – +300 nT | Beat-by-beat 3-Gaussian PPG template |
| **IBI** | Inter-Beat Interval | Event-based | ms | 300 – 1500 ms | AR(1) HRV model from beat timing |
| **ST** | Skin Temperature | 4 Hz | °C | 25.0 – 40.0 °C | Exponential thermal approach |
| **ACC_X** | Accelerometer X-axis | 32 Hz | g | −4.0 – +4.0 g | Gravity + breathing + activity |
| **ACC_Y** | Accelerometer Y-axis | 32 Hz | g | −4.0 – +4.0 g | Gravity + breathing + activity |
| **ACC_Z** | Accelerometer Z-axis | 32 Hz | g | −4.0 – +4.0 g | Gravity (dominant) + breathing |

> IBI is event-based (one row per heartbeat) rather than uniformly sampled. It is stored separately and also embedded as a sparse column in the combined CSV.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Clone and install

```bash
git clone https://github.com/your-org/autism-physio-ai-pipeline.git
cd autism-physio-ai-pipeline/module_1a_data_simulation
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

All dependencies are standard scientific Python packages with no proprietary requirements.

### Verify installation

```bash
python main.py --list_emotions
```

Expected output:
```
Available emotion / behaviour labels:

  [affective]
    Happy        – Positive valence, moderate-high arousal
    Anger        – Negative valence, very high arousal, sympathetic surge
    Fear         – Negative valence, very high arousal, freeze/flight
    Disgust      – Negative valence, moderate arousal, aversive response
    Sad          – Negative valence, low arousal, parasympathetic dominance
    Surprise     – Brief high arousal orienting response

  [physiological_need]
    Hunger       – Internal state – food deprivation
    Thirst       – Internal state – fluid deprivation / dehydration
    Toilet       – Internal state – elimination need (bladder/bowel)
    Tired        – Fatigue / drowsiness – parasympathetic dominance
```

---

## Quick Start

### Default simulation (5 minutes, 5 random events, medium noise)

```bash
python main.py
```

This generates output in `./output/`:
- 9 PNG figures (8 individual + 1 combined)
- 5 per-signal CSV files
- 1 combined CSV
- 4 annotation CSVs
- 1 metadata JSON

### Single targeted emotion

```bash
python main.py --emotion Anger --n_events 4 --duration 240 --noise high --seed 42
```

### Multiple targeted emotions

```bash
python main.py --emotions "Fear,Surprise,Anger" --n_events 6 --event_dur 30 --out results/high_arousal
```

### Physiological needs study

```bash
python main.py --emotions "Hunger,Thirst,Toilet,Tired" --n_events 5 \
               --event_dur random --duration 480 --noise medium --seed 99
```

### Fully random session (count, duration, and emotion all random)

```bash
python main.py --n_events random --event_dur random --noise low --seed 7
```

### Fast CSV-only batch run (no plots)

```bash
python main.py --duration 600 --n_events 10 --no_plots --out results/batch
```

---

## CLI Reference

```
python main.py [OPTIONS]
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--duration` | `float` | `300` | Total recording duration in seconds |
| `--n_events` | `int` \| `"random"` | `5` | Number of emotion/behaviour events |
| `--event_dur` | `float` \| `"random"` | `30` | Duration of each event in seconds |
| `--emotion` | `str` | `None` | Single target emotion (repeats for all events) |
| `--emotions` | `str` (CSV) | `None` | Comma-separated emotion subset e.g. `"Anger,Fear"` |
| `--noise` | `"low"` \| `"medium"` \| `"high"` | `"medium"` | Noise injection level |
| `--seed` | `int` | `42` | Random seed for full reproducibility |
| `--out` | `str` | `"output"` | Output directory path |
| `--no_plots` | flag | off | Skip PNG generation (CSV only) |
| `--list_emotions` | flag | off | Print all valid emotion labels and exit |

**Emotion/behaviour mode logic:**

| `--emotion` | `--emotions` | Behaviour |
|-------------|--------------|-----------|
| Not set | Not set | Fully random from all 10 states |
| `--emotion Anger` | Not set | All events use `Anger` |
| Not set | `--emotions "Anger,Fear"` | Each event randomly sampled from subset |
| Both set | — | `--emotion` takes precedence |

**Random scheduling:**

When `--n_events random`, the count is sampled uniformly from [1, 10].
When `--event_dur random`, each event duration is sampled uniformly from [10, 60] s.
Events are always placed with a minimum 15 s quiet gap between them and a minimum 10 s lead-in baseline at recording start.

---

## Python API

Module 1A can be imported directly into other pipeline modules or notebooks.

### Basic usage

```python
from simulator  import DataSimulator
from visualizer import SignalVisualizer
from exporter   import DataExporter

# Create simulator
sim = DataSimulator(
    duration_s       = 300,
    n_events         = 5,
    event_duration_s = 30.0,
    emotions         = None,       # None = random; or str / list
    noise_level      = "medium",
    seed             = 42,
)

# Run simulation
result = sim.simulate()

# Visualise
viz = SignalVisualizer(result, output_dir="output")
viz.save_all()

# Export CSVs
exporter = DataExporter(result, output_dir="output")
exporter.export_all()
```

### Accessing results programmatically

```python
result = sim.simulate()

# Raw signal arrays (numpy ndarray)
eda  = result.signals["EDA"]       # shape (1200,)  at 4 Hz for 300 s
bvp  = result.signals["BVP"]       # shape (19200,) at 64 Hz
st   = result.signals["ST"]        # shape (1200,)  at 4 Hz
accx = result.signals["ACC_X"]     # shape (9600,)  at 32 Hz

# Time vectors (seconds)
t_eda = result.time_vectors["EDA"] # shape (1200,)

# IBI (event-based)
beat_times = result.ibi_times_s    # seconds, one per heartbeat
ibi_values = result.ibi_values_ms  # milliseconds

# Scheduled events
for ev in result.events:
    print(ev.emotion, ev.start_s, ev.end_s, ev.duration_s)

# Annotations (pandas DataFrames)
event_df   = result.annotations["events"]
baseline_df = result.annotations["baseline_wins"]
sqi_df     = result.annotations["signal_quality"]
labels_df  = result.annotations["sample_labels"]

# Metadata
print(result.metadata)
# {'duration_s': 300, 'n_events': 5, 'noise_level': 'medium', 'seed': 42, ...}
```

### Targeted emotion scheduling

```python
from event_scheduler import (
    make_events_specific,
    make_events_multiple,
    make_events_random,
)

# Five Anger events of 25 s each
events = make_events_specific("Anger", n_events=5, duration_s=25.0, total_dur_s=300)

# Mixed subset
events = make_events_multiple(
    ["Fear", "Anger", "Surprise"],
    n_events=6, duration_s=20.0, total_dur_s=300,
)

# Fully random
events = make_events_random(total_dur_s=300, max_events=8)
```

### Selective export

```python
exporter = DataExporter(result, output_dir="results/")

# Per-signal CSVs only
exporter.export_individual_signals()

# Combined CSV only
exporter.export_combined()

# Annotations only
exporter.export_annotations()

# Metadata JSON only
exporter.export_metadata()
```

### Selective visualisation

```python
viz = SignalVisualizer(result, output_dir="figures/", dpi=200)

# Individual plots
viz.plot_individual_signals()   # one figure per signal

# Combined 7-panel figure
viz.plot_combined()

# ACC 3-axis comparison
viz.plot_acc_combined()
```

---

## Output Files

Running a simulation produces the following file tree:

```
output/
├── figures/
│   ├── signal_EDA.png                  ← EDA with event shading
│   ├── signal_BVP.png                  ← BVP with beat markers
│   ├── signal_IBI.png                  ← IBI tachogram (inverted)
│   ├── signal_ST.png                   ← Skin temperature
│   ├── signal_ACC_X.png                ← ACC X-axis
│   ├── signal_ACC_Y.png                ← ACC Y-axis
│   ├── signal_ACC_Z.png                ← ACC Z-axis
│   ├── signal_ACC_combined.png         ← 3-axis ACC comparison
│   └── combined_signals.png            ← All 7 channels in one figure
│
├── signals/
│   ├── EDA.csv                         ← timestamp_s, EDA_uS         (4 Hz)
│   ├── BVP.csv                         ← timestamp_s, BVP_nT         (64 Hz)
│   ├── IBI.csv                         ← timestamp_s, IBI_ms         (event)
│   ├── ST.csv                          ← timestamp_s, ST_degC        (4 Hz)
│   └── ACC.csv                         ← timestamp_s, ACC_X/Y/Z_g   (32 Hz)
│
├── combined_signals.csv                ← All channels at 64 Hz (interpolated)
│
├── annotations/
│   ├── annotations_events.csv          ← Per-event label metadata
│   ├── annotations_baseline_windows.csv← Quiet period windows
│   ├── annotations_signal_quality.csv  ← 10 s window SQI scores
│   └── annotations_sample_labels.csv   ← Sample-level labels (64 Hz)
│
└── metadata.json                       ← Run parameters + signal statistics
```

### File schemas

#### `EDA.csv`
```
timestamp_s, EDA_uS
0.000000, 3.241879
0.250000, 3.198432
...
```

#### `IBI.csv`
```
timestamp_s, IBI_ms
0.812341, 806.221
1.618562, 798.440
...
```

#### `combined_signals.csv`
```
timestamp_s, BVP_nT, EDA_uS, ST_degC, ACC_X_g, ACC_Y_g, ACC_Z_g, IBI_ms, label, event_id, category
0.000000,  12.341,  3.241,  32.410,  0.021, -0.003, 0.978,     NaN, baseline, 0, baseline
0.015625,  18.902,  3.242,  32.411,  0.020, -0.001, 0.980,     NaN, baseline, 0, baseline
...
0.812500,  45.123,  3.260,  32.415,  0.022, -0.002, 0.977, 806.221, baseline, 0, baseline
...
```
> IBI column is `NaN` except at exact beat timestamps. Label/event_id/category columns reflect the active emotion state.

#### `annotations_events.csv`
```
event_id, emotion, category, valence, arousal, start_s, end_s, duration_s, color_hex, description, expect_EDA_dir, expect_HR_dir, expect_ST_dir, expect_ACC_dir
1, Happy, affective, positive, moderate, 10.000, 40.000, 30.000, #FFD700, Positive valence moderate-high arousal, up, up, up, low
...
```

#### `annotations_signal_quality.csv`
```
signal, window_start_s, window_end_s, sqi, state_label
EDA,   0.00,  10.00, 0.9812, baseline
EDA,  10.00,  20.00, 0.9345, Happy
...
BVP,   0.00,  10.00, 0.8923, baseline
...
```

#### `metadata.json`
```json
{
  "duration_s": 300,
  "n_events": 5,
  "noise_level": "medium",
  "seed": 42,
  "emotion_mode": "random_all",
  "event_duration_s": 30.0,
  "elapsed_s": 0.11,
  "events": [
    { "event_id": 1, "emotion": "Happy", "category": "affective",
      "start_s": 10.0, "end_s": 40.0, "duration_s": 30.0 }
  ],
  "signal_stats": {
    "EDA":   { "n_samples": 1200,  "mean": 6.904, "std": 7.119, "min": 0.01,  "max": 30.0 },
    "BVP":   { "n_samples": 19200, "mean": 20.33, "std": 30.14, "min": -300,  "max": 300  }
  },
  "ibi_n_beats": 413
}
```

---

## Module Architecture

Module 1A is composed of 8 Python source files plus an entry point. Each file has a single well-defined responsibility.

```
module_1a_data_simulation/
├── config.py           ← Constants: sampling rates, signal ranges,
│                          baseline params, all 10 emotion profiles
├── signal_models.py    ← Low-level waveform generation primitives
│                          (SCR kernels, PPG beat templates, HRV model,
│                           ST thermal model, ACC breathing artefact)
├── event_scheduler.py  ← Event timing engine (EventConfig dataclass,
│                          EventScheduler class, factory helpers)
├── simulator.py        ← Master orchestrator — 6-step pipeline,
│                          returns SimulationResult dataclass
├── noise_injector.py   ← Three-tier noise injection for all channels
├── annotator.py        ← Auto-annotation: SQI, event labels,
│                          baseline windows, sample-level labels
├── visualizer.py       ← PNG figure generation (individual + combined)
├── exporter.py         ← CSV and JSON export engine
├── main.py             ← CLI entry point + demo scenario runner
└── __init__.py         ← Public package API
```

### Data flow

```
main.py / API call
        │
        ▼
  EventScheduler ──── config.py (emotion profiles)
        │
        ▼ List[EventConfig]
  DataSimulator ─────────────────────────────────────────────────────
        │                                                            │
        ├─ signal_models.generate_eda_baseline()                    │
        ├─ signal_models.generate_ibi_sequence()                    │
        ├─ signal_models.generate_bvp_from_beats()                  │
        ├─ signal_models.generate_st_baseline()                     │
        ├─ signal_models.generate_acc_baseline()                    │
        │         ← baseline signals generated                      │
        │                                                            │
        ├─ signal_models.generate_eda_event_signal()  ┐             │
        ├─ signal_models.generate_bvp_from_beats()    │ per event   │
        ├─ signal_models.generate_st_event_signal()   │ window      │
        ├─ signal_models.generate_acc_event_signal()  ┘             │
        │         ← event modulations superimposed                  │
        │                                                            │
        ├─ NoiseInjector.inject()                                   │
        │         ← noise added                                      │
        │                                                            │
        ├─ clip_to_range()                                          │
        │         ← physiological range enforced                    │
        │                                                            │
        └─ AutoAnnotator.annotate()                                 │
                  ← SimulationResult ◄──────────────────────────────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
       SignalVisualizer      DataExporter
       (PNG figures)         (CSV / JSON)
```

### Key classes

| Class | File | Description |
|-------|------|-------------|
| `DataSimulator` | `simulator.py` | Master entry point. Accepts all user parameters. Returns `SimulationResult`. |
| `SimulationResult` | `simulator.py` | Dataclass holding all signal arrays, time vectors, events, annotations, and metadata. |
| `EventConfig` | `event_scheduler.py` | Dataclass for a single event: emotion, start, duration. Provides access to the full profile dict. |
| `EventScheduler` | `event_scheduler.py` | Generates validated non-overlapping event lists. Supports all three scheduling modes. |
| `NoiseInjector` | `noise_injector.py` | Adds layered realistic noise to all channels at a chosen level. |
| `AutoAnnotator` | `annotator.py` | Builds four structured annotation DataFrames from a completed simulation. |
| `SignalVisualizer` | `visualizer.py` | Generates and saves all PNG figures. |
| `DataExporter` | `exporter.py` | Writes all CSV and JSON output files. |

---

## Physiological Realism

Every signal is modelled from first principles of human autonomic physiology.

### EDA — Electrodermal Activity

The EDA signal follows the two-component model standard in psychophysiology research:

- **Tonic component (SCL):** Slow-drifting skin conductance level, modulated by the sympathetic nervous system. Baseline ~3.0 µS. Emotion-specific elevations or depressions are modelled with exponential approach curves.
- **Phasic component (SCR):** Individual Skin Conductance Responses modelled as:

  ```
  SCR(t) = A · (1 − e^(−t/τ_rise)) · e^(−t/τ_decay)
  ```

  Where `τ_rise` (~0.5 s) and `τ_decay` (~5 s) match empirical measurements. SCR rate and amplitude are emotion-specific (e.g. Fear: 2.0 SCR/s at 3.4 µS peak; Sadness: 0.12 SCR/s at 0.28 µS peak).

### BVP / IBI — Blood Volume Pulse and Heart Rate Variability

- **Beat placement:** Uses an **AR(1) auto-regressive model** to generate correlated RR intervals, producing realistic HRV rather than independent Gaussian noise. AR(1) coefficient φ = 0.55.
- **PPG waveform:** Each heartbeat is synthesised from a **three-Gaussian model**:
  - G1: Systolic peak (largest, ~18% into beat cycle)
  - G2: Dicrotic notch (small negative deflection, ~42%)
  - G3: Diastolic peak (~54%)
- **Emotion modulation:** Heart rate shifts exponentially toward the target HR (e.g. +36 bpm for Fear), with emotion-specific HRV reduction (Fear: 0.42× baseline HRV; Sadness: 1.38× baseline HRV).

### Skin Temperature

Temperature is the slowest-changing signal. Modelled as an **exponential thermal approach** toward an emotion-specific target:

```
ΔST(t) = Δ_target · (1 − e^(−rate · t))
```

Physiological effects are captured: Fear produces peripheral vasoconstriction (ΔST = −0.65°C), Anger produces cutaneous vasodilation (+0.60°C), Tired produces mild cooling (−0.50°C).

### Accelerometer

- **Z-axis bias:** Gravity is dominant (~0.98 g on Z-axis for wrist-worn device).
- **Breathing artefact:** All three axes include a 0.25 Hz respiratory component (0.04 g amplitude) — a well-documented artefact in wrist-worn accelerometers.
- **Activity modulation:** Each emotion has a characteristic movement amplitude, dominant frequency (e.g. Anger: 1.8 g at 3.5 Hz; Fear: 0.9 g at 9 Hz for tremor; Tired: 0.04 g at 0.2 Hz for near-stillness), and aperiodic jitter.

### Baseline Parameters

| Signal | Parameter | Value | Source |
|--------|-----------|-------|--------|
| EDA | Resting tonic level | 3.0 µS | Boucsein (2012) |
| EDA | Spontaneous SCR rate | 0.05 /s | Dawson et al. (2017) |
| BVP | Resting HR | 75 bpm | Standard physiological range |
| IBI | Resting SDNN (HRV) | 28 ms | Task Force (1996) |
| ST | Resting wrist temperature | 33.0°C | Marins et al. (2014) |
| ACC | Z-axis gravity | 0.98 g | Wrist angle modelling |
| ACC | Breathing frequency | 0.25 Hz | ~15 breaths/min at rest |

---

## Noise Model

Three noise profiles simulate real-world data quality conditions encountered in wearable research.

### Noise Components by Signal

| Signal | Noise Component | Low | Medium | High |
|--------|----------------|-----|--------|------|
| EDA | Gaussian sensor floor | 0.03 µS | 0.10 µS | 0.25 µS |
| EDA | Electrode drift (random walk, <0.05 Hz) | 0.05 µS | 0.15 µS | 0.40 µS |
| EDA | Powerline interference (50 Hz) | None | 0.02 µS | 0.08 µS |
| BVP | Gaussian noise | 3.0 nT | 8.0 nT | 18.0 nT |
| BVP | Powerline interference | None | 1.5 nT | 4.0 nT |
| BVP | Motion artefact bursts | None | 5.0 nT | 15.0 nT |
| BVP | Baseline wander | Low | Medium | High |
| ST | Thermal sensor noise | 0.01°C | 0.03°C | 0.08°C |
| ACC | Gaussian (all axes) | 0.02 g | 0.06 g | 0.15 g |
| ACC | Correlated cross-axis (vibration) | Present | Present | Present |
| IBI | Quantisation uncertainty | ±0.5 ms | ±1.5 ms | ±3.0 ms |

> **Powerline frequency:** Default 50 Hz (EU/UK). Change `POWERLINE_FREQ_HZ = 60.0` in `config.py` for North American deployments.

### Signal Quality Index (SQI)

The `AutoAnnotator` computes a [0.0, 1.0] SQI score for every 10-second window of each signal using four heuristics:

1. **Range check** — fraction of samples outside physiological limits → penalty up to −0.4
2. **Flatline detection** — near-zero variance (dead sensor / saturation) → penalty −0.5
3. **Clipping fraction** — samples at hard rails → penalty up to −0.3
4. **Short vs. long variance ratio** — for BVP and ACC, detects sustained motion artefact → penalty −0.2

Scores are stored in `annotations_signal_quality.csv` and can be used by downstream modules to weight or reject windows during feature extraction.

---

## Annotation System

Four annotation files are produced automatically after every simulation.

### 1. Event Annotations (`annotations_events.csv`)

One row per emotion event. Includes:
- Timing: `start_s`, `end_s`, `duration_s`
- Classification: `emotion`, `category`, `valence`, `arousal`
- Expected signal directions for downstream validation: `expect_EDA_dir`, `expect_HR_dir`, `expect_ST_dir`, `expect_ACC_dir`
- Visualisation: `color_hex`

### 2. Baseline Windows (`annotations_baseline_windows.csv`)

Identifies all continuous quiet periods (no active event) ≥ 5 seconds. Used by downstream modules to extract resting-state feature distributions per subject.

### 3. Signal Quality (`annotations_signal_quality.csv`)

SQI scores for every 10-second window of every signal channel. Downstream modules (feature extraction, model training) should filter or down-weight windows with `sqi < 0.6`.

### 4. Sample Labels (`annotations_sample_labels.csv`)

A label for every sample on the 64 Hz reference grid. Columns: `time_s`, `label`, `event_id`, `category`. This file is the primary ground-truth annotation used in Module 3 (Feature Extraction) and Module 4 (Model Training).

---

## Configuration

All parameters are centralised in `config.py`. No hard-coded values exist elsewhere.

### Key configuration sections

```python
# Sampling rates (Hz) — edit to match your target device
SAMPLING_RATES = {
    "EDA": 4, "BVP": 64, "IBI": None, "ST": 4,
    "ACC_X": 32, "ACC_Y": 32, "ACC_Z": 32,
}

# Physiological ranges — used for clipping and SQI validation
SIGNAL_RANGES = {
    "EDA": (0.01, 30.0), "BVP": (-300.0, 300.0),
    "IBI": (300.0, 1500.0), "ST": (25.0, 40.0),
    "ACC_X": (-4.0, 4.0), "ACC_Y": (-4.0, 4.0), "ACC_Z": (-4.0, 4.0),
}

# Powerline frequency — change to 60.0 for North America
POWERLINE_FREQ_HZ = 50.0

# Default simulation parameters
DEFAULT_DURATION_S  = 300
DEFAULT_N_EVENTS    = 5
DEFAULT_EVENT_DUR_S = 30.0
DEFAULT_NOISE_LEVEL = "medium"
DEFAULT_SEED        = 42
```

### Adding a new emotion

To add a custom target state, append a new entry to `EMOTION_PROFILES` in `config.py`:

```python
EMOTION_PROFILES["CustomState"] = {
    "category":    "affective",            # or "physiological_need"
    "description": "Brief description",
    "color":       "#AABBCC",              # hex for plots
    "valence":     "positive",
    "arousal":     "moderate",

    "EDA": {
        "tonic_delta":        2.0,         # µS change from baseline
        "scr_amplitude_mean": 1.0,         # µS per SCR peak
        "scr_amplitude_std":  0.30,
        "scr_rate_hz":        0.50,        # SCRs/second during event
        "rise_time_s":        1.5,         # onset ramp constant
        "recovery_factor":    0.60,        # fractional recovery at event end
    },
    "BVP": {
        "hr_delta_bpm":      10.0,         # +/- bpm from baseline
        "hrv_factor":         0.90,        # multiplier on HRV std
        "amplitude_factor":   1.05,        # multiplier on BVP amplitude
    },
    "IBI": {
        "delta_mean_ms":    -60.0,         # ms change in mean IBI
        "delta_std_ms":       8.0,
    },
    "ST": {
        "delta_celsius":     0.20,         # +/- °C from baseline
        "change_rate":       0.04,         # °C/s approach speed
    },
    "ACC": {
        "activity_amp":      0.40,         # g — movement amplitude
        "dominant_freq_hz":  1.5,          # Hz — characteristic movement
        "jitter":            0.10,         # g — aperiodic component
    },
}
```

---

## Extending the Module

### Adapting to a different wearable device

1. Update `SAMPLING_RATES` in `config.py` to match your device.
2. Update `SIGNAL_RANGES` if your device has different ADC ranges or units.
3. If your device uses different units, update `SIGNAL_UNITS` and the column headers in `exporter.py`.

### Generating a batch dataset

```python
from simulator import DataSimulator
from exporter  import DataExporter

for seed in range(50):
    sim = DataSimulator(
        duration_s   = 300,
        n_events     = "random",
        event_duration_s = "random",
        emotions     = None,
        noise_level  = "medium",
        seed         = seed,
    )
    result = sim.simulate()
    DataExporter(result, output_dir=f"dataset/session_{seed:03d}").export_all()
```

### Integration with downstream modules

`SimulationResult` is designed to be passed directly to Module 2 (Preprocessing):

```python
result = DataSimulator(...).simulate()

# Pass to Module 2
from module_2_preprocessing import Preprocessor
preprocessed = Preprocessor(result).run()
```

The `combined_signals.csv` format is identical to the output of Module 1B (Live Ingestion), so any downstream module that reads CSV will work with both synthetic and real data without modification.

---

## References

The physiological profiles and signal models in this module are grounded in the following peer-reviewed literature:

1. **Kreibig, S. D. (2010).** Autonomic nervous system activity in emotion: A review. *Biological Psychology, 84*(3), 394–421.
   — Primary reference for all emotion-specific ANS profiles.

2. **Stephens, C. L., et al. (2022).** Electrodermal activity in individuals with autism spectrum disorder: A systematic review. *Autism Research.*
   — Baseline EDA parameters and event responses specific to autistic populations.

3. **Kushki, A., et al. (2013).** Investigating the autonomic nervous system response to anxiety in children with autism spectrum disorders. *PLOS ONE, 8*(4), e59730.
   — Physiological need state profiles for autistic children.

4. **Task Force of the European Society of Cardiology. (1996).** Heart rate variability: Standards of measurement, physiological interpretation, and clinical use. *Circulation, 93*(5), 1043–1065.
   — IBI/HRV parameters, SDNN definition.

5. **Boucsein, W. (2012).** *Electrodermal Activity* (2nd ed.). Springer.
   — SCR kernel model, tonic/phasic decomposition.

6. **Dawson, M. E., et al. (2017).** The electrodermal system. In *Handbook of Psychophysiology* (4th ed.). Cambridge University Press.
   — Spontaneous SCR rate and baseline EDA reference values.

7. **Marins, J. C., et al. (2014).** Thermographic profile of the skin surface of the upper limbs. *Thermology International.*
   — Resting wrist skin temperature reference values.

8. **Clifford, G. D., Azuaje, F., & McSharry, P. (2006).** *Advanced Methods and Tools for ECG Data Analysis.* Artech House.
   — AR(1) HRV model and PPG waveform generation.

---

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before submitting a pull request.

**Development priorities for Module 1A:**
- Additional ethnic/demographic baseline parameter sets
- Subject-variability sampling (drawing each simulation from a population distribution)
- Additional physiological need states (Pain, Sensory Overload)
- Real device data validation against synthetic profiles

---

## License

MIT License. See [`LICENSE`](LICENSE) for details.

---

## Citation

If you use this module in your research, please cite:

```bibtex
@software{autism_physio_ai_module1a_2025,
  title   = {Autism Physio-AI Pipeline: Module 1A -- Physiological Signal Data Simulation},
  year    = {2025},
  version = {1.0.0},
  url     = {https://github.com/your-org/autism-physio-ai-pipeline}
}
```

---

<p align="center">
  Part of the <strong>Autism Physio-AI Pipeline</strong> · Module 1A of 9
</p>
