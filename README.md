# Autism Physio-AI Pipeline

<p align="center">
  <img src="module_2a_data_simulation/assets/combined_signals_preview.png" alt="Physiological Signal Preview" width="860"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
  <img src="https://img.shields.io/badge/Domain-Autism%20%7C%20Affective%20Computing-purple" />
  <img src="https://img.shields.io/badge/Signals-EDA%20%7C%20BVP%20%7C%20IBI%20%7C%20ST%20%7C%20ACC-teal" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## Overview

An end-to-end AI pipeline for **predicting emotions and behaviours in autistic children** using physiological signals from wrist-worn wearable devices (Empatica E4 and equivalents).

Autistic children — particularly those who are non-verbal or minimally verbal — often cannot communicate internal states such as fear, pain, hunger, or distress. This pipeline provides caregivers and clinicians with an objective, continuous, non-invasive window into those states.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      AUTISM PHYSIO-AI PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Module   │    │ Module   │    │ Module   │    │ Module   │
  │   2A     │──▶│   1      │──▶│   3      │──▶│   4      │
  │ Simulate │    │ Acquire  │    │ Preproc. │    │ Feature  │
  │ ✅v1.1.0 │    │ ✅v1.0.0 │    │ ✅v1.0.0 │    │ Engineer.│
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
                       │ is_annotated=False               │
                       │ → skip to M9                     ▼
                       │                          ┌──────────┐
                       │                          │ Module   │
                       │                          │   5      │
                       │                          │ Training │
                       │                          └──────────┘
                       │                               │
                       │                    ┌──────────┼──────────┐
                       │                    ▼          ▼          ▼
                       │             unimodal    multimodal   Module 6
                       │              models      model      Evaluate
                       │                    └──────────┼──────────┘
                       │                               ▼
                       │                          Module 7
                       │                          Decision
                       │                          Fusion
                       │                               │
                       │                          Module 8
                       │                          Model Mgmt
                       │                               │
                       └──────────────────────▶  Module 9
                        is_annotated=False        Deployment
                                                  + Inference
```

### Module Status

| Module | Name | Status | Description |
|--------|------|--------|-------------|
| **2A** | Data Simulation | ✅ v1.1.0 | Synthetic physiological signal generation with participant demographics |
| **1** | Data Acquisition | ✅ v1.0.0 | Gateway routing all data sources into a standardised `PipelinePacket` |
| **3** | Preprocessing | ✅ v1.0.0 | Signal cleaning, filtering, 80-feature extraction, and RobustScaler normalisation |
| **4** | Feature Engineering | 🔜 Planned | Cross-signal features, temporal dynamics, dimensionality reduction |
| **5** | Model Training | 🔜 Planned | Unimodal and multimodal classifier training with stratified cross-validation |
| **6** | Model Evaluation | 🔜 Planned | Clinical metrics, per-demographic analysis, calibration |
| **7** | Decision Fusion | 🔜 Planned | Combine unimodal model predictions via weighted/stacked fusion |
| **8** | Model Management | 🔜 Planned | Versioned model registry, retraining triggers, ONNX export |
| **9** | Deployment | 🔜 Planned | Real-time inference, caregiver alerts, edge deployment |

---

## Target States

The pipeline targets **10 emotion and behaviour states** relevant to autistic children:

**Affective Emotions:** Happy · Anger · Fear · Disgust · Sad · Surprise

**Physiological Needs:** Hunger · Thirst · Toilet · Tired

---

## Physiological Signals

All modules operate on five wearable sensor modalities from the Empatica E4:

| Signal | Description | Sample Rate | Unit |
|--------|-------------|-------------|------|
| EDA | Electrodermal Activity (Skin Conductance) | 4 Hz | µS |
| BVP | Blood Volume Pulse (PPG) | 64 Hz | nT |
| IBI | Inter-Beat Interval | Event-based | ms |
| ST | Skin Temperature | 4 Hz | °C |
| ACC | 3-Axis Accelerometer | 32 Hz | g |

---

## PipelinePacket — Inter-Module Data Contract

All modules exchange data through a single standardised `PipelinePacket`:

```python
PipelinePacket:
  signals:       Dict[str, pd.DataFrame]   # {signal_name: DataFrame at native rate}
  combined:      pd.DataFrame              # all channels resampled to 64 Hz
  metadata:      dict                      # session info, demographics, SQI
  source_type:   'imported' | 'simulated' | 'live' | 'deployment'
  is_annotated:  bool                      # False → deployment inference path
  session_id:    str
  user_id:       str
```

- `is_annotated = True` → training/evaluation path (Modules 3–8)
- `is_annotated = False` → live deployment path (Module 9 only)

---

## Getting Started

### Prerequisites

```bash
# Windows
scripts\setup.bat

# Mac / Linux
bash scripts/setup.sh

# Activate virtual environment (Windows)
.venv\Scripts\activate
```

### Run the full pipeline (simulation → preprocessing)

```bash
python pipeline_main.py
```

### Run individual modules

```bash
# Module 2A — generate synthetic data
cd module_2a_data_simulation
python main.py

# Module 1 — acquire / route data
.\run_data_acquisition_module.bat     # Windows
./run_data_acquisition_module.sh      # Mac/Linux

# Module 3 — preprocess signals
.\run_data_preprocessing.bat          # Windows
./run_data_preprocessing.sh           # Mac/Linux
```

---

## Repository Structure

```
autism-physio-ai-pipeline/
├── CLAUDE.md                          ← Full AI-assistant context
├── README.md                          ← This file
├── pipeline_main.py                   ← Master orchestrator (M2A → M1 → M3)
│
├── run_full_pipeline.bat / .sh        ← Full pipeline launcher
├── run_data_acquisition_module.bat/.sh← Module 1 standalone
├── run_data_preprocessing.bat / .sh   ← Module 3 standalone
│
├── scripts/
│   ├── setup.bat                      ← Windows environment setup
│   └── setup.sh                       ← Mac/Linux environment setup
│
├── tests/
│   └── test_module_1a.py              ← Module 2A test suite (35 tests)
│
├── module_2a_data_simulation/         ← ✅ Built v1.1.0
│   ├── config.py                      ← Sampling rates, signal ranges, emotion profiles
│   ├── signal_models.py               ← Waveform generators (SCR, PPG, AR-HRV, thermal, ACC)
│   ├── event_scheduler.py             ← Non-overlapping emotion event placement
│   ├── simulator.py                   ← 6-step simulation pipeline
│   ├── noise_injector.py              ← Low / medium / high noise tiers
│   ├── annotator.py                   ← Event labels, SQI, sample-level labels
│   ├── visualizer.py                  ← Signal plots with event timeline bar
│   ├── exporter.py                    ← Per-signal CSVs, combined CSV, metadata JSON
│   ├── user_profiles.py               ← Participant demographics + physiology
│   └── main.py                        ← CLI + interactive menu
│
├── module_1_data_acquisition/         ← ✅ Built v1.0.0
│   ├── pipeline_packet.py             ← PipelinePacket dataclass (inter-module contract)
│   ├── mode_1_1_import.py             ← Load existing CSV folders
│   ├── mode_1_2_simulate.py           ← Call Module 2A in-process
│   ├── mode_1_3_live.py               ← Live BLE streaming + real-time annotation
│   ├── mode_1_4_deployment.py         ← Strip labels → deployment path
│   ├── acquisition_module.py          ← Orchestrator
│   └── main.py                        ← Interactive menu CLI
│
├── module_3_preprocessing/            ← ✅ Built v1.0.0
│   ├── config.py                      ← Filter defaults, cleaning thresholds, scaler rationale
│   ├── signal_cleaner.py              ← Missing data, out-of-range, flatline detection
│   ├── signal_filters.py              ← Butterworth, Hampel, Kalman (RTS smoother)
│   ├── feature_extractor.py           ← 80 features across EDA/BVP/IBI/ST/ACC
│   ├── normaliser.py                  ← RobustScaler + demographic encoding + feature fusion
│   ├── visualiser.py                  ← Processed signal plots, raw-vs-processed comparison
│   ├── exporter.py                    ← 4 CSV sets (raw/normalised × per-signal/combined)
│   └── preprocessor.py                ← 6-step master orchestrator
│
├── module_4_feature_engineering/      ← 🔜 Planned
├── module_5_model_training/           ← 🔜 Planned
├── module_6_model_evaluation/         ← 🔜 Planned
├── module_7_decision_fusion/          ← 🔜 Planned
├── module_8_model_management/         ← 🔜 Planned
└── module_9_deployment/               ← 🔜 Planned
```

---

## Module 2A — Simulation Output

Each simulation run produces:

```
module_2a_data_simulation/outputs/M2A_v1.1.0_run_NNN/
  EDA.csv, BVP.csv, IBI.csv, ST.csv, ACC.csv   ← native sampling rate
  combined_signals.csv                           ← all channels at 64 Hz
  signal_EDA.png ... combined_signals.png        ← with event timeline bar
  annotations_events.csv
  annotations_signal_quality.csv
  annotations_sample_labels.csv
  metadata.json
```

Participant demographics (age, gender, autism severity, verbal status, comorbidity) are generated per user and propagated through all downstream modules.

## Module 3 — Preprocessing Output

Each run produces:

```
module_3_preprocessing/outputs/M3_v1.0.0_run_NNN/
  features_raw/
    EDA_features.csv, BVP_features.csv, IBI_features.csv
    ST_features.csv, ACC_features.csv
    combined_features.csv                         ← 80 features fused
  features_normalised/
    EDA_features_norm.csv ... combined_features_norm.csv
  processed_EDA.png ... comparison_all_signals.png
  cleaning_report.csv
  preprocessing_metadata.json
```

The 80 features span time-domain, frequency-domain, and non-linear measures across all five signals, with demographic encodings appended to every row.

---

## Key Dependencies

```
numpy>=1.24.0        scipy>=1.10.0       pandas>=2.0.0
matplotlib>=3.7.0    seaborn>=0.12.0     scikit-learn>=1.3.0
antropy>=0.1.9       pykalman>=0.9.7     pytest>=7.4.0
```

---

## Testing

```bash
python -m pytest tests/ -v --tb=short
```

35 tests currently cover Module 2A signal generation, event scheduling, noise injection, annotation, and export.

---

## References

1. Kreibig, S. D. (2010). Autonomic nervous system activity in emotion. *Biological Psychology, 84*(3), 394–421.
2. Stephens, C. L., et al. (2022). Electrodermal activity in individuals with ASD. *Autism Research.*
3. Kushki, A., et al. (2013). Investigating ANS response to anxiety in ASD. *PLOS ONE, 8*(4).
4. Task Force of the European Society of Cardiology. (1996). Heart rate variability standards. *Circulation, 93*(5).
5. Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer.
6. Schoen, S. A., et al. (2008). Sensory over-responsivity and autonomic arousal in ASD. *Journal of Autism and Developmental Disorders.*
7. Loomes, R., et al. (2017). What is the male-to-female ratio in ASD? *Journal of Child Psychology and Psychiatry, 58*(4).

---

## License

MIT License — see [LICENSE](LICENSE).
