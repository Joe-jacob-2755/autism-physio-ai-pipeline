# 🧠 Autism Physio-AI Pipeline

<p align="center">
  <img src="module_1a_data_simulation/assets/combined_signals_preview.png" alt="Physiological Signal Preview" width="860"/>
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

An end-to-end AI pipeline for **predicting emotions and behaviours in autistic children** using physiological signals acquired from wearable devices. The pipeline covers the full lifecycle from synthetic data generation through to live inference and deployment.

Autistic children — particularly those who are non-verbal or minimally verbal — often cannot communicate internal states such as fear, pain, hunger, or distress. This pipeline aims to provide caregivers and clinicians with an objective, continuous, non-invasive window into those states using wrist-worn physiological sensors.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AUTISM PHYSIO-AI PIPELINE                            │
├──────────┬──────────┬──────────┬──────────┬──────────┬───────────────────  │
│ Module   │ Module   │ Module   │ Module   │ Module   │ Modules             │
│   1A     │   1B     │   2      │   3      │   4      │  5 – 9              │
│ Data     │ Live     │ Pre-     │ Feature  │ Model    │ Inference / App /   │
│ Simu-    │ Data     │ process- │ Extract- │ Training │ Validation /        │
│ lation   │ Ingest.  │ ing      │ ion      │          │ Deployment          │
│ ✅ v1.0  │ Planned  │ Planned  │ Planned  │ Planned  │ Planned             │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────────────────┘
```

### Module Descriptions

| Module | Name | Status | Description |
|--------|------|--------|-------------|
| **1A** | Data Simulation | ✅ v1.0.0 | Synthetic physiological signal generation for pipeline development and model validation |
| **1B** | Live Data Ingestion | 🔜 Planned | Real-time data acquisition from wearable devices |
| **2** | Preprocessing | 🔜 Planned | Filtering, artefact removal, segmentation, normalisation |
| **3** | Feature Extraction | 🔜 Planned | Time-domain, frequency-domain, and non-linear features per window |
| **4** | Model Training | 🔜 Planned | Multi-class classifier training, validation, and cross-validation |
| **5** | Inference Engine | 🔜 Planned | Real-time prediction from live signals |
| **6** | Alert & Output | 🔜 Planned | Caregiver notifications, logging, and dashboard |
| **7** | Validation | 🔜 Planned | Clinical validation framework and metrics |
| **8** | Model Management | 🔜 Planned | Versioning, retraining, and model registry |
| **9** | Deployment | 🔜 Planned | Edge deployment and device integration |

---

## Target States

The pipeline targets **10 emotion and behaviour states** relevant to autistic children:

**Affective Emotions:** Happy · Anger · Fear · Disgust · Sad · Surprise

**Physiological Needs:** Hunger · Thirst · Toilet · Tired

---

## Physiological Signals

All modules operate on five wearable sensor modalities:

| Signal | Description | Sample Rate |
|--------|-------------|------------|
| EDA | Electrodermal Activity (Skin Conductance) | 4 Hz |
| BVP | Blood Volume Pulse (PPG) | 64 Hz |
| IBI | Inter-Beat Interval | Event-based |
| ST | Skin Temperature | 4 Hz |
| ACC | 3-Axis Accelerometer | 32 Hz |

---

## Getting Started

Start with **Module 1A** to generate synthetic training data:

```bash
cd module_1a_data_simulation
pip install -r requirements.txt
python main.py
```

See the [Module 1A README](module_1a_data_simulation/README.md) for full documentation.

---

## Repository Structure

```
autism-physio-ai-pipeline/
├── module_1a_data_simulation/   ← Synthetic data generation (v1.0.0)
│   ├── README.md
│   ├── main.py
│   ├── config.py
│   ├── simulator.py
│   ├── signal_models.py
│   ├── event_scheduler.py
│   ├── noise_injector.py
│   ├── annotator.py
│   ├── visualizer.py
│   ├── exporter.py
│   ├── requirements.txt
│   ├── docs/
│   │   ├── QUICKSTART.md
│   │   ├── ARCHITECTURE.md
│   │   ├── API_REFERENCE.md
│   │   ├── SIGNAL_SPECIFICATIONS.md
│   │   └── EMOTION_PROFILES.md
│   └── assets/
└── (further modules to follow)
```

---

## References

1. Kreibig, S. D. (2010). Autonomic nervous system activity in emotion: A review. *Biological Psychology, 84*(3), 394–421.
2. Stephens, C. L., et al. (2022). Electrodermal activity in individuals with autism spectrum disorder. *Autism Research.*
3. Kushki, A., et al. (2013). Investigating autonomic nervous system response to anxiety in ASD. *PLOS ONE, 8*(4).
4. Task Force of the European Society of Cardiology. (1996). Heart rate variability standards. *Circulation, 93*(5).

---

## License

MIT License — see [LICENSE](LICENSE).
