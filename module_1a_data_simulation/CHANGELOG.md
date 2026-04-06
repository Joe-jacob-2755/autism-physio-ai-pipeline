# Changelog

All notable changes to Module 1A are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2025

### Added

**Signal Generation**
- EDA simulation with tonic SCL baseline, slow drift, spontaneous SCRs, and emotion-specific phasic SCR events using biphasic kernel model
- BVP simulation using beat-by-beat 3-Gaussian PPG template synthesis from AR(1) HRV model
- IBI event-based series derived from beat placement with short-range auto-regressive correlation (φ=0.55)
- ST simulation with exponential thermal approach model, slow sinusoidal drift, and thermoregulatory random walk
- ACC 3-axis simulation with gravity component, 0.25 Hz breathing artefact, and emotion-specific activity modulation

**Emotion Profiles**
- 6 affective states: Happy, Anger, Fear, Disgust, Sad, Surprise
- 4 physiological need states: Hunger, Thirst, Toilet, Tired
- All profiles parameterised from peer-reviewed autonomic physiology literature
- Profiles cover EDA, BVP, IBI, ST, and ACC modulations

**Event Scheduling**
- Fixed or random event count
- Fixed or random event duration (per-event independent sampling)
- Three emotion modes: specific single emotion, subset list, fully random
- Minimum inter-event gap enforcement
- Minimum lead-in baseline period

**Noise Injection**
- Three noise tiers: low, medium, high
- EDA: Gaussian floor, slow electrode drift, powerline interference
- BVP: Gaussian noise, powerline interference, motion artefact bursts, baseline wander
- ST: Thermal sensor Gaussian noise
- ACC: Gaussian noise with correlated cross-axis vibration component
- IBI: Measurement quantisation uncertainty
- Configurable powerline frequency (default 50 Hz EU/UK)

**Auto-Annotation**
- Per-event annotation table with timing, emotion, category, valence, arousal, expected signal directions
- Baseline window identification (continuous quiet periods ≥ 5 s)
- Signal Quality Index (SQI) in 10-second windows for every channel
- Sample-level label array at 64 Hz reference grid

**Visualisation**
- Individual signal figures for all 7 channels (EDA, BVP, IBI, ST, ACC_X, ACC_Y, ACC_Z)
- Combined 7-panel figure with all channels
- 3-axis ACC comparison figure
- Colour-coded event shading on all plots
- IBI beat markers overlaid on BVP
- Emotion legend on every figure

**Export**
- Per-signal CSVs at native sampling rates
- Combined CSV with all channels interpolated to 64 Hz reference grid
- Label and event_id columns in combined CSV
- Four annotation CSVs
- Metadata JSON with signal statistics

**Interface**
- Full CLI with `--duration`, `--n_events`, `--event_dur`, `--emotion`, `--emotions`, `--noise`, `--seed`, `--out`, `--no_plots`, `--list_emotions`
- Python API via `DataSimulator`, `SignalVisualizer`, `DataExporter`
- Package-level imports via `__init__.py`
- Fully reproducible from integer seed
- Simulation runtime < 200 ms for 5-minute recordings

---

## Planned: [1.1.0]

- Subject-variability population sampling
- Sensory overload state
- Pain state
- Unit test suite (`tests/`)
- Batch dataset generation utility

## Planned: [1.2.0]

- Multi-device sampling rate profiles (E4+, Shimmer, Polar)
- Structured artefact injection (electrode peel-off, gross motion saturation)
- Transition modelling (gradual emotion onset/offset)
