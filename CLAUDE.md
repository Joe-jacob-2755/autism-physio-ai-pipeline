# CLAUDE.md — Autism Physio-AI Pipeline
## Project Context for AI-Assisted Development

This file provides full context for Claude (and any AI assistant) working on this codebase. Read this before writing, modifying, or reviewing any code.

---

## Project Overview

**Goal:** End-to-end AI pipeline to predict **emotions and behaviours in autistic children** using physiological signals from wrist-worn wearable devices (Empatica E4 and equivalents).

**Clinical motivation:** Non-verbal and minimally verbal autistic children often cannot communicate internal states — fear, pain, hunger, distress — verbally. This pipeline provides caregivers and clinicians with an objective, continuous, non-invasive window into those states.

**Architecture:** 9 independent but sequentially connected modules. Each module is self-contained, independently runnable, and passes data forward via a standardised `PipelinePacket` data contract.

**Repository:** `https://github.com/Joe-jacob-2755/autism-physio-ai-pipeline`

---

## Repository Structure

```
autism-physio-ai-pipeline/                    ← repo root
│
├── CLAUDE.md                                 ← this file
├── README.md                                 ← project overview
├── pipeline_main.py                          ← master orchestrator (M2 → M3)
│
├── run_full_pipeline.bat / .sh               ← launch full pipeline
├── run_data_acquisition_module.bat / .sh     ← launch Module 2 standalone
├── run_data_preprocessing.bat / .sh          ← launch Module 3 standalone
├── run_preprocessing_module.bat / .sh        ← Module 3 alternate launcher
│
├── scripts/
│   ├── setup.bat                             ← Windows environment setup
│   └── setup.sh                             ← Mac/Linux environment setup
│
├── tests/
│   └── test_module_1a.py                     ← Module 2A test suite (35 tests)
│
├── module_2a_data_simulation/               ← ✅ BUILT v1.1.0
├── module_1_data_acquisition/               ← ✅ BUILT v1.0.0
├── module_3_preprocessing/                  ← ✅ BUILT v1.0.0
├── module_4_feature_engineering/            ← 🔜 PLANNED
├── module_5_model_training/                 ← 🔜 PLANNED
├── module_6_model_evaluation/               ← 🔜 PLANNED
├── module_7_decision_fusion/                ← 🔜 PLANNED
├── module_8_model_management/               ← 🔜 PLANNED
└── module_9_deployment/                     ← 🔜 PLANNED
```

---

## Critical Architecture Rules

### 1. Config file naming collision
Every module has a `config.py`. **Never add multiple module directories to `sys.path` simultaneously.** Use the `_isolated_import()` context manager in `pipeline_main.py` when importing across modules:

```python
# CORRECT — in pipeline_main.py
with _isolated_import(M3_DIR):
    from preprocessor import DataPreprocessor

# WRONG — causes wrong config.py to be loaded
sys.path.insert(0, str(M2A_DIR))
sys.path.insert(0, str(M3_DIR))
from preprocessor import DataPreprocessor  # may get M1A's config
```

### 2. PipelinePacket data contract
All modules pass data through `module_1_data_acquisition/pipeline_packet.py`. The `PipelinePacket` dataclass is the **only** inter-module data format. Do not bypass it.

```python
PipelinePacket:
  signals:       Dict[str, pd.DataFrame]   # {signal_name: DataFrame}
  combined:      pd.DataFrame              # all channels at 64 Hz
  metadata:      dict
  source_type:   'imported' | 'simulated' | 'live' | 'deployment'
  is_annotated:  bool                      # False → deployment path
  session_id:    str
  user_id:       str
```

**Routing rule:**
- `is_annotated = True`  → training/testing path (Modules 3–8)
- `is_annotated = False` → deployment inference path (Module 9)

### 3. Auto-numbered output folders
Every module creates versioned, auto-numbered output folders. Never hardcode output paths.

```
module_Xa_name/outputs/Ma_vX.Y.Z_run_NNN/
```

Pattern used everywhere:
```python
# Find next available run number
existing = [int(m.group(1)) for d in output_root.iterdir()
            if (m := pattern.match(d.name))]
next_num = (max(existing) + 1) if existing else 1
```

### 4. Virtual environment
Always run from `.venv`. Scripts are in `scripts/setup.bat` (Windows) and `scripts/setup.sh` (Mac/Linux).

```powershell
# Windows — activate before running anything
.venv\Scripts\activate
```

### 5. Numpy 2.x compatibility
Use `np.trapezoid()` not `np.trapz()` (removed in NumPy 2.0). Use `arr.max() - arr.min()` not `arr.ptp()` (removed in NumPy 2.0).

---

## Physiological Signals

All modules operate on five wearable sensor modalities from Empatica E4 / equivalent:

| Signal | Full Name | Sample Rate | Unit | Range |
|--------|-----------|------------|------|-------|
| EDA | Electrodermal Activity (Skin Conductance) | 4 Hz | µS | 0.01–30.0 |
| BVP | Blood Volume Pulse (PPG) | 64 Hz | nT | −300 to +300 |
| IBI | Inter-Beat Interval | Event-based | ms | 300–1500 |
| ST | Skin Temperature | 4 Hz | °C | 25.0–40.0 |
| ACC_X/Y/Z | 3-Axis Accelerometer | 32 Hz | g | −4.0 to +4.0 |

**CSV column names** (used consistently across all modules):
- `timestamp_s`, `EDA_uS`, `BVP_nT`, `IBI_ms`, `ST_degC`, `ACC_X_g`, `ACC_Y_g`, `ACC_Z_g`
- `target_label`, `event_id`, `category` (annotation columns — may be absent in deployment mode)

---

## Target States (10 total)

### Affective Emotions
`Happy`, `Anger`, `Fear`, `Disgust`, `Sad`, `Surprise`

### Physiological Need States
`Hunger`, `Thirst`, `Toilet`, `Tired`

**Valence–Arousal mapping:**
- Very high arousal: Fear, Anger, Surprise
- Moderate arousal: Happy, Disgust, Toilet
- Low arousal: Sad, Hunger, Thirst
- Very low arousal: Tired

---

## Participant Demographics

Generated with every simulated user profile (Module 2A v1.1.0+). Passed through to Module 3 feature matrices.

| Field | Values | Note |
|-------|--------|------|
| `age` | 5–15 years | Uniform distribution |
| `gender` | Male / Female / Non-binary | 75% / 22% / 3% (ASD prevalence ratio) |
| `ethnicity` | 5 categories | UK population-weighted |
| `autism_severity` | Low / Medium / Severe | DSM-5 Level 1/2/3 |
| `verbal_status` | Verbal / Minimally verbal / Non-verbal | 55% / 25% / 20% |
| `comorbidity` | Yes / No | 70% / 30% (realistic ASD comorbidity rate) |

**Severity → reactivity:** Severity modulates EDA/HR reactivity in the simulator:
`Low=1.0×`, `Medium=1.3×`, `Severe=1.6×`

---

## 80 Physiological Features (Module 3 output)

| Signal | L1 Derived | L2 Time/Freq/Nonlinear | Total |
|--------|-----------|----------------------|-------|
| EDA | 9 (SCL, SCR decomposition) | 9 | 18 |
| BVP | 7 (beat morphology) | 6 | 13 |
| IBI/HRV | 5 (SDNN, RMSSD, pNN50…) | 12 | 17 |
| ST | 4 (delta, rate, slope…) | 6 | 10 |
| ACC | 6 (SVM, dominant freq…) | 8 | 14 |
| **Total** | **31** | **41** | **72 core** |

**Key discriminative features** (highest importance for this population):
`eda_scr_rate`, `eda_scr_auc`, `ibi_rmssd`, `ibi_lf_hf`, `ibi_sampen`, `bvp_hr_mean`, `acc_dom_freq`, `acc_pow_tremor`, `st_delta`

**Downstream model architecture** (informed by feature structure):
- **Per-signal CSVs** → unimodal models (one per signal) → decision fusion (Module 7)
- **Combined CSV** → early fusion multimodal model (Module 5)

---

---

# MODULE REFERENCE

---

## Module 2A — Data Simulation
**Status:** ✅ Built | **Version:** 1.1.0 | **Files:** 10

### Purpose
Generate synthetic, physiologically realistic physiological signals for pipeline development and model validation **before real device data is available**. Output format is identical to Module 1B (live ingestion) so all downstream modules work unchanged.

### Location
```
module_2a_data_simulation/
├── config.py           ← ALL constants: sampling rates, signal ranges,
│                          baseline params, 10 emotion profiles,
│                          population distributions, demographics
├── signal_models.py    ← Low-level waveform generators:
│                          SCR kernel (EDA), 3-Gaussian PPG (BVP),
│                          AR(1) HRV model (IBI), thermal model (ST),
│                          breathing artefact + activity (ACC)
├── event_scheduler.py  ← EventConfig dataclass, EventScheduler class,
│                          factory helpers (specific/multiple/random)
├── simulator.py        ← DataSimulator (6-step pipeline),
│                          SimulationResult dataclass
├── noise_injector.py   ← Three noise tiers: low/medium/high
│                          EDA: drift + powerline | BVP: motion + wander
│                          ACC: correlated cross-axis vibration
├── annotator.py        ← AutoAnnotator: event labels, baseline windows,
│                          SQI (0–1 per 10 s window), sample-level labels
├── visualizer.py       ← Event timeline bar, signal plots, combined figure,
│                          ACC 3-axis figure — all with event shading
├── exporter.py         ← Per-signal CSVs + combined CSV + annotation CSVs
│                          + metadata JSON. All with target_label column.
├── user_profiles.py    ← UserProfile dataclass (physiology + demographics),
│                          UserProfileGenerator draws from POPULATION
└── main.py             ← CLI + run_simulation() + interactive menu
```

### Key Classes / Functions

```python
DataSimulator(
    duration_s, n_events, event_duration_s, emotions,
    noise_level, seed, user_profile, min_gap_s, min_lead_s
)
result = sim.simulate()  # → SimulationResult

UserProfileGenerator(n_users=10, master_seed=42).generate()
# → List[UserProfile] with physiology + demographics

run_simulation(duration_s, n_events, event_duration_s, emotions,
               noise_level, seed, n_users, shared_events_flag)
# → List[SimulationResult]
```

### Simulation Pipeline (6 steps)
1. `EventScheduler` — place non-overlapping emotion events
2. Generate baseline signals (EDA SCL, BVP beats, ST drift, ACC gravity)
3. Apply event modulations per emotion profile
4. `NoiseInjector` — add Hampel/Gaussian/powerline/motion noise
5. Clip to physiological ranges
6. `AutoAnnotator` — event labels, SQI, sample-level labels

### Output Structure
```
module_2a_data_simulation/outputs/M2A_v1.1.0_run_NNN/
  EDA.csv, BVP.csv, IBI.csv, ST.csv, ACC.csv  ← native sampling rate
  combined_signals.csv                          ← all channels at 64 Hz
  signal_EDA.png ... combined_signals.png       ← with event timeline bar
  annotations_events.csv
  annotations_signal_quality.csv
  annotations_sample_labels.csv
  metadata.json
```

### Launch Commands
```powershell
cd module_2a_data_simulation
python main.py                                        # interactive menu
python main.py --duration 300 --n_users 5 --seed 42
python main.py --emotion Fear --n_events 4 --noise high
python main.py --n_users 10 --shared_events --emotions "Fear,Anger"
python main.py --list_emotions
```

### Config Constants (module_2a_data_simulation/config.py)
```python
SAMPLING_RATES     # {EDA:4, BVP:64, IBI:None, ST:4, ACC_X:32, ...}
SIGNAL_RANGES      # physiological min/max per channel
BASELINE           # resting-state parameters per channel
EMOTION_PROFILES   # 10 emotion profiles with per-signal parameters
POPULATION         # paediatric population distributions for UserProfile
PARTICIPANT_DEMOGRAPHICS  # age, gender, severity, verbal, comorbidity
MODULE_VERSION = "1.1.0"
DEFAULT_N_USERS = 1
POWERLINE_FREQ_HZ = 50.0   # change to 60.0 for North America
```

### Important Notes
- Emotion severity reactivity: `Low=1.0×`, `Medium=1.3×`, `Severe=1.6×` on EDA/HR
- Age effect on HR: `+1.2 bpm/yr` younger (5-yr-old has ~12 bpm higher HR than 15-yr-old)
- All outputs include `target_label`, `event_id`, `category` columns
- `UserProfile.demographic_dict()` — returns only demographic fields for M3 fusion
- Module 1B (live ingestion) will produce **identical output format** — downstream modules are format-agnostic

---

## Module 1 — Data Acquisition
**Status:** ✅ Built | **Version:** 1.0.0 | **Files:** 7

### Purpose
Gateway for all data entering the pipeline. Routes from four sources into a standardised `PipelinePacket`. All downstream modules receive `PipelinePacket` regardless of data source.

### Location
```
module_1_data_acquisition/
├── pipeline_packet.py    ← PipelinePacket dataclass (inter-module contract)
│                            source_type, is_annotated, signals, combined,
│                            metadata, session_id, user_id
│                            .save() / .load() for persistence
├── mode_1_1_import.py    ← DataImporter: load existing CSVs/folders
│                            Auto-detects 4 folder layouts (M1A, individual,
│                            combined-only, single CSV)
├── mode_1_2_simulate.py  ← SimulationConnector: calls Module 2A in-process
│                            InteractiveSimSetup: guided prompts
├── mode_1_3_live.py      ← LiveDataCollector + DeviceAdapter ABC
│                            FileStreamAdapter: replay CSV at speed (testing)
│                            EmpaticaE4Adapter: BLE Streaming Server (TCP)
│                            SessionAnnotator: real-time keyboard annotation
├── mode_1_4_deployment.py← DeploymentIngester: strips labels,
│                            sets is_annotated=False → Module 9 path
├── acquisition_module.py ← DataAcquisitionModule orchestrator
│                            Auto-numbered outputs: M1_v1.0.0_mode2_X_run_NNN/
└── main.py               ← Interactive menu-driven CLI
```

### Key Classes / Functions

```python
# Master orchestrator
acq = DataAcquisitionModule(mode="2.1", save_packets=True)
packet  = acq.run_import(source_path, user_id)
packets = acq.run_simulate(duration_s, n_events, n_users, ...)
packet  = acq.run_live_file(csv_path, speed_factor=10.0)
packet  = acq.run_live_e4(host, port)
packet  = acq.run_deployment(source_path, strip_labels=True)

# PipelinePacket — the data contract
packet.signals       # {name: DataFrame}
packet.combined      # all channels at 64 Hz
packet.is_annotated  # True = training path | False = deployment path
packet.save(output_dir)
PipelinePacket.load(input_dir)
```

### Acquisition Modes

| Mode | Class | is_annotated | Notes |
|------|-------|-------------|-------|
| 2.1 Import | `DataImporter` | Detected from columns | Reads existing CSVs |
| 2.2 Simulate | `SimulationConnector` | True | Calls M1A DataSimulator |
| 2.3 Live | `LiveDataCollector` | True (if events annotated) | Requires researcher annotation |
| 2.4 Deployment | `DeploymentIngester` | False | Strips labels → Module 9 |

### Live Annotation Commands (Mode 2.3)
```
start <emotion>   — begin event (e.g. start Fear)
end               — close current event
events            — show events logged so far
list              — list all valid emotion names
stop              — end recording session
```

### Launch Command
```powershell
.\run_data_acquisition_module      # from repo root (Windows)
./run_data_acquisition_module.sh   # Mac/Linux
```

### Output Structure
```
module_1_data_acquisition/outputs/M1_v1.0.0_mode2_X_run_NNN/
  EDA.csv, BVP.csv, IBI.csv, ST.csv, ACC.csv   ← with target_label
  combined_signals.csv
  packet_metadata.json
  module2_run_summary.json
```

### Important Notes
- `pipeline_packet.py` is the **only** file that should be imported by downstream modules (not mode_2_X files directly)
- Mode 2.2 imports Module 2A in-process using `_isolated_import()` in pipeline_main.py to avoid config.py collision
- `FileStreamAdapter` replays any `combined_signals.csv` at configurable speed — use `speed_factor=10.0` for testing without a device
- E4 Streaming Server must be running locally for Mode 2.3 E4 (default: `127.0.0.1:28000`)

---

## Module 3 — Data Preprocessing
**Status:** ✅ Built | **Version:** 1.0.0 | **Files:** 9

### Purpose
Transform raw physiological signals into clean, normalised, feature-rich DataFrames ready for model training. Produces 4 CSV output sets (raw/normalised × individual/combined) supporting both unimodal and multimodal model architectures.

### Location
```
module_3_preprocessing/
├── config.py           ← Preprocessing constants: sampling rates, filter
│                          defaults, cleaning thresholds, feature windows,
│                          demographic encodings, RobustScaler rationale
├── signal_cleaner.py   ← SignalCleaner: missing data (70% discard rule),
│                          out-of-range → NaN, linear interpolation,
│                          flatline detection, CleaningReport dataclass
├── signal_filters.py   ← Three filter implementations:
│                          butterworth_lowpass/bandpass/highpass (zero-phase)
│                          hampel_filter (sliding MAD, 1.4826 scaling)
│                          kalman_filter_1d (RTS smoother)
│                          SignalFilterManager: Hampel → main filter
├── feature_extractor.py← All 80 features extracted in configurable windows
│                          EDA: SCL/SCR decomposition, 18 features
│                          BVP: beat detection, 13 features
│                          IBI: ectopic removal, HRV spectral, 17 features
│                          ST: thermal delta/slope, 10 features
│                          ACC: SVM, spectral bands, 14 features
│                          FeatureExtractor: window-based extraction
├── normaliser.py       ← FeatureNormaliser (RobustScaler — see rationale)
│                          DemographicEncoder (ordinal, clinically ordered)
│                          FeatureFuser (merge_asof window alignment)
├── visualiser.py       ← PreprocessingVisualiser:
│                          plot_processed_signals() — with event timeline bar
│                          plot_raw_vs_processed() — SNR gain annotation
├── exporter.py         ← PreprocessingExporter: 4 CSV sets + reports
└── preprocessor.py     ← DataPreprocessor master orchestrator (6 steps)
```

### 6-Step Pipeline

```
Input (PipelinePacket | folder path | dict of DataFrames)
   │
   ├─ Step 1: SignalCleaner
   │    Missing > 70% → DISCARD | < 70% → linear interpolation
   │    Out-of-range → NaN → re-fill | Flatline detection
   │    → CleaningReport per channel
   │
   ├─ Step 2: SignalFilterManager
   │    Stage A: Hampel filter  → impulse artefact removal (optional)
   │    Stage B: Butterworth    → zero-phase LP/BP per signal
   │          OR Kalman         → RTS smoother (optimal for EDA/ST)
   │
   ├─ Step 3: FeatureExtractor
   │    Window: 60 s default, 50% overlap
   │    → 80 features per window per signal
   │
   ├─ Step 4: DemographicEncoder
   │    age, gender, severity, verbal, comorbidity → appended to all rows
   │    Severity: Low=1, Medium=2, Severe=3
   │    Verbal: Verbal=0, Minimally verbal=1, Non-verbal=2
   │
   ├─ Step 5: FeatureNormaliser (RobustScaler)
   │    Per-signal scaling: median (Q2) + IQR (Q1–Q3)
   │    Reason: outlier resistance for SCR peaks, ACC impulses, HR surges
   │    FeatureFuser → per-signal DFs + combined DF (merge_asof alignment)
   │
   └─ Step 6: PreprocessingVisualiser + PreprocessingExporter
```

### Why RobustScaler (not StandardScaler or MinMaxScaler)

1. **Outlier resistance**: SCR peaks (3–5× baseline), HR +36 bpm in Fear, ACC startle impulses all inflate mean/std in StandardScaler. Median + IQR are unaffected.
2. **Inter-subject variability**: EDA 0.3–18 µS across users. RobustScaler centres on population median without collapsing individual variation.
3. **Non-Gaussian distributions**: SCR rate, SVM, spectral powers are log-normal. RobustScaler makes no Gaussian assumption.
4. **Preserves outlier signal**: Unlike MinMaxScaler [0,1], extreme emotional responses remain visible beyond ±1 range.
5. **Clinical consistency**: IQR-based normalisation recommended in HRV literature (Task Force 1996).

### Default Filter Parameters

| Signal | Type | Cutoff(s) | Rationale |
|--------|------|-----------|-----------|
| EDA | Low-pass | 1.0 Hz | Removes high-freq noise; EDA changes slowly |
| BVP | Band-pass | 0.5–8.0 Hz | Removes baseline wander; keeps cardiac harmonics |
| IBI | None | — | Ectopic beat removal only |
| ST | Low-pass | 0.1 Hz | ST changes very slowly (thermal inertia) |
| ACC | Band-pass | 0.1–15.0 Hz | Removes DC gravity; keeps up to tremor range |

### Key Classes / Functions

```python
DataPreprocessor(
    filter_type='butterworth',  # 'butterworth'|'kalman'|'hampel_only'|'none'
    apply_hampel=True,
    window_s=60.0,
    overlap=0.5,
    scaler_type='robust',       # 'robust'|'standard'|'minmax'
    generate_plots=True,
)
result = pp.run(
    signals_input,    # PipelinePacket | folder path | dict
    demographics,     # age, gender, severity, verbal, comorbidity
    session_id,
    user_id,
)
# result keys: run_folder, raw_features, norm_features,
#              raw_combined, norm_combined, cleaning_reports,
#              signals_cleaned, signals_filtered, metadata
```

### Output Structure
```
module_3_preprocessing/outputs/M3_v1.0.0_run_NNN/
  features_raw/
    EDA_features.csv          ← 18 EDA features + demographics (unimodal)
    BVP_features.csv          ← 13 BVP features + demographics
    IBI_features.csv          ← 17 IBI/HRV features + demographics
    ST_features.csv           ← 10 ST features + demographics
    ACC_features.csv          ← 14 ACC features + demographics
    combined_features.csv     ← all 80 features fused (multimodal)
  features_normalised/
    EDA_features_norm.csv     ← RobustScaler-normalised (unimodal input)
    ...
    combined_features_norm.csv← RobustScaler-normalised (multimodal input)
  processed_EDA.png ... processed_combined.png
  comparison_EDA.png ... comparison_all_signals.png
  cleaning_report.csv
  preprocessing_metadata.json
```

### Launch Commands
```powershell
.\run_data_preprocessing          # M3 standalone — browse for data
.\run_preprocessing_module        # M3 via its own interactive menu
python main.py                    # direct (from module_3_preprocessing/)
python main.py --source <path> --filter butterworth --window 60
```

### Important Notes
- Input accepts PipelinePacket (from M2), folder path, or dict of DataFrames
- Demographics auto-read from `metadata.json` if present; otherwise prompted
- `FeatureFuser` uses `merge_asof` with 30 s tolerance for window alignment across signals of different sampling rates
- `FeatureNormaliser.save()` / `.load()` persist fitted scalers — use the **same fitted scaler** when transforming test data and deployment data
- IBI features require minimum 4 beats; short windows may produce NaN — handled by `_safe()` wrapper

---

## Module 4 — Feature Engineering
**Status:** 🔜 Planned

### Planned Purpose
Advanced feature engineering beyond the 80 base features from Module 3. Produces engineered features for improved model performance.

### Planned Functionality
- **Cross-signal features**: EDA–HR coupling, ST–ACC correlation, sympatho-vagal index
- **Temporal dynamics**: Rate of change features, onset/offset velocity, sustained vs transient responses
- **Window-level aggregation**: Mean, std, min, max, slope across multiple windows per session
- **Interaction features**: Age × severity modulation terms, verbal_status × EDA_reactivity
- **Dimensionality reduction**: PCA, UMAP for visualisation; optional feature selection (mutual information, SHAP-based)
- **Stationarity tests**: ADF, KPSS — flag non-stationary features before model training

### Expected Input
`combined_features.csv` or `combined_features_norm.csv` from Module 3

### Expected Output
```
module_4_feature_engineering/outputs/M4_v1.0.0_run_NNN/
  engineered_features.csv
  feature_importance.csv
  pca_components.csv
  feature_engineering_report.json
```

---

## Module 5 — Model Training
**Status:** 🔜 Planned

### Planned Purpose
Train and validate emotion/behaviour classifiers. Supports both unimodal (per-signal) and multimodal (early fusion) architectures.

### Planned Model Architectures

**Unimodal models** (one per signal — feeds Module 7 decision fusion):
- Random Forest (baseline, interpretable)
- SVM with RBF kernel
- XGBoost
- LSTM / GRU (temporal window sequences)
- 1D CNN (spectral patterns)

**Multimodal early fusion model** (combined feature matrix):
- Gradient Boosting (XGBoost / LightGBM)
- Multi-layer Perceptron
- Transformer encoder on feature sequences

**Training strategy:**
- Stratified k-fold cross-validation (k=5) on autism severity + verbal status
- Leave-one-subject-out (LOSO) for user-dependent vs global model comparison
- Class balancing: SMOTE for minority emotion states (Disgust, Toilet)
- Hyperparameter optimisation: Optuna or GridSearchCV

### Expected Input
- `features_raw/` or `features_normalised/` from Module 3
- `engineered_features.csv` from Module 4

### Expected Output
```
module_5_model_training/outputs/M5_v1.0.0_run_NNN/
  models/
    EDA_rf_model.pkl
    BVP_rf_model.pkl
    ... (one per signal × architecture)
    multimodal_xgb_model.pkl
  training_metrics.csv
  cross_validation_results.csv
  feature_importance_per_model.csv
  confusion_matrices.png
  roc_curves.png
  training_metadata.json
```

---

## Module 6 — Model Evaluation
**Status:** 🔜 Planned

### Planned Purpose
Rigorous evaluation of trained models on held-out test data with clinically relevant metrics.

### Planned Functionality
- **Metrics**: Accuracy, F1 (macro + weighted), precision, recall, Cohen's κ, AUC-ROC per class
- **Clinical metrics**: Sensitivity/specificity per state, false negative rate for high-priority states (Fear, Toilet)
- **Per-demographic analysis**: Performance breakdown by severity, verbal status, age group
- **Calibration**: Reliability diagrams, Expected Calibration Error (ECE)
- **Robustness testing**: Performance under different noise levels, missing signal channels
- **Statistical testing**: McNemar's test for model comparison, Wilcoxon signed-rank

### Expected Input
Trained models from Module 5 + test data from Module 3

### Expected Output
```
module_6_model_evaluation/outputs/M6_v1.0.0_run_NNN/
  evaluation_report.html
  metrics_by_class.csv
  metrics_by_demographic.csv
  confusion_matrices/
  roc_curves/
  calibration_plots/
```

---

## Module 7 — Decision Fusion
**Status:** 🔜 Planned

### Planned Purpose
Combine predictions from unimodal models (EDA, BVP, IBI, ST, ACC) using decision-level fusion strategies to produce a final prediction that outperforms any single-signal model.

### Planned Fusion Strategies
- **Majority voting**: Simple vote across 5 unimodal models
- **Weighted majority voting**: Weight by signal quality index (SQI) from Module 3
- **Confidence-weighted fusion**: Weight by model posterior probabilities
- **Stacking (meta-learner)**: Train a meta-classifier on unimodal outputs
- **Dynamic fusion**: Select subset of signals based on real-time SQI thresholds
- **Dempster-Shafer evidence combination**: Principled uncertainty handling when signals disagree

### Design Rationale
Unimodal models are trained per-signal (Module 5) specifically to enable this fusion. Signal-specific features (EDA SCR rate, IBI RMSSD, ACC dominant frequency) capture complementary aspects of the autonomic response — fusion is expected to give ~8–15% F1 improvement over the best single-signal model based on literature.

### Expected Output
Final fused prediction + confidence scores + per-signal contribution weights

---

## Module 8 — Model Management
**Status:** 🔜 Planned

### Planned Purpose
Version control for trained models, retraining triggers, and model registry.

### Planned Functionality
- **Model registry**: SQLite or MLflow-based versioned model store
- **Performance tracking**: Monitor metric drift across retraining runs
- **Retraining triggers**: Automatic retraining when new labelled data accumulates
- **User-dependent vs global models**: Separate registry entries per user_id
- **Model export**: ONNX export for edge deployment
- **A/B comparison**: Side-by-side evaluation of model versions

### Expected Output
```
module_8_model_management/
  registry/
    models.db             ← SQLite model registry
    model_NNN/            ← versioned model artefacts
  retraining_log.csv
```

---

## Module 9 — Deployment / Inference
**Status:** 🔜 Planned

### Planned Purpose
Real-time emotion/behaviour prediction from live physiological signals on the deployed system. Accepts non-annotated `PipelinePacket` (is_annotated=False from Module 2 Mode 2.4).

### Planned Functionality
- **Real-time inference**: Process incoming signals in sliding windows at the feature extraction rate
- **Alert system**: Trigger caregiver notifications when high-priority states detected (Fear, Toilet, Hunger)
- **Confidence thresholding**: Only output predictions above configurable confidence threshold
- **Multi-user support**: Maintain separate model instances per participant
- **Graceful degradation**: Continue with available signals if one channel is lost/noisy
- **Logging**: Timestamped prediction log for retrospective analysis
- **Edge deployment**: Lightweight model version for Raspberry Pi / tablet deployment

### Routing Logic
```python
# Module 2 routes here via is_annotated=False
if not packet.is_annotated:
    # Deployment path — skip all training modules
    inference_engine.predict(packet)
```

### Expected Output
```
Live predictions → caregiver interface
module_9_deployment/logs/
  predictions_YYYYMMDD.csv
  session_SESSIONID_log.json
```

---

## Pipeline Data Flow

```
                    ┌─────────────────────────────────────────────┐
                    │         AUTISM PHYSIO-AI PIPELINE           │
                    └─────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Module   │    │ Module   │    │ Module   │    │ Module   │
  │   1A     │──▶│   2      │──▶│   3      │──▶│   4      │
  │ Simulate │    │ Acquire  │    │ Preproc. │    │ Feature  │
  │ ✅v1.1.0 │    │ ✅v1.0.0 │    │ ✅v1.0.0 │    │ Engineer.│
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
       ▲               │ is_annotated=False               │
  Also feeds           │ → skip to M9                     ▼
  Module 1B            │                          ┌──────────┐
  (live ingestion)     │                          │ Module   │
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

---

## Development Guidelines

### Adding a New Module

1. Create `module_N_name/` directory
2. Always include:
   - `config.py` — module-specific constants (NOTE: use unique constant names to avoid collision)
   - `main.py` — entry point with both interactive and CLI modes
   - `__init__.py` — public API exports
   - `requirements.txt` — additional dependencies
3. Accept `PipelinePacket` as input (or folder path for standalone use)
4. Output to auto-numbered folder: `outputs/MN_vX.Y.Z_run_NNN/`
5. Add launcher scripts to repo root: `run_module_N_name.bat` / `.sh`
6. Update `pipeline_main.py` with `_isolated_import()` wrapper
7. Update this `CLAUDE.md`

### Code Standards
- PEP 8, max line length 100
- Type hints on all public functions
- NumPy-style docstrings
- All random operations must accept a seeded `np.random.Generator`
- No magic numbers — all constants in `config.py`

### Testing
```powershell
# From repo root
python -m pytest tests/ -v --tb=short
```

Current tests: `tests/test_module_1a.py` — 35 tests covering Module 2A.

---

## Key Dependencies

```
numpy>=1.24.0        # Use trapezoid() not trapz(); ptp() removed
scipy>=1.10.0        # Signal processing, spectral analysis
pandas>=2.0.0        # Data manipulation
matplotlib>=3.7.0    # Visualisation
seaborn>=0.12.0      # Plot styling
scikit-learn>=1.3.0  # RobustScaler, model utilities
antropy>=0.1.9       # Sample entropy, DFA
pykalman>=0.9.7      # Kalman filter (fallback implementation also available)
pytest>=7.4.0        # Testing
```

---

## Literature References

1. **Kreibig (2010)** — Autonomic NS reactions to emotions (primary emotion→physiology reference)
2. **Stephens et al. (2022)** — EDA in ASD: baseline and reactivity
3. **Kushki et al. (2013)** — ANS and severity in autistic children
4. **Task Force ESC (1996)** — HRV standards (SDNN, RMSSD, spectral bands)
5. **Boucsein (2012)** — Electrodermal Activity textbook (SCR kernel model)
6. **Schoen et al. (2008)** — Sensory over-responsivity and arousal in ASD (severity reactivity modifiers)
7. **Loomes et al. (2017)** — ASD gender ratio meta-analysis (4:1 M:F for demographics)
8. **Clifford et al. (2006)** — ECG/PPG analysis methods (AR(1) HRV, PPG waveform)

---

## Development Philosophy

### Seven Principles for Clinical AI Pipeline Development

These principles govern all development on this codebase. They are derived from IEC 62304 (medical device software), FDA Good Machine Learning Practice, and established scientific computing standards.

**1. Every number has a provenance trail.**
Every data point flowing through the pipeline must be traceable from raw sensor sample to final prediction. Log all transformations with timestamps, parameter values, and software versions. When a signal value changes (cleaning, filtering, feature extraction), record what changed it and why.

**2. Determinism is non-negotiable.**
Given the same input and seed, the pipeline must produce bit-identical output. All `np.random.Generator` instances must be seeded and passed explicitly — never rely on global random state. Pin all dependency versions.

**3. Fail loudly on physiological implausibility.**
Silent data corruption is the worst failure mode in clinical software. Validate signal ranges at every boundary. EDA cannot be negative. IBI cannot be 50 ms. If a value exits its physiological range after a transformation, raise an error — do not clamp and continue silently.

**4. Defend the vulnerable population.**
This pipeline processes data from non-verbal autistic children who cannot report errors. Performance must be validated per demographic subgroup: severity level, verbal status, age bracket. A model that works well on average but fails for severe/non-verbal participants is clinically dangerous.

**5. Separate what you know from what you infer.**
Raw signals, cleaned signals, extracted features, and model predictions are fundamentally different epistemic categories. Never mix them in the same DataFrame without explicit column naming. The `is_annotated` flag enforces the critical boundary between training data and deployment data.

**6. Code review must verify numerical correctness, not just style.**
For signal processing code, reviewers must check: filter stability, boundary effects, window alignment across sampling rates, and NaN propagation. A Butterworth filter with wrong cutoff silently degrades every downstream feature.

**7. Design for the audit, not just the experiment.**
Structure outputs so a regulatory reviewer or ethics board can, without running code, understand what happened: which data went in, what parameters were used, what came out, and what software version produced it. Auto-numbered output folders with full metadata serve this purpose. Never overwrite previous runs.

---

## Development Workflow

### The Develop-Audit Ratchet

This project follows a **develop-audit-advance** workflow that ensures quality gates at every step. The pattern prevents regressions and catches domain-specific issues that standard development practices miss.

### Workflow Cycle — Tiered Audit System

The workflow uses four tiers of review, triggered at different cadences depending on what's being built.

```
TIER 1 — EVERY FEATURE (core quality gate)
  ├── Code Review Agent         — style, data integrity, performance, contracts
  ├── Logic Review Agent        — signal processing, physiology, statistics
  └── Security Audit Agent      — privacy, injection, dependencies

TIER 2 — EVERY ML FEATURE (modules 5-7 only)
  └── ML Experiment Validator   — leakage, stratification, class imbalance, metrics

TIER 3 — EVERY 2 FEATURES (integration retro)
  ├── Integration Architect     — cross-module contracts, output compatibility
  └── Documentation Auditor     — CLAUDE.md drift, metadata completeness

TIER 4 — MILESTONE GATES (before major releases)
  ├── Deployment Readiness      — latency, recovery, edge viability, alerts
  └── Frontend & Accessibility  — WCAG, clinical language, tablet UX
```

### Step-by-Step Protocol

#### Phase 1: DEVELOP
- Build the feature following the module structure and code standards
- Write tests alongside the implementation
- Run `python -m pytest tests/ -v --tb=short` — all tests must pass
- Commit working code

#### Phase 2: AUDIT (Single Feature)
Run Tier 1 agents independently (they have no shared context to avoid bias):

1. **Code Review** (`.agents/code-review.md`)
   - Structure, style, data integrity, performance, inter-module contract

2. **Logic Review** (`.agents/logic-review.md`)
   - Signal processing validity, feature correctness, physiological plausibility

3. **Security Audit** (`.agents/security-audit.md`)
   - Data privacy, input validation, dependency safety, clinical data protection

**If building an ML feature (Modules 5-7)**, also run:

4. **ML Experiment Validator** (`.agents/ml-experiment-validator.md`)
   - Data leakage prevention, stratification, class imbalance, experiment reproducibility

#### Phase 3: FIX
- Address all **CRITICAL** findings immediately
- Address all **HIGH** findings before proceeding
- **MEDIUM** findings: fix in the same cycle if quick, otherwise track for next cycle
- **LOW** findings: document and address in batch cleanup cycles

#### Phase 4: RETRO (Every Two Features)
After every second feature, run Tier 3 agents on BOTH features together:

5. **Integration Architect** (`.agents/integration-architect.md`)
   - Cross-module data flow, PipelinePacket contracts, output format compatibility

6. **Documentation & Compliance Auditor** (`.agents/documentation-compliance.md`)
   - CLAUDE.md accuracy, metadata completeness, regulatory evidence trail

This catches **integration drift** and **documentation rot** that single-feature audits miss.

#### Phase 5: MILESTONE GATE (Before Releases)
Before deployment or frontend milestones, run Tier 4 agents:

7. **Deployment Readiness Reviewer** (`.agents/deployment-readiness.md`)
   - Latency budgets, failure recovery, graceful degradation, edge deployment viability

8. **Frontend & Accessibility Reviewer** (`.agents/frontend-accessibility.md`)
   - WCAG 2.1 AA, clinical language, prediction uncertainty display, tablet responsiveness

### Triggering Reviews

**Automatic (during development cycle):**
- Tier 1: After completing any feature, before marking it done
- Tier 2: After completing any ML feature (Modules 5-7)
- Tier 3: After completing every second feature, on the pair
- Tier 4: Before any deployment or UI release milestone

**Manual (anytime):**
- When modifying existing modules → Tier 1
- Before merging any branch → Tier 1
- When changing shared infrastructure (`pipeline_main.py`, `PipelinePacket`, `config.py`) → Tier 1 + Tier 3
- When adding or updating dependencies → Tier 1 (Security) + Tier 3 (Integration)
- When touching any module's output format (contract change) → Tier 3 (Integration + Documentation)
- When updating CLAUDE.md or README → Tier 3 (Documentation)

### Audit Verdict Escalation

| Agent Verdict | Action Required |
|---------------|----------------|
| APPROVE / SOUND / PASS / VALID / COHERENT / ACCURATE / CLINICALLY APPROPRIATE / PRODUCTION READY | Advance to next phase |
| REQUEST CHANGES / REVISE / CONDITIONAL PASS / METHODOLOGICAL CONCERNS / DRIFT DETECTED / REVISIONS NEEDED / NEEDS HARDENING | Fix findings, re-run that agent |
| BLOCK / REJECT / FAIL / INVALID / BROKEN CONTRACT / MISLEADING / UNSAFE FOR CLINICAL USE / NOT DEPLOYABLE | Stop. Fix all findings. Re-run ALL agents in the relevant tier |

---

## Review Agents

Eight independent review agents live in `.agents/`. Each operates with NO prior context — they review only what the code says, preventing confirmation bias.

### Tier 1 — Core Quality Gate (Every Feature)

| Agent | File | Focus | Verdict |
|-------|------|-------|---------|
| **Code Review** | `.agents/code-review.md` | Quality, style, data integrity, performance, contracts | APPROVE / REQUEST CHANGES / BLOCK |
| **Logic Review** | `.agents/logic-review.md` | Algorithms, physiology, math, statistics | SOUND / REVISE / REJECT |
| **Security Audit** | `.agents/security-audit.md` | Privacy, injection, dependencies, clinical data | PASS / CONDITIONAL PASS / FAIL |

### Tier 2 — ML Methodology (ML Features Only)

| Agent | File | Focus | Verdict |
|-------|------|-------|---------|
| **ML Experiment Validator** | `.agents/ml-experiment-validator.md` | Leakage, stratification, imbalance, reproducibility, metrics | VALID / METHODOLOGICAL CONCERNS / INVALID |

### Tier 3 — Integration & Documentation (Every 2 Features)

| Agent | File | Focus | Verdict |
|-------|------|-------|---------|
| **Integration Architect** | `.agents/integration-architect.md` | Cross-module data flow, PipelinePacket contracts, output compatibility | COHERENT / DRIFT DETECTED / BROKEN CONTRACT |
| **Documentation Auditor** | `.agents/documentation-compliance.md` | CLAUDE.md accuracy, metadata completeness, regulatory evidence | ACCURATE / DRIFT DETECTED / MISLEADING |

### Tier 4 — Milestone Gates (Before Releases)

| Agent | File | Focus | Verdict |
|-------|------|-------|---------|
| **Deployment Readiness** | `.agents/deployment-readiness.md` | Latency, recovery, degradation, edge viability, alerts | PRODUCTION READY / NEEDS HARDENING / NOT DEPLOYABLE |
| **Frontend & Accessibility** | `.agents/frontend-accessibility.md` | WCAG 2.1, clinical language, uncertainty display, tablet UX | CLINICALLY APPROPRIATE / REVISIONS NEEDED / UNSAFE FOR CLINICAL USE |

### Why Eight Separate Agents?

1. **Separation of concerns**: Each agent owns a distinct expertise domain — code quality, physiological correctness, security, ML methodology, systems integration, documentation accuracy, production operations, and clinical UX.
2. **No bias**: Each agent starts fresh — no memory of design discussions, no attachment to implementation decisions.
3. **Tiered efficiency**: Not all agents run every time. Core agents run on every feature; specialist agents activate only when their domain is relevant.
4. **Complete lifecycle coverage**: The eight agents together cover build → validate → integrate → deploy → present — the full clinical AI pipeline lifecycle.

---

## Development Rules

Six rule files in `.claude/rules/` codify the non-negotiable standards:

| Rule | File | Summary |
|------|------|---------|
| Clinical Data Integrity | `01-clinical-data-integrity.md` | Validate ranges, no silent drops, preserve timestamps, NaN handling |
| Reproducibility | `02-reproducibility.md` | Seeded randomness, versioned outputs, full parameter recording |
| Module Contract | `03-module-contract.md` | PipelinePacket only, isolated imports, annotation routing |
| Numerical Safety | `04-numerical-safety.md` | NumPy 2.x compat, division guards, filter stability, float tolerance |
| Testing Standards | `05-testing-standards.md` | Every module tested, deterministic, plausibility checks |
| Security Baseline | `06-security-baseline.md` | No eval/pickle on untrusted input, path validation, data minimisation |

These rules are enforced by the review agents during every audit cycle.

---

## Glossary

| Term | Definition |
|------|------------|
| EDA | Electrodermal Activity — skin conductance |
| SCL | Skin Conductance Level — tonic EDA component |
| SCR | Skin Conductance Response — phasic EDA component |
| BVP | Blood Volume Pulse — PPG signal from Empatica E4 |
| IBI | Inter-Beat Interval — time between consecutive heartbeats (ms) |
| HRV | Heart Rate Variability — variation in IBI |
| RMSSD | Root Mean Square of Successive Differences — key HRV metric |
| SDNN | Standard Deviation of NN intervals — overall HRV |
| ST | Skin Temperature — wrist peripheral temperature |
| ACC | Accelerometer — 3-axis wrist movement |
| SVM | Signal Vector Magnitude — √(X²+Y²+Z²) for ACC |
| SQI | Signal Quality Index — 0–1 per window |
| PipelinePacket | Standardised data container passed between modules |
| LOSO | Leave-One-Subject-Out cross-validation |
| M2A | Module 2A (Data Simulation) |
| M1 | Module 1 (Data Acquisition) |
| M3 | Module 3 (Data Preprocessing) |
| ASD | Autism Spectrum Disorder |
| ANS | Autonomic Nervous System |
| DSM-5 | Diagnostic and Statistical Manual 5th edition (severity levels) |
