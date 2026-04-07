# Architecture — Module 1A: Data Simulation

## Design Philosophy

Module 1A is built around three principles:

**1. Format parity with real devices.** Every output file — column names, units, sampling rates, file structure — exactly matches what Module 1B (Live Data Ingestion) will produce from a physical Empatica E4 or equivalent device. This means that every downstream module (preprocessing, feature extraction, model training) operates identically whether data came from the simulator or a real child.

**2. Physical realism over simplicity.** Signals are generated from first-principles physiological models rather than statistical noise around a mean. This produces artefacts, correlations, and dynamics that real models must learn to handle — making the synthetic data genuinely useful for pipeline validation.

**3. Full reproducibility.** A single integer seed controls every random process in the pipeline end-to-end, including event scheduling, baseline generation, event modulation, and noise injection. Given the same seed and parameters, the bit-identical output is produced every time.

---

## File Dependency Map

```
config.py
    ├── signal_models.py
    ├── event_scheduler.py
    │       └── simulator.py
    │               ├── noise_injector.py
    │               ├── annotator.py
    │               ├── visualizer.py       ← independent consumer of SimulationResult
    │               └── exporter.py         ← independent consumer of SimulationResult
    └── main.py  ← CLI / entry point
```

No circular imports. `config.py` is the only shared dependency across all files.

---

## Simulation Pipeline — Step by Step

### Step 1 — Event Scheduling

`EventScheduler` resolves three user-facing parameters into a concrete list of `EventConfig` objects:

**n_events resolution:**
- Integer → used directly
- `"random"` → sampled from `Uniform(1, max_events)`

**Event duration resolution:**
- Float → used for all events
- `"random"` → each event independently sampled from `Uniform(event_dur_min_s, event_dur_max_s)`

**Emotion resolution:**
- `None` → each event randomly sampled from all 10 states, with no immediate repeats
- Single string `"Anger"` → all events use `Anger`
- List `["Anger", "Fear"]` → each event randomly sampled from the provided subset

**Placement:** Events are placed greedily from left to right with:
- A minimum `min_lead_s` (default 10 s) quiet period before the first event
- A minimum `min_gap_s` (default 15 s) gap between consecutive events
- Events that cannot fit within the recording duration are silently dropped with a console warning

### Step 2 — Baseline Signal Generation

All signals start from a neutral resting state. Baselines are subject-specific: each simulation draw slightly different tonic levels, temperature baselines, and ACC orientations via the seeded RNG, simulating inter-subject variability.

| Signal | Baseline Components |
|--------|-------------------|
| EDA | Mean tonic level (3.0 µS ± σ), slow sinusoidal ultradian drift (0.004 Hz), spontaneous SCR events at 0.05/s |
| BVP | Beat-by-beat reconstruction from AR(1) IBI sequence at 75 bpm, SDNN 28 ms |
| IBI | AR(1) RR interval series: `RR_n = µ + φ·(RR_{n-1} − µ) + ε`, φ=0.55 |
| ST | Mean (33.0°C ± 0.4), slow sinusoidal drift (0.002 Hz), low-pass filtered random walk |
| ACC_X/Y/Z | Gravity component (Z≈0.98g), 0.25 Hz breathing artefact, white noise floor |

### Step 3 — Event Modulation

For each event window, the baseline signal is additively modified by the emotion-specific perturbation:

**EDA modulation:**
- Tonic elevation: `Δtonic(t) = tonic_delta · (1 − e^{-t/τ_rise})`
- Partial recovery at event end: `−tonic_delta · recovery_factor · (1 − e^{-t/τ_rec})`
- Phasic SCRs at emotion-specific rate and amplitude, placed using exponential inter-arrival times

**BVP/IBI modulation:**
- The entire event window is regenerated at the target HR and HRV (rather than adding to the baseline), ensuring physiological beat timing throughout
- Target HR = `baseline_hr + hr_delta_bpm` (clipped to 40–200 bpm)
- HRV = `baseline_hrv_std × hrv_factor`
- IBI arrays are patched: baseline beats in the event window are replaced with event-rate beats

**ST modulation:**
- Additive: `ΔST(t) = delta_celsius · (1 − e^{-rate·t})`
- Very slow approach (rate 0.005–0.20 °C/s) reflecting thermal inertia of skin

**ACC modulation:**
- Emotion-specific sinusoidal activity at `dominant_freq_hz` and `activity_amp`
- Smooth onset/offset envelope using `tanh` shaping to avoid step artefacts
- Irregular jitter component added on top

### Step 4 — Noise Injection

`NoiseInjector` adds layered realistic noise after all signal generation is complete. This ordering (signal first, then noise) ensures clean ground-truth signals exist internally before degradation, which is used by the annotator for SQI computation.

Noise is additive. Multiple independent noise components are summed:

```
signal_noisy = signal_clean + gaussian_noise + drift_noise + powerline + motion_artefact
```

Each component is independently seeded from the main RNG.

### Step 5 — Physiological Range Clipping

After noise injection, all signals are hard-clipped to their physiological ranges using `SIGNAL_RANGES` from `config.py`. This prevents physically impossible values (e.g. negative EDA, temperature above 40°C) from propagating to downstream modules.

### Step 6 — Auto-Annotation

`AutoAnnotator` derives four structured annotation tables from the completed simulation. It does not have access to the clean pre-noise signals — SQI is computed on the final noisy signal, as a downstream module would receive it.

---

## Signal Model Equations

### SCR Kernel (EDA phasic component)

```
SCR(t) = A · (1 − e^{−t/τ_rise}) · e^{−t/τ_decay}

A         = amplitude (µS), emotion-specific
τ_rise    ≈ 0.5 s  (fast onset)
τ_decay   ≈ 5.0 s  (slow recovery)
max_dur   = 20 s   (kernel truncation)
```

### AR(1) HRV Model (IBI)

```
RR_n = µ_RR + φ · (RR_{n-1} − µ_RR) + ε_n

µ_RR = 60 / HR_bpm  (mean RR interval in seconds)
φ    = 0.55          (correlation between consecutive beats)
ε_n  ~ N(0, σ_RR)   σ_RR = SDNN / 1000
```

The AR(1) structure produces short-range HRV correlations consistent with healthy autonomic regulation. A pure Gaussian model would produce unrealistically independent beat-to-beat variation.

### 3-Gaussian PPG Template

```
BVP_beat(t) = G1(t) + G2(t) + G3(t)

G1(t) = A        · exp(−(t−0.18)² / 2·0.06²)   systolic peak
G2(t) = −0.12·A  · exp(−(t−0.42)² / 2·0.03²)   dicrotic notch
G3(t) = 0.30·A   · exp(−(t−0.54)² / 2·0.08²)   diastolic peak

t ∈ [0, 1] normalised beat cycle
A = amplitude (nT), emotion-specific factor applied
```

### Thermal Approach (ST)

```
ST(t) = ST_baseline + ΔT · (1 − e^{−rate·t})

ΔT   = emotion-specific delta (°C), e.g. −0.65 for Fear
rate = emotion-specific approach rate (°C/s)
```

---

## SimulationResult Data Contract

`SimulationResult` is the data contract between Module 1A and all downstream modules. Its structure is version-controlled and must not be changed without a corresponding version bump.

```python
@dataclass
class SimulationResult:
    signals:       Dict[str, np.ndarray]   # post-noise, clipped
    time_vectors:  Dict[str, np.ndarray]   # seconds from recording start
    ibi_times_s:   np.ndarray              # beat onset times
    ibi_values_ms: np.ndarray              # inter-beat intervals in ms
    events:        List[EventConfig]        # scheduled events
    annotations:   Dict[str, pd.DataFrame] # four annotation tables
    metadata:      dict                     # run parameters + stats
    duration_s:    float
```

**Stability guarantee:** The keys of `signals`, `time_vectors`, and `annotations` are fixed constants defined in `config.py`:

```python
SIGNAL_NAMES = ["EDA", "BVP", "IBI", "ST", "ACC_X", "ACC_Y", "ACC_Z"]
```

Downstream modules may hard-code these keys.

---

## Random State Management

A single `numpy.random.default_rng(seed)` is created in `DataSimulator.__init__()` and passed by reference to all generators. Sub-generators (e.g. in `NoiseInjector`) derive a new seed from the master RNG:

```python
child_seed = int(self._rng.integers(0, 9999))
noise_rng  = np.random.default_rng(child_seed)
```

This ensures that:
1. The full simulation is reproducible from a single integer seed.
2. Adding or removing noise components does not change the signal generation sequence (the master RNG is not consumed by noise generation).

---

## Performance Characteristics

| Configuration | Duration | Events | Signals Generated | Simulation Time |
|---|---|---|---|---|
| Default (medium noise) | 300 s | 5 | ~50,000 samples | ~0.11 s |
| Extended (low noise) | 600 s | 8 | ~100,000 samples | ~0.20 s |
| Long session | 3600 s | 20 | ~600,000 samples | ~1.2 s |

Visualisation (PNG generation) typically adds 2–5 seconds depending on signal length. Use `--no_plots` for batch generation.

The dominant cost is BVP generation (64 Hz × duration) and matplotlib rendering. All signal generation is vectorised NumPy; no Python loops over samples exist except for SCR kernel placement (sparse event loop).

---

## Testing Strategy

The module is designed for straightforward unit testing:

- **Signal shape validation:** `len(signals["EDA"]) == int(duration_s * 4)`
- **Range validation:** `signals["EDA"].min() >= 0.01` and `signals["EDA"].max() <= 30.0`
- **Event modulation validation:** Mean EDA in event windows should exceed mean EDA in baseline windows for high-arousal emotions
- **Annotation completeness:** `annotations["sample_labels"]` should have exactly `int(duration_s * 64)` rows
- **Reproducibility:** Two runs with the same seed and parameters should produce bit-identical outputs
- **CSV roundtrip:** Re-reading a written CSV should produce arrays within floating-point precision of the originals

A test suite (`tests/`) is planned for a future release.
