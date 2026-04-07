# API Reference — Module 1A: Data Simulation

Complete reference for all public classes, methods, and functions.

---

## `DataSimulator`

*`simulator.py`*

Master simulation orchestrator. Accepts all user-facing parameters and returns a `SimulationResult`.

### Constructor

```python
DataSimulator(
    duration_s        : float                         = 300,
    n_events          : Union[int, str]               = 5,
    event_duration_s  : Union[float, str]             = 30.0,
    emotions          : Optional[Union[str, list]]    = None,
    noise_level       : str                           = "medium",
    seed              : Optional[int]                 = 42,
    min_gap_s         : float                         = 15.0,
    min_lead_s        : float                         = 10.0,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `duration_s` | `float` | `300` | Total recording duration in seconds. Minimum recommended: 60 s. |
| `n_events` | `int` or `"random"` | `5` | Number of emotion/behaviour events to schedule. `"random"` samples from [1, 10]. |
| `event_duration_s` | `float` or `"random"` | `30.0` | Duration of each event in seconds. `"random"` samples each event from [10, 60] s. Minimum: 5 s. |
| `emotions` | `None`, `str`, or `list[str]` | `None` | Target emotion(s). `None` = fully random from all 10 states. Single string = that emotion repeated. List = random sample from subset. |
| `noise_level` | `"low"`, `"medium"`, `"high"` | `"medium"` | Noise injection intensity. See [Noise Model](README.md#noise-model). |
| `seed` | `int` or `None` | `42` | Master random seed. `None` = non-deterministic. |
| `min_gap_s` | `float` | `15.0` | Minimum quiet gap between consecutive events (seconds). |
| `min_lead_s` | `float` | `10.0` | Quiet baseline period before the first event (seconds). |

#### Methods

**`simulate() → SimulationResult`**

Run the complete 6-step simulation pipeline and return results.

```python
sim    = DataSimulator(duration_s=300, n_events=5, seed=42)
result = sim.simulate()
```

---

## `SimulationResult`

*`simulator.py`*

Dataclass returned by `DataSimulator.simulate()`. All downstream modules consume this object.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `signals` | `Dict[str, np.ndarray]` | Post-noise, clipped signal arrays keyed by signal name (`"EDA"`, `"BVP"`, `"ST"`, `"ACC_X"`, `"ACC_Y"`, `"ACC_Z"`). |
| `time_vectors` | `Dict[str, np.ndarray]` | Time vectors in seconds for each signal, at native sampling rate. Same keys as `signals`. Also contains `"IBI"` key for beat times. |
| `ibi_times_s` | `np.ndarray` | Beat onset times in seconds (event-based, not uniform). |
| `ibi_values_ms` | `np.ndarray` | Inter-beat interval values in milliseconds (same length as `ibi_times_s`). |
| `events` | `List[EventConfig]` | Ordered list of scheduled events. |
| `annotations` | `Dict[str, pd.DataFrame]` | Four annotation tables: `"events"`, `"baseline_wins"`, `"signal_quality"`, `"sample_labels"`. |
| `metadata` | `dict` | Run parameters, signal statistics, and elapsed time. |
| `duration_s` | `float` | Recording duration in seconds. |

### Methods

**`summary() → str`**

Returns a formatted text summary of the simulation result, suitable for console output.

```python
print(result.summary())
```

---

## `EventConfig`

*`event_scheduler.py`*

Dataclass representing a single labelled physiological event window.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `emotion` | `str` | Emotion/behaviour label (must be in `EMOTIONS_ALL`). |
| `start_s` | `float` | Event start time in seconds from recording onset. |
| `duration_s` | `float` | Event duration in seconds. |

### Properties (read-only)

| Property | Type | Description |
|----------|------|-------------|
| `end_s` | `float` | `start_s + duration_s` |
| `profile` | `dict` | Full emotion profile dict from `EMOTION_PROFILES` |
| `color` | `str` | Hex colour string for plotting |
| `bvp_hr_delta` | `float` | HR change in bpm for this emotion |
| `bvp_hr_rise_s` | `float` | HR onset time constant in seconds |

---

## `EventScheduler`

*`event_scheduler.py`*

Generates a validated list of non-overlapping `EventConfig` objects.

### Constructor

```python
EventScheduler(
    duration_s        : float,
    n_events          : Union[int, str]               = 5,
    event_duration_s  : Union[float, str]             = 30.0,
    emotions          : Optional[Union[str, list]]    = None,
    rng               : Optional[np.random.Generator] = None,
    min_gap_s         : float                         = 15.0,
    min_lead_s        : float                         = 10.0,
    max_events        : int                           = 10,
    event_dur_min_s   : float                         = 10.0,
    event_dur_max_s   : float                         = 60.0,
)
```

### Methods

**`schedule() → List[EventConfig]`**

Generate and return the event list. Returns an empty list if no events fit within the recording.

```python
scheduler = EventScheduler(duration_s=300, n_events=5, emotions="Anger")
events    = scheduler.schedule()
```

---

## Event Factory Functions

*`event_scheduler.py`*

Convenience shortcuts for common scheduling patterns.

### `make_events_specific`

```python
make_events_specific(
    emotion     : str,
    n_events    : int,
    duration_s  : float,
    total_dur_s : float,
    rng         : Optional[np.random.Generator] = None,
) → List[EventConfig]
```

Schedule `n_events` identical events of a single emotion.

```python
events = make_events_specific("Anger", n_events=4, duration_s=25.0, total_dur_s=300)
```

### `make_events_multiple`

```python
make_events_multiple(
    emotions    : List[str],
    n_events    : int,
    duration_s  : float,
    total_dur_s : float,
    rng         : Optional[np.random.Generator] = None,
) → List[EventConfig]
```

Schedule `n_events` randomly drawn from an emotion subset.

```python
events = make_events_multiple(["Fear", "Anger", "Surprise"],
                               n_events=6, duration_s=20.0, total_dur_s=300)
```

### `make_events_random`

```python
make_events_random(
    total_dur_s     : float,
    max_events      : int   = 8,
    event_dur_min_s : float = 15.0,
    event_dur_max_s : float = 45.0,
    rng             : Optional[np.random.Generator] = None,
) → List[EventConfig]
```

Fully random schedule: random count, random durations, random emotions.

```python
events = make_events_random(total_dur_s=300, max_events=8)
```

---

## `NoiseInjector`

*`noise_injector.py`*

Adds realistic layered noise to a dictionary of signal arrays.

### Constructor

```python
NoiseInjector(
    level : str              = "medium",   # "low" | "medium" | "high"
    seed  : Optional[int]    = None,
)
```

### Methods

**`inject(signals: Dict[str, np.ndarray]) → Dict[str, np.ndarray]`**

Add noise to every signal in the dictionary. Returns a new dict (does not modify in-place).

```python
injector      = NoiseInjector(level="medium", seed=99)
noisy_signals = injector.inject(raw_signals)
```

---

## `AutoAnnotator`

*`annotator.py`*

Generates structured annotation DataFrames for a completed simulation.

### Constructor

```python
AutoAnnotator(
    duration_s : float,
    events     : List[EventConfig],
    signals    : Dict[str, np.ndarray],
)
```

### Methods

**`annotate() → Dict[str, pd.DataFrame]`**

Run full annotation pipeline and return four DataFrames.

```python
annotator   = AutoAnnotator(duration_s=300, events=events, signals=signals)
annotations = annotator.annotate()

# Keys: "events", "baseline_wins", "signal_quality", "sample_labels"
event_df = annotations["events"]
sqi_df   = annotations["signal_quality"]
```

---

## `compute_sqi`

*`annotator.py`*

```python
compute_sqi(
    segment     : np.ndarray,
    signal_name : str,
    fs          : int,
) → float
```

Compute a [0.0, 1.0] Signal Quality Index for a signal segment.

| Return Value | Interpretation |
|---|---|
| 0.90 – 1.00 | Excellent quality |
| 0.70 – 0.89 | Good quality, minor artefacts |
| 0.50 – 0.69 | Moderate quality, use with caution |
| < 0.50 | Poor quality, consider rejection |

```python
sqi = compute_sqi(eda_segment, "EDA", fs=4)
```

---

## `SignalVisualizer`

*`visualizer.py`*

Generates publication-quality PNG figures from a `SimulationResult`.

### Constructor

```python
SignalVisualizer(
    result     : SimulationResult,
    output_dir : str = "output",
    dpi        : int = 150,
)
```

### Methods

**`save_all() → Dict[str, Path]`**

Generate and save all figures. Returns `{label: file_path}`.

```python
viz   = SignalVisualizer(result, output_dir="figures/", dpi=200)
paths = viz.save_all()
```

**`plot_individual_signals() → Dict[str, Path]`**

Generate one figure per signal channel (EDA, BVP, IBI, ST, ACC_X, ACC_Y, ACC_Z).

**`plot_combined() → Path`**

Generate a single 7-panel figure with all channels.

**`plot_acc_combined() → Path`**

Generate a 3-panel figure comparing all three ACC axes.

---

## `DataExporter`

*`exporter.py`*

Writes simulation output to CSV and JSON files.

### Constructor

```python
DataExporter(
    result     : SimulationResult,
    output_dir : str = "output",
)
```

### Methods

**`export_all() → Dict[str, Path]`**

Export all files. Returns `{label: file_path}`.

```python
exporter = DataExporter(result, output_dir="results/")
files    = exporter.export_all()
```

**`export_individual_signals() → Dict[str, Path]`**

Write per-signal CSVs: `EDA.csv`, `BVP.csv`, `IBI.csv`, `ST.csv`, `ACC.csv`.

**`export_combined() → Path`**

Write `combined_signals.csv` — all channels interpolated to 64 Hz reference grid with label columns.

**`export_annotations() → Dict[str, Path]`**

Write all four annotation CSVs.

**`export_metadata() → Path`**

Write `metadata.json` with run parameters and signal statistics.

---

## `run_simulation`

*`main.py`*

High-level function combining simulate + visualise + export in a single call.

```python
run_simulation(
    duration_s       : float,
    n_events         : Union[int, str],
    event_duration_s : Union[float, str],
    emotions         : Optional[Union[str, list]],
    noise_level      : str,
    seed             : int,
    output_dir       : str,
    generate_plots   : bool = True,
) → SimulationResult
```

```python
from main import run_simulation

result = run_simulation(
    duration_s       = 300,
    n_events         = 5,
    event_duration_s = 30.0,
    emotions         = ["Anger", "Fear"],
    noise_level      = "medium",
    seed             = 42,
    output_dir       = "my_output/",
    generate_plots   = True,
)
```

---

## Configuration Constants

*`config.py`*

All constants are importable directly:

```python
from config import (
    SAMPLING_RATES,    # Dict[str, Optional[int]]
    SIGNAL_UNITS,      # Dict[str, str]
    SIGNAL_RANGES,     # Dict[str, Tuple[float, float]]
    BASELINE,          # Dict[str, dict]
    NOISE_PROFILES,    # Dict[str, Dict[str, dict]]
    EMOTION_PROFILES,  # Dict[str, dict]
    EMOTIONS_ALL,      # List[str]  — all 10 states
    EMOTIONS_AFFECTIVE,# List[str]  — 6 affective emotions only
    EMOTIONS_NEEDS,    # List[str]  — 4 physiological needs only
    SIGNAL_NAMES,      # List[str]
    DEFAULT_DURATION_S,
    DEFAULT_N_EVENTS,
    DEFAULT_EVENT_DUR_S,
    DEFAULT_NOISE_LEVEL,
    DEFAULT_SEED,
    POWERLINE_FREQ_HZ,
)
```

### `EMOTIONS_ALL`
```python
['Happy', 'Anger', 'Fear', 'Disgust', 'Sad', 'Surprise',
 'Hunger', 'Thirst', 'Toilet', 'Tired']
```

### `SAMPLING_RATES`
```python
{'EDA': 4, 'BVP': 64, 'IBI': None, 'ST': 4, 'ACC_X': 32, 'ACC_Y': 32, 'ACC_Z': 32}
```

### `SIGNAL_RANGES`
```python
{'EDA': (0.01, 30.0), 'BVP': (-300.0, 300.0), 'IBI': (300.0, 1500.0),
 'ST': (25.0, 40.0), 'ACC_X': (-4.0, 4.0), 'ACC_Y': (-4.0, 4.0), 'ACC_Z': (-4.0, 4.0)}
```

---

## Package-Level Imports

```python
from module_1a_data_simulation import (
    DataSimulator,
    SimulationResult,
    SignalVisualizer,
    DataExporter,
    EventConfig,
    EventScheduler,
    make_events_specific,
    make_events_multiple,
    make_events_random,
    EMOTIONS_ALL,
    EMOTION_PROFILES,
    SAMPLING_RATES,
)
```
