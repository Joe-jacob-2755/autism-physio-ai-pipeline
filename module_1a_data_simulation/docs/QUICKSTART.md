# Quickstart Guide — Module 1A

Get up and running in 5 minutes.

---

## 1. Install

```bash
git clone https://github.com/your-org/autism-physio-ai-pipeline.git
cd autism-physio-ai-pipeline/module_1a_data_simulation
pip install -r requirements.txt
```

---

## 2. Run your first simulation

```bash
python main.py
```

This runs a 5-minute recording with 5 randomly selected emotion events at medium noise. Output goes to `./output/`.

---

## 3. Check the outputs

```
output/
├── combined_signals.png      ← Open this first
├── signal_EDA.png
├── signal_BVP.png
├── signal_IBI.png
├── signal_ST.png
├── signal_ACC_combined.png
├── combined_signals.csv
├── EDA.csv, BVP.csv, IBI.csv, ST.csv, ACC.csv
├── annotations_events.csv
└── metadata.json
```

Open `combined_signals.png` to see all 7 signal channels with emotion event shading.

---

## 4. Try a targeted scenario

```bash
# Simulate a child showing Fear and Anger (high-arousal states)
python main.py \
  --emotions "Fear,Anger" \
  --n_events 4 \
  --duration 240 \
  --noise medium \
  --seed 42 \
  --out output_fear_anger
```

---

## 5. See available emotion labels

```bash
python main.py --list_emotions
```

The 10 available states are:

**Affective:** `Happy`, `Anger`, `Fear`, `Disgust`, `Sad`, `Surprise`

**Physiological needs:** `Hunger`, `Thirst`, `Toilet`, `Tired`

---

## 6. Use in Python

```python
from simulator  import DataSimulator
from visualizer import SignalVisualizer
from exporter   import DataExporter

# Create and run simulation
result = DataSimulator(
    duration_s   = 300,
    n_events     = 5,
    emotions     = None,       # random
    noise_level  = "medium",
    seed         = 42,
).simulate()

# Access signal data directly
eda = result.signals["EDA"]          # numpy array, 4 Hz
bvp = result.signals["BVP"]          # numpy array, 64 Hz
print(f"EDA mean: {eda.mean():.2f} µS")
print(f"Beats detected: {len(result.ibi_times_s)}")

# Save plots and CSVs
SignalVisualizer(result, output_dir="my_output").save_all()
DataExporter(result, output_dir="my_output").export_all()
```

---

## 7. What's next?

- Read the full [CLI Reference](README.md#cli-reference)
- Understand the [Signal Specifications](docs/SIGNAL_SPECIFICATIONS.md)
- Study the [Emotion Profiles](docs/EMOTION_PROFILES.md)
- Review the [Architecture](docs/ARCHITECTURE.md) before integrating with downstream modules
- Check the [API Reference](docs/API_REFERENCE.md) for all available classes and methods
