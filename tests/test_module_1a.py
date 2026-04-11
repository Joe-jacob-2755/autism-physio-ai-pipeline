"""
=============================================================================
MODULE 1A — TEST SUITE
=============================================================================
Run with:
    pytest tests/ -v --tb=short

From the repo root, with the virtual environment active.
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module_2a_data_simulation"))

import pytest
import pandas as pd
import numpy as np

from annotator import AutoAnnotator, compute_sqi
from noise_injector import NoiseInjector
from event_scheduler import (
    EventScheduler, EventConfig,
    make_events_specific, make_events_multiple, make_events_random,
)
from simulator import DataSimulator, SimulationResult
from config import (
    SAMPLING_RATES, SIGNAL_RANGES, EMOTIONS_ALL,
    EMOTIONS_AFFECTIVE, EMOTIONS_NEEDS,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def default_result() -> SimulationResult:
    """Run a short simulation once and share across tests."""
    sim = DataSimulator(
        duration_s=120,
        n_events=3,
        event_duration_s=20.0,
        emotions=None,
        noise_level="medium",
        seed=42,
    )
    return sim.simulate()


@pytest.fixture(scope="module")
def anger_result() -> SimulationResult:
    """Single-emotion simulation for targeted signal tests."""
    sim = DataSimulator(
        duration_s=90,
        n_events=2,
        event_duration_s=20.0,
        emotions="Anger",
        noise_level="low",
        seed=7,
    )
    return sim.simulate()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_all_emotions_present(self):
        assert len(EMOTIONS_ALL) == 13

    def test_affective_emotions(self):
        expected = {"Happy", "Anger", "Fear", "Disgust", "Sad", "Surprise"}
        assert set(EMOTIONS_AFFECTIVE) == expected

    def test_need_states(self):
        expected = {"Hunger", "Thirst", "Toilet", "Tired"}
        assert set(EMOTIONS_NEEDS) == expected

    def test_sampling_rates(self):
        assert SAMPLING_RATES["EDA"] == 4
        assert SAMPLING_RATES["BVP"] == 64
        assert SAMPLING_RATES["ST"] == 4
        assert SAMPLING_RATES["ACC_X"] == 32

    def test_signal_ranges_valid(self):
        for name, (lo, hi) in SIGNAL_RANGES.items():
            assert lo < hi, f"{name}: lo ({lo}) must be < hi ({hi})"


# ─────────────────────────────────────────────────────────────────────────────
# EVENT SCHEDULER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEventScheduler:

    def test_specific_emotion(self):
        events = make_events_specific("Anger", n_events=3, duration_s=20.0, total_dur_s=180)
        assert len(events) == 3
        assert all(e.emotion == "Anger" for e in events)

    def test_multiple_emotions(self):
        events = make_events_multiple(["Anger", "Fear"], n_events=4, duration_s=15.0, total_dur_s=200)
        assert len(events) == 4
        assert all(e.emotion in ["Anger", "Fear"] for e in events)

    def test_random_events(self):
        rng = np.random.default_rng(42)
        events = make_events_random(total_dur_s=300, max_events=6, rng=rng)
        assert 1 <= len(events) <= 6
        assert all(e.emotion in EMOTIONS_ALL for e in events)

    def test_no_overlaps(self):
        events = make_events_specific("Happy", n_events=5, duration_s=15.0, total_dur_s=300)
        for i in range(len(events) - 1):
            assert events[i].end_s <= events[i + 1].start_s, \
                f"Events {i} and {i + 1} overlap"

    def test_events_within_duration(self):
        dur = 120.0
        events = make_events_specific("Sad", n_events=3, duration_s=15.0, total_dur_s=dur)
        for e in events:
            assert e.start_s >= 0
            assert e.end_s <= dur

    def test_random_n_events(self):
        sched = EventScheduler(duration_s=300, n_events="random", rng=np.random.default_rng(0))
        events = sched.schedule()
        assert len(events) >= 1

    def test_random_duration(self):
        sched = EventScheduler(duration_s=300, n_events=3, event_duration_s="random",
                               rng=np.random.default_rng(5))
        events = sched.schedule()
        durs = [e.duration_s for e in events]
        assert len(set(durs)) > 1 or len(events) == 1   # durations should vary

    def test_invalid_emotion_raises(self):
        with pytest.raises(ValueError):
            make_events_specific("NotAnEmotion", n_events=1, duration_s=10, total_dur_s=60)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION RESULT STRUCTURE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulationResultStructure:

    def test_signal_keys_present(self, default_result):
        expected = {"EDA", "BVP", "ST", "ACC_X", "ACC_Y", "ACC_Z"}
        assert expected.issubset(set(default_result.signals.keys()))

    def test_time_vector_keys_present(self, default_result):
        expected = {"EDA", "BVP", "ST", "ACC_X", "ACC_Y", "ACC_Z", "IBI"}
        assert expected.issubset(set(default_result.time_vectors.keys()))

    def test_ibi_arrays_exist(self, default_result):
        assert len(default_result.ibi_times_s) > 0
        assert len(default_result.ibi_values_ms) > 0
        assert len(default_result.ibi_times_s) == len(default_result.ibi_values_ms)

    def test_annotation_keys_present(self, default_result):
        expected = {"events", "baseline_wins", "signal_quality", "sample_labels"}
        assert expected == set(default_result.annotations.keys())

    def test_metadata_keys_present(self, default_result):
        for key in ("duration_s", "n_events", "noise_level", "seed"):
            assert key in default_result.metadata

    def test_events_list(self, default_result):
        assert isinstance(default_result.events, list)
        assert all(isinstance(e, EventConfig) for e in default_result.events)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL SHAPE AND RANGE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalShapesAndRanges:

    @pytest.mark.parametrize("sig_name,fs", [
        ("EDA", 4),
        ("BVP", 64),
        ("ST", 4),
        ("ACC_X", 32),
        ("ACC_Y", 32),
        ("ACC_Z", 32),
    ])
    def test_signal_length(self, default_result, sig_name, fs):
        duration = default_result.duration_s
        expected = int(duration * fs)
        actual = len(default_result.signals[sig_name])
        assert actual == expected, \
            f"{sig_name}: expected {expected} samples, got {actual}"

    @pytest.mark.parametrize("sig_name", ["EDA", "BVP", "ST", "ACC_X", "ACC_Y", "ACC_Z"])
    def test_signal_within_physiological_range(self, default_result, sig_name):
        lo, hi = SIGNAL_RANGES[sig_name]
        arr = default_result.signals[sig_name]
        assert arr.min() >= lo, f"{sig_name} min {arr.min():.4f} < physiological lo {lo}"
        assert arr.max() <= hi, f"{sig_name} max {arr.max():.4f} > physiological hi {hi}"

    def test_ibi_within_range(self, default_result):
        lo, hi = SIGNAL_RANGES["IBI"]
        vals = default_result.ibi_values_ms
        assert vals.min() >= lo
        assert vals.max() <= hi

    def test_time_vectors_start_at_zero(self, default_result):
        for name, tvec in default_result.time_vectors.items():
            if name == "IBI":
                continue
            assert tvec[0] == pytest.approx(0.0, abs=0.01), \
                f"{name} time vector does not start at 0"

    def test_no_nan_in_signals(self, default_result):
        for name, arr in default_result.signals.items():
            assert not np.any(np.isnan(arr)), f"{name} contains NaN values"

    def test_no_inf_in_signals(self, default_result):
        for name, arr in default_result.signals.items():
            assert not np.any(np.isinf(arr)), f"{name} contains Inf values"


# ─────────────────────────────────────────────────────────────────────────────
# PHYSIOLOGICAL PLAUSIBILITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPhysiologicalPlausibility:

    def test_anger_eda_elevated_during_events(self, anger_result):
        """EDA should be higher during Anger events than during baseline."""
        eda = anger_result.signals["EDA"]
        fs = SAMPLING_RATES["EDA"]
        events = anger_result.events

        event_vals = []
        baseline_vals = []

        for ev in events:
            i0 = int(ev.start_s * fs)
            i1 = int(ev.end_s * fs)
            event_vals.extend(eda[i0:i1].tolist())

        # Use first 10 s as baseline
        baseline_vals = eda[:int(10 * fs)].tolist()

        assert np.mean(event_vals) > np.mean(baseline_vals), \
            "Anger EDA during events should exceed baseline EDA"

    def test_ibi_shorter_during_anger(self, anger_result):
        """IBI should be shorter (faster HR) during Anger events."""
        events = anger_result.events
        ibi_times = anger_result.ibi_times_s
        ibi_vals = anger_result.ibi_values_ms

        for ev in events:
            mask_ev = (ibi_times >= ev.start_s) & (ibi_times < ev.end_s)
            mask_base = ibi_times < ev.start_s - 5.0

            if mask_ev.sum() > 2 and mask_base.sum() > 2:
                mean_ev = ibi_vals[mask_ev].mean()
                mean_base = ibi_vals[mask_base].mean()
                assert mean_ev < mean_base, \
                    f"IBI during Anger ({mean_ev:.1f}ms) should be shorter than baseline ({mean_base:.1f}ms)"
                break

    def test_tired_eda_below_baseline(self):
        """Tired should produce EDA below neutral baseline."""
        sim = DataSimulator(
            duration_s=90, n_events=1, event_duration_s=30.0,
            emotions="Tired", noise_level="low", seed=10,
        )
        result = sim.simulate()
        eda = result.signals["EDA"]
        fs = SAMPLING_RATES["EDA"]
        ev = result.events[0]

        base_mean = eda[:int(ev.start_s * fs)].mean()
        event_mean = eda[int(ev.start_s * fs):int(ev.end_s * fs)].mean()

        assert event_mean < base_mean, \
            f"Tired EDA ({event_mean:.3f}) should be below baseline EDA ({base_mean:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:

    def test_same_seed_same_output(self):
        kwargs = dict(duration_s=60, n_events=2, event_duration_s=10.0,
                      emotions=None, noise_level="medium", seed=99)
        r1 = DataSimulator(**kwargs).simulate()
        r2 = DataSimulator(**kwargs).simulate()

        for sig_name in r1.signals:
            np.testing.assert_array_equal(
                r1.signals[sig_name],
                r2.signals[sig_name],
                err_msg=f"{sig_name}: same seed produced different output",
            )

    def test_different_seeds_different_output(self):
        base = dict(duration_s=60, n_events=2, event_duration_s=10.0,
                    emotions=None, noise_level="medium")
        r1 = DataSimulator(**base, seed=1).simulate()
        r2 = DataSimulator(**base, seed=2).simulate()

        # At least one signal should differ
        diffs = [
            not np.array_equal(r1.signals[s], r2.signals[s])
            for s in r1.signals
        ]
        assert any(diffs), "Different seeds should produce different signals"

    def test_same_events_with_seed(self):
        kwargs = dict(duration_s=120, n_events="random", event_duration_s="random",
                      emotions=None, noise_level="low", seed=55)
        r1 = DataSimulator(**kwargs).simulate()
        r2 = DataSimulator(**kwargs).simulate()

        assert len(r1.events) == len(r2.events)
        for e1, e2 in zip(r1.events, r2.events):
            assert e1.emotion == e2.emotion
            assert e1.start_s == e2.start_s
            assert e1.duration_s == e2.duration_s


# ─────────────────────────────────────────────────────────────────────────────
# NOISE INJECTOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseInjector:

    def test_noise_increases_variance(self):
        clean = {"EDA": np.ones(400) * 3.0, "ST": np.ones(400) * 33.0}
        noisy = NoiseInjector(level="high", seed=1).inject(clean)
        assert noisy["EDA"].std() > clean["EDA"].std()
        assert noisy["ST"].std() > clean["ST"].std()

    def test_high_noise_greater_than_low(self):
        sig = {"EDA": np.full(400, 3.0), "ST": np.full(400, 33.0)}
        low = NoiseInjector("low", seed=1).inject({"EDA": sig["EDA"].copy(), "ST": sig["ST"].copy()})
        high = NoiseInjector("high", seed=1).inject({"EDA": sig["EDA"].copy(), "ST": sig["ST"].copy()})
        assert high["EDA"].std() > low["EDA"].std()

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            NoiseInjector(level="extreme")


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnotations:

    def test_event_annotation_row_count(self, default_result):
        df = default_result.annotations["events"]
        assert len(df) == len(default_result.events)

    def test_event_annotation_columns(self, default_result):
        df = default_result.annotations["events"]
        required = {"emotion", "start_s", "end_s", "duration_s", "category"}
        assert required.issubset(set(df.columns))

    def test_sample_labels_length(self, default_result):
        df = default_result.annotations["sample_labels"]
        expected = int(default_result.duration_s * SAMPLING_RATES["BVP"])
        assert len(df) == expected

    def test_sample_labels_all_emotions_valid(self, default_result):
        df = default_result.annotations["sample_labels"]
        labels = set(df["label"].unique())
        valid = set(EMOTIONS_ALL) | {"baseline"}
        assert labels.issubset(valid)

    def test_sqi_values_in_range(self, default_result):
        df = default_result.annotations["signal_quality"]
        assert df["sqi"].between(0.0, 1.0).all()

    def test_baseline_windows_no_overlap_with_events(self, default_result):
        ev_df = default_result.annotations["events"]
        base_df = default_result.annotations["baseline_wins"]
        for _, brow in base_df.iterrows():
            for _, erow in ev_df.iterrows():
                # Baseline window must not overlap with any event window
                overlap = not (brow["end_s"] <= erow["start_s"]
                               or brow["start_s"] >= erow["end_s"])
                assert not overlap, \
                    f"Baseline window {brow['start_s']}-{brow['end_s']} overlaps event {erow['emotion']}"

    def test_compute_sqi_perfect_signal(self):
        """A clean in-range signal should score near 1.0."""
        seg = np.linspace(3.0, 5.0, 40)   # clean EDA segment
        sqi = compute_sqi(seg, "EDA", fs=4)
        assert sqi >= 0.7

    def test_compute_sqi_flatline(self):
        """A flatline signal should score low."""
        seg = np.full(40, 3.0)
        sqi = compute_sqi(seg, "EDA", fs=4)
        assert sqi < 0.6


# ─────────────────────────────────────────────────────────────────────────────
# NOISE LEVEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseLevels:

    @pytest.mark.parametrize("level", ["low", "medium", "high"])
    def test_all_noise_levels_run(self, level):
        result = DataSimulator(
            duration_s=60, n_events=1, event_duration_s=15.0,
            emotions="Happy", noise_level=level, seed=42,
        ).simulate()
        assert result is not None
        assert "EDA" in result.signals
