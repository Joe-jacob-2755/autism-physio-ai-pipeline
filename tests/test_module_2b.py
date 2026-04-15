"""
=============================================================================
MODULE 2B — TEST SUITE
=============================================================================
Run with:
    pytest tests/test_module_2b.py -v --tb=short

From the repo root, with the virtual environment active.
=============================================================================
"""

import sys
import os
import importlib

# ── Isolated import of M2B modules ──────────────────────────────────────────
# Both M2A and M2B have config.py (and other colliding names).  Python caches
# modules by name, so once M2A's config is imported the cache wins.  We must
# temporarily remove M2A from sys.path, flush the cached modules, insert M2B,
# import what we need, then restore the original state.
_M2B_DIR = os.path.join(os.path.dirname(__file__), "..", "module_2b_annotation")

# Save original path and remove M2A entries
_orig_path = list(sys.path)
sys.path = [p for p in sys.path if "module_2a_data_simulation" not in p]
sys.path.insert(0, _M2B_DIR)

# Flush any cached modules that collide
_COLLIDING = ["config", "annotator", "validator", "exporter", "visualizer"]
_cached = {name: sys.modules.pop(name) for name in _COLLIDING if name in sys.modules}

import pytest  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from datetime import datetime  # noqa: E402

from config import (  # noqa: E402
    EMOTIONS_ALL, EMOTIONS_AFFECTIVE, EMOTIONS_NEEDS, EMOTIONS_BEHAVIOURAL,
    EMOTION_METADATA, CATEGORY_MAP, SAMPLING_RATES, COMBINED_RATE_HZ,
    INTENSITY_LEVELS, INTENSITY_REACTIVITY, MODULE_VERSION,
    MIN_EVENT_DURATION_S, SHORT_EVENT_WARN_S,
    BATCH_CSV_REQUIRED_COLS, EXCEL_COL_ALIASES, ANNOTATION_COLS,
)
from annotator import AnnotationEvent, ManualAnnotator  # noqa: E402
from validator import AnnotationValidator, ValidationResult  # noqa: E402
from loader import SignalLoader, _MinimalPacket  # noqa: E402
from exporter import AnnotationExporter, next_run_folder  # noqa: E402
from excel_importer import ExcelImporter, detect_recording_start  # noqa: E402

# Restore original sys.path so M2A tests still work if collected in same session
sys.path = _orig_path


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_packet():
    """Build a minimal packet-like object with 120s of data."""
    duration_s = 120.0
    n_samples_64 = int(duration_s * 64)
    n_samples_4 = int(duration_s * 4)
    n_samples_32 = int(duration_s * 32)

    rng = np.random.default_rng(42)
    ts_64 = np.arange(n_samples_64) / 64.0
    ts_4 = np.arange(n_samples_4) / 4.0
    ts_32 = np.arange(n_samples_32) / 32.0

    signals = {
        "EDA": pd.DataFrame({
            "timestamp_s": ts_4,
            "EDA_uS": rng.uniform(0.5, 5.0, n_samples_4),
        }),
        "BVP": pd.DataFrame({
            "timestamp_s": ts_64,
            "BVP_nT": rng.uniform(-100, 100, n_samples_64),
        }),
        "ST": pd.DataFrame({
            "timestamp_s": ts_4,
            "ST_degC": rng.uniform(30.0, 37.0, n_samples_4),
        }),
        "ACC": pd.DataFrame({
            "timestamp_s": ts_32,
            "ACC_X_g": rng.uniform(-2, 2, n_samples_32),
            "ACC_Y_g": rng.uniform(-2, 2, n_samples_32),
            "ACC_Z_g": rng.uniform(-2, 2, n_samples_32),
        }),
    }

    combined = pd.DataFrame({
        "timestamp_s": ts_64,
        "EDA_uS": rng.uniform(0.5, 5.0, n_samples_64),
        "BVP_nT": rng.uniform(-100, 100, n_samples_64),
        "ST_degC": rng.uniform(30.0, 37.0, n_samples_64),
        "ACC_X_g": rng.uniform(-2, 2, n_samples_64),
        "ACC_Y_g": rng.uniform(-2, 2, n_samples_64),
        "ACC_Z_g": rng.uniform(-2, 2, n_samples_64),
    })

    return _MinimalPacket(
        signals=signals,
        combined=combined,
        metadata={"source_type": "simulated"},
        source_type="simulated",
        is_annotated=False,
        session_id="test_session",
        user_id="test_user",
    )


@pytest.fixture
def annotator(minimal_packet):
    """ManualAnnotator with a 120s packet, no events."""
    return ManualAnnotator(minimal_packet, duration_s=120.0)


@pytest.fixture
def annotator_with_events(annotator):
    """ManualAnnotator pre-loaded with 3 non-overlapping events."""
    annotator.add_event("Fear", 10.0, 15.0)
    annotator.add_event("Happy", 40.0, 20.0, intensity="Low")
    annotator.add_event("Tired", 80.0, 25.0, intensity="High")
    return annotator


@pytest.fixture
def validator():
    """Fresh AnnotationValidator."""
    return AnnotationValidator()


@pytest.fixture
def batch_csv_df():
    """A valid batch CSV DataFrame."""
    return pd.DataFrame({
        "start_s": [5.0, 30.0, 70.0],
        "emotion": ["Fear", "Happy", "Anger"],
        "duration_s": [10.0, 15.0, 20.0],
    })


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

    def test_behavioural_states(self):
        expected = {"SIB", "ATO", "GAB"}
        assert set(EMOTIONS_BEHAVIOURAL) == expected

    def test_emotion_categories_consistent(self):
        for em in EMOTIONS_ALL:
            assert em in CATEGORY_MAP, f"{em} missing from CATEGORY_MAP"
            assert em in EMOTION_METADATA, f"{em} missing from EMOTION_METADATA"

    def test_emotion_metadata_fields(self):
        required_fields = {
            "category", "color", "valence", "arousal", "description",
            "expect_EDA_dir", "expect_HR_dir", "expect_ST_dir", "expect_ACC_dir",
        }
        for em, meta in EMOTION_METADATA.items():
            for field in required_fields:
                assert field in meta, f"{em} missing field '{field}'"

    def test_sampling_rates(self):
        assert SAMPLING_RATES["EDA"] == 4
        assert SAMPLING_RATES["BVP"] == 64
        assert SAMPLING_RATES["ST"] == 4
        assert SAMPLING_RATES["ACC_X"] == 32

    def test_combined_rate(self):
        assert COMBINED_RATE_HZ == 64

    def test_intensity_levels(self):
        assert set(INTENSITY_LEVELS) == {"Low", "Medium", "High"}
        for level in INTENSITY_LEVELS:
            assert level in INTENSITY_REACTIVITY

    def test_intensity_reactivity_order(self):
        assert INTENSITY_REACTIVITY["Low"] < INTENSITY_REACTIVITY["Medium"]
        assert INTENSITY_REACTIVITY["Medium"] < INTENSITY_REACTIVITY["High"]

    def test_module_version(self):
        assert MODULE_VERSION == "1.0.0"

    def test_annotation_cols(self):
        assert "target_label" in ANNOTATION_COLS
        assert "event_id" in ANNOTATION_COLS
        assert "category" in ANNOTATION_COLS

    def test_batch_csv_required_cols(self):
        assert "start_s" in BATCH_CSV_REQUIRED_COLS
        assert "emotion" in BATCH_CSV_REQUIRED_COLS

    def test_excel_col_aliases_keys(self):
        expected_keys = {"start", "end", "duration", "emotion", "intensity"}
        assert set(EXCEL_COL_ALIASES.keys()) == expected_keys


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION EVENT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnotationEvent:

    def test_basic_creation(self):
        ev = AnnotationEvent(event_id=1, emotion="Fear", start_s=10.0, duration_s=20.0)
        assert ev.event_id == 1
        assert ev.emotion == "Fear"
        assert ev.start_s == 10.0
        assert ev.duration_s == 20.0
        assert ev.intensity == "Medium"  # default

    def test_end_s_property(self):
        ev = AnnotationEvent(event_id=1, emotion="Fear", start_s=10.0, duration_s=20.0)
        assert ev.end_s == 30.0

    def test_category_property(self):
        ev_aff = AnnotationEvent(event_id=1, emotion="Fear", start_s=0, duration_s=10)
        assert ev_aff.category == "affective"

        ev_need = AnnotationEvent(event_id=2, emotion="Hunger", start_s=0, duration_s=10)
        assert ev_need.category == "physiological_need"

        ev_beh = AnnotationEvent(event_id=3, emotion="SIB", start_s=0, duration_s=10)
        assert ev_beh.category == "behavioural"

    def test_color_property(self):
        ev = AnnotationEvent(event_id=1, emotion="Fear", start_s=0, duration_s=10)
        assert ev.color == EMOTION_METADATA["Fear"]["color"]

    def test_reactivity_property(self):
        ev_low = AnnotationEvent(event_id=1, emotion="Fear", start_s=0,
                                 duration_s=10, intensity="Low")
        ev_high = AnnotationEvent(event_id=2, emotion="Fear", start_s=0,
                                  duration_s=10, intensity="High")
        assert ev_low.reactivity == 1.0
        assert ev_high.reactivity == 1.6

    def test_to_dict(self):
        ev = AnnotationEvent(event_id=1, emotion="Anger", start_s=5.0,
                             duration_s=15.0, intensity="High")
        d = ev.to_dict()
        assert d["event_id"] == 1
        assert d["emotion"] == "Anger"
        assert d["category"] == "affective"
        assert d["intensity"] == "High"
        assert d["start_s"] == 5.0
        assert d["end_s"] == 20.0
        assert d["duration_s"] == 15.0
        assert d["reactivity"] == 1.6
        assert "color_hex" in d
        assert "description" in d

    def test_to_dict_all_emotions(self):
        """Every emotion produces a valid dict with all expected keys."""
        expected_keys = {
            "event_id", "emotion", "category", "intensity", "reactivity",
            "valence", "arousal", "start_s", "end_s", "duration_s",
            "color_hex", "description", "expect_EDA_dir", "expect_HR_dir",
            "expect_ST_dir", "expect_ACC_dir",
        }
        for emotion in EMOTIONS_ALL:
            ev = AnnotationEvent(event_id=1, emotion=emotion,
                                 start_s=0, duration_s=10)
            d = ev.to_dict()
            assert set(d.keys()) == expected_keys, f"{emotion} dict keys mismatch"


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestValidator:

    def test_valid_event(self, validator):
        ev = AnnotationEvent(event_id=1, emotion="Fear", start_s=10.0, duration_s=20.0)
        result = validator.validate_event(ev, [], 120.0)
        assert result.is_valid

    def test_invalid_emotion(self, validator):
        ev = AnnotationEvent(event_id=1, emotion="InvalidEmo", start_s=10.0, duration_s=20.0)
        result = validator.validate_event(ev, [], 120.0)
        assert not result.is_valid
        assert any("Unknown emotion" in e for e in result.errors)

    def test_negative_start(self, validator):
        ev = AnnotationEvent(event_id=1, emotion="Fear", start_s=-5.0, duration_s=20.0)
        result = validator.validate_event(ev, [], 120.0)
        assert not result.is_valid
        assert any("negative" in e.lower() for e in result.errors)

    def test_exceeds_duration(self, validator):
        ev = AnnotationEvent(event_id=1, emotion="Fear", start_s=110.0, duration_s=20.0)
        result = validator.validate_event(ev, [], 120.0)
        assert not result.is_valid
        assert any("exceeds" in e.lower() for e in result.errors)

    def test_too_short_event(self, validator):
        ev = AnnotationEvent(event_id=1, emotion="Fear", start_s=10.0, duration_s=1.0)
        result = validator.validate_event(ev, [], 120.0)
        assert not result.is_valid
        assert any("below minimum" in e.lower() for e in result.errors)

    def test_minimum_duration_boundary(self, validator):
        """Event exactly at MIN_EVENT_DURATION_S should pass."""
        ev = AnnotationEvent(event_id=1, emotion="Fear", start_s=10.0,
                             duration_s=MIN_EVENT_DURATION_S)
        result = validator.validate_event(ev, [], 120.0)
        assert result.is_valid

    def test_overlap_detection(self, validator):
        existing = AnnotationEvent(event_id=1, emotion="Fear", start_s=10.0, duration_s=20.0)
        new = AnnotationEvent(event_id=2, emotion="Happy", start_s=25.0, duration_s=10.0)
        result = validator.validate_event(new, [existing], 120.0)
        assert not result.is_valid
        assert any("overlaps" in e.lower() for e in result.errors)

    def test_no_overlap_adjacent_events(self, validator):
        """Events touching but not overlapping should pass."""
        existing = AnnotationEvent(event_id=1, emotion="Fear", start_s=10.0, duration_s=20.0)
        new = AnnotationEvent(event_id=2, emotion="Happy", start_s=30.0, duration_s=10.0)
        result = validator.validate_event(new, [existing], 120.0)
        assert result.is_valid

    def test_short_event_warning(self, validator):
        """Event between MIN_EVENT_DURATION_S and SHORT_EVENT_WARN_S gets warning."""
        dur = (MIN_EVENT_DURATION_S + SHORT_EVENT_WARN_S) / 2.0
        ev = AnnotationEvent(event_id=1, emotion="Fear", start_s=10.0, duration_s=dur)
        result = validator.validate_event(ev, [], 120.0)
        assert result.is_valid
        assert any("short" in w.lower() for w in result.warnings)

    def test_boundary_proximity_warning(self, validator):
        ev = AnnotationEvent(event_id=1, emotion="Fear", start_s=0.5,
                             duration_s=MIN_EVENT_DURATION_S)
        result = validator.validate_event(ev, [], 120.0)
        assert result.is_valid
        assert any("close to recording start" in w.lower() for w in result.warnings)

    def test_end_boundary_proximity_warning(self, validator):
        ev = AnnotationEvent(event_id=1, emotion="Fear", start_s=115.0, duration_s=4.5)
        result = validator.validate_event(ev, [], 120.0)
        assert result.is_valid
        assert any("close to recording end" in w.lower() for w in result.warnings)

    # ── validate_all ─────────────────────────────────────────────────────

    def test_validate_all_empty(self, validator):
        result = validator.validate_all([], 120.0)
        assert not result.is_valid
        assert any("no events" in e.lower() for e in result.errors)

    def test_validate_all_valid_set(self, validator):
        events = [
            AnnotationEvent(1, "Fear", 10.0, 15.0),
            AnnotationEvent(2, "Happy", 40.0, 20.0),
            AnnotationEvent(3, "Anger", 80.0, 15.0),
        ]
        result = validator.validate_all(events, 120.0)
        assert result.is_valid

    def test_validate_all_single_emotion_warning(self, validator):
        events = [
            AnnotationEvent(1, "Fear", 10.0, 15.0),
            AnnotationEvent(2, "Fear", 40.0, 15.0),
        ]
        result = validator.validate_all(events, 120.0)
        assert result.is_valid
        assert any("same label" in w.lower() for w in result.warnings)

    def test_validate_all_low_density_warning(self, validator):
        """One short event in a long recording → low density."""
        events = [
            AnnotationEvent(1, "Fear", 10.0, MIN_EVENT_DURATION_S),
        ]
        result = validator.validate_all(events, 1000.0)
        assert result.is_valid
        assert any("density" in w.lower() for w in result.warnings)

    # ── validate_batch_csv ───────────────────────────────────────────────

    def test_batch_csv_valid(self, validator, batch_csv_df):
        result = validator.validate_batch_csv(batch_csv_df)
        assert result.is_valid

    def test_batch_csv_missing_cols(self, validator):
        df = pd.DataFrame({"time": [1.0], "label": ["Fear"]})
        result = validator.validate_batch_csv(df)
        assert not result.is_valid
        assert any("missing required" in e.lower() for e in result.errors)

    def test_batch_csv_empty(self, validator):
        df = pd.DataFrame({"start_s": pd.Series(dtype=float),
                           "emotion": pd.Series(dtype=str)})
        result = validator.validate_batch_csv(df)
        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_batch_csv_invalid_emotion(self, validator):
        df = pd.DataFrame({
            "start_s": [5.0],
            "emotion": ["NonExistent"],
            "duration_s": [10.0],
        })
        result = validator.validate_batch_csv(df)
        assert not result.is_valid
        assert any("invalid emotion" in e.lower() for e in result.errors)

    def test_batch_csv_no_duration_col_warns(self, validator):
        df = pd.DataFrame({
            "start_s": [5.0],
            "emotion": ["Fear"],
        })
        result = validator.validate_batch_csv(df)
        assert result.is_valid
        assert any("neither" in w.lower() for w in result.warnings)

    def test_batch_csv_extra_cols_warns(self, validator):
        df = pd.DataFrame({
            "start_s": [5.0],
            "emotion": ["Fear"],
            "duration_s": [10.0],
            "notes": ["some note"],
        })
        result = validator.validate_batch_csv(df)
        assert result.is_valid
        assert any("unrecognised" in w.lower() for w in result.warnings)

    # ── compute_annotation_density ───────────────────────────────────────

    def test_density_empty(self, validator):
        assert validator.compute_annotation_density([], 120.0) == 0.0

    def test_density_zero_duration(self, validator):
        assert validator.compute_annotation_density([], 0.0) == 0.0

    def test_density_correct(self, validator):
        events = [
            AnnotationEvent(1, "Fear", 10.0, 30.0),
            AnnotationEvent(2, "Happy", 50.0, 30.0),
        ]
        density = validator.compute_annotation_density(events, 120.0)
        assert abs(density - 0.5) < 1e-6

    def test_density_capped_at_1(self, validator):
        """Overlapping events (if somehow present) shouldn't exceed 1.0."""
        events = [
            AnnotationEvent(1, "Fear", 0, 120),
            AnnotationEvent(2, "Happy", 0, 120),
        ]
        density = validator.compute_annotation_density(events, 120.0)
        assert density <= 1.0

    # ── ValidationResult merge ───────────────────────────────────────────

    def test_merge_both_valid(self):
        a = ValidationResult(is_valid=True, warnings=["w1"])
        b = ValidationResult(is_valid=True, warnings=["w2"])
        merged = a.merge(b)
        assert merged.is_valid
        assert len(merged.warnings) == 2

    def test_merge_one_invalid(self):
        a = ValidationResult(is_valid=True)
        b = ValidationResult(is_valid=False, errors=["e1"])
        merged = a.merge(b)
        assert not merged.is_valid
        assert len(merged.errors) == 1


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL ANNOTATOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestManualAnnotator:

    def test_add_event_valid(self, annotator):
        result = annotator.add_event("Fear", 10.0, 20.0)
        assert result.is_valid
        assert len(annotator.events) == 1
        assert annotator.events[0].emotion == "Fear"

    def test_add_event_invalid_rejected(self, annotator):
        result = annotator.add_event("InvalidEmo", 10.0, 20.0)
        assert not result.is_valid
        assert len(annotator.events) == 0

    def test_add_event_auto_increments_id(self, annotator):
        annotator.add_event("Fear", 10.0, 15.0)
        annotator.add_event("Happy", 40.0, 15.0)
        assert annotator.events[0].event_id == 1
        assert annotator.events[1].event_id == 2

    def test_add_event_with_intensity(self, annotator):
        annotator.add_event("Fear", 10.0, 20.0, intensity="High")
        assert annotator.events[0].intensity == "High"

    def test_add_event_overlap_rejected(self, annotator):
        annotator.add_event("Fear", 10.0, 20.0)
        result = annotator.add_event("Happy", 15.0, 10.0)
        assert not result.is_valid
        assert len(annotator.events) == 1

    def test_edit_event(self, annotator):
        annotator.add_event("Fear", 10.0, 20.0)
        result = annotator.edit_event(1, emotion="Happy")
        assert result.is_valid
        assert annotator.events[0].emotion == "Happy"

    def test_edit_event_invalid_id(self, annotator):
        result = annotator.edit_event(999, emotion="Happy")
        assert not result.is_valid

    def test_edit_event_revalidates(self, annotator):
        annotator.add_event("Fear", 10.0, 15.0)
        annotator.add_event("Happy", 40.0, 15.0)
        # Try to move event 2 to overlap with event 1
        result = annotator.edit_event(2, start_s=12.0)
        assert not result.is_valid
        # Event 2 should be unchanged
        assert annotator.events[1].start_s == 40.0

    def test_delete_event(self, annotator_with_events):
        assert len(annotator_with_events.events) == 3
        removed = annotator_with_events.delete_event(2)
        assert removed is True
        assert len(annotator_with_events.events) == 2
        # IDs renumbered
        ids = [ev.event_id for ev in annotator_with_events.events]
        assert ids == [1, 2]

    def test_delete_event_invalid_id(self, annotator):
        removed = annotator.delete_event(999)
        assert removed is False

    def test_clear_all_events(self, annotator_with_events):
        annotator_with_events.clear_all_events()
        assert len(annotator_with_events.events) == 0

    def test_replace_event_removes_overlapping(self, annotator):
        annotator.add_event("Fear", 10.0, 20.0)
        result, removed = annotator.replace_event("Happy", 15.0, 25.0)
        assert result.is_valid
        assert len(removed) == 1
        assert removed[0].emotion == "Fear"
        assert len(annotator.events) == 1
        assert annotator.events[0].emotion == "Happy"

    # ── Batch import ─────────────────────────────────────────────────────

    def test_import_from_dataframe(self, annotator, batch_csv_df):
        n, result = annotator.import_from_dataframe(batch_csv_df)
        assert n == 3
        assert result.is_valid
        assert len(annotator.events) == 3

    def test_import_from_dataframe_with_end_s(self, annotator):
        df = pd.DataFrame({
            "start_s": [5.0, 30.0],
            "emotion": ["Fear", "Happy"],
            "end_s": [15.0, 50.0],
        })
        n, result = annotator.import_from_dataframe(df)
        assert n == 2
        assert annotator.events[0].duration_s == 10.0
        assert annotator.events[1].duration_s == 20.0

    def test_import_with_default_duration(self, annotator):
        df = pd.DataFrame({
            "start_s": [5.0, 30.0],
            "emotion": ["Fear", "Happy"],
        })
        n, result = annotator.import_from_dataframe(df, default_duration_s=10.0)
        assert n == 2
        assert all(ev.duration_s == 10.0 for ev in annotator.events)

    def test_import_no_duration_no_default(self, annotator):
        df = pd.DataFrame({
            "start_s": [5.0],
            "emotion": ["Fear"],
        })
        n, result = annotator.import_from_dataframe(df)
        assert n == 0
        assert not result.is_valid

    def test_import_invalid_csv_schema(self, annotator):
        df = pd.DataFrame({"time": [5.0], "label": ["Fear"]})
        n, result = annotator.import_from_dataframe(df)
        assert n == 0
        assert not result.is_valid

    def test_load_existing_events(self, annotator):
        events_df = pd.DataFrame({
            "emotion": ["Fear", "Happy"],
            "start_s": [10.0, 50.0],
            "end_s": [25.0, 70.0],
            "intensity": ["Medium", "Low"],
        })
        n = annotator.load_existing_events(events_df)
        assert n == 2
        assert annotator.events[0].emotion == "Fear"
        assert annotator.events[1].intensity == "Low"

    # ── Validation ───────────────────────────────────────────────────────

    def test_validate_current_state(self, annotator_with_events):
        result = annotator_with_events.validate_current_state()
        assert result.is_valid

    def test_validate_empty_state(self, annotator):
        result = annotator.validate_current_state()
        assert not result.is_valid

    # ── Display ──────────────────────────────────────────────────────────

    def test_summary_table_empty(self, annotator):
        table = annotator.summary_table()
        assert "no events" in table.lower()

    def test_summary_table_with_events(self, annotator_with_events):
        table = annotator_with_events.summary_table()
        assert "Fear" in table
        assert "Happy" in table
        assert "Tired" in table
        assert "3 events" in table

    # ── Apply to packet ──────────────────────────────────────────────────

    def test_apply_to_packet(self, annotator_with_events):
        packet = annotator_with_events.apply_to_packet()
        assert packet.is_annotated is True

        # Check annotation columns exist on combined
        assert "target_label" in packet.combined.columns
        assert "event_id" in packet.combined.columns
        assert "category" in packet.combined.columns

        # Check annotation columns on each signal
        for name, df in packet.signals.items():
            assert "target_label" in df.columns, f"{name} missing target_label"

    def test_apply_to_packet_labels_correct(self, annotator_with_events):
        packet = annotator_with_events.apply_to_packet()
        combined = packet.combined
        ts = combined["timestamp_s"].values

        # Samples inside Fear event (10-25s) should be "Fear"
        fear_mask = (ts >= 10.0) & (ts < 25.0)
        fear_labels = combined.loc[fear_mask, "target_label"]
        assert (fear_labels == "Fear").all()

        # Samples in baseline region should be "baseline"
        baseline_mask = (ts >= 30.0) & (ts < 40.0)
        baseline_labels = combined.loc[baseline_mask, "target_label"]
        assert (baseline_labels == "baseline").all()

    # ── Event / sample label DataFrames ──────────────────────────────────

    def test_to_event_dataframe(self, annotator_with_events):
        df = annotator_with_events.to_event_dataframe()
        assert len(df) == 3
        assert "emotion" in df.columns
        assert "start_s" in df.columns
        assert "end_s" in df.columns
        assert list(df["emotion"]) == ["Fear", "Happy", "Tired"]  # sorted by start_s

    def test_to_event_dataframe_empty(self, annotator):
        df = annotator.to_event_dataframe()
        assert df.empty

    def test_to_sample_labels_dataframe(self, annotator_with_events):
        df = annotator_with_events.to_sample_labels_dataframe()
        assert len(df) == int(120.0 * COMBINED_RATE_HZ)
        assert set(df.columns) == {"time_s", "label", "event_id", "category"}

        # Check some labels
        fear_rows = df[df["label"] == "Fear"]
        assert len(fear_rows) > 0
        baseline_rows = df[df["label"] == "baseline"]
        assert len(baseline_rows) > 0

    def test_to_sample_labels_dataframe_empty(self, annotator):
        df = annotator.to_sample_labels_dataframe()
        assert len(df) == int(120.0 * COMBINED_RATE_HZ)
        assert (df["label"] == "baseline").all()


# ─────────────────────────────────────────────────────────────────────────────
# LOADER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalLoader:

    def test_load_from_packet(self, minimal_packet):
        loader = SignalLoader(verbose=False)
        packet, duration_s, existing_events, warnings = loader.load(minimal_packet)
        assert duration_s > 0
        assert existing_events is None
        assert packet.signals is not None

    def test_detect_duration(self, minimal_packet):
        loader = SignalLoader(verbose=False)
        duration = loader._detect_duration(minimal_packet)
        # 120s of data at 64 Hz → max timestamp ~ 119.98
        assert 119.0 < duration <= 120.0

    def test_strip_annotation_cols(self, minimal_packet):
        # Add annotation columns
        for df in minimal_packet.signals.values():
            df["target_label"] = "baseline"
            df["event_id"] = 0
            df["category"] = "baseline"
        minimal_packet.combined["target_label"] = "baseline"
        minimal_packet.is_annotated = True

        loader = SignalLoader(verbose=False)
        loader._strip_annotation_cols(minimal_packet)

        assert minimal_packet.is_annotated is False
        for df in minimal_packet.signals.values():
            assert "target_label" not in df.columns

    def test_check_already_annotated(self, minimal_packet):
        loader = SignalLoader(verbose=False)
        assert loader._check_already_annotated(minimal_packet) is False

        minimal_packet.is_annotated = True
        assert loader._check_already_annotated(minimal_packet) is True

    def test_load_folder_not_found(self):
        loader = SignalLoader(verbose=False)
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/folder/path")


class TestMinimalPacket:

    def test_attributes(self, minimal_packet):
        assert hasattr(minimal_packet, "signals")
        assert hasattr(minimal_packet, "combined")
        assert hasattr(minimal_packet, "metadata")
        assert hasattr(minimal_packet, "source_type")
        assert hasattr(minimal_packet, "is_annotated")
        assert hasattr(minimal_packet, "session_id")
        assert hasattr(minimal_packet, "user_id")

    def test_signal_names(self, minimal_packet):
        names = minimal_packet.signal_names
        assert "EDA" in names
        assert "BVP" in names


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestExporter:

    def test_next_run_folder(self, tmp_path):
        folder1 = next_run_folder(tmp_path)
        assert folder1.exists()
        assert "run_001" in folder1.name

        folder2 = next_run_folder(tmp_path)
        assert "run_002" in folder2.name

    def test_next_run_folder_numbering(self, tmp_path):
        """Non-matching folders are ignored."""
        (tmp_path / "random_folder").mkdir()
        folder = next_run_folder(tmp_path)
        assert "run_001" in folder.name

    def test_export_metadata(self, annotator_with_events, tmp_path):
        exporter = AnnotationExporter(verbose=False)
        annotator_with_events.apply_to_packet()
        path = exporter._export_metadata(
            annotator_with_events.packet, annotator_with_events, tmp_path
        )
        assert path.exists()
        import json
        with open(path) as f:
            meta = json.load(f)
        assert meta["is_annotated"] is True
        assert meta["n_events"] == 3
        assert meta["module"] == "M2B"

    def test_export_signals(self, annotator_with_events, tmp_path):
        exporter = AnnotationExporter(verbose=False)
        annotator_with_events.apply_to_packet()
        paths = exporter._export_signals(annotator_with_events.packet, tmp_path)
        for name in annotator_with_events.packet.signals:
            assert name in paths
            assert paths[name].exists()

    def test_export_combined(self, annotator_with_events, tmp_path):
        exporter = AnnotationExporter(verbose=False)
        annotator_with_events.apply_to_packet()
        path = exporter._export_combined(annotator_with_events.packet, tmp_path)
        assert path is not None
        assert path.exists()
        df = pd.read_csv(path)
        assert "target_label" in df.columns

    def test_export_events_csv(self, annotator_with_events, tmp_path):
        exporter = AnnotationExporter(verbose=False)
        path = exporter._export_events(annotator_with_events, tmp_path)
        assert path is not None
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 3

    def test_export_sample_labels(self, annotator_with_events, tmp_path):
        exporter = AnnotationExporter(verbose=False)
        path = exporter._export_sample_labels(annotator_with_events, tmp_path)
        assert path is not None
        assert path.exists()
        df = pd.read_csv(path)
        assert "label" in df.columns
        assert len(df) == int(120.0 * COMBINED_RATE_HZ)


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL IMPORTER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestExcelImporter:

    def test_import_seconds_format(self):
        df = pd.DataFrame({
            "start_time": [10.0, 50.0],
            "end_time": [25.0, 70.0],
            "targeted_behaviour": ["Fear", "Happy"],
            "intensity": ["Low", "High"],
        })
        importer = ExcelImporter()
        result = importer.import_dataframe(df)
        assert result.is_valid
        assert result.n_imported == 2
        assert result.events[0]["emotion"] == "Fear"
        assert result.events[0]["start_s"] == 10.0
        assert result.events[0]["duration_s"] == 15.0
        assert result.events[0]["intensity"] == "Low"

    def test_import_with_duration_column(self):
        df = pd.DataFrame({
            "onset": [10.0, 50.0],
            "duration": [15.0, 20.0],
            "label": ["Fear", "Happy"],
        })
        importer = ExcelImporter()
        result = importer.import_dataframe(df)
        assert result.is_valid
        assert result.n_imported == 2

    def test_import_default_duration(self):
        df = pd.DataFrame({
            "start_s": [10.0, 50.0],
            "emotion": ["Fear", "Happy"],
        })
        importer = ExcelImporter(default_duration_s=15.0)
        result = importer.import_dataframe(df)
        assert result.is_valid
        assert all(ev["duration_s"] == 15.0 for ev in result.events)

    def test_import_no_emotion_col(self):
        df = pd.DataFrame({
            "start_s": [10.0],
            "value": [42.0],
        })
        importer = ExcelImporter()
        result = importer.import_dataframe(df)
        assert not result.is_valid
        assert any("emotion" in e.lower() for e in result.errors)

    def test_import_no_start_col(self):
        df = pd.DataFrame({
            "emotion": ["Fear"],
            "value": [42.0],
        })
        importer = ExcelImporter()
        result = importer.import_dataframe(df)
        assert not result.is_valid
        assert any("start time" in e.lower() for e in result.errors)

    def test_import_clock_times_with_start(self):
        recording_start = datetime(2025, 3, 15, 14, 0, 0)
        df = pd.DataFrame({
            "start_time": ["14:05:00", "14:15:30"],
            "end_time": ["14:10:00", "14:20:30"],
            "emotion": ["Fear", "Happy"],
        })
        importer = ExcelImporter(recording_start=recording_start)
        result = importer.import_dataframe(df)
        assert result.is_valid
        assert result.n_imported == 2
        assert abs(result.events[0]["start_s"] - 300.0) < 1.0  # 5 min
        assert abs(result.events[0]["duration_s"] - 300.0) < 1.0  # 5 min

    def test_import_clock_times_without_start_errors(self):
        df = pd.DataFrame({
            "start_time": ["14:05:00"],
            "end_time": ["14:10:00"],
            "emotion": ["Fear"],
        })
        importer = ExcelImporter()  # no recording_start
        result = importer.import_dataframe(df)
        assert not result.is_valid
        assert any("clock time" in e.lower() for e in result.errors)

    def test_emotion_normalisation(self):
        """Aliases and case-insensitive matching should work."""
        df = pd.DataFrame({
            "start_s": [10.0, 30.0, 50.0, 70.0],
            "duration_s": [10.0, 10.0, 10.0, 10.0],
            "emotion": ["angry", "scared", "self-injurious", "tantrum"],
        })
        importer = ExcelImporter()
        result = importer.import_dataframe(df)
        assert result.is_valid
        emotions = [ev["emotion"] for ev in result.events]
        assert emotions == ["Anger", "Fear", "SIB", "GAB"]

    def test_unrecognised_emotion_error(self):
        df = pd.DataFrame({
            "start_s": [10.0],
            "duration_s": [10.0],
            "emotion": ["TotallyFake"],
        })
        importer = ExcelImporter()
        result = importer.import_dataframe(df)
        assert result.n_imported == 0
        assert any("unrecognised" in e.lower() for e in result.errors)

    def test_intensity_parsing(self):
        df = pd.DataFrame({
            "start_s": [10.0, 30.0, 50.0],
            "duration_s": [10.0, 10.0, 10.0],
            "emotion": ["Fear", "Happy", "Anger"],
            "intensity": ["low", "HIGH", "unknown"],
        })
        importer = ExcelImporter()
        result = importer.import_dataframe(df)
        assert result.events[0]["intensity"] == "Low"
        assert result.events[1]["intensity"] == "High"
        assert result.events[2]["intensity"] == "Medium"  # fallback

    def test_file_not_found(self, tmp_path):
        importer = ExcelImporter()
        result = importer.import_file(tmp_path / "nonexistent.csv")
        assert not result.is_valid
        assert any("not found" in e.lower() for e in result.errors)

    def test_import_csv_file(self, tmp_path):
        csv_path = tmp_path / "events.csv"
        df = pd.DataFrame({
            "start_s": [10.0, 40.0],
            "duration_s": [15.0, 20.0],
            "emotion": ["Fear", "Happy"],
        })
        df.to_csv(csv_path, index=False)

        importer = ExcelImporter()
        result = importer.import_file(csv_path)
        assert result.is_valid
        assert result.n_imported == 2

    def test_unsupported_file_type(self, tmp_path):
        bad_path = tmp_path / "events.json"
        bad_path.write_text("{}")
        importer = ExcelImporter()
        result = importer.import_file(bad_path)
        assert not result.is_valid
        assert any("unsupported" in e.lower() for e in result.errors)

    def test_column_detection_case_insensitive(self):
        df = pd.DataFrame({
            "START_TIME": [10.0],
            "DURATION": [15.0],
            "EMOTION": ["Fear"],
            "INTENSITY": ["High"],
        })
        importer = ExcelImporter()
        result = importer.import_dataframe(df)
        assert result.is_valid

    def test_end_before_start_error(self):
        df = pd.DataFrame({
            "start_s": [50.0],
            "end_s": [40.0],
            "emotion": ["Fear"],
        })
        importer = ExcelImporter()
        result = importer.import_dataframe(df)
        assert result.n_imported == 0
        assert any("before" in e.lower() for e in result.errors)


class TestDetectRecordingStart:

    def test_iso_format(self):
        packet = _MinimalPacket(
            signals={}, combined=pd.DataFrame(),
            metadata={"recording_start": "2025-03-15T14:30:00"},
        )
        dt = detect_recording_start(packet)
        assert dt is not None
        assert dt.hour == 14
        assert dt.minute == 30

    def test_datetime_object(self):
        expected = datetime(2025, 3, 15, 14, 30, 0)
        packet = _MinimalPacket(
            signals={}, combined=pd.DataFrame(),
            metadata={"recording_start": expected},
        )
        dt = detect_recording_start(packet)
        assert dt == expected

    def test_no_metadata(self):
        packet = _MinimalPacket(
            signals={}, combined=pd.DataFrame(), metadata={},
        )
        dt = detect_recording_start(packet)
        assert dt is None

    def test_alternative_keys(self):
        packet = _MinimalPacket(
            signals={}, combined=pd.DataFrame(),
            metadata={"session_start": "2025-03-15 10:00:00"},
        )
        dt = detect_recording_start(packet)
        assert dt is not None
        assert dt.hour == 10


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION: ANNOTATOR → EXPORTER ROUND-TRIP
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnotationRoundTrip:

    def test_annotate_export_reload(self, annotator_with_events, tmp_path):
        """Full round-trip: annotate → export → reload events."""
        # Apply annotations
        packet = annotator_with_events.apply_to_packet()

        # Export events CSV
        exporter = AnnotationExporter(verbose=False)
        events_path = exporter._export_events(annotator_with_events, tmp_path)
        assert events_path.exists()

        # Reload events into a fresh annotator
        events_df = pd.read_csv(events_path)
        fresh_annotator = ManualAnnotator(packet, duration_s=120.0)
        n_loaded = fresh_annotator.load_existing_events(events_df)
        assert n_loaded == 3

        # Verify events match
        original_emotions = sorted(ev.emotion for ev in annotator_with_events.events)
        reloaded_emotions = sorted(ev.emotion for ev in fresh_annotator.events)
        assert original_emotions == reloaded_emotions

    def test_sample_labels_match_events(self, annotator_with_events):
        """Sample labels at 64 Hz should be consistent with events."""
        events_df = annotator_with_events.to_event_dataframe()
        labels_df = annotator_with_events.to_sample_labels_dataframe()

        for _, row in events_df.iterrows():
            emotion = row["emotion"]
            start = row["start_s"]
            end = row["end_s"]

            # All samples in [start, end) should have this emotion
            mask = (labels_df["time_s"] >= start) & (labels_df["time_s"] < end)
            event_labels = labels_df.loc[mask, "label"]
            assert (event_labels == emotion).all(), (
                f"Mismatch for {emotion} in [{start}, {end})"
            )

    def test_full_export(self, annotator_with_events, tmp_path):
        """Full export produces all expected files (except plots)."""
        annotator_with_events.apply_to_packet()
        exporter = AnnotationExporter(verbose=False)

        # Use a minimal visualizer mock that raises to skip plots
        class _NoPlotViz:
            def plot_overview(self, path): raise RuntimeError("skip")
            def plot_annotated(self, events, path): raise RuntimeError("skip")

        outputs = exporter.export_all(
            annotator_with_events.packet,
            annotator_with_events,
            _NoPlotViz(),
            tmp_path,
        )

        # Check non-plot outputs
        assert "annotations_events" in outputs
        assert "annotations_sample_labels" in outputs
        assert "packet_metadata" in outputs
        for name in annotator_with_events.packet.signals:
            assert name in outputs
