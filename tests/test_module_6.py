"""
=============================================================================
TESTS — MODULE 6: FEATURE ENGINEERING
=============================================================================
Comprehensive test suite covering:
  - Feature extraction (all 5 signals)
  - Normalisation (RobustScaler fit/transform)
  - Demographic encoding (ordinal + one-hot)
  - Feature fusion (per-signal + combined)
  - Feature selection (all methods + ensemble)
  - Dimensionality reduction (PCA, PLS-VIP, SVD)
  - Exporter (output structure)
  - FeatureEngineer orchestrator (end-to-end)
  - Edge cases (empty, NaN, single-sample)
  - Reproducibility (seeded determinism)

Run:  python -m pytest tests/test_module_6.py -v --tb=short
=============================================================================
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Setup: add module_6 to path ──────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
M6_DIR = REPO_ROOT / "module_6_feature_engineering"

# Clean up any config.py from a previous test module (e.g. M2A, M3)
for _stale in ["config", "exporter", "splitter", "annotator", "simulator",
               "noise_injector", "event_scheduler", "normaliser", "encoder",
               "feature_extractor", "feature_selector", "feature_engineer",
               "dimensionality_reducer"]:
    if _stale in sys.modules:
        del sys.modules[_stale]

# Remove any other module paths that could cause config collision
sys.path = [p for p in sys.path
            if "module_2a_data_simulation" not in p
            and "module_3_data_splitting" not in p]
sys.path.insert(0, str(M6_DIR))

from feature_extractor import (
    FeatureExtractor,
    extract_eda_features,
    extract_bvp_features,
    extract_ibi_features,
    extract_st_features,
    extract_acc_features,
    _safe,
    _band_power,
    _dominant_freq,
    _spectral_centroid,
    _spectral_entropy,
    _sample_entropy,
    _zero_crossing_rate,
    _decompose_eda,
    _poincare,
)
from normaliser import FeatureNormaliser, DemographicEncoder, FeatureFuser
from encoder import OneHotEncoder
from feature_selector import FeatureSelector
from dimensionality_reducer import DimensionalityReducer
from exporter import FeatureEngineeringExporter
from feature_engineer import FeatureEngineer, next_run_folder
from config import META_COLS


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES — synthetic signal data
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def eda_signal(rng):
    """60s of synthetic EDA at 4 Hz with a simulated SCR event."""
    fs = 4
    n = 60 * fs  # 240 samples
    t = np.arange(n) / fs
    baseline = 2.0 + 0.3 * np.sin(2 * np.pi * 0.01 * t)
    # Add SCR-like peak around t=30s
    scr = 1.5 * np.exp(-((t - 30) ** 2) / (2 * 3 ** 2))
    noise = rng.normal(0, 0.05, n)
    return np.clip(baseline + scr + noise, 0.01, 30.0), fs


@pytest.fixture
def bvp_signal(rng):
    """60s of synthetic BVP at 64 Hz with quasi-periodic peaks."""
    fs = 64
    n = 60 * fs
    t = np.arange(n) / fs
    # ~75 bpm -> 1.25 Hz
    sig = 50 * np.sin(2 * np.pi * 1.25 * t)
    sig += 15 * np.sin(2 * np.pi * 2.5 * t)  # 2nd harmonic
    noise = rng.normal(0, 5, n)
    return sig + noise, fs


@pytest.fixture
def ibi_data(rng):
    """Synthetic IBI series (event-based) — ~60s of heartbeats at ~75 bpm."""
    n_beats = 75
    ibi_ms = 800 + rng.normal(0, 30, n_beats)  # ~800 ms mean
    ibi_ms = np.clip(ibi_ms, 500, 1200)
    times = np.cumsum(ibi_ms / 1000)
    return times, ibi_ms


@pytest.fixture
def st_signal(rng):
    """60s of synthetic skin temperature at 4 Hz."""
    fs = 4
    n = 60 * fs
    t = np.arange(n) / fs
    sig = 33.0 + 0.5 * t / 60 + rng.normal(0, 0.02, n)
    return np.clip(sig, 25.0, 40.0), fs


@pytest.fixture
def acc_signal(rng):
    """60s of synthetic 3-axis ACC at 32 Hz."""
    fs = 32
    n = 60 * fs
    t = np.arange(n) / fs
    ax = 0.1 * np.sin(2 * np.pi * 2.0 * t) + rng.normal(0, 0.05, n)
    ay = -0.05 + rng.normal(0, 0.03, n)
    az = 1.0 + 0.08 * np.sin(2 * np.pi * 1.5 * t) + rng.normal(0, 0.04, n)
    return ax, ay, az, fs


@pytest.fixture
def sample_signals(eda_signal, bvp_signal, ibi_data, st_signal, acc_signal):
    """Dict of DataFrames mimicking Module 3 output."""
    eda_vals, eda_fs = eda_signal
    bvp_vals, bvp_fs = bvp_signal
    ibi_times, ibi_vals = ibi_data
    st_vals, st_fs = st_signal
    ax, ay, az, acc_fs = acc_signal

    labels_eda = ["Happy"] * (len(eda_vals) // 2) + ["Fear"] * (len(eda_vals) - len(eda_vals) // 2)
    labels_bvp = ["Happy"] * (len(bvp_vals) // 2) + ["Fear"] * (len(bvp_vals) - len(bvp_vals) // 2)
    labels_ibi = ["Happy"] * (len(ibi_vals) // 2) + ["Fear"] * (len(ibi_vals) - len(ibi_vals) // 2)
    labels_st = ["Happy"] * (len(st_vals) // 2) + ["Fear"] * (len(st_vals) - len(st_vals) // 2)
    labels_acc = ["Happy"] * (len(ax) // 2) + ["Fear"] * (len(ax) - len(ax) // 2)

    return {
        "EDA": pd.DataFrame({
            "timestamp_s": np.arange(len(eda_vals)) / eda_fs,
            "EDA_uS": eda_vals,
            "target_label": labels_eda,
        }),
        "BVP": pd.DataFrame({
            "timestamp_s": np.arange(len(bvp_vals)) / bvp_fs,
            "BVP_nT": bvp_vals,
            "target_label": labels_bvp,
        }),
        "IBI": pd.DataFrame({
            "timestamp_s": ibi_times,
            "IBI_ms": ibi_vals,
            "target_label": labels_ibi,
        }),
        "ST": pd.DataFrame({
            "timestamp_s": np.arange(len(st_vals)) / st_fs,
            "ST_degC": st_vals,
            "target_label": labels_st,
        }),
        "ACC": pd.DataFrame({
            "timestamp_s": np.arange(len(ax)) / acc_fs,
            "ACC_X_g": ax,
            "ACC_Y_g": ay,
            "ACC_Z_g": az,
            "target_label": labels_acc,
        }),
    }


@pytest.fixture
def sample_demographics():
    return {
        "age": 10,
        "gender": "Male",
        "ethnicity": "White British",
        "autism_severity": "Medium",
        "verbal_status": "Minimally verbal",
        "comorbidity": "Yes",
    }


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "m4_test_output"


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestUtilities:
    def test_safe_finite(self):
        assert _safe(3.14) == 3.14

    def test_safe_nan(self):
        assert np.isnan(_safe(np.nan))

    def test_safe_inf(self):
        assert np.isnan(_safe(np.inf))

    def test_safe_default(self):
        assert _safe(np.nan, default=0.0) == 0.0

    def test_safe_none(self):
        assert np.isnan(_safe(None))

    def test_band_power_short_signal(self):
        assert np.isnan(_band_power(np.array([1, 2, 3]), 4.0, 0.0, 1.0))

    def test_band_power_valid(self, rng):
        sig = rng.normal(0, 1, 256)
        power = _band_power(sig, 64.0, 0.5, 8.0)
        assert np.isfinite(power)
        assert power >= 0

    def test_dominant_freq_sinusoid(self):
        fs = 64
        t = np.arange(512) / fs
        sig = np.sin(2 * np.pi * 5.0 * t)
        freq = _dominant_freq(sig, fs, 0.0, 32.0)
        assert abs(freq - 5.0) < 1.0  # within 1 Hz

    def test_spectral_centroid_valid(self, rng):
        sig = rng.normal(0, 1, 256)
        sc = _spectral_centroid(sig, 64.0)
        assert np.isfinite(sc) and sc > 0

    def test_spectral_entropy_valid(self, rng):
        sig = rng.normal(0, 1, 256)
        se = _spectral_entropy(sig, 64.0)
        assert np.isfinite(se) and se > 0

    def test_sample_entropy_valid(self, rng):
        sig = rng.normal(0, 1, 200)
        se = _sample_entropy(sig)
        assert np.isfinite(se)

    def test_sample_entropy_constant_signal(self):
        sig = np.ones(100)
        se = _sample_entropy(sig)
        assert np.isnan(se)  # zero std -> nan

    def test_sample_entropy_short(self):
        assert np.isnan(_sample_entropy(np.array([1, 2, 3])))

    def test_zero_crossing_rate_sine(self):
        fs = 100
        t = np.arange(1000) / fs
        sig = np.sin(2 * np.pi * 5.0 * t)
        zcr = _zero_crossing_rate(sig)
        # ~10 crossings per second / 100 samples per second ~ 0.1
        assert 0.05 < zcr < 0.15

    def test_decompose_eda(self, eda_signal):
        sig, fs = eda_signal
        tonic, phasic = _decompose_eda(sig, fs)
        assert len(tonic) == len(sig)
        assert len(phasic) == len(sig)
        assert np.all(phasic >= 0)

    def test_poincare_valid(self):
        rr = np.array([800, 810, 795, 820, 805, 815, 790, 825])
        sd1, sd2 = _poincare(rr)
        assert np.isfinite(sd1) and sd1 > 0
        assert np.isfinite(sd2) and sd2 > 0

    def test_poincare_too_short(self):
        sd1, sd2 = _poincare(np.array([800, 810]))
        assert np.isnan(sd1)


# ─────────────────────────────────────────────────────────────────────────────
# PER-SIGNAL FEATURE EXTRACTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEDAFeatures:
    def test_eda_feature_count(self, eda_signal):
        sig, fs = eda_signal
        feats = extract_eda_features(sig, fs)
        assert len(feats) >= 18, f"Expected >= 18, got {len(feats)}"

    def test_eda_scl_mean_plausible(self, eda_signal):
        sig, fs = eda_signal
        feats = extract_eda_features(sig, fs)
        assert 0.01 <= feats["eda_scl_mean"] <= 30.0

    def test_eda_scr_rate_nonnegative(self, eda_signal):
        sig, fs = eda_signal
        feats = extract_eda_features(sig, fs)
        assert feats["eda_scr_rate"] >= 0

    def test_eda_empty_signal(self):
        feats = extract_eda_features(np.array([]), 4.0)
        assert feats == {}

    def test_eda_short_signal(self):
        feats = extract_eda_features(np.array([1.0, 2.0, 3.0]), 4.0)
        assert feats == {}

    def test_eda_all_values_finite(self, eda_signal):
        sig, fs = eda_signal
        feats = extract_eda_features(sig, fs)
        for k, v in feats.items():
            if v is not None:
                assert np.isfinite(v) or np.isnan(v), f"{k} = {v}"


class TestBVPFeatures:
    def test_bvp_feature_count(self, bvp_signal):
        sig, fs = bvp_signal
        feats = extract_bvp_features(sig, fs)
        assert len(feats) >= 13

    def test_bvp_hr_plausible(self, bvp_signal):
        sig, fs = bvp_signal
        feats = extract_bvp_features(sig, fs)
        hr = feats.get("bvp_hr_mean", np.nan)
        if np.isfinite(hr):
            assert 40 <= hr <= 200, f"HR={hr} outside plausible range"

    def test_bvp_sqi_range(self, bvp_signal):
        sig, fs = bvp_signal
        feats = extract_bvp_features(sig, fs)
        sqi = feats.get("bvp_sqi", 0)
        assert 0 <= sqi <= 1

    def test_bvp_empty(self):
        assert extract_bvp_features(np.array([]), 64.0) == {}


class TestIBIFeatures:
    def test_ibi_feature_count(self, ibi_data):
        times, values = ibi_data
        feats = extract_ibi_features(times, values)
        assert len(feats) >= 17

    def test_ibi_sdnn_positive(self, ibi_data):
        times, values = ibi_data
        feats = extract_ibi_features(times, values)
        assert feats["ibi_sdnn"] > 0

    def test_ibi_rmssd_positive(self, ibi_data):
        times, values = ibi_data
        feats = extract_ibi_features(times, values)
        assert feats["ibi_rmssd"] > 0

    def test_ibi_pnn50_range(self, ibi_data):
        times, values = ibi_data
        feats = extract_ibi_features(times, values)
        assert 0 <= feats["ibi_pnn50"] <= 100

    def test_ibi_lf_hf_nonnegative(self, ibi_data):
        times, values = ibi_data
        feats = extract_ibi_features(times, values)
        if np.isfinite(feats.get("ibi_lf_hf", np.nan)):
            assert feats["ibi_lf_hf"] >= 0

    def test_ibi_too_few_beats(self):
        feats = extract_ibi_features(np.array([0, 1]), np.array([800, 810]))
        assert feats == {}


class TestSTFeatures:
    def test_st_feature_count(self, st_signal):
        sig, fs = st_signal
        feats = extract_st_features(sig, fs)
        assert len(feats) >= 10

    def test_st_mean_plausible(self, st_signal):
        sig, fs = st_signal
        feats = extract_st_features(sig, fs)
        assert 25 <= feats["st_mean"] <= 40

    def test_st_empty(self):
        assert extract_st_features(np.array([]), 4.0) == {}


class TestACCFeatures:
    def test_acc_feature_count(self, acc_signal):
        ax, ay, az, fs = acc_signal
        feats = extract_acc_features(ax, ay, az, fs)
        assert len(feats) >= 14

    def test_acc_svm_mean_positive(self, acc_signal):
        ax, ay, az, fs = acc_signal
        feats = extract_acc_features(ax, ay, az, fs)
        assert feats["acc_svm_mean"] > 0

    def test_acc_dom_freq_in_range(self, acc_signal):
        ax, ay, az, fs = acc_signal
        feats = extract_acc_features(ax, ay, az, fs)
        df = feats.get("acc_dom_freq", np.nan)
        if np.isfinite(df):
            assert 0 <= df <= fs / 2

    def test_acc_empty(self):
        e = np.array([])
        assert extract_acc_features(e, e, e, 32.0) == {}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureExtractor:
    def test_extract_all_returns_all_signals(self, sample_signals):
        ext = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        result = ext.extract_all(sample_signals, session_id="test", user_id="u1")
        assert set(result.keys()) == {"EDA", "BVP", "IBI", "ST", "ACC"}

    def test_extract_all_has_meta_columns(self, sample_signals):
        ext = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        result = ext.extract_all(sample_signals)
        for sig_name, df in result.items():
            if len(df) > 0:
                assert "session_id" in df.columns
                assert "user_id" in df.columns
                assert "window_start_s" in df.columns
                assert "target_label" in df.columns

    def test_extract_all_preserves_labels(self, sample_signals):
        ext = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        result = ext.extract_all(sample_signals)
        for sig_name, df in result.items():
            if len(df) > 0:
                labels = df["target_label"].unique()
                assert len(labels) >= 1

    def test_smaller_window_more_rows(self, sample_signals):
        ext30 = FeatureExtractor(window_s=30.0, overlap=0.5, verbose=False)
        ext60 = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        r30 = ext30.extract_all(sample_signals)
        r60 = ext60.extract_all(sample_signals)
        # 30s windows should produce more rows than 60s on same data
        for sig in ("EDA", "BVP", "ST", "ACC"):
            if sig in r30 and sig in r60 and len(r60[sig]) > 0:
                assert len(r30[sig]) >= len(r60[sig])

    def test_partial_signals(self):
        """Only EDA provided — other signals should be absent."""
        signals = {
            "EDA": pd.DataFrame({
                "timestamp_s": np.arange(240) / 4.0,
                "EDA_uS": np.random.default_rng(42).normal(3, 0.5, 240),
            })
        }
        ext = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        result = ext.extract_all(signals)
        assert "EDA" in result
        assert "BVP" not in result


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestNormaliser:
    def test_fit_transform_returns_same_structure(self, sample_signals):
        ext = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        raw = ext.extract_all(sample_signals)

        norm = FeatureNormaliser(scaler_type="robust", verbose=False)
        scaled = norm.fit_transform(raw)
        assert set(scaled.keys()) == set(raw.keys())
        for sig in raw:
            assert scaled[sig].shape == raw[sig].shape

    def test_transform_after_fit(self, sample_signals):
        ext = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        raw = ext.extract_all(sample_signals)

        norm = FeatureNormaliser(scaler_type="robust", verbose=False)
        norm.fit_transform(raw)
        # Transform again (simulating test data)
        scaled2 = norm.transform(raw)
        assert set(scaled2.keys()) == set(raw.keys())

    def test_transform_without_fit_raises(self):
        norm = FeatureNormaliser(verbose=False)
        with pytest.raises(ValueError, match="No fitted scaler"):
            norm.transform({"EDA": pd.DataFrame({"a": [1, 2, 3]})})

    def test_save_load_roundtrip(self, sample_signals, tmp_path):
        ext = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        raw = ext.extract_all(sample_signals)

        norm = FeatureNormaliser(scaler_type="robust", verbose=False)
        scaled1 = norm.fit_transform(raw)
        path = tmp_path / "scaler.joblib"
        norm.save(path)

        norm2 = FeatureNormaliser.load(path)
        scaled2 = norm2.transform(raw)
        for sig in scaled1:
            if len(scaled1[sig]) > 0:
                pd.testing.assert_frame_equal(scaled1[sig], scaled2[sig])

    def test_all_scaler_types(self, sample_signals):
        ext = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        raw = ext.extract_all(sample_signals)
        for stype in ("robust", "standard", "minmax"):
            norm = FeatureNormaliser(scaler_type=stype, verbose=False)
            scaled = norm.fit_transform(raw)
            assert len(scaled) == len(raw)

    def test_meta_cols_not_scaled(self, sample_signals):
        ext = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        raw = ext.extract_all(sample_signals)

        norm = FeatureNormaliser(scaler_type="robust", verbose=False)
        scaled = norm.fit_transform(raw)
        for sig in raw:
            if "session_id" in raw[sig].columns:
                assert list(raw[sig]["session_id"]) == list(scaled[sig]["session_id"])
            if "target_label" in raw[sig].columns:
                assert list(raw[sig]["target_label"]) == list(scaled[sig]["target_label"])


# ─────────────────────────────────────────────────────────────────────────────
# DEMOGRAPHIC ENCODER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDemographicEncoder:
    def test_encode_all_fields(self, sample_demographics):
        enc = DemographicEncoder.encode(sample_demographics)
        assert enc["age"] == 10
        assert enc["gender_enc"] == 0  # Male
        assert enc["autism_severity_enc"] == 2  # Medium
        assert enc["verbal_status_enc"] == 1  # Minimally verbal
        assert enc["comorbidity_enc"] == 1  # Yes

    def test_unknown_value_returns_minus_one(self):
        demo = {"gender": "Unknown", "age": 8}
        enc = DemographicEncoder.encode(demo)
        assert enc["gender_enc"] == -1

    def test_encode_dataframe(self):
        df = pd.DataFrame({
            "val": [1, 2, 3],
            "gender": ["Male", "Female", "Non-binary"],
            "autism_severity": ["Low", "Medium", "Severe"],
        })
        out = DemographicEncoder.encode_dataframe(df)
        assert "gender_enc" in out.columns
        assert list(out["gender_enc"]) == [0, 1, 2]
        assert list(out["autism_severity_enc"]) == [1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# ONE-HOT ENCODER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestOneHotEncoder:
    def test_encode_creates_binary_cols(self):
        df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0],
            "gender": ["Male", "Female", "Male"],
            "autism_severity": ["Low", "Medium", "Severe"],
        })
        ohe = OneHotEncoder(verbose=False)
        out = ohe.encode_dataframe(df)
        assert "gender_Male" in out.columns
        assert "gender_Female" in out.columns
        assert "gender" not in out.columns
        assert out["gender_Male"].dtype == int

    def test_encode_no_categorical_cols(self):
        df = pd.DataFrame({"feat1": [1.0, 2.0], "feat2": [3.0, 4.0]})
        ohe = OneHotEncoder(verbose=False)
        out = ohe.encode_dataframe(df)
        pd.testing.assert_frame_equal(out, df)

    def test_encode_all_dict(self):
        dfs = {
            "EDA": pd.DataFrame({
                "feat": [1.0, 2.0],
                "gender": ["Male", "Female"],
            }),
            "BVP": pd.DataFrame({
                "feat": [3.0, 4.0],
                "gender": ["Female", "Male"],
            }),
        }
        ohe = OneHotEncoder(verbose=False)
        encoded = ohe.encode_all(dfs)
        assert "EDA" in encoded and "BVP" in encoded
        assert "gender_Male" in encoded["EDA"].columns

    def test_drop_first(self):
        df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0],
            "gender": ["Male", "Female", "Non-binary"],
        })
        ohe = OneHotEncoder(drop_first=True, verbose=False)
        out = ohe.encode_dataframe(df)
        # With 3 categories and drop_first, should have 2 columns
        gender_cols = [c for c in out.columns if c.startswith("gender_")]
        assert len(gender_cols) == 2


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE FUSER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureFuser:
    def test_fuse_returns_combined(self, sample_signals, sample_demographics):
        ext = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        raw = ext.extract_all(sample_signals)

        fuser = FeatureFuser()
        per_signal, combined = fuser.fuse(raw, sample_demographics)
        assert len(per_signal) == len(raw)
        assert isinstance(combined, pd.DataFrame)
        if len(combined) > 0:
            assert combined.shape[1] > max(df.shape[1] for df in raw.values() if len(df) > 0)

    def test_fuse_without_demographics(self, sample_signals):
        ext = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        raw = ext.extract_all(sample_signals)
        fuser = FeatureFuser()
        per_signal, combined = fuser.fuse(raw, None)
        assert isinstance(combined, pd.DataFrame)

    def test_fuse_single_signal(self):
        df = pd.DataFrame({
            "session_id": ["s1", "s1"],
            "user_id": ["u1", "u1"],
            "window_start_s": [0.0, 30.0],
            "target_label": ["Happy", "Fear"],
            "feat_a": [1.0, 2.0],
        })
        fuser = FeatureFuser()
        _, combined = fuser.fuse({"EDA": df}, None)
        assert len(combined) == 2


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureSelector:
    @pytest.fixture
    def selector_df(self, rng):
        """DataFrame with known features and target labels."""
        n = 100
        return pd.DataFrame({
            "session_id": ["s1"] * n,
            "user_id": ["u1"] * n,
            "window_start_s": np.arange(n, dtype=float),
            "target_label": rng.choice(["Happy", "Fear", "Sad"], n),
            "feat_informative": rng.choice([0.0, 1.0, 2.0], n,
                                           p=[0.33, 0.33, 0.34]),
            "feat_noise_1": rng.normal(0, 1, n),
            "feat_noise_2": rng.normal(0, 1, n),
            "feat_noise_3": rng.normal(0, 1, n),
            "feat_constant": np.ones(n),
        })

    def test_ensemble_ranking(self, selector_df):
        sel = FeatureSelector(top_n=3, method="ensemble", verbose=False)
        imp, selected, full = sel.rank_and_select(selector_df)
        assert len(imp) > 0
        assert "rank" in imp.columns
        assert imp["rank"].iloc[0] == 1

    def test_mutual_information(self, selector_df):
        sel = FeatureSelector(top_n=2, method="mutual_information", verbose=False)
        imp, selected, _ = sel.rank_and_select(selector_df)
        assert len(imp) > 0

    def test_random_forest(self, selector_df):
        sel = FeatureSelector(top_n=2, method="random_forest", verbose=False)
        imp, selected, _ = sel.rank_and_select(selector_df)
        assert len(imp) > 0

    def test_f_statistic(self, selector_df):
        sel = FeatureSelector(top_n=2, method="f_statistic", verbose=False)
        imp, selected, _ = sel.rank_and_select(selector_df)
        assert len(imp) > 0

    def test_variance_threshold(self, selector_df):
        sel = FeatureSelector(top_n=2, method="variance_threshold", verbose=False)
        imp, selected, _ = sel.rank_and_select(selector_df)
        assert len(imp) > 0

    def test_selected_df_has_top_n_features(self, selector_df):
        sel = FeatureSelector(top_n=2, method="ensemble", verbose=False)
        imp, selected, _ = sel.rank_and_select(selector_df)
        # Selected should have meta cols + top 2 features
        feat_cols = [c for c in selected.columns
                     if c not in META_COLS and c != "target_label"]
        assert len(feat_cols) == 2

    def test_no_target_column_falls_back_to_variance(self):
        df = pd.DataFrame({
            "session_id": ["s1"] * 10,
            "feat_a": np.random.default_rng(42).normal(0, 1, 10),
            "feat_b": np.random.default_rng(42).normal(0, 0.01, 10),
        })
        sel = FeatureSelector(top_n=1, verbose=False)
        imp, selected, _ = sel.rank_and_select(df)
        assert len(imp) > 0

    def test_empty_df(self):
        sel = FeatureSelector(verbose=False)
        imp, selected, full = sel.rank_and_select(pd.DataFrame())
        assert len(imp) == 0


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONALITY REDUCTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDimensionalityReducer:
    @pytest.fixture
    def dimred_df(self, rng):
        n = 80
        return pd.DataFrame({
            "session_id": ["s1"] * n,
            "user_id": ["u1"] * n,
            "window_start_s": np.arange(n, dtype=float),
            "target_label": rng.choice(["Happy", "Fear", "Sad", "Anger"], n),
            **{f"feat_{i}": rng.normal(0, 1, n) for i in range(30)},
        })

    def test_pca_reduces_dimensions(self, dimred_df):
        reducer = DimensionalityReducer(pca_variance=0.95, verbose=False)
        results = reducer.reduce_and_compare(dimred_df)
        pca = results["pca_result"]
        assert pca["n_components"] <= 30
        assert pca["total_variance"] >= 0.94

    def test_pls_vip_identifies_features(self, dimred_df):
        reducer = DimensionalityReducer(verbose=False)
        results = reducer.reduce_and_compare(dimred_df)
        vip = results["pls_vip_result"]
        assert vip is not None
        assert "vip_ranking" in vip
        assert len(vip["vip_ranking"]) == 30

    def test_svd_runs(self, dimred_df):
        reducer = DimensionalityReducer(verbose=False)
        results = reducer.reduce_and_compare(dimred_df)
        svd = results["svd_result"]
        assert svd["transformed"] is not None

    def test_comparison_recommends(self, dimred_df):
        reducer = DimensionalityReducer(verbose=False)
        results = reducer.reduce_and_compare(dimred_df)
        assert "best_method" in results
        assert results["best_method"] in ("pca", "pls_vip")

    def test_no_target_skips_pls(self):
        df = pd.DataFrame({
            "session_id": ["s1"] * 20,
            **{f"feat_{i}": np.random.default_rng(42).normal(0, 1, 20) for i in range(10)},
        })
        reducer = DimensionalityReducer(verbose=False)
        results = reducer.reduce_and_compare(df)
        assert results["pls_vip_result"] is None
        assert results["best_method"] == "pca"


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestExporter:
    def test_export_creates_directories(self, tmp_output):
        FeatureEngineeringExporter(tmp_output)
        assert (tmp_output / "features_raw").exists()
        assert (tmp_output / "features_normalised").exists()
        assert (tmp_output / "features_encoded").exists()
        assert (tmp_output / "features_selected").exists()
        assert (tmp_output / "dimensionality_reduction").exists()

    def test_export_all_writes_files(self, tmp_output):
        exporter = FeatureEngineeringExporter(tmp_output)
        raw = {"EDA": pd.DataFrame({"feat": [1, 2, 3]})}
        norm = {"EDA": pd.DataFrame({"feat": [0.1, 0.2, 0.3]})}
        encoded = {"EDA": pd.DataFrame({"feat": [0.1, 0.2, 0.3]})}
        combined = pd.DataFrame({"feat": [1, 2, 3]})
        imp = pd.DataFrame({"feature": ["a"], "rank": [1]})
        selected = pd.DataFrame({"feat": [1, 2, 3]})
        dimred = {}

        exporter.export_all(
            raw_features=raw,
            norm_features=norm,
            raw_combined=combined,
            norm_combined=combined,
            encoded_features=encoded,
            encoded_combined=combined,
            importance_df=imp,
            selected_df=selected,
            selected_encoded_df=selected,
            dimred_results=dimred,
            metadata={"test": True},
        )
        assert (tmp_output / "features_raw" / "EDA_features.csv").exists()
        assert (tmp_output / "feature_engineering_metadata.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
# END-TO-END ORCHESTRATOR TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineerE2E:
    def test_full_pipeline(self, sample_signals, sample_demographics, tmp_output):
        engineer = FeatureEngineer(
            window_s=30.0,
            overlap=0.5,
            scaler_type="robust",
            top_n_features=5,
            selection_method="ensemble",
            verbose=False,
        )
        result = engineer.run(
            signals_input=sample_signals,
            demographics=sample_demographics,
            session_id="test_session",
            user_id="test_user",
            output_dir=str(tmp_output),
        )

        # All expected keys in result
        assert "run_folder" in result
        assert "raw_features" in result
        assert "norm_features" in result
        assert "raw_combined" in result
        assert "norm_combined" in result
        assert "encoded_features" in result
        assert "importance_df" in result
        assert "selected_df" in result
        assert "dimred_results" in result
        assert "metadata" in result

        # Check output files were written
        assert (tmp_output / "features_raw").exists()
        assert (tmp_output / "features_normalised").exists()
        assert (tmp_output / "features_encoded").exists()
        assert (tmp_output / "feature_engineering_metadata.json").exists()

    def test_pipeline_without_demographics(self, sample_signals, tmp_path):
        out = tmp_path / "no_demo"
        engineer = FeatureEngineer(
            window_s=30.0, overlap=0.5, verbose=False
        )
        result = engineer.run(
            signals_input=sample_signals,
            demographics=None,
            session_id="s1",
            user_id="u1",
            output_dir=str(out),
        )
        assert result["metadata"]["demographics"] == {}

    def test_pipeline_different_scaler(self, sample_signals, tmp_path):
        for stype in ("robust", "standard", "minmax"):
            out = tmp_path / f"scaler_{stype}"
            eng = FeatureEngineer(
                window_s=30.0, overlap=0.5, scaler_type=stype, verbose=False
            )
            result = eng.run(
                signals_input=sample_signals,
                output_dir=str(out),
            )
            assert result["metadata"]["scaler_type"] == stype


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    def test_feature_extraction_deterministic(self, sample_signals):
        ext = FeatureExtractor(window_s=60.0, overlap=0.5, verbose=False)
        r1 = ext.extract_all(sample_signals)
        r2 = ext.extract_all(sample_signals)
        for sig in r1:
            if len(r1[sig]) > 0:
                pd.testing.assert_frame_equal(r1[sig], r2[sig])

    def test_selector_seeded_determinism(self):
        rng = np.random.default_rng(99)
        n = 50
        df = pd.DataFrame({
            "session_id": ["s1"] * n,
            "target_label": rng.choice(["A", "B", "C"], n),
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "f3": rng.normal(0, 1, n),
        })
        s1 = FeatureSelector(top_n=2, seed=42, verbose=False)
        s2 = FeatureSelector(top_n=2, seed=42, verbose=False)
        imp1, _, _ = s1.rank_and_select(df)
        imp2, _, _ = s2.rank_and_select(df)
        pd.testing.assert_frame_equal(imp1, imp2)


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_nan_signal(self):
        sig = np.full(240, np.nan)
        feats = extract_eda_features(sig, 4.0)
        # Should return features but values should be NaN
        if feats:
            for v in feats.values():
                if v is not None:
                    assert np.isnan(v) or np.isfinite(v)

    def test_constant_signal(self):
        sig = np.full(240, 5.0)
        feats = extract_eda_features(sig, 4.0)
        assert feats.get("eda_std", -1) == 0.0

    def test_single_value_per_signal(self):
        """Signals with < MIN_WINDOW_SAMPLES should return empty."""
        assert extract_eda_features(np.array([1.0]), 4.0) == {}
        assert extract_bvp_features(np.array([1.0]), 64.0) == {}
        assert extract_st_features(np.array([1.0]), 4.0) == {}
        assert extract_acc_features(
            np.array([0.0]), np.array([0.0]), np.array([1.0]), 32.0
        ) == {}

    def test_inf_in_signal(self):
        sig = np.array([1.0, 2.0, np.inf, 3.0] * 60)
        feats = extract_eda_features(sig, 4.0)
        # Should not crash — features may be nan but no exception
        assert isinstance(feats, dict)

    def test_negative_eda(self):
        """EDA should not be negative in real data, but extractor should handle it."""
        sig = np.array([-0.5, -0.3, 0.1, 0.5] * 60)
        feats = extract_eda_features(sig, 4.0)
        assert isinstance(feats, dict)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FOLDER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

class TestRunFolder:
    def test_next_run_folder_auto_increments(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "feature_engineer.MODULE_DIR", tmp_path
        )
        f1 = next_run_folder()
        f2 = next_run_folder()
        assert f1 != f2
        assert f1.exists()
        assert f2.exists()
        # Second folder number should be higher
        n1 = int(f1.name.split("_")[-1])
        n2 = int(f2.name.split("_")[-1])
        assert n2 == n1 + 1
