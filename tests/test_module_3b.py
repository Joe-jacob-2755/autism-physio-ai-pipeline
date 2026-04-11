"""
=============================================================================
TESTS — MODULE 3B: DATA SPLITTING
=============================================================================
Comprehensive test suite covering:
  - DataSplitter configuration (ratios, modes, validation)
  - User-level splitting (random)
  - Stratified splitting (by demographics)
  - Data partitioning (split_data)
  - SplitResult dataclass
  - SplitExporter (output structure, metadata)
  - Edge cases (minimum users, single stratum, fallback)
  - Reproducibility (seeded determinism)
  - M3 output loading (load_from_m3_outputs)
  - Run folder auto-numbering

Run:  python -m pytest tests/test_module_3b.py -v --tb=short
=============================================================================
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Setup: add module_3 to path (isolated to avoid config.py collision) ──
REPO_ROOT = Path(__file__).resolve().parent.parent
M3_DIR = REPO_ROOT / "module_3_data_splitting"

# Clean up any config.py from a previous test module (e.g. M2A)
for _stale in ["config", "exporter", "splitter", "annotator",
               "simulator", "noise_injector", "event_scheduler"]:
    if _stale in sys.modules:
        del sys.modules[_stale]

# Save original sys.path and sys.modules state to prevent config.py
# from M3 leaking into other test modules (see CLAUDE.md Rule 3.2).
_original_path = sys.path.copy()
_original_modules = set(sys.modules.keys())

sys.path.insert(0, str(M3_DIR))

from splitter import DataSplitter, SplitResult, _StratificationError  # noqa: E402
from exporter import SplitExporter  # noqa: E402
import config as m3b_config  # noqa: E402

# Pull out the constants we need, then clean up sys.path
DEFAULT_SPLIT_MODE = m3b_config.DEFAULT_SPLIT_MODE
DEFAULT_TRAIN_RATIO = m3b_config.DEFAULT_TRAIN_RATIO
DEFAULT_VAL_RATIO = m3b_config.DEFAULT_VAL_RATIO
DEFAULT_TEST_RATIO = m3b_config.DEFAULT_TEST_RATIO
DEFAULT_TRAIN_RATIO_2WAY = m3b_config.DEFAULT_TRAIN_RATIO_2WAY
DEFAULT_TEST_RATIO_2WAY = m3b_config.DEFAULT_TEST_RATIO_2WAY
DEFAULT_SEED = m3b_config.DEFAULT_SEED
USER_COL = m3b_config.USER_COL
STRATIFY_COLUMNS = m3b_config.STRATIFY_COLUMNS
MIN_USERS_PER_STRATUM = m3b_config.MIN_USERS_PER_STRATUM
MODULE_VERSION = m3b_config.MODULE_VERSION
MODULE_LABEL = m3b_config.MODULE_LABEL

# Remove M3 from sys.path so 'config' doesn't shadow other modules' config.
# Also remove bare module names from sys.modules cache so that when M6 tests
# import 'config', 'exporter', 'splitter', they get their own versions.
if str(M3_DIR) in sys.path:
    sys.path.remove(str(M3_DIR))
for _mod_name in ["config", "exporter", "splitter"]:
    if _mod_name in sys.modules:
        del sys.modules[_mod_name]


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES — synthetic user demographics and feature data
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def user_demographics_10():
    """10 users with varied demographics for stratification testing."""
    return pd.DataFrame({
        "user_id": [f"user_{i:03d}" for i in range(1, 11)],
        "age": [6, 8, 10, 12, 7, 9, 11, 5, 14, 13],
        "gender": ["Male"] * 7 + ["Female"] * 3,
        "ethnicity": ["White"] * 5 + ["Asian"] * 3 + ["Black"] * 2,
        "autism_severity": ["Low", "Medium", "Severe", "Low", "Medium",
                            "Severe", "Low", "Medium", "Severe", "Low"],
        "verbal_status": ["Verbal", "Minimally verbal", "Non-verbal",
                          "Verbal", "Minimally verbal", "Non-verbal",
                          "Verbal", "Minimally verbal", "Non-verbal",
                          "Verbal"],
        "comorbidity": ["Yes", "No"] * 5,
    })


@pytest.fixture
def user_demographics_4():
    """4 users — minimum for meaningful splitting."""
    return pd.DataFrame({
        "user_id": ["user_001", "user_002", "user_003", "user_004"],
        "age": [6, 10, 8, 12],
        "autism_severity": ["Low", "Medium", "Severe", "Low"],
        "verbal_status": ["Verbal", "Non-verbal", "Verbal", "Non-verbal"],
    })


@pytest.fixture
def user_demographics_2():
    """2 users — absolute minimum."""
    return pd.DataFrame({
        "user_id": ["user_001", "user_002"],
        "age": [6, 10],
        "autism_severity": ["Low", "Severe"],
        "verbal_status": ["Verbal", "Non-verbal"],
    })


@pytest.fixture
def feature_dfs_10():
    """Per-signal feature DataFrames for 10 users, 5 windows each."""
    rng = np.random.default_rng(42)
    dfs = {}
    for sig in ["EDA", "BVP", "IBI", "ST", "ACC"]:
        rows = []
        for i in range(1, 11):
            for w in range(5):
                row = {
                    "user_id": f"user_{i:03d}",
                    "session_id": f"session_{i:03d}",
                    "window_start_s": w * 30.0,
                    "window_end_s": (w + 1) * 30.0,
                    "target_label": rng.choice(
                        ["Happy", "Fear", "Anger", "Sad"]),
                }
                for f_idx in range(5):
                    row[f"{sig.lower()}_feat_{f_idx}"] = rng.normal(0, 1)
                rows.append(row)
        dfs[sig] = pd.DataFrame(rows)
    return dfs


@pytest.fixture
def combined_df_10(feature_dfs_10):
    """Combined features DataFrame for 10 users."""
    # Take first signal and add some columns from others
    base = feature_dfs_10["EDA"].copy()
    for sig in ["BVP", "IBI", "ST", "ACC"]:
        feat_cols = [c for c in feature_dfs_10[sig].columns
                     if c.startswith(sig.lower())]
        for col in feat_cols:
            base[col] = feature_dfs_10[sig][col].values
    return base


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 1: Configuration and validation
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitterConfig:
    """Test DataSplitter initialisation and parameter validation."""

    def test_default_three_way(self):
        s = DataSplitter(verbose=False)
        assert s.split_mode == "three_way"
        assert s.train_ratio == DEFAULT_TRAIN_RATIO
        assert s.val_ratio == DEFAULT_VAL_RATIO
        assert s.test_ratio == DEFAULT_TEST_RATIO

    def test_default_two_way(self):
        s = DataSplitter(split_mode="two_way", verbose=False)
        assert s.train_ratio == DEFAULT_TRAIN_RATIO_2WAY
        assert s.val_ratio == 0.0
        assert s.test_ratio == DEFAULT_TEST_RATIO_2WAY

    def test_custom_ratios(self):
        s = DataSplitter(
            split_mode="three_way",
            train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
            verbose=False,
        )
        assert s.train_ratio == 0.5
        assert s.val_ratio == 0.25
        assert s.test_ratio == 0.25

    def test_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            DataSplitter(
                train_ratio=0.5, val_ratio=0.5, test_ratio=0.5,
                verbose=False,
            )

    def test_two_way_ignores_val(self):
        s = DataSplitter(
            split_mode="two_way",
            train_ratio=0.7, test_ratio=0.3,
            verbose=False,
        )
        assert s.val_ratio == 0.0

    def test_seed_stored(self):
        s = DataSplitter(seed=123, verbose=False)
        assert s.seed == 123

    def test_stratify_cols_default(self):
        s = DataSplitter(verbose=False)
        assert s.stratify_cols == STRATIFY_COLUMNS

    def test_stratify_cols_custom(self):
        s = DataSplitter(stratify_cols=["age"], verbose=False)
        assert s.stratify_cols == ["age"]


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 2: Random splitting
# ─────────────────────────────────────────────────────────────────────────────

class TestRandomSplit:
    """Test random (non-stratified) user splitting."""

    def test_three_way_split_10_users(self, user_demographics_10):
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)

        assert len(result.train_users) > 0
        assert len(result.test_users) > 0
        assert len(result.val_users) > 0
        # All users accounted for
        all_users = (result.train_users + result.val_users
                     + result.test_users)
        assert len(all_users) == 10
        assert len(set(all_users)) == 10  # no duplicates

    def test_two_way_split_10_users(self, user_demographics_10):
        s = DataSplitter(
            split_mode="two_way", stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)

        assert len(result.train_users) > 0
        assert len(result.test_users) > 0
        assert len(result.val_users) == 0
        assert len(result.train_users) + len(result.test_users) == 10

    def test_minimum_2_users(self, user_demographics_2):
        s = DataSplitter(
            split_mode="two_way", stratify=False, verbose=False)
        result = s.split_users(user_demographics_2)

        assert len(result.train_users) >= 1
        assert len(result.test_users) >= 1

    def test_single_user_raises(self):
        df = pd.DataFrame({"user_id": ["user_001"], "age": [10]})
        s = DataSplitter(verbose=False)
        with pytest.raises(ValueError, match="at least 2"):
            s.split_users(df)

    def test_no_user_col_raises(self):
        df = pd.DataFrame({"participant": ["p1", "p2"]})
        s = DataSplitter(verbose=False)
        with pytest.raises(ValueError, match="user_id"):
            s.split_users(df)

    def test_no_duplicate_users(self, user_demographics_10):
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        all_users = (result.train_users + result.val_users
                     + result.test_users)
        assert len(all_users) == len(set(all_users))

    def test_approximate_ratios(self, user_demographics_10):
        """With 10 users and 60/20/20, expect ~6/2/2."""
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        # Allow ±1 user tolerance
        assert abs(len(result.train_users) - 6) <= 1
        assert abs(len(result.val_users) - 2) <= 1
        assert abs(len(result.test_users) - 2) <= 1


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 3: Stratified splitting
# ─────────────────────────────────────────────────────────────────────────────

class TestStratifiedSplit:
    """Test stratified user splitting by demographics."""

    def test_stratified_10_users(self, user_demographics_10):
        s = DataSplitter(stratify=True, verbose=False)
        result = s.split_users(user_demographics_10)

        assert result.stratified is True
        all_users = (result.train_users + result.val_users
                     + result.test_users)
        assert len(set(all_users)) == 10

    def test_fallback_to_random_few_users(self, user_demographics_2):
        """With only 2 users, stratification should fall back to random."""
        s = DataSplitter(
            split_mode="two_way", stratify=True, verbose=False)
        result = s.split_users(user_demographics_2)
        # Should still produce a valid split (fell back to random)
        assert len(result.train_users) + len(result.test_users) == 2

    def test_missing_stratify_cols_falls_back(self):
        """If stratify columns don't exist, fall back to random."""
        df = pd.DataFrame({
            "user_id": [f"u{i}" for i in range(6)],
            "age": [6, 7, 8, 9, 10, 11],
        })
        s = DataSplitter(
            stratify=True,
            stratify_cols=["nonexistent_col"],
            verbose=False,
        )
        result = s.split_users(df)
        # Should produce valid split via fallback
        all_users = (result.train_users + result.val_users
                     + result.test_users)
        assert len(set(all_users)) == 6

    def test_stratified_preserves_all_users(self, user_demographics_10):
        s = DataSplitter(stratify=True, verbose=False)
        result = s.split_users(user_demographics_10)
        all_assigned = set(
            result.train_users + result.val_users + result.test_users)
        all_input = set(user_demographics_10["user_id"])
        assert all_assigned == all_input


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 4: Data partitioning
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitData:
    """Test splitting actual DataFrames by user assignment."""

    def test_split_data_partitions_correctly(
        self, user_demographics_10, feature_dfs_10, combined_df_10,
    ):
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        result = s.split_data(result, feature_dfs_10, combined_df_10)

        # Every signal should be in train and test data
        for sig in ["EDA", "BVP", "IBI", "ST", "ACC"]:
            assert sig in result.train_data
            assert sig in result.test_data
            assert len(result.train_data[sig]) > 0
            assert len(result.test_data[sig]) > 0

        # Combined should be split too
        assert "combined" in result.train_data
        assert "combined" in result.test_data

    def test_no_user_leakage(
        self, user_demographics_10, feature_dfs_10, combined_df_10,
    ):
        """No user should appear in more than one split."""
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        result = s.split_data(result, feature_dfs_10, combined_df_10)

        train_users = set(result.train_data["EDA"]["user_id"].unique())
        test_users = set(result.test_data["EDA"]["user_id"].unique())
        val_users = set()
        if "EDA" in result.val_data:
            val_users = set(result.val_data["EDA"]["user_id"].unique())

        assert train_users.isdisjoint(test_users)
        assert train_users.isdisjoint(val_users)
        assert test_users.isdisjoint(val_users)

    def test_all_rows_accounted_for(
        self, user_demographics_10, feature_dfs_10, combined_df_10,
    ):
        """Total rows across splits should equal input rows."""
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        result = s.split_data(result, feature_dfs_10, combined_df_10)

        for sig in ["EDA", "BVP", "IBI", "ST", "ACC"]:
            total_input = len(feature_dfs_10[sig])
            total_split = len(result.train_data[sig])
            total_split += len(result.test_data[sig])
            if sig in result.val_data:
                total_split += len(result.val_data[sig])
            assert total_split == total_input

    def test_df_without_user_col_goes_to_train(self):
        """DataFrames without user_id go entirely to train."""
        s = DataSplitter(split_mode="two_way", stratify=False, verbose=False)
        demo = pd.DataFrame({
            "user_id": ["u1", "u2", "u3", "u4"],
            "autism_severity": ["Low", "Medium", "Severe", "Low"],
        })
        result = s.split_users(demo)

        # Feature DF with no user_id column
        no_uid_df = pd.DataFrame({"feat_a": [1, 2, 3], "feat_b": [4, 5, 6]})
        result = s.split_data(result, {"SIG": no_uid_df})

        assert "SIG" in result.train_data
        assert len(result.train_data["SIG"]) == 3


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 5: SplitResult
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitResult:
    """Test the SplitResult dataclass."""

    def test_summary_keys(self, user_demographics_10):
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        summary = result.summary()

        assert "split_mode" in summary
        assert "n_train" in summary
        assert "n_val" in summary
        assert "n_test" in summary
        assert "actual_train_ratio" in summary
        assert "seed" in summary

    def test_summary_actual_ratios(self, user_demographics_10):
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        summary = result.summary()

        total = summary["n_train"] + summary["n_val"] + summary["n_test"]
        assert total == 10
        assert abs(
            summary["actual_train_ratio"]
            + summary["actual_val_ratio"]
            + summary["actual_test_ratio"]
            - 1.0
        ) < 0.01

    def test_demographics_populated(self, user_demographics_10):
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)

        assert len(result.train_demographics) > 0
        assert len(result.test_demographics) > 0
        # Each demographic entry should have user fields
        assert "age" in result.train_demographics[0]


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 6: Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    """Test that same seed produces identical splits."""

    def test_same_seed_same_split(self, user_demographics_10):
        s1 = DataSplitter(seed=42, stratify=False, verbose=False)
        s2 = DataSplitter(seed=42, stratify=False, verbose=False)

        r1 = s1.split_users(user_demographics_10)
        r2 = s2.split_users(user_demographics_10)

        assert r1.train_users == r2.train_users
        assert r1.val_users == r2.val_users
        assert r1.test_users == r2.test_users

    def test_different_seed_different_split(self, user_demographics_10):
        s1 = DataSplitter(seed=42, stratify=False, verbose=False)
        s2 = DataSplitter(seed=99, stratify=False, verbose=False)

        r1 = s1.split_users(user_demographics_10)
        r2 = s2.split_users(user_demographics_10)

        # Very unlikely to produce the same split with different seeds
        assert (r1.train_users != r2.train_users
                or r1.test_users != r2.test_users)

    def test_stratified_reproducibility(self, user_demographics_10):
        s1 = DataSplitter(seed=42, stratify=True, verbose=False)
        s2 = DataSplitter(seed=42, stratify=True, verbose=False)

        r1 = s1.split_users(user_demographics_10)
        r2 = s2.split_users(user_demographics_10)

        assert r1.train_users == r2.train_users
        assert r1.test_users == r2.test_users


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 7: Exporter
# ─────────────────────────────────────────────────────────────────────────────

class TestExporter:
    """Test SplitExporter output structure and metadata."""

    def test_export_creates_folder_structure(
        self, tmp_path, user_demographics_10, feature_dfs_10, combined_df_10,
    ):
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        result = s.split_data(result, feature_dfs_10, combined_df_10)

        exporter = SplitExporter(output_root=tmp_path)
        saved = exporter.export_all(result)

        # Check folder structure
        run_dir = exporter.run_dir
        assert (run_dir / "train" / "features_raw").exists()
        assert (run_dir / "test" / "features_raw").exists()
        if result.val_users:
            assert (run_dir / "val" / "features_raw").exists()

        # Check metadata
        assert (run_dir / "split_metadata.json").exists()
        assert (run_dir / "user_assignments.csv").exists()

    def test_export_train_csvs_exist(
        self, tmp_path, user_demographics_10, feature_dfs_10, combined_df_10,
    ):
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        result = s.split_data(result, feature_dfs_10, combined_df_10)

        exporter = SplitExporter(output_root=tmp_path)
        exporter.export_all(result)

        raw_dir = exporter.run_dir / "train" / "features_raw"
        for sig in ["EDA", "BVP", "IBI", "ST", "ACC"]:
            assert (raw_dir / f"{sig}_features.csv").exists()
        assert (raw_dir / "combined_features.csv").exists()

    def test_metadata_json_valid(
        self, tmp_path, user_demographics_10, feature_dfs_10, combined_df_10,
    ):
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        result = s.split_data(result, feature_dfs_10, combined_df_10)

        exporter = SplitExporter(output_root=tmp_path)
        exporter.export_all(result)

        meta_path = exporter.run_dir / "split_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert "split_mode" in meta
        assert "train_users" in meta
        assert "test_users" in meta
        assert "seed" in meta
        assert "n_train" in meta

    def test_user_assignments_csv(
        self, tmp_path, user_demographics_10, feature_dfs_10, combined_df_10,
    ):
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        result = s.split_data(result, feature_dfs_10, combined_df_10)

        exporter = SplitExporter(output_root=tmp_path)
        exporter.export_all(result)

        csv_path = exporter.run_dir / "user_assignments.csv"
        df = pd.read_csv(csv_path)
        assert "user_id" in df.columns
        assert "split" in df.columns
        assert len(df) == 10
        assert set(df["split"].unique()) <= {"train", "val", "test"}

    def test_two_way_no_val_folder(
        self, tmp_path, user_demographics_10, feature_dfs_10, combined_df_10,
    ):
        s = DataSplitter(
            split_mode="two_way", stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        result = s.split_data(result, feature_dfs_10, combined_df_10)

        exporter = SplitExporter(output_root=tmp_path)
        exporter.export_all(result)

        assert not (exporter.run_dir / "val").exists()

    def test_export_with_norm_data(
        self, tmp_path, user_demographics_10, feature_dfs_10, combined_df_10,
    ):
        """Test exporting normalised features alongside raw."""
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(user_demographics_10)
        result = s.split_data(result, feature_dfs_10, combined_df_10)

        # Use same feature_dfs as "normalised" for testing
        exporter = SplitExporter(output_root=tmp_path)
        exporter.export_all(
            result,
            norm_feature_dfs=feature_dfs_10,
            norm_combined_df=combined_df_10,
        )

        norm_dir = exporter.run_dir / "train" / "features_normalised"
        assert norm_dir.exists()
        for sig in ["EDA", "BVP", "IBI", "ST", "ACC"]:
            assert (norm_dir / f"{sig}_features_norm.csv").exists()


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 8: Run folder management
# ─────────────────────────────────────────────────────────────────────────────

class TestRunFolder:
    """Test auto-numbered run folder creation."""

    def test_first_run_is_001(self, tmp_path):
        exporter = SplitExporter(output_root=tmp_path)
        assert exporter.run_dir.name.endswith("_001")

    def test_increments_run_number(self, tmp_path):
        e1 = SplitExporter(output_root=tmp_path)
        e2 = SplitExporter(output_root=tmp_path)
        # Second run should be 002
        num1 = int(e1.run_dir.name.split("_")[-1])
        num2 = int(e2.run_dir.name.split("_")[-1])
        assert num2 == num1 + 1

    def test_run_folder_contains_version(self, tmp_path):
        exporter = SplitExporter(output_root=tmp_path)
        assert MODULE_LABEL in exporter.run_dir.name
        assert MODULE_VERSION in exporter.run_dir.name


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 9: Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_3_users_three_way(self):
        """3 users, three_way: should produce 1/1/1 or similar."""
        df = pd.DataFrame({
            "user_id": ["u1", "u2", "u3"],
            "autism_severity": ["Low", "Medium", "Severe"],
            "verbal_status": ["Verbal", "Non-verbal", "Verbal"],
        })
        s = DataSplitter(stratify=False, verbose=False)
        result = s.split_users(df)

        all_u = result.train_users + result.val_users + result.test_users
        assert len(set(all_u)) == 3
        assert len(result.test_users) >= 1

    def test_2_users_two_way(self):
        """2 users, two_way: should produce 1/1."""
        df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "autism_severity": ["Low", "Severe"],
        })
        s = DataSplitter(split_mode="two_way", stratify=False, verbose=False)
        result = s.split_users(df)

        assert len(result.train_users) == 1
        assert len(result.test_users) == 1

    def test_all_same_stratum(self, user_demographics_10):
        """All users in same stratum — stratification still works."""
        df = user_demographics_10.copy()
        df["autism_severity"] = "Low"
        df["verbal_status"] = "Verbal"

        s = DataSplitter(stratify=True, verbose=False)
        result = s.split_users(df)

        all_u = result.train_users + result.val_users + result.test_users
        assert len(set(all_u)) == 10

    def test_empty_feature_df_skipped(self, tmp_path):
        """Empty DataFrames should be skipped gracefully."""
        demo = pd.DataFrame({
            "user_id": ["u1", "u2", "u3"],
            "autism_severity": ["Low", "Medium", "Severe"],
        })
        s = DataSplitter(
            split_mode="two_way", stratify=False, verbose=False)
        result = s.split_users(demo)

        empty_dfs = {"EDA": pd.DataFrame(columns=["user_id", "feat"])}
        result = s.split_data(result, empty_dfs)

        exporter = SplitExporter(output_root=tmp_path)
        saved = exporter.export_all(result)
        assert len(saved) > 0  # metadata + assignments always written


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASS 10: M3 output loading
# ─────────────────────────────────────────────────────────────────────────────

class TestM3Loading:
    """Test loading and aggregating M3 output folders."""

    def test_load_from_m3_outputs(self, tmp_path):
        """Create mock M3 output folders and verify loading."""
        rng = np.random.default_rng(42)

        for i in range(3):
            user_dir = tmp_path / f"M3_run_{i:03d}"
            feat_dir = user_dir / "features_raw"
            feat_dir.mkdir(parents=True)

            for sig in ["EDA", "BVP", "IBI", "ST", "ACC"]:
                df = pd.DataFrame({
                    "user_id": [f"user_{i:03d}"] * 5,
                    "session_id": [f"session_{i:03d}"] * 5,
                    "target_label": ["Happy"] * 5,
                    f"{sig.lower()}_feat_0": rng.normal(0, 1, 5),
                })
                df.to_csv(feat_dir / f"{sig}_features.csv", index=False)

            combined = pd.DataFrame({
                "user_id": [f"user_{i:03d}"] * 5,
                "session_id": [f"session_{i:03d}"] * 5,
                "target_label": ["Happy"] * 5,
                "eda_feat_0": rng.normal(0, 1, 5),
                "bvp_feat_0": rng.normal(0, 1, 5),
            })
            combined.to_csv(
                feat_dir / "combined_features.csv", index=False)

        source_dirs = sorted(tmp_path.glob("M3_run_*"))
        s = DataSplitter(split_mode="two_way", verbose=False)
        feature_dfs, combined_df, user_demo_df = s.load_from_m3_outputs(
            source_dirs)

        assert len(feature_dfs) == 5
        assert len(combined_df) == 15  # 3 users × 5 rows
        assert len(user_demo_df) == 3

    def test_load_empty_folder(self, tmp_path):
        """Loading from empty folder should return empty structures."""
        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()

        s = DataSplitter(verbose=False)
        feature_dfs, combined_df, user_demo_df = s.load_from_m3_outputs(
            [empty_dir])

        assert len(feature_dfs) == 0
        assert len(combined_df) == 0
