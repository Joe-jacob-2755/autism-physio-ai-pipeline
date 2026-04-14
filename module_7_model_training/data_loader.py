"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  data_loader.py
=============================================================================
Load M6 feature-engineering output, clean, split, and prepare data for
all model families:
  - TabularData  : sklearn classifiers (DT, RF, SVM, XGBoost)
  - SequenceData : temporal models (BiLSTM, 1D-CNN, TFT)
  - TPPData      : Temporal Point Process
  - unsupervised : raw feature matrix without labels
=============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    META_COLS, TARGET_COL, SPLIT_COL, DEMOGRAPHIC_RAW, DEMOGRAPHIC_ENC,
    ALL_NON_FEATURE_COLS, KNOWN_NAN_COLS, TARGET_CLASSES,
    SEQUENCE_LENGTH, SEQUENCE_STRIDE, RANDOM_SEED,
    FORECAST_LOOKBACK, FORECAST_HORIZON, FORECAST_STRIDE,
)


# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class DataBundle:
    """Master container returned by DataLoader.load()."""
    df: pd.DataFrame
    feature_cols: List[str]
    meta_cols: List[str]
    demographic_cols: List[str]
    target_col: str
    label_encoder: LabelEncoder
    class_names: List[str]
    n_classes: int
    class_counts: Dict[str, int]
    split_counts: Dict[str, int]


@dataclass
class TabularData:
    """For sklearn classifiers — flat feature matrices."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    class_names: List[str]
    n_classes: int
    demographics_train: Optional[pd.DataFrame] = None
    demographics_val: Optional[pd.DataFrame] = None
    demographics_test: Optional[pd.DataFrame] = None


@dataclass
class SequenceData:
    """For temporal models — 3D tensors of window sequences."""
    X_train: np.ndarray   # (N, seq_len, n_features)
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray   # label of LAST window in each sequence
    y_val: np.ndarray
    y_test: np.ndarray
    seq_len: int
    n_features: int
    feature_names: List[str]
    class_names: List[str]
    n_classes: int


@dataclass
class ForecastData:
    """For time series forecasting -- predict future emotional state.

    X: (N, lookback, n_features) -- history windows
    y: (N,) -- label of the window FORECAST_HORIZON steps ahead
    """
    X_train: np.ndarray   # (N, lookback, n_features)
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray   # label HORIZON windows ahead of last history window
    y_val: np.ndarray
    y_test: np.ndarray
    lookback: int         # number of history windows (e.g. 6 = 3 min)
    horizon: int          # prediction gap in windows (e.g. 2 = 1 min ahead)
    n_features: int
    feature_names: List[str]
    class_names: List[str]
    n_classes: int


@dataclass
class TFTData:
    """For Temporal Fusion Transformer — static + temporal separation."""
    static_train: np.ndarray   # (N, n_static)
    static_val: np.ndarray
    static_test: np.ndarray
    temporal_train: np.ndarray  # (N, seq_len, n_temporal)
    temporal_val: np.ndarray
    temporal_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    static_feature_names: List[str]
    temporal_feature_names: List[str]
    class_names: List[str]
    n_classes: int
    seq_len: int


@dataclass
class TPPData:
    """For Temporal Point Process — event times and types."""
    event_times: List[np.ndarray]        # per-sequence event timestamps
    event_types: List[np.ndarray]        # per-sequence event type indices
    inter_event_times: List[np.ndarray]  # time gaps between events
    n_event_types: int
    event_type_names: List[str]
    n_sequences: int
    sufficient: bool = True  # False if not enough events to train


# ── Data Loader ──────────────────────────────────────────────────────────

class DataLoader:
    """
    Load M6 feature-engineering output and prepare data for model training.

    Parameters
    ----------
    source : path to CSV or directory containing M6 output
    seed   : random seed for reproducibility
    """

    def __init__(self, source: str | Path, seed: int = RANDOM_SEED,
                 verbose: bool = True):
        self.source = Path(source)
        self.seed = seed
        self.verbose = verbose
        self._bundle: Optional[DataBundle] = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [DataLoader] {msg}")

    # ── Load & clean ─────────────────────────────────────────────────────

    def load(self) -> DataBundle:
        """Load CSV, identify columns, clean data, return DataBundle."""
        # Resolve source to a CSV file
        csv_path = self._resolve_csv()
        self._log(f"Loading {csv_path.name} ...")

        df = pd.read_csv(csv_path)
        self._log(f"  Raw shape: {df.shape[0]} rows x {df.shape[1]} cols")

        # Drop known all-NaN columns
        nan_cols_present = [c for c in KNOWN_NAN_COLS if c in df.columns]
        if nan_cols_present:
            df = df.drop(columns=nan_cols_present)
            self._log(f"  Dropped {len(nan_cols_present)} all-NaN cols: "
                       f"{nan_cols_present}")

        # Identify column types
        meta_cols = [c for c in df.columns if c in META_COLS]
        demo_cols = [c for c in df.columns
                     if c in DEMOGRAPHIC_RAW | DEMOGRAPHIC_ENC]
        feature_cols = [c for c in df.columns
                        if c not in ALL_NON_FEATURE_COLS
                        and c not in DEMOGRAPHIC_ENC
                        and c not in KNOWN_NAN_COLS]
        self._log(f"  Features: {len(feature_cols)} | Demographics: "
                   f"{len(demo_cols)} | Meta: {len(meta_cols)}")

        # Impute remaining NaNs with column median (train-set safe)
        nan_counts = df[feature_cols].isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            for col in nan_cols.index:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            self._log(f"  Imputed NaN in {len(nan_cols)} cols: "
                       f"{list(nan_cols.index)}")

        # Check for remaining issues
        inf_mask = df[feature_cols].abs() == float("inf")
        n_inf = inf_mask.sum().sum()
        if n_inf > 0:
            df[feature_cols] = df[feature_cols].replace(
                [float("inf"), float("-inf")], np.nan)
            for col in feature_cols:
                df[col] = df[col].fillna(df[col].median())
            self._log(f"  Replaced {n_inf} inf values with median")

        # Encode target labels
        le = LabelEncoder()
        # Fit on all known classes so encoding is consistent
        present_classes = sorted(df[TARGET_COL].unique())
        le.fit(present_classes)
        class_names = list(le.classes_)
        n_classes = len(class_names)

        class_counts = df[TARGET_COL].value_counts().to_dict()
        split_counts = (df[SPLIT_COL].value_counts().to_dict()
                        if SPLIT_COL in df.columns else {})

        self._log(f"  Classes ({n_classes}): {class_counts}")
        if split_counts:
            self._log(f"  Split: {split_counts}")

        bundle = DataBundle(
            df=df,
            feature_cols=feature_cols,
            meta_cols=meta_cols,
            demographic_cols=demo_cols,
            target_col=TARGET_COL,
            label_encoder=le,
            class_names=class_names,
            n_classes=n_classes,
            class_counts=class_counts,
            split_counts=split_counts,
        )
        self._bundle = bundle
        return bundle

    # ── Tabular data ─────────────────────────────────────────────────────

    def get_tabular_data(self, bundle: Optional[DataBundle] = None) -> TabularData:
        """Return train/val/test split as flat numpy arrays."""
        b = bundle or self._bundle
        if b is None:
            b = self.load()

        df = b.df
        feat = b.feature_cols
        le = b.label_encoder

        if SPLIT_COL in df.columns:
            train_df = df[df[SPLIT_COL] == "train"]
            val_df = df[df[SPLIT_COL] == "val"]
            test_df = df[df[SPLIT_COL] == "test"]
        else:
            # No split column — create stratified split
            train_df, val_df, test_df = self._stratified_split(df)

        def extract(subset):
            X = subset[feat].values.astype(np.float32)
            y = le.transform(subset[TARGET_COL].values)
            demo_cols_present = [c for c in b.demographic_cols
                                 if c in subset.columns]
            demo = subset[demo_cols_present].reset_index(drop=True)
            return X, y, demo

        X_tr, y_tr, d_tr = extract(train_df)
        X_va, y_va, d_va = extract(val_df)
        X_te, y_te, d_te = extract(test_df)

        self._log(f"  Tabular: train={len(X_tr)}, val={len(X_va)}, "
                   f"test={len(X_te)}")

        return TabularData(
            X_train=X_tr, X_val=X_va, X_test=X_te,
            y_train=y_tr, y_val=y_va, y_test=y_te,
            feature_names=feat,
            class_names=b.class_names,
            n_classes=b.n_classes,
            demographics_train=d_tr,
            demographics_val=d_va,
            demographics_test=d_te,
        )

    # ── Sequence data ────────────────────────────────────────────────────

    def get_sequence_data(
        self,
        bundle: Optional[DataBundle] = None,
        seq_len: int = SEQUENCE_LENGTH,
        stride: int = SEQUENCE_STRIDE,
    ) -> SequenceData:
        """
        Form sequences of consecutive windows for temporal models.

        Groups by (user_id, session_id), orders by window_start_s,
        applies sliding window of length `seq_len` with `stride`.
        Label is the target of the LAST window in each sequence.
        """
        b = bundle or self._bundle
        if b is None:
            b = self.load()

        df = b.df.copy()
        feat = b.feature_cols
        le = b.label_encoder

        # Form sequences per user/session group
        sequences = []
        labels = []
        splits = []

        group_cols = []
        if "user_id" in df.columns:
            group_cols.append("user_id")
        if "session_id" in df.columns:
            group_cols.append("session_id")

        if not group_cols:
            groups = [("all", df)]
        else:
            groups = df.groupby(group_cols)

        for _, group_df in groups:
            group_df = group_df.sort_values("window_start_s").reset_index(
                drop=True)
            X_group = group_df[feat].values.astype(np.float32)
            y_group = le.transform(group_df[TARGET_COL].values)
            split_group = (group_df[SPLIT_COL].values
                           if SPLIT_COL in group_df.columns else None)

            for i in range(0, len(X_group) - seq_len + 1, stride):
                seq = X_group[i:i + seq_len]
                lbl = y_group[i + seq_len - 1]  # Last window's label
                sequences.append(seq)
                labels.append(lbl)
                if split_group is not None:
                    splits.append(split_group[i + seq_len - 1])

        X_all = np.array(sequences, dtype=np.float32)
        y_all = np.array(labels, dtype=np.int64)

        # Split
        if splits:
            splits = np.array(splits)
            tr = splits == "train"
            va = splits == "val"
            te = splits == "test"
        else:
            n = len(X_all)
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(n)
            n_tr = int(0.6 * n)
            n_va = int(0.2 * n)
            tr = np.zeros(n, dtype=bool)
            va = np.zeros(n, dtype=bool)
            te = np.zeros(n, dtype=bool)
            tr[idx[:n_tr]] = True
            va[idx[n_tr:n_tr + n_va]] = True
            te[idx[n_tr + n_va:]] = True

        self._log(f"  Sequences: {len(X_all)} total (seq_len={seq_len}, "
                   f"stride={stride})")
        self._log(f"    train={tr.sum()}, val={va.sum()}, test={te.sum()}")

        return SequenceData(
            X_train=X_all[tr], X_val=X_all[va], X_test=X_all[te],
            y_train=y_all[tr], y_val=y_all[va], y_test=y_all[te],
            seq_len=seq_len,
            n_features=X_all.shape[2] if X_all.ndim == 3 else 0,
            feature_names=feat,
            class_names=b.class_names,
            n_classes=b.n_classes,
        )

    # ── Forecast data ───────────────────────────────────────────────────

    def get_forecast_data(
        self,
        bundle: Optional[DataBundle] = None,
        lookback: int = FORECAST_LOOKBACK,
        horizon: int = FORECAST_HORIZON,
        stride: int = FORECAST_STRIDE,
    ) -> ForecastData:
        """
        Form forecast sequences: use `lookback` history windows to predict
        the emotional state `horizon` windows into the future.

        Parameters
        ----------
        lookback : number of past windows as input (6 = 3 min with 30s stride)
        horizon  : gap in windows to the prediction target (2 = 1 min ahead)
        stride   : sliding window step (1 = every 30 s)

        Label comes from window at index [i + lookback - 1 + horizon],
        i.e. `horizon` windows beyond the end of the history window.
        """
        b = bundle or self._bundle
        if b is None:
            b = self.load()

        df = b.df.copy()
        feat = b.feature_cols
        le = b.label_encoder

        sequences = []
        labels = []
        splits = []

        group_cols = [c for c in ["user_id", "session_id"]
                      if c in df.columns]
        groups = df.groupby(group_cols) if group_cols else [("all", df)]

        for _, group_df in groups:
            group_df = group_df.sort_values("window_start_s").reset_index(
                drop=True)
            X_group = group_df[feat].values.astype(np.float32)
            y_group = le.transform(group_df[TARGET_COL].values)
            split_group = (group_df[SPLIT_COL].values
                           if SPLIT_COL in group_df.columns else None)

            # Need at least lookback + horizon windows in this group
            total_needed = lookback + horizon
            if len(X_group) < total_needed:
                continue

            for i in range(0, len(X_group) - total_needed + 1, stride):
                # History: windows [i : i + lookback]
                seq = X_group[i:i + lookback]
                # Future label: window at [i + lookback - 1 + horizon]
                future_idx = i + lookback - 1 + horizon
                lbl = y_group[future_idx]
                sequences.append(seq)
                labels.append(lbl)
                if split_group is not None:
                    # Use the split of the HISTORY end (not future)
                    splits.append(split_group[i + lookback - 1])

        if len(sequences) == 0:
            self._log("  Forecast: 0 sequences -- insufficient data")
            return ForecastData(
                X_train=np.empty((0, lookback, 0), dtype=np.float32),
                X_val=np.empty((0, lookback, 0), dtype=np.float32),
                X_test=np.empty((0, lookback, 0), dtype=np.float32),
                y_train=np.empty(0, dtype=np.int64),
                y_val=np.empty(0, dtype=np.int64),
                y_test=np.empty(0, dtype=np.int64),
                lookback=lookback, horizon=horizon,
                n_features=len(feat), feature_names=feat,
                class_names=b.class_names, n_classes=b.n_classes,
            )

        X_all = np.array(sequences, dtype=np.float32)
        y_all = np.array(labels, dtype=np.int64)

        if splits:
            sp = np.array(splits)
            tr, va, te = sp == "train", sp == "val", sp == "test"
        else:
            n = len(X_all)
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(n)
            n_tr, n_va = int(0.6 * n), int(0.2 * n)
            tr = np.zeros(n, dtype=bool)
            va = np.zeros(n, dtype=bool)
            te = np.zeros(n, dtype=bool)
            tr[idx[:n_tr]] = True
            va[idx[n_tr:n_tr + n_va]] = True
            te[idx[n_tr + n_va:]] = True

        self._log(f"  Forecast: {len(X_all)} total (lookback={lookback}, "
                   f"horizon={horizon}, stride={stride})")
        self._log(f"    train={tr.sum()}, val={va.sum()}, test={te.sum()}")

        return ForecastData(
            X_train=X_all[tr], X_val=X_all[va], X_test=X_all[te],
            y_train=y_all[tr], y_val=y_all[va], y_test=y_all[te],
            lookback=lookback, horizon=horizon,
            n_features=X_all.shape[2] if X_all.ndim == 3 else 0,
            feature_names=feat,
            class_names=b.class_names,
            n_classes=b.n_classes,
        )

    # ── TFT data ─────────────────────────────────────────────────────────

    def get_tft_data(
        self,
        bundle: Optional[DataBundle] = None,
        seq_len: int = SEQUENCE_LENGTH,
        stride: int = SEQUENCE_STRIDE,
    ) -> TFTData:
        """
        Structure data for Temporal Fusion Transformer.

        Static features  = encoded demographics (constant per user)
        Temporal features = signal features (vary per window)
        """
        b = bundle or self._bundle
        if b is None:
            b = self.load()

        df = b.df.copy()
        le = b.label_encoder

        # Separate static (demographics) and temporal (signal features)
        enc_demo_cols = sorted(c for c in DEMOGRAPHIC_ENC if c in df.columns)
        # Include age as static too
        static_cols = enc_demo_cols + (["age"] if "age" in df.columns else [])
        temporal_cols = b.feature_cols  # All signal features are temporal

        # Form sequences (same grouping logic)
        static_seqs, temporal_seqs, labels, splits = [], [], [], []

        group_cols = [c for c in ["user_id", "session_id"]
                      if c in df.columns]
        groups = df.groupby(group_cols) if group_cols else [("all", df)]

        for _, group_df in groups:
            group_df = group_df.sort_values("window_start_s").reset_index(
                drop=True)
            T = group_df[temporal_cols].values.astype(np.float32)
            S = group_df[static_cols].values.astype(np.float32)
            y = le.transform(group_df[TARGET_COL].values)
            sp = (group_df[SPLIT_COL].values
                  if SPLIT_COL in group_df.columns else None)

            for i in range(0, len(T) - seq_len + 1, stride):
                temporal_seqs.append(T[i:i + seq_len])
                static_seqs.append(S[i])  # Static is same across sequence
                labels.append(y[i + seq_len - 1])
                if sp is not None:
                    splits.append(sp[i + seq_len - 1])

        T_all = np.array(temporal_seqs, dtype=np.float32)
        S_all = np.array(static_seqs, dtype=np.float32)
        y_all = np.array(labels, dtype=np.int64)

        if splits:
            sp_arr = np.array(splits)
            tr, va, te = sp_arr == "train", sp_arr == "val", sp_arr == "test"
        else:
            n = len(y_all)
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(n)
            n_tr, n_va = int(0.6 * n), int(0.2 * n)
            tr = np.zeros(n, dtype=bool)
            va = np.zeros(n, dtype=bool)
            te = np.zeros(n, dtype=bool)
            tr[idx[:n_tr]] = True
            va[idx[n_tr:n_tr + n_va]] = True
            te[idx[n_tr + n_va:]] = True

        self._log(f"  TFT data: {len(y_all)} sequences, "
                   f"static={len(static_cols)}, temporal={len(temporal_cols)}")

        return TFTData(
            static_train=S_all[tr], static_val=S_all[va],
            static_test=S_all[te],
            temporal_train=T_all[tr], temporal_val=T_all[va],
            temporal_test=T_all[te],
            y_train=y_all[tr], y_val=y_all[va], y_test=y_all[te],
            static_feature_names=static_cols,
            temporal_feature_names=temporal_cols,
            class_names=b.class_names,
            n_classes=b.n_classes,
            seq_len=seq_len,
        )

    # ── TPP data ─────────────────────────────────────────────────────────

    def get_tpp_data(
        self, bundle: Optional[DataBundle] = None,
    ) -> TPPData:
        """
        Extract event data for Temporal Point Process.

        Events = transitions from baseline to emotion states.
        """
        b = bundle or self._bundle
        if b is None:
            b = self.load()

        df = b.df.copy()
        le = b.label_encoder

        event_times_all = []
        event_types_all = []
        inter_event_all = []

        group_cols = [c for c in ["user_id", "session_id"]
                      if c in df.columns]
        groups = df.groupby(group_cols) if group_cols else [("all", df)]

        for _, group_df in groups:
            group_df = group_df.sort_values("window_start_s").reset_index(
                drop=True)
            labels = group_df[TARGET_COL].values
            times = group_df["window_start_s"].values.astype(float)

            # Find non-baseline events
            event_mask = labels != "baseline"
            if event_mask.sum() == 0:
                continue

            ev_times = times[event_mask]
            ev_types = le.transform(labels[event_mask])

            event_times_all.append(ev_times)
            event_types_all.append(ev_types)
            if len(ev_times) > 1:
                inter_event_all.append(np.diff(ev_times))
            else:
                inter_event_all.append(np.array([0.0]))

        n_events_total = sum(len(t) for t in event_times_all)
        event_type_names = b.class_names
        sufficient = n_events_total >= 10

        if not sufficient:
            self._log(f"  TPP: only {n_events_total} events -- insufficient "
                       f"(need >= 10)")
        else:
            self._log(f"  TPP: {n_events_total} events across "
                       f"{len(event_times_all)} sequences")

        return TPPData(
            event_times=event_times_all,
            event_types=event_types_all,
            inter_event_times=inter_event_all,
            n_event_types=b.n_classes,
            event_type_names=event_type_names,
            n_sequences=len(event_times_all),
            sufficient=sufficient,
        )

    # ── Unsupervised data ────────────────────────────────────────────────

    def get_unsupervised_data(
        self, bundle: Optional[DataBundle] = None,
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """Return (X_all_features, y_true_labels, feature_names)."""
        b = bundle or self._bundle
        if b is None:
            b = self.load()
        X = b.df[b.feature_cols].values.astype(np.float32)
        y = b.label_encoder.transform(b.df[TARGET_COL].values)
        self._log(f"  Unsupervised: {X.shape[0]} samples x "
                   f"{X.shape[1]} features")
        return X, y, b.feature_cols

    # ── Summary ──────────────────────────────────────────────────────────

    def describe(self, bundle: Optional[DataBundle] = None) -> dict:
        """Return a summary dict of the loaded data."""
        b = bundle or self._bundle
        if b is None:
            b = self.load()
        return {
            "n_samples": len(b.df),
            "n_features": len(b.feature_cols),
            "n_classes": b.n_classes,
            "class_names": b.class_names,
            "class_counts": b.class_counts,
            "split_counts": b.split_counts,
            "feature_names": b.feature_cols,
            "demographic_cols": b.demographic_cols,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_csv(self) -> Path:
        """Find the combined features CSV from source path."""
        p = self.source
        if p.is_file() and p.suffix == ".csv":
            return p
        # Directory — look for combined features CSV
        candidates = [
            p / "combined_features_all_users_split.csv",
            p / "combined_features_all_users.csv",
            p / "combined_features.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        # Search recursively
        csvs = list(p.glob("**/combined_features*.csv"))
        if csvs:
            return csvs[0]
        raise FileNotFoundError(
            f"No combined features CSV found in {p}")

    def _stratified_split(self, df, train_ratio=0.6, val_ratio=0.2):
        """Stratified train/val/test split when no split column exists."""
        rng = np.random.default_rng(self.seed)
        train_dfs, val_dfs, test_dfs = [], [], []

        for label, group in df.groupby(TARGET_COL):
            idx = rng.permutation(len(group))
            n_tr = max(1, int(train_ratio * len(group)))
            n_va = max(0, int(val_ratio * len(group)))
            train_dfs.append(group.iloc[idx[:n_tr]])
            val_dfs.append(group.iloc[idx[n_tr:n_tr + n_va]])
            test_dfs.append(group.iloc[idx[n_tr + n_va:]])

        return (pd.concat(train_dfs), pd.concat(val_dfs),
                pd.concat(test_dfs))
