"""
=============================================================================
MODULE 3 – DATA SPLITTING  |  splitter.py
=============================================================================
User-level, stratified data splitting for physiological signal pipelines.

This module runs IMMEDIATELY after data acquisition (M1 + M2A), splitting
raw signals BEFORE any preprocessing or feature engineering.  Test data is
kept untouched until final model evaluation.

Why user-level splitting?
-------------------------
Physiological signals from the same participant share:
  - Baseline physiology (EDA 0.3-18 uS range per individual)
  - Severity-modulated reactivity (1.0x-1.6x)
  - Temporal autocorrelation within sessions

Splitting at the window level would leak participant identity into
train/test sets. Splitting at the user_id level ensures the model
is evaluated on truly unseen participants.

Why split before preprocessing?
-------------------------------
If preprocessing (scaling, filtering) is fitted on all data, information
from test participants leaks into the training pipeline via fitted
parameters (scaler statistics, filter state).  By splitting first, all
fitting operations downstream use training data ONLY.

Why stratified?
---------------
Autism severity (Low/Medium/Severe) and verbal status (Verbal/Minimally
verbal/Non-verbal) modulate physiological reactivity. Without
stratification, the test set could over-represent one subgroup and
produce misleading evaluation metrics.

=============================================================================
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    DEFAULT_SPLIT_MODE, DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO,
    DEFAULT_TEST_RATIO, DEFAULT_TRAIN_RATIO_2WAY, DEFAULT_TEST_RATIO_2WAY,
    STRATIFY_COLUMNS, MIN_USERS_PER_STRATUM, DEFAULT_SEED,
    USER_COL, SESSION_COL, TARGET_COL, DEMOGRAPHIC_COLS,
    SIGNAL_FILES, COMBINED_FILE, FEATURE_SUBDIR, FEATURE_NORM_SUBDIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SplitResult:
    """Result of a data split operation."""
    split_mode: str                          # "two_way" | "three_way"
    train_users: List[str]
    val_users: List[str]                     # empty list if two_way
    test_users: List[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    stratified: bool
    stratify_columns: List[str]
    seed: int
    n_total_users: int
    # Per-split data (populated by DataSplitter.split_data)
    train_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    val_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    test_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    # Demographics per split
    train_demographics: List[dict] = field(default_factory=list)
    val_demographics: List[dict] = field(default_factory=list)
    test_demographics: List[dict] = field(default_factory=list)

    def summary(self) -> dict:
        """Return a summary dict (without DataFrames) for metadata."""
        return {
            "split_mode": self.split_mode,
            "train_users": self.train_users,
            "val_users": self.val_users,
            "test_users": self.test_users,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "stratified": self.stratified,
            "stratify_columns": self.stratify_columns,
            "seed": self.seed,
            "n_total_users": self.n_total_users,
            "n_train": len(self.train_users),
            "n_val": len(self.val_users),
            "n_test": len(self.test_users),
            "actual_train_ratio": round(
                len(self.train_users) / max(self.n_total_users, 1), 4),
            "actual_val_ratio": round(
                len(self.val_users) / max(self.n_total_users, 1), 4),
            "actual_test_ratio": round(
                len(self.test_users) / max(self.n_total_users, 1), 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DATA SPLITTER
# ─────────────────────────────────────────────────────────────────────────────

class DataSplitter:
    """
    Split physiological data by user into train/val/test sets.

    Parameters
    ----------
    split_mode     : 'two_way' (train/test) or 'three_way' (train/val/test)
    train_ratio    : fraction of users for training
    val_ratio      : fraction of users for validation (ignored in two_way)
    test_ratio     : fraction of users for testing
    stratify       : whether to stratify by demographics
    stratify_cols  : columns to stratify on
    seed           : random seed for reproducibility
    verbose        : print progress
    """

    def __init__(
        self,
        split_mode: str = DEFAULT_SPLIT_MODE,
        train_ratio: float = None,
        val_ratio: float = None,
        test_ratio: float = None,
        stratify: bool = True,
        stratify_cols: List[str] = None,
        seed: int = DEFAULT_SEED,
        verbose: bool = True,
    ):
        self.split_mode = split_mode
        self.stratify = stratify
        self.stratify_cols = stratify_cols or STRATIFY_COLUMNS
        self.seed = seed
        self.verbose = verbose

        # Set ratios based on mode
        if split_mode == "two_way":
            self.train_ratio = train_ratio or DEFAULT_TRAIN_RATIO_2WAY
            self.val_ratio = 0.0
            self.test_ratio = test_ratio or DEFAULT_TEST_RATIO_2WAY
        else:
            self.train_ratio = train_ratio or DEFAULT_TRAIN_RATIO
            self.val_ratio = val_ratio or DEFAULT_VAL_RATIO
            self.test_ratio = test_ratio or DEFAULT_TEST_RATIO

        # Validate ratios
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.3f} "
                f"(train={self.train_ratio}, val={self.val_ratio}, "
                f"test={self.test_ratio})"
            )

    # ── Public API ─────────────────────────────────────────────────────────

    def split_users(
        self,
        user_demographics: pd.DataFrame,
    ) -> SplitResult:
        """
        Assign users to train/val/test splits.

        Parameters
        ----------
        user_demographics : DataFrame with one row per user, columns:
                            user_id, autism_severity, verbal_status, ...

        Returns
        -------
        SplitResult with user assignments (no data yet)
        """
        if USER_COL not in user_demographics.columns:
            raise ValueError(
                f"user_demographics must contain '{USER_COL}' column"
            )

        users = user_demographics[USER_COL].unique().tolist()
        n = len(users)
        self._log(f"[Splitter] {n} users to split "
                  f"({self.split_mode}: "
                  f"train={self.train_ratio:.0%} "
                  f"val={self.val_ratio:.0%} "
                  f"test={self.test_ratio:.0%})")

        if n < 2:
            raise ValueError(
                f"Need at least 2 users to split, got {n}. "
                "Generate more simulated users or import more data."
            )

        # Attempt stratified split
        stratified = False
        if self.stratify and n >= 4:
            try:
                train_u, val_u, test_u = self._stratified_split(
                    user_demographics)
                stratified = True
                self._log(f"[Splitter] Stratified split on "
                          f"{self.stratify_cols}")
            except _StratificationError as e:
                self._log(f"[Splitter] Stratification failed: {e}")
                self._log("[Splitter] Falling back to random split.")
                train_u, val_u, test_u = self._random_split(users)
        else:
            train_u, val_u, test_u = self._random_split(users)

        # Build demographics per split
        demo_df = user_demographics.set_index(USER_COL)
        train_demo = [demo_df.loc[u].to_dict() for u in train_u
                      if u in demo_df.index]
        val_demo = [demo_df.loc[u].to_dict() for u in val_u
                    if u in demo_df.index]
        test_demo = [demo_df.loc[u].to_dict() for u in test_u
                     if u in demo_df.index]

        result = SplitResult(
            split_mode=self.split_mode,
            train_users=sorted(train_u),
            val_users=sorted(val_u),
            test_users=sorted(test_u),
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            stratified=stratified,
            stratify_columns=self.stratify_cols if stratified else [],
            seed=self.seed,
            n_total_users=n,
            train_demographics=train_demo,
            val_demographics=val_demo,
            test_demographics=test_demo,
        )

        self._log(f"[Splitter] Train: {len(train_u)} users {train_u}")
        if val_u:
            self._log(f"[Splitter] Val:   {len(val_u)} users {val_u}")
        self._log(f"[Splitter] Test:  {len(test_u)} users {test_u}")

        return result

    def split_data(
        self,
        split_result: SplitResult,
        feature_dfs: Dict[str, pd.DataFrame],
        combined_df: pd.DataFrame = None,
    ) -> SplitResult:
        """
        Partition feature DataFrames by the user assignments in split_result.

        Parameters
        ----------
        split_result : SplitResult from split_users()
        feature_dfs  : {signal_name: DataFrame} — per-signal feature CSVs
        combined_df  : optional combined features DataFrame

        Returns
        -------
        SplitResult with train_data, val_data, test_data populated
        """
        for sig_name, df in feature_dfs.items():
            if USER_COL not in df.columns:
                self._log(f"  WARNING: {sig_name} has no '{USER_COL}' column, "
                          "including all rows in train.")
                split_result.train_data[sig_name] = df
                continue

            split_result.train_data[sig_name] = df[
                df[USER_COL].isin(split_result.train_users)
            ].copy()
            split_result.test_data[sig_name] = df[
                df[USER_COL].isin(split_result.test_users)
            ].copy()
            if split_result.val_users:
                split_result.val_data[sig_name] = df[
                    df[USER_COL].isin(split_result.val_users)
                ].copy()

        # Combined DataFrame
        if combined_df is not None and USER_COL in combined_df.columns:
            split_result.train_data["combined"] = combined_df[
                combined_df[USER_COL].isin(split_result.train_users)
            ].copy()
            split_result.test_data["combined"] = combined_df[
                combined_df[USER_COL].isin(split_result.test_users)
            ].copy()
            if split_result.val_users:
                split_result.val_data["combined"] = combined_df[
                    combined_df[USER_COL].isin(split_result.val_users)
                ].copy()

        # Log split sizes
        for label, data in [("Train", split_result.train_data),
                            ("Val", split_result.val_data),
                            ("Test", split_result.test_data)]:
            if data:
                total_rows = sum(len(df) for df in data.values())
                self._log(f"  {label}: {total_rows:,} total rows "
                          f"across {len(data)} DataFrames")

        return split_result

    # ── Splitting strategies ──────────────────────────────────────────────

    def _random_split(
        self, users: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Simple random split of users."""
        rng = np.random.default_rng(self.seed)
        shuffled = list(users)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, round(n * self.train_ratio))
        n_test = max(1, round(n * self.test_ratio))
        n_val = n - n_train - n_test

        # Ensure at least 1 in test, adjust train if needed
        if n_val < 0:
            n_val = 0
            n_train = n - n_test

        train_u = shuffled[:n_train]
        val_u = shuffled[n_train:n_train + n_val] if n_val > 0 else []
        test_u = shuffled[n_train + n_val:]

        return train_u, val_u, test_u

    def _stratified_split(
        self,
        user_demographics: pd.DataFrame,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Stratified split: ensures each split has representative demographics.

        Strategy:
          1. Create a composite stratum key from stratify_cols
          2. Within each stratum, randomly assign users to train/val/test
          3. If a stratum has too few users, assign all to train
        """
        rng = np.random.default_rng(self.seed)
        df = user_demographics.copy()

        # Build composite stratum key
        available_cols = [c for c in self.stratify_cols if c in df.columns]
        if not available_cols:
            raise _StratificationError(
                f"None of {self.stratify_cols} found in demographics"
            )

        df["_stratum"] = df[available_cols].astype(str).agg("_".join, axis=1)
        strata = df.groupby("_stratum")[USER_COL].apply(list).to_dict()

        train_all, val_all, test_all = [], [], []

        for stratum_key, stratum_users in strata.items():
            n_s = len(stratum_users)
            shuffled = list(stratum_users)
            rng.shuffle(shuffled)

            if n_s < MIN_USERS_PER_STRATUM:
                # Too few users — put all in train
                train_all.extend(shuffled)
                self._log(f"  Stratum '{stratum_key}': {n_s} users "
                          f"(< {MIN_USERS_PER_STRATUM}) -> all to train")
                continue

            n_train = max(1, round(n_s * self.train_ratio))
            n_test = max(1, round(n_s * self.test_ratio))
            n_val = n_s - n_train - n_test
            if n_val < 0:
                n_val = 0
                n_train = n_s - n_test

            train_all.extend(shuffled[:n_train])
            if n_val > 0:
                val_all.extend(shuffled[n_train:n_train + n_val])
            test_all.extend(shuffled[n_train + n_val:])

        if not test_all:
            raise _StratificationError(
                "Stratification produced empty test set. "
                "Need more users or fewer strata."
            )

        return train_all, val_all, test_all

    # ── Data loading ──────────────────────────────────────────────────────

    def load_from_m3_outputs(
        self,
        source_dirs: List[Path],
        metadata_list: List[dict] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        """
        Load and aggregate feature CSVs from multiple M3 output folders.

        Parameters
        ----------
        source_dirs    : list of M3 run folders (one per user)
        metadata_list  : optional list of metadata dicts (one per user)

        Returns
        -------
        feature_dfs   : {signal_name: concatenated DataFrame}
        combined_df   : concatenated combined features
        user_demo_df  : one row per user with demographics
        """
        all_features = {}
        all_combined = []
        user_rows = []

        for i, src in enumerate(source_dirs):
            src = Path(src)
            feat_dir = src / FEATURE_SUBDIR
            if not feat_dir.exists():
                feat_dir = src  # flat layout

            # Load per-signal features
            for sig_file in ["EDA_features.csv", "BVP_features.csv",
                             "IBI_features.csv", "ST_features.csv",
                             "ACC_features.csv"]:
                fpath = feat_dir / sig_file
                if fpath.exists():
                    sig_name = sig_file.replace("_features.csv", "")
                    df = pd.read_csv(fpath)
                    if sig_name not in all_features:
                        all_features[sig_name] = []
                    all_features[sig_name].append(df)

            # Load combined
            combo_path = feat_dir / "combined_features.csv"
            if combo_path.exists():
                all_combined.append(pd.read_csv(combo_path))

            # Extract user demographics from first row or metadata
            demo = {}
            if metadata_list and i < len(metadata_list):
                meta = metadata_list[i]
                demo = meta.get("demographics", {})
                demo[USER_COL] = meta.get("user_id", f"user_{i+1:03d}")
            elif all_combined:
                row = all_combined[-1].iloc[0]
                demo[USER_COL] = row.get(USER_COL, f"user_{i+1:03d}")
                for col in DEMOGRAPHIC_COLS:
                    if col in row.index:
                        demo[col] = row[col]

            if demo:
                user_rows.append(demo)

        # Concatenate
        feature_dfs = {
            sig: pd.concat(dfs, ignore_index=True)
            for sig, dfs in all_features.items()
        }
        combined_df = (pd.concat(all_combined, ignore_index=True)
                       if all_combined else pd.DataFrame())
        user_demo_df = pd.DataFrame(user_rows) if user_rows else pd.DataFrame()

        self._log(f"[Splitter] Loaded {len(source_dirs)} M3 runs: "
                  f"{len(feature_dfs)} signals, "
                  f"{len(combined_df)} combined rows, "
                  f"{len(user_demo_df)} users")

        return feature_dfs, combined_df, user_demo_df

    # ── Raw-signal loading (new pipeline: split before preprocessing) ───

    def load_from_packets(
        self,
        packets: list,
    ) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame]:
        """
        Extract raw signals and demographics from PipelinePackets.

        Parameters
        ----------
        packets : list of PipelinePacket objects from M1

        Returns
        -------
        signals_by_user : {user_id: {signal_name: DataFrame}}
        user_demo_df    : one row per user with demographics
        """
        signals_by_user: Dict[str, Dict[str, pd.DataFrame]] = {}
        user_rows = []

        for i, pkt in enumerate(packets):
            uid = getattr(pkt, "user_id", f"user_{i+1:03d}")
            signals_by_user[uid] = {}

            # Extract signals from packet
            if hasattr(pkt, "signals") and pkt.signals:
                for sig_name, df in pkt.signals.items():
                    if df is not None and len(df) > 0:
                        signals_by_user[uid][sig_name] = df

            # Extract demographics from packet metadata
            demo = {USER_COL: uid}
            if hasattr(pkt, "metadata") and pkt.metadata:
                meta_user = pkt.metadata.get("user", {})
                for col in DEMOGRAPHIC_COLS:
                    if col in meta_user and meta_user[col] is not None:
                        demo[col] = meta_user[col]
            user_rows.append(demo)

        user_demo_df = pd.DataFrame(user_rows)
        n_sigs = sum(len(s) for s in signals_by_user.values())
        self._log(f"[M3 Splitter] Loaded {len(packets)} packets: "
                  f"{n_sigs} total signal DataFrames, "
                  f"{len(user_demo_df)} users")

        return signals_by_user, user_demo_df

    def load_from_acquisition_outputs(
        self,
        source_dirs: List[Path],
        metadata_list: List[dict] = None,
    ) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame]:
        """
        Load raw signal CSVs from M1/M2A output folders (one per user).

        Parameters
        ----------
        source_dirs    : list of M1 output folders (one per user)
        metadata_list  : optional list of metadata dicts (one per user)

        Returns
        -------
        signals_by_user : {user_id: {signal_name: DataFrame}}
        user_demo_df    : one row per user with demographics
        """
        signals_by_user: Dict[str, Dict[str, pd.DataFrame]] = {}
        user_rows = []

        for i, src in enumerate(source_dirs):
            src = Path(src)

            # Determine user_id
            uid = f"user_{i+1:03d}"
            if metadata_list and i < len(metadata_list):
                uid = metadata_list[i].get("user_id", uid)

            signals_by_user[uid] = {}

            # Load raw signal CSVs
            for sig_file in SIGNAL_FILES:
                fpath = src / sig_file
                if fpath.exists():
                    df = pd.read_csv(fpath)
                    if len(df) > 0:
                        sig_name = sig_file.replace(".csv", "")
                        signals_by_user[uid][sig_name] = df

            # Extract demographics
            demo = {USER_COL: uid}
            if metadata_list and i < len(metadata_list):
                meta = metadata_list[i]
                user_meta = meta.get("demographics", meta.get("user", meta))
                for col in DEMOGRAPHIC_COLS:
                    if col in user_meta and user_meta[col] is not None:
                        demo[col] = user_meta[col]
            else:
                # Try reading from metadata.json in the folder
                meta_path = src / "metadata.json"
                if not meta_path.exists():
                    meta_path = src / "packet_metadata.json"
                if meta_path.exists():
                    import json
                    try:
                        meta = json.load(open(meta_path))
                        user_meta = meta.get("user", meta.get("demographics", {}))
                        for col in DEMOGRAPHIC_COLS:
                            if col in user_meta and user_meta[col] is not None:
                                demo[col] = user_meta[col]
                        demo[USER_COL] = meta.get("user_id", uid)
                    except Exception:
                        pass

            user_rows.append(demo)

        user_demo_df = pd.DataFrame(user_rows)
        n_sigs = sum(len(s) for s in signals_by_user.values())
        self._log(f"[M3 Splitter] Loaded {len(source_dirs)} directories: "
                  f"{n_sigs} signal files, {len(user_demo_df)} users")

        return signals_by_user, user_demo_df

    def split_raw_signals(
        self,
        split_result: 'SplitResult',
        signals_by_user: Dict[str, Dict[str, pd.DataFrame]],
    ) -> 'SplitResult':
        """
        Partition raw signal DataFrames by user assignments.

        Parameters
        ----------
        split_result     : SplitResult from split_users()
        signals_by_user  : {user_id: {signal_name: DataFrame}}

        Returns
        -------
        SplitResult with train_data, val_data, test_data populated.
        Each split's data is {user_id: {signal_name: DataFrame}}.
        """
        split_result.train_data = {
            uid: signals_by_user[uid]
            for uid in split_result.train_users
            if uid in signals_by_user
        }
        split_result.test_data = {
            uid: signals_by_user[uid]
            for uid in split_result.test_users
            if uid in signals_by_user
        }
        if split_result.val_users:
            split_result.val_data = {
                uid: signals_by_user[uid]
                for uid in split_result.val_users
                if uid in signals_by_user
            }

        for label, data in [("Train", split_result.train_data),
                            ("Val", split_result.val_data),
                            ("Test", split_result.test_data)]:
            if data:
                n_users = len(data)
                n_signals = sum(len(sigs) for sigs in data.values())
                self._log(f"  {label}: {n_users} users, "
                          f"{n_signals} signal DataFrames")

        return split_result

    def _log(self, msg):
        if self.verbose:
            print(msg)


class _StratificationError(Exception):
    """Raised when stratification cannot be performed."""
    pass
