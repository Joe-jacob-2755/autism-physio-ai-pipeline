"""
=============================================================================
MODULE 4 - EXPLORATORY DATA ANALYSIS  |  combined_loader.py
=============================================================================
Load and concatenate all training users' raw physiological signals into a
single combined dataset with user_id columns for cross-user analysis.
=============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import VALUE_COLS, SIGNAL_SAMPLING_RATES, DEMOGRAPHIC_FIELDS

log = logging.getLogger(__name__)


@dataclass
class CombinedData:
    """Container for combined multi-user signal data."""
    signals: Dict[str, pd.DataFrame]      # {signal_name: DF with user_id}
    demographics: pd.DataFrame             # user_id, age, gender, ...
    user_ids: List[str] = field(default_factory=list)
    n_users: int = 0
    total_duration_s: float = 0.0
    events_metadata: List[dict] = field(default_factory=list)


class CombinedDataLoader:
    """Load and concatenate all training users' raw signals."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Loader] {msg}")

    def load_from_packets(
        self,
        packets_by_uid: Dict[str, Any],
        train_users: List[str],
        demographics_by_uid: Optional[Dict[str, dict]] = None,
    ) -> CombinedData:
        """
        Concatenate all training users' signals from PipelinePackets.

        Parameters
        ----------
        packets_by_uid : {user_id: PipelinePacket}
        train_users    : list of user_ids in the training set
        demographics_by_uid : {user_id: {age, gender, ...}}
        """
        signals_combined: Dict[str, list] = {}
        demo_rows = []
        total_dur = 0.0
        events_meta = []

        for uid in sorted(train_users):
            pkt = packets_by_uid.get(uid)
            if pkt is None:
                self._log(f"  WARNING: No packet for {uid}, skipping")
                continue

            for sig_name, df in pkt.signals.items():
                if sig_name not in signals_combined:
                    signals_combined[sig_name] = []
                df_copy = df.copy()
                df_copy["user_id"] = uid
                signals_combined[sig_name].append(df_copy)

            # Duration
            if hasattr(pkt, "metadata") and isinstance(pkt.metadata, dict):
                dur = pkt.metadata.get("duration_s", 0)
                total_dur += dur
                # Extract event metadata
                evts = pkt.metadata.get("events", [])
                for evt in evts:
                    evt_copy = dict(evt)
                    evt_copy["user_id"] = uid
                    events_meta.append(evt_copy)
            else:
                # Estimate duration from signals
                for df in pkt.signals.values():
                    if "timestamp_s" in df.columns and len(df) > 0:
                        total_dur += df["timestamp_s"].max()
                        break

            # Demographics
            if demographics_by_uid and uid in demographics_by_uid:
                row = {"user_id": uid}
                row.update(demographics_by_uid[uid])
                demo_rows.append(row)

        # Concatenate each signal
        result_signals = {}
        for sig_name, dfs in signals_combined.items():
            result_signals[sig_name] = pd.concat(dfs, ignore_index=True)
            self._log(f"  {sig_name}: {len(result_signals[sig_name]):,} rows "
                      f"from {len(dfs)} users")

        demo_df = pd.DataFrame(demo_rows) if demo_rows else pd.DataFrame()
        user_ids = sorted(train_users)

        self._log(f"  Combined: {len(user_ids)} users, "
                  f"{total_dur:.0f}s total duration")

        return CombinedData(
            signals=result_signals,
            demographics=demo_df,
            user_ids=user_ids,
            n_users=len(user_ids),
            total_duration_s=total_dur,
            events_metadata=events_meta,
        )

    def load_from_folders(
        self,
        user_folders: Dict[str, Path],
        demographics_by_uid: Optional[Dict[str, dict]] = None,
    ) -> CombinedData:
        """
        Load from M3 output directories (train/user_NNN/*.csv).

        Parameters
        ----------
        user_folders : {user_id: Path to folder containing signal CSVs}
        demographics_by_uid : optional demographics dict
        """
        signals_combined: Dict[str, list] = {}
        demo_rows = []
        total_dur = 0.0

        for uid, folder in sorted(user_folders.items()):
            folder = Path(folder)
            for sig_name in VALUE_COLS:
                csv_path = folder / f"{sig_name}.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                df["user_id"] = uid
                if sig_name not in signals_combined:
                    signals_combined[sig_name] = []
                signals_combined[sig_name].append(df)

                # Estimate duration from first signal
                if "timestamp_s" in df.columns and len(df) > 0:
                    dur = df["timestamp_s"].max()
                    if sig_name == list(VALUE_COLS.keys())[0]:
                        total_dur += dur

            if demographics_by_uid and uid in demographics_by_uid:
                row = {"user_id": uid}
                row.update(demographics_by_uid[uid])
                demo_rows.append(row)

        result_signals = {}
        for sig_name, dfs in signals_combined.items():
            result_signals[sig_name] = pd.concat(dfs, ignore_index=True)
            self._log(f"  {sig_name}: {len(result_signals[sig_name]):,} rows "
                      f"from {len(dfs)} users")

        demo_df = pd.DataFrame(demo_rows) if demo_rows else pd.DataFrame()
        user_ids = sorted(user_folders.keys())

        return CombinedData(
            signals=result_signals,
            demographics=demo_df,
            user_ids=user_ids,
            n_users=len(user_ids),
            total_duration_s=total_dur,
        )
