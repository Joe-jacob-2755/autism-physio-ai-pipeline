"""
=============================================================================
MODULE 5B - DATA AUGMENTATION  |  signal_generator.py
=============================================================================
Generate synthetic physiological signal segments using a trained CT-TimeGAN.

Responsibilities:
  1. Extract real event segments from M5 cleaned signals
  2. Normalise segments to [0, 1] for TGAN input
  3. Generate synthetic segments conditioned on (emotion, severity, verbal)
  4. Denormalise back to physiological scale
  5. Attach metadata (synthetic flag, source emotion, condition)

Each signal type has its own TGAN — this module handles the per-signal
training data preparation and generation loop.
=============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BASELINE_CONTEXT_S,
    EMOTION_TO_IDX,
    MODULE_LABEL,
    SEGMENT_DURATION_S,
    SEVERITY_TO_IDX,
    SIGNAL_SPECS,
    TGAN,
    VERBAL_TO_IDX,
)

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SegmentData:
    """Collection of extracted signal segments ready for TGAN training."""
    signal_name: str
    segments: np.ndarray           # (n_segments, seq_len, n_features)
    emotions: np.ndarray           # (n_segments,) int indices
    severities: np.ndarray         # (n_segments,) int indices
    verbals: np.ndarray            # (n_segments,) int indices
    emotion_labels: List[str]      # original string labels
    user_ids: List[str]            # which user each segment came from
    min_vals: np.ndarray           # per-feature min for denormalisation
    max_vals: np.ndarray           # per-feature max for denormalisation


@dataclass
class SyntheticSegment:
    """A single generated synthetic signal segment."""
    signal_name: str
    values: np.ndarray             # (seq_len, n_features) denormalised
    emotion_label: str
    emotion_idx: int
    severity_idx: int
    verbal_idx: int
    timestamps: np.ndarray         # (seq_len,) in seconds


# ═════════════════════════════════════════════════════════════════════════════
# Segment Extractor
# ═════════════════════════════════════════════════════════════════════════════

class SegmentExtractor:
    """
    Extract fixed-length event segments from cleaned signal CSVs.

    Reads M5 output CSVs and extracts segments around each event,
    padding/truncating to SEGMENT_DURATION_S.
    """

    def __init__(self, segment_duration_s: float = SEGMENT_DURATION_S,
                 baseline_context_s: float = BASELINE_CONTEXT_S):
        self.segment_duration_s = segment_duration_s
        self.baseline_context_s = baseline_context_s

    def extract_for_signal(
        self,
        signal_name: str,
        user_dirs: Dict[str, Path],
        demographics: Dict[str, dict],
    ) -> Optional[SegmentData]:
        """
        Extract segments for one signal type across all training users.

        Parameters
        ----------
        signal_name : str
            One of "EDA", "BVP", "IBI", "ST", "ACC".
        user_dirs : dict
            {user_id: Path to user's cleaned_signals/ directory}
        demographics : dict
            {user_id: {"autism_severity": ..., "verbal_status": ...}}

        Returns
        -------
        SegmentData or None if no segments could be extracted.
        """
        spec = SIGNAL_SPECS.get(signal_name)
        if spec is None:
            log.warning(f"[{MODULE_LABEL}] Unknown signal: {signal_name}")
            return None

        value_cols = spec["value_cols"]
        sampling_rate = spec["sampling_rate"]

        # IBI is event-based — use a fixed segment count instead of time
        if sampling_rate is None:
            seq_len = 30  # fixed number of IBI values per segment
        else:
            seq_len = int(self.segment_duration_s * sampling_rate)

        all_segments = []
        all_emotions = []
        all_severities = []
        all_verbals = []
        all_emotion_labels = []
        all_user_ids = []

        for uid, udir in user_dirs.items():
            csv_path = udir / f"{signal_name}.csv"
            if not csv_path.exists():
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                log.warning(f"[{MODULE_LABEL}] Error reading {csv_path}: {e}")
                continue

            # Check required columns
            missing = [c for c in value_cols if c not in df.columns]
            if missing:
                continue
            if "target_label" not in df.columns:
                continue

            # Get demographic info
            demo = demographics.get(uid, {})
            sev_str = demo.get("autism_severity", "Medium")
            vrb_str = demo.get("verbal_status", "Verbal")
            sev_idx = SEVERITY_TO_IDX.get(sev_str, 1)
            vrb_idx = VERBAL_TO_IDX.get(vrb_str, 0)

            # Extract event segments
            segments = self._extract_events_from_df(
                df, value_cols, seq_len, sampling_rate
            )

            for seg_values, emotion_label in segments:
                emo_idx = EMOTION_TO_IDX.get(emotion_label)
                if emo_idx is None:
                    continue  # skip unknown emotion labels

                all_segments.append(seg_values)
                all_emotions.append(emo_idx)
                all_severities.append(sev_idx)
                all_verbals.append(vrb_idx)
                all_emotion_labels.append(emotion_label)
                all_user_ids.append(uid)

        if not all_segments:
            log.warning(f"[{MODULE_LABEL}] No segments extracted for "
                        f"{signal_name}")
            return None

        segments_arr = np.stack(all_segments, axis=0)  # (N, seq_len, feat)

        # Compute min/max for normalisation
        n_feat = segments_arr.shape[2]
        min_vals = segments_arr.reshape(-1, n_feat).min(axis=0)
        max_vals = segments_arr.reshape(-1, n_feat).max(axis=0)

        log.info(f"[{MODULE_LABEL}] Extracted {len(all_segments)} segments "
                 f"for {signal_name} (seq_len={seq_len})")

        return SegmentData(
            signal_name=signal_name,
            segments=segments_arr,
            emotions=np.array(all_emotions),
            severities=np.array(all_severities),
            verbals=np.array(all_verbals),
            emotion_labels=all_emotion_labels,
            user_ids=all_user_ids,
            min_vals=min_vals,
            max_vals=max_vals,
        )

    def _extract_events_from_df(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        seq_len: int,
        sampling_rate: Optional[int],
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Extract fixed-length segments around each event in a signal DataFrame.

        Returns list of (segment_array, emotion_label) tuples.
        """
        results = []

        if "event_id" not in df.columns:
            return results

        # Filter to labelled events (not baseline)
        mask = df["target_label"].notna()
        mask &= df["target_label"].astype(str).str.lower() != "baseline"
        labelled = df.loc[mask]

        if labelled.empty:
            return results

        event_ids = labelled["event_id"].dropna().unique()

        for eid in event_ids:
            event_rows = labelled[labelled["event_id"] == eid]
            if event_rows.empty:
                continue

            emotion = str(event_rows["target_label"].iloc[0])
            start_idx = event_rows.index[0]

            # Include baseline context before the event
            if sampling_rate is not None:
                context_samples = int(self.baseline_context_s * sampling_rate)
            else:
                context_samples = 5  # fixed context for IBI

            seg_start = max(0, start_idx - context_samples)
            seg_end = seg_start + seq_len

            # Extract values
            vals = df.loc[seg_start:seg_end - 1, value_cols].values

            if len(vals) == 0:
                continue

            # Pad or truncate to seq_len
            if len(vals) < seq_len:
                pad = np.full(
                    (seq_len - len(vals), len(value_cols)),
                    np.nan
                )
                vals = np.concatenate([vals, pad], axis=0)
                # Forward-fill NaN from padding
                for col_i in range(vals.shape[1]):
                    col = vals[:, col_i]
                    mask_nan = np.isnan(col)
                    if mask_nan.any() and not mask_nan.all():
                        last_valid = col[~mask_nan][-1]
                        col[mask_nan] = last_valid
                        vals[:, col_i] = col
            elif len(vals) > seq_len:
                vals = vals[:seq_len]

            # Replace any remaining NaN with column mean
            for col_i in range(vals.shape[1]):
                col = vals[:, col_i]
                nan_mask = np.isnan(col)
                if nan_mask.any():
                    col_mean = np.nanmean(col) if not nan_mask.all() else 0.0
                    col[nan_mask] = col_mean
                    vals[:, col_i] = col

            results.append((vals.astype(np.float32), emotion))

        return results


# ═════════════════════════════════════════════════════════════════════════════
# Signal Generator
# ═════════════════════════════════════════════════════════════════════════════

class SignalGenerator:
    """
    Generate synthetic signal segments using a trained CTTimeGAN.

    Parameters
    ----------
    model : CTTimeGAN (trained, in eval mode)
    segment_data : SegmentData
        The real data used for training (needed for denormalisation).
    device : str
    seed : int
    """

    def __init__(
        self,
        model,
        segment_data: SegmentData,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.model = model
        self.model.eval()
        self.segment_data = segment_data
        self.device = device
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def normalise(segments: np.ndarray, min_vals: np.ndarray,
                  max_vals: np.ndarray) -> np.ndarray:
        """Normalise segments to [0, 1] using min-max scaling."""
        range_vals = max_vals - min_vals
        range_vals[range_vals < 1e-8] = 1.0  # avoid division by zero
        return (segments - min_vals) / range_vals

    @staticmethod
    def denormalise(segments: np.ndarray, min_vals: np.ndarray,
                    max_vals: np.ndarray) -> np.ndarray:
        """Denormalise segments from [0, 1] back to original scale."""
        range_vals = max_vals - min_vals
        range_vals[range_vals < 1e-8] = 1.0
        return segments * range_vals + min_vals

    def generate(
        self,
        n_samples: int,
        emotion_idx: int,
        severity_idx: int = 1,
        verbal_idx: int = 0,
    ) -> List[SyntheticSegment]:
        """
        Generate n_samples synthetic segments for a given condition.

        Parameters
        ----------
        n_samples : int
        emotion_idx : int
        severity_idx : int
        verbal_idx : int

        Returns
        -------
        List of SyntheticSegment
        """
        import torch

        sd = self.segment_data
        seq_len = sd.segments.shape[1]
        noise_dim = self.model.noise_dim
        spec = SIGNAL_SPECS[sd.signal_name]
        sampling_rate = spec["sampling_rate"] or 1  # IBI fallback

        results = []
        batch_size = min(n_samples, TGAN["batch_size"])

        with torch.no_grad():
            generated = 0
            while generated < n_samples:
                bs = min(batch_size, n_samples - generated)

                # Random noise
                z = torch.randn(
                    bs, seq_len, noise_dim, device=self.device
                )
                emo_t = torch.full((bs,), emotion_idx,
                                   dtype=torch.long, device=self.device)
                sev_t = torch.full((bs,), severity_idx,
                                   dtype=torch.long, device=self.device)
                vrb_t = torch.full((bs,), verbal_idx,
                                   dtype=torch.long, device=self.device)

                # Generate
                fake_signal = self.model.generate_signal(
                    z, emo_t, sev_t, vrb_t
                )
                fake_np = fake_signal.cpu().numpy()

                # Denormalise
                fake_denorm = self.denormalise(
                    fake_np, sd.min_vals, sd.max_vals
                )

                # Create timestamps
                timestamps = np.arange(seq_len) / sampling_rate

                # Lookup emotion label
                from config import IDX_TO_EMOTION
                emo_label = IDX_TO_EMOTION.get(emotion_idx, f"class_{emotion_idx}")

                for i in range(bs):
                    results.append(SyntheticSegment(
                        signal_name=sd.signal_name,
                        values=fake_denorm[i],
                        emotion_label=emo_label,
                        emotion_idx=emotion_idx,
                        severity_idx=severity_idx,
                        verbal_idx=verbal_idx,
                        timestamps=timestamps,
                    ))

                generated += bs

        log.info(f"[{MODULE_LABEL}] Generated {len(results)} synthetic "
                 f"{sd.signal_name} segments for '{emo_label}'")

        return results

    def generate_for_plan(
        self,
        synthetic_needed: Dict[str, int],
        default_severity: int = 1,
        default_verbal: int = 0,
    ) -> Dict[str, List[SyntheticSegment]]:
        """
        Generate segments for all classes in the augmentation plan.

        Parameters
        ----------
        synthetic_needed : dict
            {emotion_label: n_to_generate}
        default_severity : int
            Default severity index when not specified per-sample.
        default_verbal : int
            Default verbal index when not specified per-sample.

        Returns
        -------
        {emotion_label: [SyntheticSegment, ...]}
        """
        all_synthetic = {}
        for emo_label, n_needed in synthetic_needed.items():
            if n_needed <= 0:
                continue
            emo_idx = EMOTION_TO_IDX.get(emo_label)
            if emo_idx is None:
                log.warning(f"[{MODULE_LABEL}] Unknown emotion '{emo_label}', "
                            "skipping generation")
                continue

            segments = self.generate(
                n_needed, emo_idx, default_severity, default_verbal
            )
            all_synthetic[emo_label] = segments

        return all_synthetic
