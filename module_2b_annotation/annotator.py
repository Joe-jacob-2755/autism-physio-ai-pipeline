"""
=============================================================================
MODULE 2B -- DATA ANNOTATION  |  annotator.py
=============================================================================
Core annotation engine.

ManualAnnotator holds the mutable event list and coordinates all annotation
operations.  It does NOT own any I/O -- the CLI in main.py calls these
methods and handles printing.

Author  : Autism Physio-AI Pipeline
Version : 1.0.0
=============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    ANNOTATION_COLS,
    CATEGORY_MAP,
    COMBINED_RATE_HZ,
    EMOTION_METADATA,
    EMOTIONS_ALL,
    INTENSITY_REACTIVITY,
)
from validator import AnnotationValidator, ValidationResult


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION EVENT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnnotationEvent:
    """A single annotation event.

    Duck-types the same attribute interface as M2A's EventConfig
    (start_s, end_s, emotion, color, category) so that visualizer
    helpers can work without importing from M2A.
    """

    event_id: int
    emotion: str
    start_s: float
    duration_s: float
    intensity: str = "Medium"  # Low / Medium / High

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    @property
    def category(self) -> str:
        return CATEGORY_MAP.get(self.emotion, "unknown")

    @property
    def color(self) -> str:
        meta = EMOTION_METADATA.get(self.emotion, {})
        return meta.get("color", "#999999")

    @property
    def reactivity(self) -> float:
        """Reactivity multiplier derived from intensity level."""
        return INTENSITY_REACTIVITY.get(self.intensity, 1.0)

    def to_dict(self) -> dict:
        """Serialise to a flat dictionary for DataFrame construction."""
        meta = EMOTION_METADATA.get(self.emotion, {})
        return {
            "event_id": self.event_id,
            "emotion": self.emotion,
            "category": self.category,
            "intensity": self.intensity,
            "reactivity": self.reactivity,
            "valence": meta.get("valence", ""),
            "arousal": meta.get("arousal", ""),
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "duration_s": round(self.duration_s, 6),
            "color_hex": self.color,
            "description": meta.get("description", ""),
            "expect_EDA_dir": meta.get("expect_EDA_dir", ""),
            "expect_HR_dir": meta.get("expect_HR_dir", ""),
            "expect_ST_dir": meta.get("expect_ST_dir", ""),
            "expect_ACC_dir": meta.get("expect_ACC_dir", ""),
        }


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL ANNOTATOR
# ─────────────────────────────────────────────────────────────────────────────

class ManualAnnotator:
    """Core annotation state machine.

    Holds the mutable list of AnnotationEvents and coordinates all
    annotation operations.

    Parameters
    ----------
    packet : object
        PipelinePacket (unannotated or stripped).
    duration_s : float
        Recording duration in seconds.
    """

    def __init__(self, packet, duration_s: float):
        self.packet = packet
        self.duration_s = duration_s
        self.events: List[AnnotationEvent] = []
        self.validator = AnnotationValidator()
        self._next_id = 1

    # ── Add / edit / delete ───────────────────────────────────────────────

    def find_overlapping(
        self, start_s: float, end_s: float,
    ) -> List[AnnotationEvent]:
        """Return all existing events that overlap [start_s, end_s)."""
        overlapping = []
        for ev in self.events:
            if start_s < ev.end_s and end_s > ev.start_s:
                overlapping.append(ev)
        return overlapping

    def add_event(
        self,
        emotion: str,
        start_s: float,
        duration_s: float,
        intensity: str = "Medium",
    ) -> ValidationResult:
        """Create and validate a new event. Appends on success."""
        event = AnnotationEvent(
            event_id=self._next_id,
            emotion=emotion,
            start_s=start_s,
            duration_s=duration_s,
            intensity=intensity,
        )
        result = self.validator.validate_event(
            event, self.events, self.duration_s
        )
        if result.is_valid:
            self.events.append(event)
            self._next_id += 1
        return result

    def replace_event(
        self,
        emotion: str,
        start_s: float,
        duration_s: float,
        intensity: str = "Medium",
    ) -> Tuple[ValidationResult, List[AnnotationEvent]]:
        """Add a new event, auto-removing any overlapping events first.

        Returns (validation_result, list_of_removed_events).
        """
        end_s = start_s + duration_s
        removed = self.find_overlapping(start_s, end_s)
        for ev in removed:
            self.delete_event(ev.event_id)

        result = self.add_event(emotion, start_s, duration_s, intensity)
        return result, removed

    def edit_event(
        self,
        event_id: int,
        emotion: Optional[str] = None,
        start_s: Optional[float] = None,
        duration_s: Optional[float] = None,
        intensity: Optional[str] = None,
    ) -> ValidationResult:
        """Edit an existing event by ID. Validates before applying."""
        idx = self._find_index(event_id)
        if idx is None:
            return ValidationResult(
                is_valid=False,
                errors=[f"Event ID {event_id} not found."],
            )

        ev = self.events[idx]
        new_ev = AnnotationEvent(
            event_id=ev.event_id,
            emotion=emotion if emotion is not None else ev.emotion,
            start_s=start_s if start_s is not None else ev.start_s,
            duration_s=duration_s if duration_s is not None else ev.duration_s,
            intensity=intensity if intensity is not None else ev.intensity,
        )

        others = [e for e in self.events if e.event_id != event_id]
        result = self.validator.validate_event(
            new_ev, others, self.duration_s
        )
        if result.is_valid:
            self.events[idx] = new_ev
        return result

    def delete_event(self, event_id: int) -> bool:
        """Remove an event by ID. Returns True if found and removed."""
        idx = self._find_index(event_id)
        if idx is None:
            return False
        self.events.pop(idx)
        self._renumber()
        return True

    def clear_all_events(self) -> None:
        """Remove all events."""
        self.events.clear()
        self._next_id = 1

    # ── Batch import ──────────────────────────────────────────────────────

    def import_from_csv(
        self, path: Path, default_duration_s: Optional[float] = None,
    ) -> Tuple[int, ValidationResult]:
        """Import events from a batch CSV.

        Parameters
        ----------
        path : Path
            Path to CSV file.
        default_duration_s : float, optional
            Default duration if CSV lacks end_s and duration_s columns.

        Returns
        -------
        (n_imported, ValidationResult)
        """
        df = pd.read_csv(path)
        return self.import_from_dataframe(df, default_duration_s)

    def import_from_dataframe(
        self, df: pd.DataFrame, default_duration_s: Optional[float] = None,
    ) -> Tuple[int, ValidationResult]:
        """Import events from a DataFrame.

        Returns
        -------
        (n_imported, ValidationResult)
        """
        csv_result = self.validator.validate_batch_csv(df)
        if not csv_result.is_valid:
            return 0, csv_result

        all_warnings = list(csv_result.warnings)
        n_imported = 0
        row_errors: List[str] = []

        for i, row in df.iterrows():
            emotion = str(row["emotion"]).strip().title()
            start_s = float(row["start_s"])

            # Resolve duration
            if "duration_s" in df.columns and pd.notna(row.get("duration_s")):
                dur = float(row["duration_s"])
            elif "end_s" in df.columns and pd.notna(row.get("end_s")):
                dur = float(row["end_s"]) - start_s
            elif default_duration_s is not None:
                dur = default_duration_s
            else:
                row_errors.append(
                    f"Row {i + 1}: No duration or end_s, and no default provided."
                )
                continue

            result = self.add_event(emotion, start_s, dur)
            if result.is_valid:
                n_imported += 1
            else:
                for err in result.errors:
                    row_errors.append(f"Row {i + 1}: {err}")
                all_warnings.extend(result.warnings)

        combined = ValidationResult(
            is_valid=len(row_errors) == 0,
            errors=row_errors,
            warnings=all_warnings,
        )
        return n_imported, combined

    def load_existing_events(self, events_df: pd.DataFrame) -> int:
        """Load events from an existing annotations_events.csv DataFrame.

        Returns the number of events loaded.
        """
        self.clear_all_events()
        n = 0
        for _, row in events_df.iterrows():
            emotion = str(row.get("emotion", "")).strip()
            start_s = float(row.get("start_s", 0))
            end_s = float(row.get("end_s", start_s))
            dur = end_s - start_s
            intensity = str(row.get("intensity", "Medium")).strip()
            if intensity not in ("Low", "Medium", "High"):
                intensity = "Medium"
            if emotion in EMOTIONS_ALL and dur > 0:
                self.events.append(AnnotationEvent(
                    event_id=self._next_id,
                    emotion=emotion,
                    start_s=start_s,
                    duration_s=dur,
                    intensity=intensity,
                ))
                self._next_id += 1
                n += 1
        return n

    # ── Validation ────────────────────────────────────────────────────────

    def validate_current_state(self) -> ValidationResult:
        """Validate all current events as a set."""
        return self.validator.validate_all(self.events, self.duration_s)

    # ── Display ───────────────────────────────────────────────────────────

    def summary_table(self) -> str:
        """Return a formatted table of all events for terminal display."""
        if not self.events:
            return "  (no events)"

        lines = []
        hdr = (
            f"  {'ID':>3}  {'Emotion':<12} {'Category':<20} "
            f"{'Intensity':<10} "
            f"{'Start (s)':>10} {'End (s)':>10} {'Duration':>10}"
        )
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))

        sorted_events = sorted(self.events, key=lambda e: e.start_s)
        for ev in sorted_events:
            lines.append(
                f"  {ev.event_id:>3}  {ev.emotion:<12} {ev.category:<20} "
                f"{ev.intensity:<10} "
                f"{ev.start_s:>10.2f} {ev.end_s:>10.2f} {ev.duration_s:>10.2f}"
            )

        density = self.validator.compute_annotation_density(
            self.events, self.duration_s
        )
        lines.append("")
        lines.append(
            f"  Total: {len(self.events)} events  |  "
            f"Coverage: {density:.1%} of {self.duration_s:.1f}s recording"
        )
        return "\n".join(lines)

    # ── Apply annotations to packet ───────────────────────────────────────

    def apply_to_packet(self) -> object:
        """Add annotation columns to all signal DataFrames in the packet.

        Modifies packet in-place and returns it with is_annotated=True.
        """
        sorted_events = sorted(self.events, key=lambda e: e.start_s)

        # Annotate each signal DataFrame
        for name, df in self.packet.signals.items():
            self._annotate_dataframe(df, sorted_events)

        # Annotate combined DataFrame
        if self.packet.combined is not None and len(self.packet.combined) > 0:
            self._annotate_dataframe(self.packet.combined, sorted_events)

        self.packet.is_annotated = True
        return self.packet

    # ── Annotation DataFrames (for export) ────────────────────────────────

    def to_event_dataframe(self) -> pd.DataFrame:
        """Build annotations_events.csv DataFrame."""
        if not self.events:
            return pd.DataFrame()
        sorted_events = sorted(self.events, key=lambda e: e.start_s)
        rows = [ev.to_dict() for ev in sorted_events]
        return pd.DataFrame(rows)

    def to_sample_labels_dataframe(self) -> pd.DataFrame:
        """Build annotations_sample_labels.csv at COMBINED_RATE_HZ.

        One row per sample at the combined reference rate (64 Hz).
        """
        n_samples = int(self.duration_s * COMBINED_RATE_HZ)
        if n_samples <= 0:
            return pd.DataFrame(
                columns=["time_s", "label", "event_id", "category"]
            )

        time_s = np.arange(n_samples) / COMBINED_RATE_HZ
        labels = np.full(n_samples, "baseline", dtype=object)
        event_ids = np.zeros(n_samples, dtype=int)
        categories = np.full(n_samples, "baseline", dtype=object)

        sorted_events = sorted(self.events, key=lambda e: e.start_s)
        for ev in sorted_events:
            mask = (time_s >= ev.start_s) & (time_s < ev.end_s)
            labels[mask] = ev.emotion
            event_ids[mask] = ev.event_id
            categories[mask] = ev.category

        return pd.DataFrame({
            "time_s": np.round(time_s, 6),
            "label": labels,
            "event_id": event_ids,
            "category": categories,
        })

    # ── Internal helpers ──────────────────────────────────────────────────

    def _annotate_dataframe(
        self, df: pd.DataFrame, sorted_events: List[AnnotationEvent],
    ) -> None:
        """Add target_label, event_id, category columns to a DataFrame."""
        if "timestamp_s" not in df.columns:
            return

        timestamps = df["timestamp_s"].values
        n = len(timestamps)

        target_label = np.full(n, "baseline", dtype=object)
        event_id = np.zeros(n, dtype=int)
        category = np.full(n, "baseline", dtype=object)

        for ev in sorted_events:
            mask = (timestamps >= ev.start_s) & (timestamps < ev.end_s)
            target_label[mask] = ev.emotion
            event_id[mask] = ev.event_id
            category[mask] = ev.category

        # Remove existing annotation columns if present (re-annotation)
        for col in ANNOTATION_COLS:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        df["target_label"] = target_label
        df["event_id"] = event_id
        df["category"] = category

    def _find_index(self, event_id: int) -> Optional[int]:
        """Find the list index of an event by its ID."""
        for i, ev in enumerate(self.events):
            if ev.event_id == event_id:
                return i
        return None

    def _renumber(self) -> None:
        """Renumber events 1..N after a deletion."""
        sorted_events = sorted(self.events, key=lambda e: e.start_s)
        for i, ev in enumerate(sorted_events, start=1):
            ev.event_id = i
        self.events = sorted_events
        self._next_id = len(self.events) + 1
