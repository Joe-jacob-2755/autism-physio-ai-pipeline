"""
=============================================================================
MODULE 2B -- DATA ANNOTATION  |  validator.py
=============================================================================
Stateless validation for annotation events.

All methods are pure functions that take the event list and recording
duration and return ValidationResult objects.

Author  : Autism Physio-AI Pipeline
Version : 1.0.0
=============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from config import (
    EMOTIONS_ALL,
    BATCH_CSV_REQUIRED_COLS,
    MIN_EVENT_DURATION_S,
    MIN_ANNOTATION_DENSITY,
    BOUNDARY_MARGIN_S,
    SHORT_EVENT_WARN_S,
)


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of a validation check."""

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Combine two results (errors from either make it invalid)."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationValidator:
    """Stateless annotation validation helper.

    Rule set
    --------
    R1  Emotion label must be in EMOTIONS_ALL                     (ERROR)
    R2  start_s >= 0                                               (ERROR)
    R3  end_s <= duration_s                                        (ERROR)
    R4  end_s > start_s                                            (ERROR)
    R5  duration >= MIN_EVENT_DURATION_S                            (ERROR)
    R6  No overlapping events                                      (ERROR)
    R7  Annotation density < MIN_ANNOTATION_DENSITY                (WARNING)
    R8  Event duration < SHORT_EVENT_WARN_S                        (WARNING)
    R9  Event near recording boundary (< BOUNDARY_MARGIN_S)        (WARNING)
    R10 Only one emotion class present                             (WARNING)
    R11 Batch CSV missing required columns                         (ERROR)
    """

    # ── Single-event validation ───────────────────────────────────────────

    @staticmethod
    def validate_event(
        event,
        all_events: list,
        duration_s: float,
    ) -> ValidationResult:
        """Validate a single event against rules and existing events.

        Parameters
        ----------
        event : AnnotationEvent
            The event to validate.
        all_events : list[AnnotationEvent]
            All previously accepted events (excluding *event* itself).
        duration_s : float
            Total recording duration in seconds.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # R1 — valid emotion label
        if event.emotion not in EMOTIONS_ALL:
            errors.append(
                f"Unknown emotion '{event.emotion}'. "
                f"Valid: {', '.join(EMOTIONS_ALL)}"
            )

        # R2 — start >= 0
        if event.start_s < 0:
            errors.append(
                f"Event start time ({event.start_s:.2f}s) cannot be negative."
            )

        # R3 — end <= duration
        if event.end_s > duration_s:
            errors.append(
                f"Event end time ({event.end_s:.2f}s) exceeds recording "
                f"duration ({duration_s:.2f}s)."
            )

        # R4 — end > start
        if event.end_s <= event.start_s:
            errors.append(
                f"Event end ({event.end_s:.2f}s) must be after start "
                f"({event.start_s:.2f}s)."
            )

        # R5 — minimum duration
        if event.duration_s < MIN_EVENT_DURATION_S:
            errors.append(
                f"Event duration ({event.duration_s:.2f}s) is below minimum "
                f"({MIN_EVENT_DURATION_S:.1f}s)."
            )

        # R6 — overlap check
        for other in all_events:
            if other.event_id == event.event_id:
                continue
            if event.start_s < other.end_s and event.end_s > other.start_s:
                errors.append(
                    f"Event {event.event_id} ({event.emotion} "
                    f"{event.start_s:.1f}-{event.end_s:.1f}s) overlaps with "
                    f"event {other.event_id} ({other.emotion} "
                    f"{other.start_s:.1f}-{other.end_s:.1f}s)."
                )

        # R8 — short event warning
        if event.duration_s < SHORT_EVENT_WARN_S and event.duration_s >= MIN_EVENT_DURATION_S:
            warnings.append(
                f"Event {event.event_id} ({event.emotion}) is short "
                f"({event.duration_s:.1f}s) -- may be suboptimal for training."
            )

        # R9 — boundary proximity
        if event.start_s < BOUNDARY_MARGIN_S:
            warnings.append(
                f"Event {event.event_id} starts very close to recording "
                f"start ({event.start_s:.1f}s < {BOUNDARY_MARGIN_S:.0f}s margin)."
            )
        if duration_s - event.end_s < BOUNDARY_MARGIN_S:
            warnings.append(
                f"Event {event.event_id} ends very close to recording "
                f"end ({event.end_s:.1f}s, recording {duration_s:.1f}s)."
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # ── Full event set validation ─────────────────────────────────────────

    @staticmethod
    def validate_all(events: list, duration_s: float) -> ValidationResult:
        """Validate the complete set of annotation events.

        Parameters
        ----------
        events : list[AnnotationEvent]
            All events to validate.
        duration_s : float
            Total recording duration in seconds.
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not events:
            errors.append("No events to annotate. Add at least one event.")
            return ValidationResult(is_valid=False, errors=errors)

        # Validate each event individually
        for ev in events:
            others = [o for o in events if o.event_id != ev.event_id]
            result = AnnotationValidator.validate_event(ev, others, duration_s)
            errors.extend(result.errors)
            warnings.extend(result.warnings)

        # Deduplicate overlap errors (A overlaps B == B overlaps A)
        errors = list(dict.fromkeys(errors))
        warnings = list(dict.fromkeys(warnings))

        # R7 — annotation density
        density = AnnotationValidator.compute_annotation_density(
            events, duration_s
        )
        if density < MIN_ANNOTATION_DENSITY:
            warnings.append(
                f"Low annotation density ({density:.1%}). Only "
                f"{density * 100:.1f}% of the recording has event labels."
            )

        # R10 — single emotion class
        unique_emotions = set(ev.emotion for ev in events)
        if len(unique_emotions) == 1:
            warnings.append(
                f"All events use the same label ('{list(unique_emotions)[0]}'). "
                f"A single-class dataset limits model training."
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # ── Batch CSV validation ──────────────────────────────────────────────

    @staticmethod
    def validate_batch_csv(df: pd.DataFrame) -> ValidationResult:
        """Validate schema of a batch annotation CSV.

        Parameters
        ----------
        df : pd.DataFrame
            Loaded batch CSV.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # R11 — required columns
        missing = [c for c in BATCH_CSV_REQUIRED_COLS if c not in df.columns]
        if missing:
            errors.append(
                f"Batch CSV missing required columns: {', '.join(missing)}. "
                f"Required: {', '.join(BATCH_CSV_REQUIRED_COLS)}"
            )
            return ValidationResult(is_valid=False, errors=errors)

        if len(df) == 0:
            errors.append("Batch CSV is empty (no rows).")
            return ValidationResult(is_valid=False, errors=errors)

        # Check duration info
        has_end = "end_s" in df.columns
        has_dur = "duration_s" in df.columns
        if not has_end and not has_dur:
            warnings.append(
                "Batch CSV has neither 'end_s' nor 'duration_s' column. "
                "You will be prompted for a default duration."
            )

        # Check emotion values
        invalid_emotions = []
        for i, row in df.iterrows():
            em = str(row["emotion"]).strip().title()
            if em not in EMOTIONS_ALL:
                invalid_emotions.append((i + 1, row["emotion"]))
        if invalid_emotions:
            sample = invalid_emotions[:5]
            msgs = [f"row {r}: '{e}'" for r, e in sample]
            errors.append(
                f"Invalid emotion labels in batch CSV: {'; '.join(msgs)}"
                + (f" (and {len(invalid_emotions) - 5} more)"
                   if len(invalid_emotions) > 5 else "")
            )

        # Check numeric columns
        for col in ["start_s"] + (["end_s"] if has_end else []) + (
            ["duration_s"] if has_dur else []
        ):
            try:
                pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError):
                errors.append(
                    f"Column '{col}' contains non-numeric values."
                )

        # Warn about unrecognised columns
        known = set(BATCH_CSV_REQUIRED_COLS) | {"end_s", "duration_s"}
        extra = set(df.columns) - known
        if extra:
            warnings.append(
                f"Unrecognised columns ignored: {', '.join(sorted(extra))}"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def compute_annotation_density(events: list, duration_s: float) -> float:
        """Fraction of recording covered by events (0.0 -- 1.0)."""
        if duration_s <= 0 or not events:
            return 0.0
        total_event_s = sum(ev.duration_s for ev in events)
        return min(total_event_s / duration_s, 1.0)
