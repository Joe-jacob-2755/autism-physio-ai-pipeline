"""
=============================================================================
MODULE 2B -- DATA ANNOTATION  |  excel_importer.py
=============================================================================
Import observer event logs from Excel (.xlsx / .xls) or CSV files.

Handles the real-world workflow where an observer records events on paper
during a session and later transcribes them into a spreadsheet with columns
like "start_time", "end_time", "targeted_behaviour", "intensity".

Features:
  - Auto-detects column names via EXCEL_COL_ALIASES (case-insensitive)
  - Auto-detects time formats (clock time like 14:32:15 or seconds like 45.0)
  - Converts clock times to seconds-from-start using recording start time
  - Parses intensity levels (Low / Medium / High)
  - Returns AnnotationEvents ready for the ManualAnnotator

Author  : Autism Physio-AI Pipeline
Version : 1.0.0
=============================================================================
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from config import (
    CLOCK_TIME_FORMATS,
    EMOTIONS_ALL,
    EXCEL_COL_ALIASES,
    INTENSITY_LEVELS,
)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

class ExcelImportResult:
    """Result of an Excel import operation."""

    def __init__(
        self,
        events: List[dict],
        n_rows: int = 0,
        n_imported: int = 0,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        detected_columns: Optional[dict] = None,
        time_format: str = "unknown",
    ):
        self.events = events
        self.n_rows = n_rows
        self.n_imported = n_imported
        self.errors = errors or []
        self.warnings = warnings or []
        self.detected_columns = detected_columns or {}
        self.time_format = time_format

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0 and self.n_imported > 0


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL IMPORTER
# ─────────────────────────────────────────────────────────────────────────────

class ExcelImporter:
    """Import observer event logs from Excel/CSV files.

    Parameters
    ----------
    recording_start : datetime, optional
        Absolute start time of the recording. Required when the observer
        used clock times (e.g. 14:32:15) instead of seconds from start.
        If None and clock times are detected, the importer will raise
        an error asking for the start time.
    default_duration_s : float, optional
        Default event duration if the spreadsheet has no end_time or
        duration column. If None and durations cannot be determined,
        an error is raised per row.
    """

    def __init__(
        self,
        recording_start: Optional[datetime] = None,
        default_duration_s: Optional[float] = None,
    ):
        self.recording_start = recording_start
        self.default_duration_s = default_duration_s

    def import_file(self, path: Path) -> ExcelImportResult:
        """Import events from an Excel or CSV file.

        Parameters
        ----------
        path : Path
            Path to .xlsx, .xls, or .csv file.

        Returns
        -------
        ExcelImportResult with parsed events and diagnostics.
        """
        path = Path(path)
        if not path.exists():
            return ExcelImportResult(
                events=[], errors=[f"File not found: {path}"]
            )

        # Read the file
        try:
            if path.suffix.lower() in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            elif path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                return ExcelImportResult(
                    events=[],
                    errors=[f"Unsupported file type: {path.suffix}. "
                            "Expected .xlsx, .xls, or .csv"],
                )
        except Exception as e:
            return ExcelImportResult(
                events=[], errors=[f"Failed to read file: {e}"]
            )

        if len(df) == 0:
            return ExcelImportResult(
                events=[], errors=["File is empty (no data rows)."]
            )

        return self.import_dataframe(df)

    def import_dataframe(self, df: pd.DataFrame) -> ExcelImportResult:
        """Import events from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Observer's event log.

        Returns
        -------
        ExcelImportResult with parsed events and diagnostics.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Step 1: Auto-detect columns
        col_map = self._detect_columns(df)

        if "emotion" not in col_map:
            errors.append(
                "Could not find an emotion/behaviour column. "
                f"Expected one of: {EXCEL_COL_ALIASES['emotion']}"
            )
            return ExcelImportResult(
                events=[], n_rows=len(df), errors=errors,
                detected_columns=col_map,
            )

        if "start" not in col_map:
            errors.append(
                "Could not find a start time column. "
                f"Expected one of: {EXCEL_COL_ALIASES['start']}"
            )
            return ExcelImportResult(
                events=[], n_rows=len(df), errors=errors,
                detected_columns=col_map,
            )

        # Step 2: Detect time format from the start column
        start_col = col_map["start"]
        time_format = self._detect_time_format(df[start_col])

        if time_format == "clock" and self.recording_start is None:
            errors.append(
                "Clock times detected (e.g. 14:32:15) but no recording "
                "start time was provided. Please provide the recording "
                "start time so clock times can be converted to seconds."
            )
            return ExcelImportResult(
                events=[], n_rows=len(df), errors=errors,
                detected_columns=col_map, time_format=time_format,
            )

        # Step 3: Parse each row
        events = []
        for i, row in df.iterrows():
            row_num = i + 2  # Excel row (1-indexed header + data)

            # Parse emotion
            raw_emotion = str(row[col_map["emotion"]]).strip()
            emotion = self._normalise_emotion(raw_emotion)
            if emotion is None:
                errors.append(
                    f"Row {row_num}: Unrecognised emotion '{raw_emotion}'. "
                    f"Valid: {', '.join(EMOTIONS_ALL)}"
                )
                continue

            # Parse start time
            raw_start = row[col_map["start"]]
            start_s = self._parse_time(raw_start, time_format)
            if start_s is None:
                errors.append(
                    f"Row {row_num}: Could not parse start time '{raw_start}'"
                )
                continue

            # Parse end time / duration
            duration_s = None
            if "duration" in col_map and pd.notna(row.get(col_map["duration"])):
                try:
                    duration_s = float(row[col_map["duration"]])
                except (ValueError, TypeError):
                    warnings.append(
                        f"Row {row_num}: Could not parse duration "
                        f"'{row[col_map['duration']]}', trying end time."
                    )

            if duration_s is None and "end" in col_map and pd.notna(
                row.get(col_map["end"])
            ):
                raw_end = row[col_map["end"]]
                end_s = self._parse_time(raw_end, time_format)
                if end_s is not None:
                    duration_s = end_s - start_s
                    if duration_s <= 0:
                        errors.append(
                            f"Row {row_num}: End time ({raw_end}) is before "
                            f"or equal to start time ({raw_start})."
                        )
                        continue
                else:
                    warnings.append(
                        f"Row {row_num}: Could not parse end time '{raw_end}'"
                    )

            if duration_s is None:
                if self.default_duration_s is not None:
                    duration_s = self.default_duration_s
                    warnings.append(
                        f"Row {row_num}: No duration or end time; "
                        f"using default {self.default_duration_s}s."
                    )
                else:
                    errors.append(
                        f"Row {row_num}: No duration or end time found, "
                        "and no default duration set."
                    )
                    continue

            # Parse intensity
            intensity = "Medium"
            if "intensity" in col_map and pd.notna(
                row.get(col_map["intensity"])
            ):
                raw_intensity = str(row[col_map["intensity"]]).strip().title()
                if raw_intensity in INTENSITY_LEVELS:
                    intensity = raw_intensity
                else:
                    warnings.append(
                        f"Row {row_num}: Unrecognised intensity "
                        f"'{raw_intensity}', using Medium."
                    )

            events.append({
                "emotion": emotion,
                "start_s": round(start_s, 3),
                "duration_s": round(duration_s, 3),
                "intensity": intensity,
            })

        return ExcelImportResult(
            events=events,
            n_rows=len(df),
            n_imported=len(events),
            errors=errors,
            warnings=warnings,
            detected_columns=col_map,
            time_format=time_format,
        )

    # ── Column detection ─────────────────────────────────────────────────

    def _detect_columns(self, df: pd.DataFrame) -> dict:
        """Auto-detect column roles from the DataFrame headers.

        Returns a dict mapping role ("start", "end", "duration",
        "emotion", "intensity") to the actual column name in the DataFrame.
        """
        col_map = {}
        df_cols_lower = {c.lower().strip(): c for c in df.columns}

        for role, aliases in EXCEL_COL_ALIASES.items():
            for alias in aliases:
                if alias.lower() in df_cols_lower:
                    col_map[role] = df_cols_lower[alias.lower()]
                    break

        return col_map

    # ── Time format detection ────────────────────────────────────────────

    def _detect_time_format(self, series: pd.Series) -> str:
        """Detect whether a time column contains seconds or clock times.

        Returns "seconds", "clock", or "unknown".
        """
        sample = series.dropna().head(5)
        if len(sample) == 0:
            return "unknown"

        # Check if values are already numeric (seconds from start)
        numeric_count = 0
        clock_count = 0

        for val in sample:
            if isinstance(val, (int, float)):
                numeric_count += 1
                continue

            val_str = str(val).strip()

            # Try parsing as float
            try:
                float(val_str)
                numeric_count += 1
                continue
            except ValueError:
                pass

            # Try parsing as clock time
            if self._parse_clock_time(val_str) is not None:
                clock_count += 1

        if numeric_count >= len(sample) * 0.6:
            return "seconds"
        if clock_count >= len(sample) * 0.6:
            return "clock"
        return "unknown"

    # ── Time parsing ─────────────────────────────────────────────────────

    def _parse_time(self, value, time_format: str) -> Optional[float]:
        """Parse a time value to seconds from recording start."""
        if pd.isna(value):
            return None

        if time_format == "seconds":
            return self._parse_seconds(value)
        elif time_format == "clock":
            return self._parse_clock_to_seconds(value)
        else:
            # Try seconds first, then clock
            result = self._parse_seconds(value)
            if result is not None:
                return result
            return self._parse_clock_to_seconds(value)

    def _parse_seconds(self, value) -> Optional[float]:
        """Parse a numeric seconds value."""
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (ValueError, TypeError):
            return None

    def _parse_clock_to_seconds(self, value) -> Optional[float]:
        """Parse a clock time string to seconds from recording start."""
        if self.recording_start is None:
            return None

        val_str = str(value).strip()
        parsed_time = self._parse_clock_time(val_str)
        if parsed_time is None:
            return None

        # Build a full datetime from the recording start date + parsed time
        event_dt = self.recording_start.replace(
            hour=parsed_time.hour,
            minute=parsed_time.minute,
            second=parsed_time.second,
            microsecond=0,
        )

        # Handle midnight crossing (event is next day)
        if event_dt < self.recording_start:
            event_dt += timedelta(days=1)

        delta = (event_dt - self.recording_start).total_seconds()
        return delta

    def _parse_clock_time(self, val_str: str) -> Optional[datetime]:
        """Try to parse a string as a clock time using known formats."""
        for fmt in CLOCK_TIME_FORMATS:
            try:
                return datetime.strptime(val_str, fmt)
            except ValueError:
                continue
        return None

    # ── Emotion normalisation ────────────────────────────────────────────

    @staticmethod
    def _normalise_emotion(raw: str) -> Optional[str]:
        """Normalise an emotion string to the canonical form.

        Handles case-insensitivity and common abbreviations.
        """
        # Direct match (case-insensitive)
        raw_lower = raw.lower()
        for emotion in EMOTIONS_ALL:
            if emotion.lower() == raw_lower:
                return emotion

        # Common abbreviations / aliases
        aliases = {
            "self-injurious": "SIB",
            "self injurious": "SIB",
            "self-injury": "SIB",
            "aggression": "ATO",
            "aggression to others": "ATO",
            "general aggression": "GAB",
            "tantrum": "GAB",
            "tantrums": "GAB",
            "property destruction": "GAB",
            "happiness": "Happy",
            "joy": "Happy",
            "angry": "Anger",
            "scared": "Fear",
            "afraid": "Fear",
            "sadness": "Sad",
            "crying": "Sad",
            "surprised": "Surprise",
            "shock": "Surprise",
            "hungry": "Hunger",
            "thirsty": "Thirst",
            "fatigue": "Tired",
            "sleepy": "Tired",
            "drowsy": "Tired",
            "toileting": "Toilet",
            "bathroom": "Toilet",
            "disgusted": "Disgust",
        }
        if raw_lower in aliases:
            return aliases[raw_lower]

        return None


# ─────────────────────────────────────────────────────────────────────────────
# RECORDING START TIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_recording_start(packet) -> Optional[datetime]:
    """Try to detect the recording start time from packet metadata.

    Checks metadata fields that Empatica E4 and other devices typically
    set: 'recording_start', 'start_time', 'start_datetime'.

    Parameters
    ----------
    packet : PipelinePacket-like
        Packet with .metadata dict.

    Returns
    -------
    datetime or None if not found.
    """
    meta = getattr(packet, "metadata", {})
    if not meta:
        return None

    # Check common metadata keys
    for key in ("recording_start", "start_time", "start_datetime",
                "recording_start_time", "session_start"):
        val = meta.get(key)
        if val is None:
            continue

        if isinstance(val, datetime):
            return val

        # Try parsing as ISO format
        val_str = str(val).strip()
        for fmt in (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ):
            try:
                return datetime.strptime(val_str, fmt)
            except ValueError:
                continue

    return None
