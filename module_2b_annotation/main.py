"""
=============================================================================
MODULE 2B -- DATA ANNOTATION  |  main.py
=============================================================================
Interactive CLI for manual annotation of physiological signals.

Provides both:
  - Standalone entry point (run from command line)
  - Pipeline-callable API via run_annotation()

Author  : Autism Physio-AI Pipeline
Version : 1.0.0
=============================================================================
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure module directory is on path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotator import ManualAnnotator  # noqa: E402
from config import (  # noqa: E402
    EMOTIONS_ALL,
    EMOTIONS_AFFECTIVE,
    EMOTIONS_NEEDS,
    EMOTIONS_BEHAVIOURAL,
    MODULE_VERSION,
    MIN_RECORDING_DURATION_S,
)
from excel_importer import ExcelImporter, detect_recording_start  # noqa: E402
from exporter import AnnotationExporter, next_run_folder  # noqa: E402
from loader import SignalLoader  # noqa: E402
from visualizer import AnnotationVisualizer  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL HELPERS (mirroring module_1/main.py pattern)
# ─────────────────────────────────────────────────────────────────────────────

WIDTH = 66


def _header(title: str):
    print()
    print("  " + "=" * (WIDTH - 2))
    print(f"  {title}")
    print("  " + "=" * (WIDTH - 2))


def _section(title: str):
    print()
    print("  " + "-" * (WIDTH - 4))
    print(f"  {title}")
    print("  " + "-" * (WIDTH - 4))


def _line():
    print("  " + "-" * (WIDTH - 4))


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"\n  {prompt}{suffix}: ").strip()
    return raw if raw else str(default)


def _ask_float(prompt: str, default: float,
               min_val: float = None, max_val: float = None) -> float:
    while True:
        raw = _ask(prompt, str(default))
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"  * Must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"  * Must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("  * Please enter a number.")


def _ask_int(prompt: str, default: int,
             min_val: int = 0, max_val: int = 999) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"  * Must be between {min_val} and {max_val}")
        except ValueError:
            print("  * Please enter a number.")


def _ask_yn(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = _ask(f"{prompt}  ({hint})", "")
    if raw == "":
        return default
    return raw.lower() in ("y", "yes")


def _choose(prompt: str, options: list, labels: list = None) -> int:
    """Numbered menu returning 0-based index."""
    display = labels or [str(o) for o in options]
    print()
    for i, lbl in enumerate(display, 1):
        print(f"    {i}.  {lbl}")
    while True:
        raw = _ask(prompt, "1")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print("  * Please enter a valid number.")


# ─────────────────────────────────────────────────────────────────────────────
# EMOTION CHOOSER
# ─────────────────────────────────────────────────────────────────────────────

def _choose_emotion() -> str:
    """Interactive numbered menu for selecting an emotion label."""
    print()
    print("  Affective Emotions:")
    for i, em in enumerate(EMOTIONS_AFFECTIVE, 1):
        print(f"    {i:>2}.  {em}")

    offset = len(EMOTIONS_AFFECTIVE)
    print("  Physiological Needs:")
    for i, em in enumerate(EMOTIONS_NEEDS, offset + 1):
        print(f"    {i:>2}.  {em}")

    offset2 = offset + len(EMOTIONS_NEEDS)
    print("  Behavioural:")
    for i, em in enumerate(EMOTIONS_BEHAVIOURAL, offset2 + 1):
        print(f"    {i:>2}.  {em}")

    while True:
        raw = _ask("Select emotion number", "1")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(EMOTIONS_ALL):
                return EMOTIONS_ALL[idx]
        except ValueError:
            pass
        print(f"  * Enter a number 1-{len(EMOTIONS_ALL)}")


# ─────────────────────────────────────────────────────────────────────────────
# RECORDING START TIME HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _get_recording_start(packet):
    """Get recording start datetime from packet metadata, or None."""
    return detect_recording_start(packet)


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION SUB-FLOWS
# ─────────────────────────────────────────────────────────────────────────────

def _flow_cli_annotation(annotator: ManualAnnotator) -> None:
    """Interactive one-by-one event entry."""
    _section("Add Events -- Interactive Entry")
    print(f"  Recording duration: {annotator.duration_s:.2f}s")
    print("  Enter events one at a time. Type 'done' to finish.\n")

    while True:
        print(f"\n  --- Event #{len(annotator.events) + 1} ---")

        start = _ask_float(
            "Start time (seconds)", 0.0,
            min_val=0.0, max_val=annotator.duration_s,
        )
        duration = _ask_float(
            "Duration (seconds)", 30.0,
            min_val=0.1,
        )
        emotion = _choose_emotion()

        result = annotator.add_event(emotion, start, duration)

        if result.is_valid:
            ev = annotator.events[-1]
            print(f"\n  + Added: {ev.emotion} ({ev.category}) "
                  f"@ {ev.start_s:.1f}-{ev.end_s:.1f}s")
            if result.warnings:
                for w in result.warnings:
                    print(f"    [WARN] {w}")
        else:
            print("\n  ! Event rejected:")
            for err in result.errors:
                print(f"    [ERROR] {err}")
            if result.warnings:
                for w in result.warnings:
                    print(f"    [WARN] {w}")

        if not _ask_yn("Add another event?", default=True):
            break


def _flow_batch_import(annotator: ManualAnnotator) -> None:
    """Import events from a batch CSV file."""
    _section("Import Events from CSV")

    print("  Expected CSV format:")
    print("    start_s,duration_s,emotion")
    print("    45.0,30.0,Fear")
    print("    120.0,25.0,Happy")
    print()

    path_str = _ask("Path to batch CSV")
    path = Path(path_str)

    if not path.exists():
        print(f"  ! File not found: {path}")
        return

    # Check if duration column is present
    import pandas as pd
    df = pd.read_csv(path)
    default_dur = None
    if "duration_s" not in df.columns and "end_s" not in df.columns:
        print("  CSV has no 'duration_s' or 'end_s' column.")
        default_dur = _ask_float(
            "Default duration for all events (seconds)", 30.0, min_val=0.1,
        )

    n_imported, result = annotator.import_from_csv(path, default_dur)

    print(f"\n  Imported: {n_imported} events")
    if result.errors:
        print("  Errors:")
        for err in result.errors:
            print(f"    [ERROR] {err}")
    if result.warnings:
        print("  Warnings:")
        for w in result.warnings:
            print(f"    [WARN] {w}")


def _flow_excel_upload(
    annotator: ManualAnnotator,
    packet,
) -> None:
    """Import events from an observer's Excel/CSV observation sheet."""
    _section("Import from Observer Observation Sheet")

    print("  Supported formats: .xlsx, .xls, .csv")
    print("  Expected columns (auto-detected, flexible naming):")
    print("    - Start time  (start_s, start_time, onset, ...)")
    print("    - End time / duration  (end_s, duration, ...)")
    print("    - Emotion / behaviour  (emotion, behaviour, target, ...)")
    print("    - Intensity  (intensity, severity)  [optional]")
    print()

    path_str = _ask("Path to observation sheet")
    path = Path(path_str)

    if not path.exists():
        print(f"  ! File not found: {path}")
        return

    # Detect or ask for recording start time
    recording_start = detect_recording_start(packet)
    if recording_start:
        print(f"  Recording start detected: {recording_start}")
    else:
        print("  No recording start time found in device metadata.")
        if _ask_yn("Does the observation sheet use clock times "
                    "(e.g. 14:32:15)?", default=False):
            while True:
                raw = _ask("Recording start date/time "
                           "(YYYY-MM-DD HH:MM:SS)", "")
                if not raw:
                    print("  * Required for clock time conversion.")
                    continue
                try:
                    recording_start = datetime.strptime(
                        raw, "%Y-%m-%d %H:%M:%S"
                    )
                    break
                except ValueError:
                    print("  * Invalid format. Use: YYYY-MM-DD HH:MM:SS")

    # Ask for default duration if needed
    default_dur = None
    if _ask_yn("Set a default duration for events without end times?",
               default=False):
        default_dur = _ask_float(
            "Default duration (seconds)", 30.0, min_val=0.1
        )

    # Run the import
    importer = ExcelImporter(
        recording_start=recording_start,
        default_duration_s=default_dur,
    )
    result = importer.import_file(path)

    # Report column detection
    if result.detected_columns:
        print("\n  Detected columns:")
        for role, col_name in result.detected_columns.items():
            print(f"    {role:<12} -> '{col_name}'")
    if result.time_format != "unknown":
        print(f"  Time format:  {result.time_format}")

    # Report errors
    if result.errors:
        print("\n  Import errors:")
        for err in result.errors:
            print(f"    [ERROR] {err}")

    if result.warnings:
        print("\n  Import warnings:")
        for w in result.warnings:
            print(f"    [WARN] {w}")

    if not result.events:
        print("\n  No events imported.")
        return

    # Add events to annotator
    n_added = 0
    for ev_dict in result.events:
        add_result = annotator.add_event(
            emotion=ev_dict["emotion"],
            start_s=ev_dict["start_s"],
            duration_s=ev_dict["duration_s"],
            intensity=ev_dict["intensity"],
        )
        if add_result.is_valid:
            n_added += 1
        else:
            for err in add_result.errors:
                print(f"    [ERROR] {err}")

    print(f"\n  Successfully imported {n_added} of "
          f"{result.n_imported} events from observation sheet.")
    if n_added > 0:
        print(annotator.summary_table())


def _flow_load_existing(annotator: ManualAnnotator, existing_df) -> None:
    """Load events from a previously saved annotations_events.csv."""
    _section("Load Existing Annotations")

    if existing_df is None or len(existing_df) == 0:
        print("  No existing annotations found.")
        return

    n = annotator.load_existing_events(existing_df)
    print(f"  Loaded {n} existing events:")
    print(annotator.summary_table())


def _flow_review_events(annotator: ManualAnnotator) -> None:
    """Review, edit, and delete events."""
    while True:
        _section("Review Events")
        print(annotator.summary_table())

        if not annotator.events:
            return

        options = [
            "Edit an event",
            "Delete an event",
            "Clear all events",
            "Continue",
        ]
        choice = _choose("Action", options)

        if choice == 0:  # Edit
            eid = _ask_int(
                "Event ID to edit", 1,
                min_val=1, max_val=len(annotator.events),
            )
            print("  Leave blank to keep current value.")

            raw_em = _ask("New emotion (blank = keep)", "")
            new_emotion = raw_em.strip().title() if raw_em else None

            raw_start = _ask("New start time (blank = keep)", "")
            new_start = float(raw_start) if raw_start else None

            raw_dur = _ask("New duration (blank = keep)", "")
            new_dur = float(raw_dur) if raw_dur else None

            result = annotator.edit_event(
                eid,
                emotion=new_emotion,
                start_s=new_start,
                duration_s=new_dur,
            )
            if result.is_valid:
                print("  + Event updated.")
            else:
                print("  ! Edit rejected:")
                for err in result.errors:
                    print(f"    [ERROR] {err}")

        elif choice == 1:  # Delete
            eid = _ask_int(
                "Event ID to delete", 1,
                min_val=1, max_val=len(annotator.events),
            )
            if annotator.delete_event(eid):
                print(f"  - Event {eid} deleted.")
            else:
                print(f"  ! Event {eid} not found.")

        elif choice == 2:  # Clear all
            if _ask_yn("Clear ALL events? This cannot be undone.", False):
                annotator.clear_all_events()
                print("  All events cleared.")

        elif choice == 3:  # Continue
            return


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANNOTATION SESSION
# ─────────────────────────────────────────────────────────────────────────────

def _run_annotation_session(
    packet,
    output_dir: Path,
    existing_events_df=None,
) -> object:
    """Drive the full interactive annotation workflow.

    Parameters
    ----------
    packet : PipelinePacket-like
        Unannotated (or stripped) packet.
    output_dir : Path
        Output folder for this run.
    existing_events_df : pd.DataFrame, optional
        Previously saved annotations_events.csv to load from.

    Returns
    -------
    Annotated PipelinePacket-like object (is_annotated=True).
    """
    # Detect duration
    duration_s = 0.0
    if (packet.combined is not None and len(packet.combined) > 0
            and "timestamp_s" in packet.combined.columns):
        duration_s = float(packet.combined["timestamp_s"].max())
    if duration_s <= 0:
        for df in packet.signals.values():
            if "timestamp_s" in df.columns and len(df) > 0:
                duration_s = max(duration_s,
                                 float(df["timestamp_s"].max()))

    if duration_s < MIN_RECORDING_DURATION_S:
        print(f"  [WARN] Recording is very short ({duration_s:.1f}s)")

    # Create annotator and visualizer
    annotator = ManualAnnotator(packet, duration_s)
    visualizer = AnnotationVisualizer(packet)

    # Step 1: Overview plot
    _section("Signal Overview")
    overview_path = output_dir / "signal_overview.png"
    visualizer.plot_overview(overview_path)
    print(f"  Signal overview saved to:\n    {overview_path}")
    print(f"  Duration: {duration_s:.2f}s")
    print(f"  Signals: {', '.join(packet.signals.keys())}")

    # Step 2: Choose annotation method
    _section("Annotation Method")

    has_existing = (existing_events_df is not None
                    and len(existing_events_df) > 0)

    options = []
    labels = []
    n = 1

    options.append("excel")
    labels.append(f"{n}. Upload observation sheet (Excel/CSV) "
                  "[RECOMMENDED]")
    n += 1

    options.append("visual")
    labels.append(f"{n}. Visual annotation (interactive plot window)")
    n += 1

    options.append("cli")
    labels.append(f"{n}. Enter events one by one (CLI)")
    n += 1

    options.append("csv")
    labels.append(f"{n}. Import from batch CSV (start_s, emotion, ...)")
    n += 1

    if has_existing:
        options.append("existing")
        labels.append(f"{n}. Load existing annotations for editing")
        n += 1

    options.append("skip")
    labels.append(f"{n}. Skip annotation (pass through unannotated)")

    choice = _choose("Choose annotation method", options, labels)
    method = options[choice]

    if method == "skip":
        print("  Annotation skipped.")
        return packet

    # Step 3: Apply chosen method
    if method == "excel":
        _flow_excel_upload(annotator, packet)

        # After Excel import, offer visual verification
        if annotator.events:
            print("\n  Events imported from observation sheet.")
            if _ask_yn("Open visual annotation tool to verify/amend?",
                       default=True):
                try:
                    from interactive_annotator import (
                        run_interactive_annotation,
                    )
                    print("\n  Opening interactive annotation window...")
                    print("  Review the events on the signals")
                    print("  DRAG to add/replace events")
                    print("  RIGHT-CLICK or D = delete  |  C = clear all")
                    print("  Press Q to finish\n")
                    annotator = run_interactive_annotation(
                        packet, annotator,
                        recording_start=_get_recording_start(packet),
                        output_dir=output_dir,
                    )
                except ImportError as e:
                    print(f"  [WARN] Interactive mode unavailable: {e}")
                except Exception as e:
                    print(f"  [WARN] Interactive mode failed: {e}")

    elif method == "visual":
        # Load existing events if available
        if has_existing:
            if _ask_yn("Load existing annotations into the visual tool?",
                       True):
                annotator.load_existing_events(existing_events_df)

        try:
            from interactive_annotator import run_interactive_annotation
            print("\n  Opening interactive annotation window...")
            print("  DRAG to add/replace events")
            print("  RIGHT-CLICK or D = delete  |  C = clear all")
            print("  Z = undo  |  Q = finish & save\n")
            annotator = run_interactive_annotation(
                packet, annotator,
                recording_start=_get_recording_start(packet),
                output_dir=output_dir,
            )
        except ImportError as e:
            print(f"  [WARN] Interactive mode unavailable: {e}")
            print("  Falling back to CLI mode.\n")
            _flow_cli_annotation(annotator)
        except Exception as e:
            print(f"  [WARN] Interactive mode failed: {e}")
            print("  Falling back to CLI mode.\n")
            _flow_cli_annotation(annotator)

    elif method == "cli":
        _flow_cli_annotation(annotator)
    elif method == "csv":
        _flow_batch_import(annotator)
    elif method == "existing":
        _flow_load_existing(annotator, existing_events_df)

    if not annotator.events:
        print("\n  No events were added.")
        if _ask_yn("Skip annotation and continue?", True):
            return packet

    # Step 4: Review and optionally add more events
    _section("Review Annotations")
    print(annotator.summary_table())

    while _ask_yn("\nEdit or add more events?", False):
        sub_options = ["Visual annotation tool", "CLI entry", "CSV import",
                       "Edit/delete events"]
        sub_choice = _choose("Method", sub_options)
        if sub_choice == 0:
            try:
                from interactive_annotator import run_interactive_annotation
                annotator = run_interactive_annotation(
                    packet, annotator,
                    recording_start=_get_recording_start(packet),
                    output_dir=output_dir,
                )
            except Exception as e:
                print(f"  [WARN] Interactive mode failed: {e}")
        elif sub_choice == 1:
            _flow_cli_annotation(annotator)
        elif sub_choice == 2:
            _flow_batch_import(annotator)
        elif sub_choice == 3:
            _flow_review_events(annotator)

        print(annotator.summary_table())

    # Step 5: Validation
    _section("Validation")
    result = annotator.validate_current_state()

    if result.errors:
        print("  ERRORS (must fix before proceeding):")
        for err in result.errors:
            print(f"    [ERROR] {err}")
        print()
        if _ask_yn("Go back and edit events?", True):
            _flow_review_events(annotator)
            result = annotator.validate_current_state()

    if result.warnings:
        print("  WARNINGS:")
        for w in result.warnings:
            print(f"    [WARN] {w}")
        if not _ask_yn("Proceed despite warnings?", True):
            return packet

    if result.errors:
        print("  Cannot proceed with unresolved errors.")
        return packet

    # Step 6: Annotated visualisation
    _section("Annotated Preview")
    annotated_path = output_dir / "signal_annotated.png"
    visualizer.plot_annotated(annotator.events, annotated_path)
    print(f"  Annotated plot saved to:\n    {annotated_path}")

    # Step 7: Confirm and export
    _section("Export")
    print(annotator.summary_table())
    print()

    if not _ask_yn("Save annotated data and proceed?", True):
        print("  Annotation cancelled.")
        return packet

    # Apply annotations to packet
    annotator.apply_to_packet()

    # Export all files
    exporter = AnnotationExporter(verbose=True)
    exporter.export_all(packet, annotator, visualizer, output_dir)

    return packet


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE-CALLABLE API (Rule R3.5)
# ─────────────────────────────────────────────────────────────────────────────

def run_annotation(
    packet,
    output_dir,
    interactive: bool = True,
    batch_csv=None,
) -> object:
    """Annotate a PipelinePacket.

    Called by run_pipeline_auto.py and pipeline_main.py.

    Parameters
    ----------
    packet : PipelinePacket-like
        Data to annotate.
    output_dir : str | Path
        Output directory for annotation results.
    interactive : bool
        If True, run full CLI workflow. If False, use batch_csv.
    batch_csv : str | Path, optional
        Path to batch CSV for non-interactive annotation.

    Returns
    -------
    Annotated PipelinePacket (is_annotated=True) or original packet
    if annotation was skipped.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = SignalLoader(verbose=True)
    packet, duration_s, existing_events_df, warnings = loader.load(packet)

    for w in warnings:
        print(f"  [WARN] {w}")

    if interactive:
        _header(f"Module 2B -- Data Annotation v{MODULE_VERSION}")
        return _run_annotation_session(
            packet, output_dir, existing_events_df
        )

    # Non-interactive batch mode
    if batch_csv is None:
        print("  [M2B] Non-interactive mode with no batch CSV -- skipping.")
        return packet

    batch_csv = Path(batch_csv)
    if not batch_csv.exists():
        print(f"  [M2B] Batch CSV not found: {batch_csv} -- skipping.")
        return packet

    annotator = ManualAnnotator(packet, duration_s)
    visualizer = AnnotationVisualizer(packet)

    n_imported, result = annotator.import_from_csv(batch_csv)
    print(f"  [M2B] Imported {n_imported} events from {batch_csv.name}")

    if result.errors:
        for err in result.errors:
            print(f"  [M2B] ERROR: {err}")

    val_result = annotator.validate_current_state()
    if not val_result.is_valid:
        for err in val_result.errors:
            print(f"  [M2B] VALIDATION ERROR: {err}")
        print("  [M2B] Annotation failed validation -- returning unannotated.")
        return packet

    annotator.apply_to_packet()

    exporter = AnnotationExporter(verbose=True)
    exporter.export_all(packet, annotator, visualizer, output_dir)

    return packet


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _discover_data_folders() -> list:
    """Find candidate data folders from known module outputs."""
    repo_root = Path(__file__).resolve().parent.parent
    candidates = []

    # M2A simulation outputs
    m2a_outputs = repo_root / "module_2a_data_simulation" / "outputs"
    if m2a_outputs.exists():
        for d in sorted(m2a_outputs.iterdir(), reverse=True):
            if d.is_dir() and (d / "combined_signals.csv").exists():
                candidates.append(d)

    # M1 acquisition outputs
    m1_outputs = repo_root / "module_1_data_acquisition" / "outputs"
    if m1_outputs.exists():
        for d in sorted(m1_outputs.iterdir(), reverse=True):
            if d.is_dir() and (d / "combined_signals.csv").exists():
                candidates.append(d)

    return candidates[:10]  # limit display


def run_standalone():
    """Run M2B as a standalone module."""
    parser = argparse.ArgumentParser(
        description="Module 2B -- Manual Data Annotation"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Path to data folder (with signal CSVs)",
    )
    parser.add_argument(
        "--batch_csv", type=str, default=None,
        help="Path to batch annotation CSV (non-interactive)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Custom output directory",
    )
    args = parser.parse_args()

    _header(f"Module 2B -- Data Annotation v{MODULE_VERSION}")

    # Determine source
    if args.source:
        source_path = Path(args.source)
    else:
        _section("Select Data Source")
        candidates = _discover_data_folders()

        if candidates:
            print("  Recent data folders found:\n")
            labels = []
            for d in candidates:
                labels.append(f"{d.parent.parent.name}/{d.name}")
            labels.append("Enter custom path")

            choice = _choose("Select data source", candidates + [None], labels)
            if choice < len(candidates):
                source_path = candidates[choice]
            else:
                source_path = Path(_ask("Enter path to data folder"))
        else:
            source_path = Path(_ask("Enter path to data folder"))

    if not source_path.exists():
        print(f"  ! Source not found: {source_path}")
        return

    # Determine output
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = next_run_folder()

    print(f"  Source:  {source_path}")
    print(f"  Output:  {output_dir}")

    # Load and annotate
    if args.batch_csv:
        result = run_annotation(
            packet=source_path,
            output_dir=output_dir,
            interactive=False,
            batch_csv=args.batch_csv,
        )
    else:
        loader = SignalLoader(verbose=True)
        packet, duration_s, existing_df, warnings = loader.load(source_path)
        for w in warnings:
            print(f"  [WARN] {w}")
        result = _run_annotation_session(packet, output_dir, existing_df)

    if hasattr(result, "is_annotated") and result.is_annotated:
        print(f"\n  Annotation complete. Output: {output_dir}")
    else:
        print("\n  Annotation was skipped or cancelled.")


if __name__ == "__main__":
    run_standalone()
