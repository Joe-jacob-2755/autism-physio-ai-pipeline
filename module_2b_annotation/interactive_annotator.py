"""
=============================================================================
MODULE 2B -- DATA ANNOTATION  |  interactive_annotator.py
=============================================================================
Interactive matplotlib-based annotation tool.

The researcher sees all physiological signals plotted, then:
  1. Click-and-drag on any signal panel to select a time region
  2. A popup dialog asks which emotion/behaviour to assign
  3. The event appears as coloured shading on all panels + timeline bar
  4. Right-click an existing event to edit or delete it
  5. Press 'S' to save, 'Z' to undo, 'Q' to finish

Workflow:
  - Observer has paper notes with timestamps and event types
  - They download physiological data from the wearable device
  - They open this tool, see the signals, and mark events visually

Author  : Autism Physio-AI Pipeline
Version : 1.0.0
=============================================================================
"""
from __future__ import annotations

import tkinter as tk
from datetime import datetime, timedelta
from tkinter import simpledialog, messagebox
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.widgets import SpanSelector
import numpy as np

from annotator import ManualAnnotator, AnnotationEvent
from config import (
    EMOTIONS_ALL,
    EMOTIONS_AFFECTIVE,
    EMOTIONS_NEEDS,
    EMOTIONS_BEHAVIOURAL,
    EMOTION_METADATA,
    INTENSITY_LEVELS,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EVENT_SHADE_ALPHA = 0.25
EVENT_BORDER_LW = 1.5
TIMELINE_HEIGHT_RATIO = 1
SIGNAL_HEIGHT_RATIO = 4
BASELINE_COLOR = "#E0E0E0"
EDGE_GRAB_FRACTION = 0.015  # fraction of x-range to detect edge grab

SIGNAL_VALUE_COLS = {
    "EDA": "EDA_uS",
    "BVP": "BVP_nT",
    "IBI": "IBI_ms",
    "ST": "ST_degC",
}

SIGNAL_COLORS = {
    "EDA": "#27AE60",
    "BVP": "#C0392B",
    "IBI": "#8E44AD",
    "ST": "#2980B9",
    "ACC_X": "#D35400",
    "ACC_Y": "#F39C12",
    "ACC_Z": "#16A085",
}

SIGNAL_LABELS = {
    "EDA": "EDA (uS)",
    "BVP": "BVP (nT)",
    "IBI": "IBI (ms)",
    "ST": "ST (degC)",
    "ACC_X": "ACC X (g)",
    "ACC_Y": "ACC Y (g)",
    "ACC_Z": "ACC Z (g)",
}


# ─────────────────────────────────────────────────────────────────────────────
# EMOTION PICKER DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class EmotionPickerDialog(simpledialog.Dialog):
    """Tkinter dialog for selecting an emotion label and intensity."""

    def __init__(self, parent, start_s: float, end_s: float):
        self.start_s = start_s
        self.end_s = end_s
        self.selected_emotion: Optional[str] = None
        self.selected_intensity: str = "Medium"
        super().__init__(parent, title="Select Emotion / Behaviour")

    def body(self, master):
        # Header
        dur = self.end_s - self.start_s
        tk.Label(
            master,
            text=f"Time: {self.start_s:.1f}s - {self.end_s:.1f}s "
                 f"({dur:.1f}s)",
            font=("Arial", 11, "bold"),
        ).pack(pady=(5, 10))

        # Emotion buttons grouped by category
        self._add_group(master, "Affective Emotions", EMOTIONS_AFFECTIVE)
        self._add_group(master, "Physiological Needs", EMOTIONS_NEEDS)
        self._add_group(master, "Behavioural", EMOTIONS_BEHAVIOURAL)

        # Intensity selector
        int_frame = tk.LabelFrame(
            master, text="Intensity", padx=8, pady=4,
        )
        int_frame.pack(fill="x", padx=5, pady=5)

        self._intensity_var = tk.StringVar(value="Medium")
        int_colors = {"Low": "#3498DB", "Medium": "#F39C12", "High": "#E74C3C"}
        row = tk.Frame(int_frame)
        row.pack()
        for level in INTENSITY_LEVELS:
            tk.Radiobutton(
                row, text=level, variable=self._intensity_var,
                value=level, font=("Arial", 10, "bold"),
                fg=int_colors.get(level, "black"),
                indicatoron=0, width=10, padx=4, pady=2,
            ).pack(side="left", padx=4)

        return None  # no initial focus widget

    def _add_group(self, master, title: str, emotions: list):
        frame = tk.LabelFrame(master, text=title, padx=8, pady=4)
        frame.pack(fill="x", padx=5, pady=3)

        row = tk.Frame(frame)
        row.pack(fill="x")

        for i, em in enumerate(emotions):
            color = EMOTION_METADATA.get(em, {}).get("color", "#999")
            btn = tk.Button(
                row,
                text=em,
                width=10,
                bg=color,
                fg="white" if em not in ("Happy", "Hunger") else "black",
                font=("Arial", 9, "bold"),
                command=lambda e=em: self._pick(e),
            )
            btn.grid(row=i // 4, column=i % 4, padx=2, pady=2, sticky="ew")

    def _pick(self, emotion: str):
        self.selected_emotion = emotion
        self.selected_intensity = self._intensity_var.get()
        self.ok()

    def buttonbox(self):
        box = tk.Frame(self)
        tk.Button(
            box, text="Cancel", width=10,
            command=self.cancel,
        ).pack(side="left", padx=5, pady=5)
        box.pack()

    def apply(self):
        self.result = self.selected_emotion


class EventEditDialog(simpledialog.Dialog):
    """Dialog for editing or deleting an existing event."""

    def __init__(self, parent, event: AnnotationEvent):
        self.event = event
        self.action: Optional[str] = None  # "edit", "delete", or None
        self.new_emotion: Optional[str] = None
        self.new_start: Optional[float] = None
        self.new_duration: Optional[float] = None
        self.new_intensity: Optional[str] = None
        super().__init__(parent, title=f"Edit Event {event.event_id}")

    def body(self, master):
        tk.Label(
            master,
            text=f"Event {self.event.event_id}: {self.event.emotion} "
                 f"[{self.event.intensity}]",
            font=("Arial", 12, "bold"),
            fg=self.event.color,
        ).pack(pady=(5, 2))

        tk.Label(
            master,
            text=f"{self.event.start_s:.1f}s - {self.event.end_s:.1f}s "
                 f"({self.event.duration_s:.1f}s)",
            font=("Arial", 10),
        ).pack(pady=(0, 10))

        # Start time
        f1 = tk.Frame(master)
        f1.pack(fill="x", padx=10, pady=2)
        tk.Label(f1, text="Start (s):", width=12, anchor="w").pack(side="left")
        self.start_var = tk.StringVar(value=f"{self.event.start_s:.1f}")
        tk.Entry(f1, textvariable=self.start_var, width=10).pack(side="left")

        # Duration
        f2 = tk.Frame(master)
        f2.pack(fill="x", padx=10, pady=2)
        tk.Label(f2, text="Duration (s):", width=12, anchor="w").pack(
            side="left")
        self.dur_var = tk.StringVar(value=f"{self.event.duration_s:.1f}")
        tk.Entry(f2, textvariable=self.dur_var, width=10).pack(side="left")

        # Intensity selector
        int_frame = tk.LabelFrame(
            master, text="Intensity", padx=5, pady=3,
        )
        int_frame.pack(fill="x", padx=5, pady=3)
        self._intensity_var = tk.StringVar(value=self.event.intensity)
        int_colors = {"Low": "#3498DB", "Medium": "#F39C12", "High": "#E74C3C"}
        row_int = tk.Frame(int_frame)
        row_int.pack()
        for level in INTENSITY_LEVELS:
            tk.Radiobutton(
                row_int, text=level, variable=self._intensity_var,
                value=level, font=("Arial", 9, "bold"),
                fg=int_colors.get(level, "black"),
                indicatoron=0, width=10, padx=3, pady=2,
            ).pack(side="left", padx=3)

        # Emotion re-label
        f3 = tk.LabelFrame(master, text="Change Label", padx=5, pady=3)
        f3.pack(fill="x", padx=5, pady=5)

        row = tk.Frame(f3)
        row.pack(fill="x")
        for i, em in enumerate(EMOTIONS_ALL):
            color = EMOTION_METADATA.get(em, {}).get("color", "#999")
            btn = tk.Button(
                row, text=em, width=8,
                bg=color,
                fg="white" if em not in ("Happy", "Hunger") else "black",
                font=("Arial", 8),
                command=lambda e=em: self._set_emotion(e),
            )
            btn.grid(row=i // 5, column=i % 5, padx=1, pady=1, sticky="ew")

        self._emotion_label = tk.Label(
            master, text="(click a button above to change label)",
            font=("Arial", 9, "italic"),
        )
        self._emotion_label.pack(pady=2)

        return None

    def _set_emotion(self, emotion: str):
        self.new_emotion = emotion
        self._emotion_label.config(
            text=f"New label: {emotion}",
            font=("Arial", 9, "bold"),
            fg=EMOTION_METADATA.get(emotion, {}).get("color", "black"),
        )

    def buttonbox(self):
        box = tk.Frame(self)
        tk.Button(
            box, text="Save Changes", width=14, bg="#2ECC71", fg="white",
            command=self._save,
        ).pack(side="left", padx=5, pady=5)
        tk.Button(
            box, text="Set to Baseline", width=14, bg="#E74C3C", fg="white",
            command=self._delete,
        ).pack(side="left", padx=5, pady=5)
        tk.Button(
            box, text="Cancel", width=10,
            command=self.cancel,
        ).pack(side="left", padx=5, pady=5)
        box.pack()

    def _save(self):
        self.action = "edit"
        try:
            self.new_start = float(self.start_var.get())
        except ValueError:
            self.new_start = None
        try:
            self.new_duration = float(self.dur_var.get())
        except ValueError:
            self.new_duration = None
        new_int = self._intensity_var.get()
        if new_int != self.event.intensity:
            self.new_intensity = new_int
        self.ok()

    def _delete(self):
        self.action = "delete"
        self.ok()

    def apply(self):
        self.result = self.action


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAP CHOICE DIALOG
# ─────────────────────────────────────────────────────────────────────────────

class _OverlapDialog(simpledialog.Dialog):
    """Dialog shown when a new annotation overlaps existing event(s).

    Gives two choices:
      - Replace: remove ONLY the overlapping event(s), then pick new label
      - Cancel: do nothing (use D key to set to baseline separately)
    """

    def __init__(self, parent, overlapping, start_s, end_s):
        self.overlapping = overlapping
        self.start_s = start_s
        self.end_s = end_s
        self.choice: str = "cancel"
        super().__init__(
            parent, title="Overlapping Event(s) Detected",
        )

    def body(self, master):
        tk.Label(
            master,
            text=f"Your selection ({self.start_s:.1f}s - "
                 f"{self.end_s:.1f}s) overlaps with:",
            font=("Arial", 10, "bold"),
        ).pack(pady=(8, 4))

        for ev in self.overlapping:
            color = EMOTION_METADATA.get(ev.emotion, {}).get(
                "color", "#999",
            )
            tk.Label(
                master,
                text=f"{ev.emotion} [{ev.intensity}]  "
                     f"({ev.start_s:.1f}s - {ev.end_s:.1f}s)",
                font=("Arial", 10),
                fg=color,
            ).pack(pady=1)

        tk.Label(
            master,
            text="Only this event will be removed.\n"
                 "All other events remain unchanged.",
            font=("Arial", 9, "italic"), fg="#666",
        ).pack(pady=(6, 2))

        return None

    def buttonbox(self):
        box = tk.Frame(self)
        tk.Button(
            box,
            text="Replace with New Event",
            width=24, bg="#E67E22", fg="white",
            font=("Arial", 10, "bold"),
            command=self._replace,
        ).pack(pady=4)
        tk.Button(
            box, text="Cancel", width=12,
            font=("Arial", 9),
            command=self.cancel,
        ).pack(pady=4)
        box.pack(padx=10, pady=5)

    def _replace(self):
        self.choice = "replace"
        self.ok()

    def apply(self):
        self.result = self.choice


def _overlap_dialog(root, overlapping, start_s, end_s) -> str:
    """Show the overlap dialog and return the user's choice.

    Returns "replace" or "cancel".
    """
    dlg = _OverlapDialog(root, overlapping, start_s, end_s)
    return dlg.choice


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE ANNOTATION UI
# ─────────────────────────────────────────────────────────────────────────────

class InteractiveAnnotationUI:
    """Interactive matplotlib-based annotation interface.

    Parameters
    ----------
    packet : PipelinePacket-like
        Must have .signals dict and .combined DataFrame.
    annotator : ManualAnnotator
        Annotation engine to add/edit/delete events.
    recording_start : datetime, optional
        If provided, a secondary x-axis shows the actual clock time.
    """

    def __init__(
        self,
        packet,
        annotator: ManualAnnotator,
        recording_start: Optional[datetime] = None,
        output_dir: Optional[str] = None,
    ):
        self.packet = packet
        self.annotator = annotator
        self.recording_start = recording_start
        self.output_dir = output_dir
        self.fig = None
        self.ax_timeline = None
        self.ax_clock = None  # secondary time axis
        self.signal_axes: list = []
        self.span_selectors: list = []
        self._undo_stack: list = []
        self._event_patches: dict = {}  # event_id -> list of patches
        self._last_click_x: Optional[float] = None  # for 'D' key delete

        # Drag-to-resize/move state
        self._drag_mode: Optional[str] = None   # "left_edge", "right_edge", "move"
        self._drag_event: Optional[AnnotationEvent] = None
        self._drag_start_x: Optional[float] = None
        self._drag_feedback_lines: list = []  # temporary visual feedback

    def run(self) -> ManualAnnotator:
        """Launch the interactive annotation window.

        Blocks until the user closes the window or presses Q.
        Returns the annotator with all events added.
        """
        signal_panels = self._get_signal_panels()
        n_panels = len(signal_panels)
        if n_panels == 0:
            print("  [WARN] No plottable signals found.")
            return self.annotator

        # Build figure
        height_ratios = (
            [TIMELINE_HEIGHT_RATIO] + [SIGNAL_HEIGHT_RATIO] * n_panels
        )
        self.fig = plt.figure(
            figsize=(18, 2.5 + 2.8 * n_panels),
            num="Module 2B -- Interactive Annotation Tool",
        )
        gs = gridspec.GridSpec(
            1 + n_panels, 1,
            height_ratios=height_ratios,
            hspace=0.08,
        )

        # X range
        all_ts = [ts for _, ts, _, _ in signal_panels]
        x_min = min(t.min() for t in all_ts)
        x_max = max(t.max() for t in all_ts)

        # Timeline bar
        self.ax_timeline = self.fig.add_subplot(gs[0])
        self._setup_timeline(x_min, x_max)

        # Signal panels
        self.signal_axes = []
        for i, (label, ts, values, color) in enumerate(signal_panels):
            ax = self.fig.add_subplot(gs[1 + i], sharex=self.ax_timeline)
            ax.plot(ts, values, color=color, linewidth=0.5, alpha=0.85)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)
            if i < n_panels - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            self.signal_axes.append(ax)

            # SpanSelector on each panel for click-drag event creation
            span = SpanSelector(
                ax, self._on_span_select, "horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="#3498DB"),
                interactive=True,
                drag_from_anywhere=True,
            )
            self.span_selectors.append(span)

        self.signal_axes[-1].set_xlabel("Time (s)", fontsize=10)

        # Add secondary clock-time axis if recording_start is available
        if self.recording_start is not None:
            self._add_clock_axis(x_min, x_max)

        # Instructions text
        self.fig.text(
            0.5, 0.005,
            "DRAG empty area to annotate  |  "
            "DRAG event edges to resize  |  "
            "DRAG event body to move  |  "
            "RIGHT-CLICK to edit  |  "
            "D = baseline  |  C = clear  |  "
            "Z = undo  |  Q = finish & save",
            ha="center", fontsize=7.5,
            style="italic", color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0",
                      edgecolor="#CCCCCC"),
        )

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Draw any pre-existing events
        for ev in self.annotator.events:
            self._draw_event(ev)

        self._update_title()

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

        return self.annotator

    # ── Event handlers ────────────────────────────────────────────────────

    def _on_span_select(self, xmin: float, xmax: float):
        """Called when user drags a span on a signal panel.

        If the region overlaps an existing event, asks to replace
        (removes ONLY that event, all others stay). Then shows the
        emotion picker for the new annotation.
        """
        if xmax - xmin < 1.0:
            return  # too small, likely accidental click

        start_s = round(xmin, 1)
        end_s = round(xmax, 1)
        duration_s = end_s - start_s
        root = self.fig.canvas.manager.window

        # Check for overlaps BEFORE showing emotion picker
        overlapping = self.annotator.find_overlapping(start_s, end_s)
        if overlapping:
            choice = _overlap_dialog(
                root, overlapping, start_s, end_s,
            )
            if choice != "replace":
                return  # user cancelled

        # Show emotion picker dialog
        dialog = EmotionPickerDialog(root, start_s, end_s)

        if dialog.selected_emotion is None:
            return

        emotion = dialog.selected_emotion
        intensity = dialog.selected_intensity

        # Save undo state
        self._push_undo()

        # Remove ONLY the overlapping events (all others untouched)
        if overlapping:
            for ev in overlapping:
                print(f"  [BASELINE] {ev.emotion} [{ev.intensity}] "
                      f"@ {ev.start_s:.1f}-{ev.end_s:.1f}s "
                      f"-> replaced")
                self.annotator.delete_event(ev.event_id)

        # Add new event
        result = self.annotator.add_event(
            emotion, start_s, duration_s, intensity=intensity,
        )

        if result.is_valid:
            print(f"  [ADDED] {emotion} [{intensity}] "
                  f"@ {start_s:.1f}-{end_s:.1f}s")
            self._redraw_all_events()
            self._update_title()
            self.fig.canvas.draw_idle()
            self._save_csvs("add")

            if result.warnings:
                messagebox.showwarning(
                    "Warning",
                    "\n".join(result.warnings),
                    parent=root,
                )
        else:
            messagebox.showerror(
                "Cannot Add Event",
                "\n".join(result.errors),
                parent=root,
            )

    def _on_click(self, event):
        """Handle mouse clicks.

        Left-click near event edge  → start drag-to-resize
        Left-click inside event     → start drag-to-move
        Right-click on event        → open edit/delete dialog
        """
        # Track cursor position for keyboard shortcuts
        if event.inaxes is not None and event.xdata is not None:
            self._last_click_x = event.xdata

        if event.inaxes is None or event.xdata is None:
            return

        click_x = event.xdata

        # ── Left-click: check for drag-to-resize/move ───────────────
        if event.button == 1:
            x_min, x_max = self.signal_axes[0].get_xlim()
            threshold = (x_max - x_min) * EDGE_GRAB_FRACTION

            for ev in self.annotator.events:
                # Near left edge?
                if abs(click_x - ev.start_s) < threshold:
                    self._start_drag("left_edge", ev, click_x)
                    return
                # Near right edge?
                if abs(click_x - ev.end_s) < threshold:
                    self._start_drag("right_edge", ev, click_x)
                    return
                # Inside event body? (for move)
                if (ev.start_s + threshold) <= click_x <= (ev.end_s - threshold):
                    self._start_drag("move", ev, click_x)
                    return
            # Not on any event edge/body → let SpanSelector handle it
            return

        # ── Right-click: edit/delete dialog ──────────────────────────
        if event.button != 3:
            return

        # Find which event was clicked
        clicked_event = None
        for ev in self.annotator.events:
            if ev.start_s <= click_x <= ev.end_s:
                clicked_event = ev
                break

        if clicked_event is None:
            return

        root = self.fig.canvas.manager.window
        dialog = EventEditDialog(root, clicked_event)

        if dialog.action == "delete":
            self._push_undo()
            self._clear_event_patches(clicked_event.event_id)
            self.annotator.delete_event(clicked_event.event_id)
            self._redraw_all_events()
            self._update_title()
            self.fig.canvas.draw_idle()
            print(f"  [BASELINE] {clicked_event.emotion} "
                  f"[{clicked_event.intensity}] "
                  f"@ {clicked_event.start_s:.1f}-"
                  f"{clicked_event.end_s:.1f}s -> set to baseline")
            self._save_csvs("baseline")

        elif dialog.action == "edit":
            self._push_undo()
            result = self.annotator.edit_event(
                clicked_event.event_id,
                emotion=dialog.new_emotion,
                start_s=dialog.new_start,
                duration_s=dialog.new_duration,
                intensity=dialog.new_intensity,
            )
            if result.is_valid:
                self._redraw_all_events()
                self._update_title()
                self.fig.canvas.draw_idle()
                print(f"  [EDITED] Event {clicked_event.event_id} updated")
                self._save_csvs("edit")
            else:
                messagebox.showerror(
                    "Edit Failed",
                    "\n".join(result.errors),
                    parent=root,
                )

    # ── Drag-to-resize/move ──────────────────────────────────────────────

    def _start_drag(self, mode: str, ev: AnnotationEvent, click_x: float):
        """Begin a drag operation, disabling SpanSelector temporarily."""
        self._drag_mode = mode
        self._drag_event = ev
        self._drag_start_x = click_x

        # Disable SpanSelector so it doesn't conflict
        for ss in self.span_selectors:
            ss.set_active(False)

        # Change cursor to indicate drag mode
        cursor = "sb_h_double_arrow" if mode != "move" else "fleur"
        try:
            self.fig.canvas.get_tk_widget().config(cursor=cursor)
        except (AttributeError, tk.TclError):
            pass

    def _on_motion(self, event):
        """Update visual feedback during drag operations."""
        if self._drag_mode is None or event.xdata is None:
            # When not dragging, update cursor on hover near event edges
            if event.inaxes is not None and event.xdata is not None:
                self._update_cursor_on_hover(event.xdata)
            return

        ev = self._drag_event
        dx = event.xdata - self._drag_start_x

        # Compute preview boundaries
        if self._drag_mode == "left_edge":
            new_start = max(0, ev.start_s + dx)
            new_end = ev.end_s
            if new_start >= new_end - 1.0:
                return  # minimum 1s duration
        elif self._drag_mode == "right_edge":
            new_start = ev.start_s
            new_end = min(self.annotator.duration_s, ev.end_s + dx)
            if new_end <= new_start + 1.0:
                return
        elif self._drag_mode == "move":
            dur = ev.duration_s
            new_start = max(0, ev.start_s + dx)
            new_end = new_start + dur
            if new_end > self.annotator.duration_s:
                new_end = self.annotator.duration_s
                new_start = new_end - dur
        else:
            return

        # Clear previous feedback lines
        self._clear_drag_feedback()

        # Draw temporary feedback lines on all axes
        feedback = []
        all_axes = [self.ax_timeline] + self.signal_axes
        for ax in all_axes:
            l1 = ax.axvline(new_start, color="#E74C3C", linewidth=2,
                            linestyle=":", alpha=0.8)
            l2 = ax.axvline(new_end, color="#E74C3C", linewidth=2,
                            linestyle=":", alpha=0.8)
            feedback.extend([l1, l2])
        self._drag_feedback_lines = feedback
        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        """Finalize drag operation and update the event."""
        if self._drag_mode is None:
            return

        # Re-enable SpanSelector
        for ss in self.span_selectors:
            ss.set_active(True)

        # Reset cursor
        try:
            self.fig.canvas.get_tk_widget().config(cursor="")
        except (AttributeError, tk.TclError):
            pass

        # Clear feedback lines
        self._clear_drag_feedback()

        if event.xdata is None:
            self._drag_mode = None
            self._drag_event = None
            self.fig.canvas.draw_idle()
            return

        ev = self._drag_event
        dx = event.xdata - self._drag_start_x

        # Skip if barely moved (accidental)
        if abs(dx) < 0.5:
            self._drag_mode = None
            self._drag_event = None
            self.fig.canvas.draw_idle()
            return

        # Compute final boundaries
        if self._drag_mode == "left_edge":
            new_start = round(max(0, ev.start_s + dx), 1)
            new_dur = round(ev.end_s - new_start, 1)
            if new_dur < 1.0:
                new_start = round(ev.end_s - 1.0, 1)
                new_dur = 1.0
        elif self._drag_mode == "right_edge":
            new_start = ev.start_s
            new_end = round(
                min(self.annotator.duration_s, ev.end_s + dx), 1,
            )
            new_dur = round(new_end - new_start, 1)
            if new_dur < 1.0:
                new_dur = 1.0
        elif self._drag_mode == "move":
            dur = ev.duration_s
            new_start = round(max(0, ev.start_s + dx), 1)
            if new_start + dur > self.annotator.duration_s:
                new_start = round(self.annotator.duration_s - dur, 1)
            new_dur = dur
        else:
            self._drag_mode = None
            self._drag_event = None
            return

        # Apply the edit
        self._push_undo()
        old_desc = (f"{ev.emotion} [{ev.intensity}] "
                    f"@ {ev.start_s:.1f}-{ev.end_s:.1f}s")
        result = self.annotator.edit_event(
            ev.event_id, start_s=new_start, duration_s=new_dur,
        )

        mode = self._drag_mode
        self._drag_mode = None
        self._drag_event = None

        if result.is_valid:
            self._redraw_all_events()
            self._update_title()
            self.fig.canvas.draw_idle()
            action = "RESIZED" if mode != "move" else "MOVED"
            new_end = new_start + new_dur
            print(f"  [{action}] {old_desc} -> "
                  f"{new_start:.1f}-{new_end:.1f}s")
            self._save_csvs("drag")
        else:
            # Revert undo on failure
            self._pop_undo()
            root = self.fig.canvas.manager.window
            messagebox.showerror(
                "Drag Failed",
                "\n".join(result.errors),
                parent=root,
            )

    def _clear_drag_feedback(self):
        """Remove temporary drag feedback lines."""
        for line in self._drag_feedback_lines:
            try:
                line.remove()
            except ValueError:
                pass
        self._drag_feedback_lines.clear()

    def _update_cursor_on_hover(self, x: float):
        """Change cursor when hovering near event edges."""
        x_min, x_max = self.signal_axes[0].get_xlim()
        threshold = (x_max - x_min) * EDGE_GRAB_FRACTION

        cursor = ""
        for ev in self.annotator.events:
            if (abs(x - ev.start_s) < threshold
                    or abs(x - ev.end_s) < threshold):
                cursor = "sb_h_double_arrow"
                break
            if ev.start_s + threshold <= x <= ev.end_s - threshold:
                cursor = "fleur"
                break

        try:
            self.fig.canvas.get_tk_widget().config(cursor=cursor)
        except (AttributeError, tk.TclError):
            pass

    def _on_key(self, event):
        """Handle keyboard shortcuts.

        Q = finish and save | Z = undo | D = delete event at cursor
        C = clear all events
        """
        if event.key == "q":
            self._save_csvs("final")
            plt.close(self.fig)

        elif event.key == "z":
            self._pop_undo()
            self._save_csvs("undo")

        elif event.key == "d":
            # Delete the event under the cursor
            if self._last_click_x is None and event.xdata is not None:
                self._last_click_x = event.xdata
            if self._last_click_x is None:
                return
            target_ev = None
            for ev in self.annotator.events:
                if ev.start_s <= self._last_click_x <= ev.end_s:
                    target_ev = ev
                    break
            if target_ev is None:
                return
            self._push_undo()
            self._clear_event_patches(target_ev.event_id)
            self.annotator.delete_event(target_ev.event_id)
            self._redraw_all_events()
            self._update_title()
            self.fig.canvas.draw_idle()
            print(f"  [BASELINE] {target_ev.emotion} "
                  f"[{target_ev.intensity}] "
                  f"@ {target_ev.start_s:.1f}-{target_ev.end_s:.1f}s "
                  f"-> set to baseline")
            self._save_csvs("baseline")

        elif event.key == "c":
            # Clear all events
            if not self.annotator.events:
                return
            root = self.fig.canvas.manager.window
            ok = messagebox.askyesno(
                "Clear All Events",
                f"Delete all {len(self.annotator.events)} events?\n"
                "This can be undone with Z.",
                parent=root,
            )
            if not ok:
                return
            self._push_undo()
            self.annotator.clear_all_events()
            self._redraw_all_events()
            self._update_title()
            self.fig.canvas.draw_idle()
            print("  [CLEARED] All events removed")
            self._save_csvs("clear")

    # ── Drawing helpers ───────────────────────────────────────────────────

    def _setup_timeline(self, x_min: float, x_max: float):
        """Configure the timeline bar axis."""
        ax = self.ax_timeline
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.axhspan(0, 1, color=BASELINE_COLOR, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    def _add_clock_axis(self, x_min: float, x_max: float):
        """Add a secondary x-axis at the top showing actual date/time."""
        ax_top = self.ax_timeline.twiny()
        self.ax_clock = ax_top

        # Convert seconds range to datetime range
        dt_start = self.recording_start + timedelta(seconds=x_min)
        dt_end = self.recording_start + timedelta(seconds=x_max)

        ax_top.set_xlim(mdates.date2num(dt_start), mdates.date2num(dt_end))

        # Choose appropriate tick format based on duration
        duration = x_max - x_min
        if duration < 600:  # < 10 min
            ax_top.xaxis.set_major_formatter(
                mdates.DateFormatter("%H:%M:%S")
            )
        elif duration < 7200:  # < 2 hours
            ax_top.xaxis.set_major_formatter(
                mdates.DateFormatter("%H:%M")
            )
        else:
            ax_top.xaxis.set_major_formatter(
                mdates.DateFormatter("%H:%M\n%d %b")
            )

        ax_top.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax_top.tick_params(labelsize=8, pad=1)
        ax_top.set_xlabel(
            f"Clock Time (start: {dt_start:%Y-%m-%d %H:%M:%S})",
            fontsize=9,
        )

    def _draw_event(self, ev: AnnotationEvent):
        """Draw a single event on the timeline and all signal panels."""
        patches = []

        # Timeline bar
        rect = self.ax_timeline.barh(
            0.5, ev.duration_s, left=ev.start_s, height=0.9,
            color=ev.color, alpha=0.75, edgecolor="white",
            linewidth=0.5, align="center",
        )
        patches.append(rect)

        # Timeline label (show emotion + intensity initial)
        mid = ev.start_s + ev.duration_s / 2
        int_char = ev.intensity[0] if ev.intensity else ""
        label_text = f"{ev.emotion} [{int_char}]"
        txt = self.ax_timeline.text(
            mid, 0.5, label_text,
            ha="center", va="center",
            fontsize=7, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.12", facecolor=ev.color,
                      alpha=0.85, edgecolor="none"),
        )
        patches.append(txt)

        # Borders on timeline
        for x in (ev.start_s, ev.end_s):
            line = self.ax_timeline.axvline(
                x, color=ev.color, linewidth=EVENT_BORDER_LW, alpha=0.9,
            )
            patches.append(line)

        # Shade on all signal panels
        for ax in self.signal_axes:
            span = ax.axvspan(
                ev.start_s, ev.end_s,
                color=ev.color, alpha=EVENT_SHADE_ALPHA,
            )
            patches.append(span)
            for x in (ev.start_s, ev.end_s):
                line = ax.axvline(
                    x, color=ev.color, linewidth=EVENT_BORDER_LW,
                    alpha=0.5, linestyle="--",
                )
                patches.append(line)

        self._event_patches[ev.event_id] = patches

    def _clear_event_patches(self, event_id: int):
        """Remove drawn patches for an event."""
        patches = self._event_patches.pop(event_id, [])
        for p in patches:
            try:
                p.remove()
            except (ValueError, AttributeError):
                # BarContainer needs special handling
                try:
                    for bar in p:
                        bar.remove()
                except (TypeError, ValueError):
                    pass

    def _redraw_all_events(self):
        """Clear and redraw all event patches."""
        for eid in list(self._event_patches.keys()):
            self._clear_event_patches(eid)

        # Reset timeline
        self.ax_timeline.clear()
        x_min, x_max = self.signal_axes[0].get_xlim()
        self._setup_timeline(x_min, x_max)

        # Clear signal panel event artists (spans/vlines) but keep signal
        for ax in self.signal_axes:
            # Remove only patches and extra lines, keep the signal Line2D
            to_remove = []
            for artist in ax.patches:
                to_remove.append(artist)
            for artist in ax.lines[1:]:  # skip first line (the signal)
                to_remove.append(artist)
            for artist in to_remove:
                try:
                    artist.remove()
                except ValueError:
                    pass

        self._event_patches.clear()
        for ev in self.annotator.events:
            self._draw_event(ev)

    def _update_title(self):
        """Update the figure title with event count and density."""
        n = len(self.annotator.events)
        density = self.annotator.validator.compute_annotation_density(
            self.annotator.events, self.annotator.duration_s
        )
        self.fig.suptitle(
            f"Interactive Annotation  |  "
            f"{n} event{'s' if n != 1 else ''}  |  "
            f"{density:.0%} coverage  |  "
            f"Recording: {self.annotator.duration_s:.0f}s",
            fontsize=12, fontweight="bold",
        )

    # ── Undo ──────────────────────────────────────────────────────────────

    def _push_undo(self):
        """Save current event state for undo."""
        snapshot = [
            AnnotationEvent(
                event_id=ev.event_id,
                emotion=ev.emotion,
                start_s=ev.start_s,
                duration_s=ev.duration_s,
                intensity=ev.intensity,
            )
            for ev in self.annotator.events
        ]
        self._undo_stack.append(snapshot)
        # Limit stack size
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    def _pop_undo(self):
        """Restore previous event state."""
        if not self._undo_stack:
            return
        snapshot = self._undo_stack.pop()
        self.annotator.events = snapshot
        self.annotator._next_id = (
            max(ev.event_id for ev in snapshot) + 1 if snapshot else 1
        )
        self._redraw_all_events()
        self._update_title()
        self.fig.canvas.draw_idle()

    # ── Live CSV saving ────────────────────────────────────────────────────

    def _save_csvs(self, action: str):
        """Save annotation CSVs and update signal CSVs after every change.

        Parameters
        ----------
        action : str
            Label for the log message (add/edit/delete/clear/undo/final).
        """
        from pathlib import Path

        if self.output_dir is None:
            return

        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        n_events = len(self.annotator.events)

        # 1. Save annotations_events.csv (lightweight, always up-to-date)
        events_df = self.annotator.to_event_dataframe()
        events_path = out / "annotations_events.csv"
        if not events_df.empty:
            events_df.to_csv(events_path, index=False, float_format="%.6f")
        else:
            # Write empty file with headers to indicate clearing
            import pandas as pd
            pd.DataFrame(columns=[
                "event_id", "emotion", "category", "intensity",
                "reactivity", "valence", "arousal", "start_s", "end_s",
                "duration_s", "color_hex", "description",
            ]).to_csv(events_path, index=False)

        # 2. Save annotations_sample_labels.csv
        labels_df = self.annotator.to_sample_labels_dataframe()
        labels_path = out / "annotations_sample_labels.csv"
        if not labels_df.empty:
            labels_df.to_csv(labels_path, index=False, float_format="%.6f")

        # 3. Apply annotations to packet and save signal CSVs
        self.annotator.apply_to_packet()

        for name, df in self.packet.signals.items():
            sig_path = out / f"{name}.csv"
            df.to_csv(sig_path, index=False, float_format="%.6f")

        if (self.packet.combined is not None
                and len(self.packet.combined) > 0):
            combined_path = out / "combined_signals.csv"
            self.packet.combined.to_csv(
                combined_path, index=False, float_format="%.6f",
            )

        # Confirmation message
        print(f"  [CSV SAVED] {action.upper()} -- "
              f"{n_events} event(s) -> {out.name}/")

    # ── Signal extraction ─────────────────────────────────────────────────

    def _get_signal_panels(self) -> list:
        """Build (label, timestamps, values, color) for each signal."""
        panels = []
        signals = self.packet.signals

        for sig_name, val_col in SIGNAL_VALUE_COLS.items():
            if sig_name in signals:
                df = signals[sig_name]
                if "timestamp_s" in df.columns and val_col in df.columns:
                    panels.append((
                        SIGNAL_LABELS.get(sig_name, sig_name),
                        df["timestamp_s"].values,
                        df[val_col].values,
                        SIGNAL_COLORS.get(sig_name, "#333"),
                    ))

        if "ACC" in signals:
            df = signals["ACC"]
            if "timestamp_s" in df.columns:
                ts = df["timestamp_s"].values
                for axis in ("ACC_X_g", "ACC_Y_g", "ACC_Z_g"):
                    if axis in df.columns:
                        key = axis.replace("_g", "")
                        panels.append((
                            SIGNAL_LABELS.get(key, axis),
                            ts, df[axis].values,
                            SIGNAL_COLORS.get(key, "#333"),
                        ))

        return panels


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive_annotation(
    packet,
    annotator: ManualAnnotator,
    recording_start: Optional[datetime] = None,
    output_dir: Optional[str] = None,
) -> ManualAnnotator:
    """Launch the interactive annotation UI.

    Parameters
    ----------
    packet : PipelinePacket-like
        Signal data to annotate.
    annotator : ManualAnnotator
        Annotator to add events to.
    recording_start : datetime, optional
        Recording start time for clock-time display on the timeline.
    output_dir : str or Path, optional
        Directory for live CSV saving. If provided, CSVs are updated
        after every add/edit/delete with a confirmation message.

    Returns
    -------
    The same annotator, with events added by the user.
    """
    ui = InteractiveAnnotationUI(
        packet, annotator, recording_start, output_dir,
    )
    return ui.run()
