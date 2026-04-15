"""
=============================================================================
MODULE 2B -- DATA ANNOTATION  |  visualizer.py
=============================================================================
Static matplotlib visualisations for the annotation module.

Two plot types:
  1. overview_plot()   -- raw unannotated signals for inspection
  2. annotated_plot()  -- signals with coloured event spans overlaid

Both follow the M2A visualizer.py pattern: GridSpec layout, event
timeline bar at top, axvspan fills, boxed emotion labels.

Author  : Autism Physio-AI Pipeline
Version : 1.0.0
=============================================================================
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from config import EMOTION_METADATA


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EVENT_SHADE_ALPHA = 0.25
EVENT_BORDER_LW = 1.5
LABEL_FONTSIZE = 8.5
TIMELINE_HEIGHT_RATIO = 1
SIGNAL_HEIGHT_RATIO = 4

SIGNAL_COLORS: dict = {
    "EDA": "#27AE60",
    "BVP": "#C0392B",
    "IBI": "#8E44AD",
    "ST": "#2980B9",
    "ACC_X": "#D35400",
    "ACC_Y": "#F39C12",
    "ACC_Z": "#16A085",
}

SIGNAL_LABELS: dict = {
    "EDA": "EDA (uS)",
    "BVP": "BVP (nT)",
    "IBI": "IBI (ms)",
    "ST": "ST (degC)",
    "ACC_X": "ACC X (g)",
    "ACC_Y": "ACC Y (g)",
    "ACC_Z": "ACC Z (g)",
}

SIGNAL_VALUE_COLS: dict = {
    "EDA": "EDA_uS",
    "BVP": "BVP_nT",
    "IBI": "IBI_ms",
    "ST": "ST_degC",
}

BASELINE_COLOR = "#E0E0E0"


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZER
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationVisualizer:
    """Static matplotlib visualisation for annotation review.

    Parameters
    ----------
    packet : PipelinePacket-like
        Must have .signals dict and .combined DataFrame.
    """

    def __init__(self, packet):
        self.packet = packet

    # ── Public API ────────────────────────────────────────────────────────

    def plot_overview(self, output_path: Path) -> Path:
        """Plot raw unannotated signals for inspection.

        Parameters
        ----------
        output_path : Path
            Full path for the output PNG.

        Returns
        -------
        Path to the saved figure.
        """
        signal_panels = self._get_signal_panels()
        n_panels = len(signal_panels)
        if n_panels == 0:
            return output_path

        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(16, 3.0 * n_panels),
            sharex=True,
        )
        if n_panels == 1:
            axes = [axes]

        for ax, (label, ts, values, color) in zip(axes, signal_panels):
            ax.plot(ts, values, color=color, linewidth=0.5, alpha=0.85)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)", fontsize=10)
        fig.suptitle("Signal Overview (unannotated)", fontsize=13, y=0.995)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def plot_annotated(
        self,
        events: list,
        output_path: Path,
    ) -> Path:
        """Plot signals with annotation events overlaid.

        Parameters
        ----------
        events : list[AnnotationEvent]
            Events to visualise.
        output_path : Path
            Full path for the output PNG.

        Returns
        -------
        Path to the saved figure.
        """
        signal_panels = self._get_signal_panels()
        n_panels = len(signal_panels)
        if n_panels == 0:
            return output_path

        # Layout: 1 timeline bar + n signal panels
        height_ratios = [TIMELINE_HEIGHT_RATIO] + [SIGNAL_HEIGHT_RATIO] * n_panels
        fig = plt.figure(figsize=(18, 2.5 + 3.0 * n_panels))
        gs = gridspec.GridSpec(
            1 + n_panels, 1,
            height_ratios=height_ratios,
            hspace=0.08,
        )

        # Detect x range
        all_ts = [ts for _, ts, _, _ in signal_panels]
        x_min = min(t.min() for t in all_ts) if all_ts else 0
        x_max = max(t.max() for t in all_ts) if all_ts else 1

        # Timeline bar
        ax_timeline = fig.add_subplot(gs[0])
        self._draw_timeline_bar(ax_timeline, events, x_min, x_max)

        # Signal panels
        for i, (label, ts, values, color) in enumerate(signal_panels):
            ax = fig.add_subplot(gs[1 + i], sharex=ax_timeline)
            ax.plot(ts, values, color=color, linewidth=0.5, alpha=0.85)
            self._shade_events(ax, events)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(True, alpha=0.3)
            if i < n_panels - 1:
                plt.setp(ax.get_xticklabels(), visible=False)

        # X-axis label on last panel
        fig.axes[-1].set_xlabel("Time (s)", fontsize=10)

        fig.suptitle("Annotated Signals", fontsize=13, y=0.995)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_signal_panels(self) -> list:
        """Build list of (label, timestamps, values, color) for plotting.

        Extracts plottable signal arrays from the packet.
        """
        panels = []
        signals = self.packet.signals

        # EDA, BVP, IBI, ST
        for sig_name, val_col in SIGNAL_VALUE_COLS.items():
            if sig_name in signals:
                df = signals[sig_name]
                if "timestamp_s" in df.columns and val_col in df.columns:
                    ts = df["timestamp_s"].values
                    vals = df[val_col].values
                    panels.append((
                        SIGNAL_LABELS.get(sig_name, sig_name),
                        ts, vals,
                        SIGNAL_COLORS.get(sig_name, "#333333"),
                    ))

        # ACC — 3 axes
        if "ACC" in signals:
            df = signals["ACC"]
            if "timestamp_s" in df.columns:
                ts = df["timestamp_s"].values
                for axis in ("ACC_X_g", "ACC_Y_g", "ACC_Z_g"):
                    if axis in df.columns:
                        key = axis.replace("_g", "").replace("ACC_", "ACC_")
                        panels.append((
                            SIGNAL_LABELS.get(
                                axis.replace("_g", ""), axis
                            ),
                            ts, df[axis].values,
                            SIGNAL_COLORS.get(
                                axis.replace("_g", ""), "#333333"
                            ),
                        ))

        return panels

    def _draw_timeline_bar(self, ax, events, x_min, x_max):
        """Draw the event timeline bar at the top of the figure."""
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title("Event Timeline", fontsize=10, loc="left", pad=4)

        # Grey baseline background
        ax.axhspan(0, 1, color=BASELINE_COLOR, alpha=0.5)

        sorted_events = sorted(events, key=lambda e: e.start_s)

        for ev in sorted_events:
            color = ev.color
            width = ev.end_s - ev.start_s

            # Colored event block
            ax.barh(
                0.5, width, left=ev.start_s, height=0.9,
                color=color, alpha=0.75, edgecolor="white", linewidth=0.5,
                align="center",
            )

            # Vertical boundaries
            ax.axvline(ev.start_s, color=color, linewidth=EVENT_BORDER_LW,
                       alpha=0.9)
            ax.axvline(ev.end_s, color=color, linewidth=EVENT_BORDER_LW,
                       alpha=0.9)

            # Centred emotion label
            mid = ev.start_s + width / 2
            ax.text(
                mid, 0.5, ev.emotion,
                ha="center", va="center",
                fontsize=LABEL_FONTSIZE, fontweight="bold",
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor=color, alpha=0.85, edgecolor="none",
                ),
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    def _shade_events(self, ax, events):
        """Shade event regions on a signal panel."""
        for ev in events:
            color = ev.color
            ax.axvspan(
                ev.start_s, ev.end_s,
                color=color, alpha=EVENT_SHADE_ALPHA,
            )
            ax.axvline(ev.start_s, color=color, linewidth=EVENT_BORDER_LW,
                       alpha=0.6, linestyle="--")
            ax.axvline(ev.end_s, color=color, linewidth=EVENT_BORDER_LW,
                       alpha=0.6, linestyle="--")
