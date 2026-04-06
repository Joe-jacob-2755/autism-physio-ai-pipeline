"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  visualizer.py
=============================================================================
Visualization engine.

Produces:
  1. Individual signal plots (one figure per signal, saved to output_dir)
  2. Combined multi-panel figure (all signals in one figure)

Both plot types include:
  • Shaded event windows (colour-coded by emotion)
  • Event labels at the top of each axis
  • Annotation overlays showing baseline / event periods
  • Clear axis labels with units

=============================================================================
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    SAMPLING_RATES, SIGNAL_UNITS, EMOTION_PROFILES, EMOTIONS_ALL,
)
from event_scheduler import EventConfig
from simulator import SimulationResult


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR / STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "EDA":   "#2ECC71",
    "BVP":   "#E74C3C",
    "ST":    "#F39C12",
    "ACC_X": "#3498DB",
    "ACC_Y": "#9B59B6",
    "ACC_Z": "#1ABC9C",
}

SIGNAL_LABELS = {
    "EDA":   "EDA (µS)",
    "BVP":   "BVP (nT)",
    "ST":    "ST (°C)",
    "ACC_X": "ACC X (g)",
    "ACC_Y": "ACC Y (g)",
    "ACC_Z": "ACC Z (g)",
}

FIGSIZE_INDIVIDUAL  = (14, 4)
FIGSIZE_COMBINED    = (16, 22)
PLOT_STYLE          = "seaborn-v0_8-whitegrid"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER – DRAW EVENT SHADING ON AN AXES
# ─────────────────────────────────────────────────────────────────────────────

def _shade_events(ax: plt.Axes, events: List[EventConfig], ymin: float, ymax: float):
    """Draw translucent event shading and emotion labels on an axes."""
    for ev in events:
        color = ev.color
        ax.axvspan(
            ev.start_s, ev.end_s,
            alpha     = 0.18,
            color     = color,
            linewidth = 0,
        )
        # Vertical dashed borders
        for t in (ev.start_s, ev.end_s):
            ax.axvline(t, color=color, linewidth=0.8, linestyle="--", alpha=0.6)

        # Emotion label at top
        mid = (ev.start_s + ev.end_s) / 2.0
        ax.text(
            mid, ymax,
            ev.emotion,
            ha="center", va="bottom",
            fontsize=7.5, color=color,
            fontweight="bold",
            clip_on=True,
            rotation=0,
        )


def _build_legend(events: List[EventConfig]) -> List[mpatches.Patch]:
    """Build deduplicated legend patches for all emotions present."""
    seen   = {}
    for ev in events:
        if ev.emotion not in seen:
            seen[ev.emotion] = mpatches.Patch(
                facecolor=ev.color, alpha=0.5, label=ev.emotion
            )
    return list(seen.values())


def _format_axes(
    ax: plt.Axes,
    t: np.ndarray,
    sig: np.ndarray,
    label: str,
    color: str,
    events: List[EventConfig],
    downsample_factor: int = 1,
):
    """Plot signal on axes with event shading."""
    # Downsample for display if very high rate
    if downsample_factor > 1:
        t_d   = t[::downsample_factor]
        sig_d = sig[::downsample_factor]
    else:
        t_d, sig_d = t, sig

    ax.plot(t_d, sig_d, color=color, linewidth=0.7, alpha=0.9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_xlim(0, t[-1])

    margin = 0.08 * (sig_d.max() - sig_d.min() + 1e-6)
    ymin   = sig_d.min() - margin
    ymax   = sig_d.max() + margin * 3   # extra headroom for labels
    ax.set_ylim(ymin, ymax)

    _shade_events(ax, events, ymin, ymax)
    ax.tick_params(labelsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VISUALIZER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SignalVisualizer:
    """
    Generate publication-quality plots from a SimulationResult.

    Parameters
    ----------
    result     : SimulationResult from DataSimulator.simulate()
    output_dir : directory where figures are saved (created if not exists)
    dpi        : figure DPI for saved images
    """

    def __init__(
        self,
        result:     SimulationResult,
        output_dir: str  = "output",
        dpi:        int  = 150,
    ):
        self.result     = result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi        = dpi
        self.events     = result.events

        try:
            plt.style.use(PLOT_STYLE)
        except OSError:
            plt.style.use("ggplot")

    # ── Individual signal figures ─────────────────────────────────────────

    def plot_individual_signals(self) -> Dict[str, Path]:
        """
        Generate one figure per signal channel.

        Returns
        -------
        dict mapping signal name → saved file path.
        """
        saved = {}
        signals_to_plot = self._get_plottable_signals()

        for sig_name, (t, sig) in signals_to_plot.items():
            fig, ax = plt.subplots(figsize=FIGSIZE_INDIVIDUAL)

            color = SIGNAL_COLORS.get(sig_name, "#555555")
            label = SIGNAL_LABELS.get(sig_name, sig_name)
            ds    = self._downsample_factor(sig_name)
            _format_axes(ax, t, sig, label, color, self.events, ds)

            # IBI scatter overlay on BVP plot (if this is BVP)
            if sig_name == "BVP":
                self._overlay_ibi_on_bvp(ax)

            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_title(
                f"Simulated {sig_name} Signal  "
                f"(fs={SAMPLING_RATES.get(sig_name, 'event')} Hz, "
                f"dur={self.result.duration_s:.0f}s)",
                fontsize=11, fontweight="bold",
            )

            # Legend
            patches = _build_legend(self.events)
            if patches:
                ax.legend(
                    handles=patches, loc="upper right",
                    fontsize=7, framealpha=0.85,
                    ncol=min(len(patches), 5),
                )

            fig.tight_layout()
            out_path = self.output_dir / f"signal_{sig_name}.png"
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            saved[sig_name] = out_path
            print(f"  [Visualizer] Saved {out_path.name}")

        # IBI standalone
        ibi_path = self._plot_ibi_standalone()
        saved["IBI"] = ibi_path

        return saved

    # ── Combined multi-panel figure ──────────────────────────────────────

    def plot_combined(self) -> Path:
        """
        Generate one combined figure with all 7 panels (EDA, BVP, IBI, ST, ACC×3).

        Returns
        -------
        Path to saved file.
        """
        fig = plt.figure(figsize=FIGSIZE_COMBINED)
        fig.suptitle(
            "Module 1A – Simulated Physiological Signals\n"
            f"Duration: {self.result.duration_s:.0f}s  |  "
            f"Events: {len(self.events)}  |  "
            f"Noise: {self.result.metadata.get('noise_level','?')}  |  "
            f"Seed: {self.result.metadata.get('seed','?')}",
            fontsize=13, fontweight="bold", y=0.995,
        )

        # 7 rows: EDA, BVP, IBI, ST, ACC_X, ACC_Y, ACC_Z
        # Heights: emphasise BVP and ACC slightly
        heights = [2, 2.5, 1.8, 2, 2, 2, 2]
        gs = GridSpec(7, 1, figure=fig, hspace=0.42, height_ratios=heights)
        axes = [fig.add_subplot(gs[i]) for i in range(7)]

        order     = ["EDA", "BVP", "IBI", "ST", "ACC_X", "ACC_Y", "ACC_Z"]
        signals   = self._get_plottable_signals()
        ibi_t, ibi_v = self.result.ibi_times_s, self.result.ibi_values_ms

        for idx, sig_name in enumerate(order):
            ax = axes[idx]
            color = SIGNAL_COLORS.get(sig_name, "#555555")
            label = SIGNAL_LABELS.get(sig_name, sig_name)

            if sig_name == "IBI":
                # Scatter plot
                mask = (ibi_t >= 0) & (ibi_t <= self.result.duration_s)
                ax.scatter(ibi_t[mask], ibi_v[mask], s=4, color="#E74C3C", alpha=0.6)
                ax.set_ylabel("IBI (ms)", fontsize=9)
                ax.set_xlim(0, self.result.duration_s)
                margin = 0.1 * (np.ptp(ibi_v[mask]) + 1e-6)
                ymin = ibi_v[mask].min() - margin
                ymax = ibi_v[mask].max() + margin * 3
                ax.set_ylim(ymin, ymax)
                _shade_events(ax, self.events, ymin, ymax)
                ax.invert_yaxis()   # shorter IBI = faster HR → intuitive direction
                ax.set_title("IBI – Inter-Beat Interval  (inverted: ↑ = faster HR)",
                             fontsize=8, loc="left", pad=3)
            else:
                if sig_name not in signals:
                    ax.set_visible(False)
                    continue
                t, sig = signals[sig_name]
                ds    = self._downsample_factor(sig_name)
                _format_axes(ax, t, sig, label, color, self.events, ds)
                ax.set_title(sig_name, fontsize=8, loc="left", pad=3)

                if sig_name == "BVP":
                    self._overlay_ibi_on_bvp(ax)

            # X-label only on bottom panel
            if idx == 6:
                ax.set_xlabel("Time (s)", fontsize=9)
            else:
                ax.set_xticklabels([])

        # Global legend at bottom
        patches = _build_legend(self.events)
        if patches:
            fig.legend(
                handles  = patches,
                loc      = "lower center",
                ncol     = min(len(patches), 6),
                fontsize = 8,
                framealpha = 0.9,
                title    = "Emotion / Behaviour Events",
                title_fontsize = 9,
                bbox_to_anchor = (0.5, -0.005),
            )

        out_path = self.output_dir / "combined_signals.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualizer] Saved {out_path.name}")
        return out_path

    # ── ACC combined plot ─────────────────────────────────────────────────

    def plot_acc_combined(self) -> Path:
        """Three ACC axes in one figure for easy comparison."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        fig.suptitle("Accelerometer (3-axis)", fontsize=11, fontweight="bold")

        for ax, ax_name in zip(axes, ["ACC_X", "ACC_Y", "ACC_Z"]):
            t   = self.result.time_vectors[ax_name]
            sig = self.result.signals[ax_name]
            color = SIGNAL_COLORS[ax_name]
            ds    = self._downsample_factor(ax_name)
            _format_axes(ax, t, sig, SIGNAL_LABELS[ax_name], color, self.events, ds)

        axes[-1].set_xlabel("Time (s)", fontsize=9)
        patches = _build_legend(self.events)
        if patches:
            fig.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.85)

        fig.tight_layout()
        out_path = self.output_dir / "signal_ACC_combined.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualizer] Saved {out_path.name}")
        return out_path

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_plottable_signals(self) -> Dict[str, tuple]:
        """Return {name: (time_array, signal_array)} for all standard signals."""
        out = {}
        for name in ("EDA", "BVP", "ST", "ACC_X", "ACC_Y", "ACC_Z"):
            if name in self.result.signals and name in self.result.time_vectors:
                out[name] = (self.result.time_vectors[name], self.result.signals[name])
        return out

    def _downsample_factor(self, name: str) -> int:
        """Downsample BVP / ACC for display to avoid slow rendering."""
        fs_map = {"BVP": 4, "ACC_X": 2, "ACC_Y": 2, "ACC_Z": 2}
        return fs_map.get(name, 1)

    def _overlay_ibi_on_bvp(self, ax: plt.Axes):
        """Mark R-peaks (beat times) as small ticks on BVP axis."""
        ibi_t = self.result.ibi_times_s
        mask  = (ibi_t >= 0) & (ibi_t <= self.result.duration_s)
        yvals = np.full(mask.sum(), ax.get_ylim()[1] * 0.95)
        ax.scatter(ibi_t[mask], yvals, s=6, color="#800000",
                   marker="|", linewidths=0.6, alpha=0.5, label="Beat")

    def _plot_ibi_standalone(self) -> Path:
        """Standalone IBI (RR interval) tachogram."""
        fig, ax = plt.subplots(figsize=FIGSIZE_INDIVIDUAL)
        ibi_t = self.result.ibi_times_s
        ibi_v = self.result.ibi_values_ms
        mask  = (ibi_t >= 0) & (ibi_t <= self.result.duration_s)

        ax.scatter(ibi_t[mask], ibi_v[mask], s=6, color="#E74C3C", alpha=0.7)
        ax.plot(   ibi_t[mask], ibi_v[mask], color="#E74C3C", lw=0.5, alpha=0.4)

        ax.set_ylabel("IBI (ms)", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, self.result.duration_s)

        margin = 0.1 * (np.ptp(ibi_v[mask]) + 1e-6)
        _shade_events(ax, self.events, ibi_v[mask].min() - margin,
                      ibi_v[mask].max() + margin * 3)

        ax.set_title(
            "Simulated IBI (RR intervals) – inverted: ↑ = faster HR",
            fontsize=11, fontweight="bold"
        )

        patches = _build_legend(self.events)
        if patches:
            ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.85)

        fig.tight_layout()
        out_path = self.output_dir / "signal_IBI.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualizer] Saved {out_path.name}")
        return out_path

    def save_all(self) -> Dict[str, Path]:
        """Convenience: generate and save all plots."""
        print("[Visualizer] Generating individual signal plots …")
        individual = self.plot_individual_signals()

        print("[Visualizer] Generating combined plot …")
        combined   = self.plot_combined()

        print("[Visualizer] Generating ACC 3-axis plot …")
        acc_comb   = self.plot_acc_combined()

        return {**individual, "combined": combined, "acc_combined": acc_comb}
