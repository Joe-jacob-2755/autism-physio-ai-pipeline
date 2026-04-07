"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  visualizer.py
=============================================================================
Visualization engine.

Every figure contains:
  - A dedicated EVENT TIMELINE BAR at the top of the figure showing
    exactly which emotion/behaviour was active at every point in time
  - Bold colour-filled event blocks on every signal panel
  - Large, boxed emotion labels directly above each event block
  - Thick vertical boundary lines at event start/end
  - A colour-coded legend for all emotions present
=============================================================================
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from typing import Dict, List

from config import SAMPLING_RATES, EMOTION_PROFILES
from event_scheduler import EventConfig
from simulator import SimulationResult


# ─────────────────────────────────────────────────────────────────────────────
# STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "EDA":   "#27AE60",
    "BVP":   "#C0392B",
    "ST":    "#E67E22",
    "ACC_X": "#2980B9",
    "ACC_Y": "#8E44AD",
    "ACC_Z": "#16A085",
}

SIGNAL_LABELS = {
    "EDA":   "EDA (µS)",
    "BVP":   "BVP (nT)",
    "ST":    "ST (°C)",
    "ACC_X": "ACC X (g)",
    "ACC_Y": "ACC Y (g)",
    "ACC_Z": "ACC Z (g)",
}

FIGSIZE_INDIVIDUAL = (16, 5.5)
FIGSIZE_COMBINED   = (18, 28)
PLOT_STYLE         = "seaborn-v0_8-whitegrid"

EVENT_SHADE_ALPHA  = 0.30   # fill inside event windows
EVENT_BORDER_LW    = 1.8    # vertical boundary line width
LABEL_FONTSIZE     = 9.5    # emotion label inside signal panel
TIMELINE_HEIGHT    = 0.13   # fraction of figure height for the timeline bar


# ─────────────────────────────────────────────────────────────────────────────
# EVENT TIMELINE BAR  (the dedicated annotation strip)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_timeline_bar(ax: plt.Axes, events: List[EventConfig], duration_s: float):
    """
    Draw a solid colour-coded event timeline bar on the given axes.

    The bar shows:
      - Grey fill for baseline periods
      - Solid emotion-coloured fill for each event window
      - Emotion name centred inside each event block
      - 'Baseline' label in quiet periods wide enough to hold text
    """
    ax.set_xlim(0, duration_s)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("Events", fontsize=9, rotation=0, labelpad=48, va="center")
    ax.tick_params(left=False, bottom=False)
    ax.set_facecolor("#ECECEC")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Shade every second of time ──────────────────────────────────────────
    # First shade entire bar grey (baseline)
    ax.barh(0.5, duration_s, left=0, height=1.0,
            color="#CCCCCC", alpha=0.5, linewidth=0)

    # ── Draw each event as a solid coloured block ────────────────────────────
    for ev in events:
        width = ev.end_s - ev.start_s
        ax.barh(0.5, width, left=ev.start_s, height=1.0,
                color=ev.color, alpha=0.85, linewidth=0)

        # Vertical borders
        for t in (ev.start_s, ev.end_s):
            ax.axvline(t, color=ev.color, linewidth=2.0, alpha=1.0, ymin=0, ymax=1)

        # Emotion label centred in block
        mid = ev.start_s + width / 2.0
        # Only draw label if block is wide enough
        ax.text(mid, 0.50, ev.emotion,
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor=ev.color, alpha=0.0,
                          edgecolor="none"),
                clip_on=True)

    # ── Label baseline gaps wide enough (≥10 s) ─────────────────────────────
    occupied = sorted([(e.start_s, e.end_s) for e in events])
    gaps = []
    cursor = 0.0
    for (s, e) in occupied:
        if s - cursor >= 5.0:
            gaps.append((cursor, s))
        cursor = e
    if duration_s - cursor >= 5.0:
        gaps.append((cursor, duration_s))

    for (g0, g1) in gaps:
        if g1 - g0 >= 10.0:
            ax.text((g0 + g1) / 2, 0.50, "baseline",
                    ha="center", va="center",
                    fontsize=8, color="#555555", style="italic",
                    clip_on=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PANEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _shade_events_on_signal(ax: plt.Axes, events: List[EventConfig],
                             ymin: float, ymax: float):
    """
    Draw event shading on a signal axes panel.

    Includes:
      - Translucent fill across the event window
      - Thick vertical boundaries
      - Boxed emotion label near the top of the panel
    """
    for ev in events:
        c = ev.color
        # Translucent fill
        ax.axvspan(ev.start_s, ev.end_s,
                   alpha=EVENT_SHADE_ALPHA, color=c, linewidth=0, zorder=1)

        # Thick vertical borders
        for t in (ev.start_s, ev.end_s):
            ax.axvline(t, color=c, linewidth=EVENT_BORDER_LW,
                       linestyle="-", alpha=0.85, zorder=2)

        # Boxed label near top of panel
        mid   = (ev.start_s + ev.end_s) / 2.0
        label_y = ymax - 0.02 * (ymax - ymin)
        ax.text(mid, label_y, ev.emotion,
                ha="center", va="top",
                fontsize=LABEL_FONTSIZE, fontweight="bold", color=c,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor=c,
                          linewidth=1.2, alpha=0.88),
                zorder=3, clip_on=True)


def _plot_signal_on_ax(ax: plt.Axes,
                        t: np.ndarray,
                        sig: np.ndarray,
                        ylabel: str,
                        color: str,
                        events: List[EventConfig],
                        ds: int = 1,
                        xlabel: bool = False):
    """Full signal panel: trace + event shading + labels."""
    td  = t[::ds]
    sd  = sig[::ds]

    ax.plot(td, sd, color=color, linewidth=0.85, alpha=0.92, zorder=4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(0, t[-1])

    pad = 0.10 * (sd.max() - sd.min() + 1e-6)
    ymin = sd.min() - pad
    ymax = sd.max() + pad * 2.5

    ax.set_ylim(ymin, ymax)
    ax.tick_params(labelsize=8)

    _shade_events_on_signal(ax, events, ymin, ymax)

    if xlabel:
        ax.set_xlabel("Time (s)", fontsize=9)
    else:
        ax.set_xticklabels([])


def _build_legend(events: List[EventConfig]) -> List[mpatches.Patch]:
    """Deduplicated legend patches."""
    seen = {}
    for ev in events:
        if ev.emotion not in seen:
            seen[ev.emotion] = mpatches.Patch(
                facecolor=ev.color, alpha=0.75,
                edgecolor="black", linewidth=0.5,
                label=ev.emotion,
            )
    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VISUALIZER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class SignalVisualizer:
    """
    Generate publication-quality annotated plots from a SimulationResult.

    Every figure has a dedicated event timeline bar at the top showing
    which emotion/behaviour was active at every point in time, plus
    bold shading and boxed labels on every signal panel.
    """

    def __init__(self, result: SimulationResult,
                 output_dir: str = "output", dpi: int = 150):
        self.result     = result
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi        = dpi
        self.events     = result.events
        self.duration   = result.duration_s

        try:
            plt.style.use(PLOT_STYLE)
        except OSError:
            plt.style.use("ggplot")

    # ── Individual signal plots ───────────────────────────────────────────────

    def plot_individual_signals(self) -> Dict[str, Path]:
        """One figure per signal channel, each with a timeline bar on top."""
        saved   = {}
        signals = self._get_plottable_signals()

        for sig_name, (t, sig) in signals.items():
            color  = SIGNAL_COLORS.get(sig_name, "#555555")
            ylabel = SIGNAL_LABELS.get(sig_name, sig_name)
            ds     = self._ds(sig_name)

            # Figure: timeline bar (top) + signal panel (bottom)
            fig = plt.figure(figsize=FIGSIZE_INDIVIDUAL)
            gs  = GridSpec(2, 1, figure=fig,
                           height_ratios=[1, 5], hspace=0.08)
            ax_tl = fig.add_subplot(gs[0])
            ax_sg = fig.add_subplot(gs[1], sharex=ax_tl)

            # ── Timeline bar ──────────────────────────────────────────────
            _draw_timeline_bar(ax_tl, self.events, self.duration)
            ax_tl.set_title(
                f"Module 1A  –  Simulated {sig_name}  "
                f"|  fs = {SAMPLING_RATES.get(sig_name, 'event')} Hz  "
                f"|  Duration = {self.duration:.0f} s  "
                f"|  Noise = {self.result.metadata.get('noise_level','?')}  "
                f"|  Seed = {self.result.metadata.get('seed','?')}",
                fontsize=10, fontweight="bold", pad=6,
            )

            # ── Signal panel ──────────────────────────────────────────────
            _plot_signal_on_ax(ax_sg, t, sig, ylabel, color,
                               self.events, ds, xlabel=True)

            if sig_name == "BVP":
                self._overlay_beats(ax_sg)

            # ── Legend ────────────────────────────────────────────────────
            patches = _build_legend(self.events)
            if patches:
                ax_sg.legend(handles=patches, loc="upper right",
                             fontsize=8, framealpha=0.90,
                             ncol=min(len(patches), 5),
                             title="Target Behaviour / Emotion",
                             title_fontsize=8)

            out_path = self.output_dir / f"signal_{sig_name}.png"
            fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            saved[sig_name] = out_path
            print(f"  [Visualizer] Saved {out_path.name}")

        # IBI standalone
        saved["IBI"] = self._plot_ibi_standalone()
        return saved

    # ── Combined multi-panel figure ───────────────────────────────────────────

    def plot_combined(self) -> Path:
        """
        All 7 signals in one figure.
        Row 0: event timeline bar
        Rows 1-7: EDA, BVP, IBI, ST, ACC_X, ACC_Y, ACC_Z
        """
        n_sig    = 7
        heights  = [1.2, 2.2, 2.8, 2.0, 2.2, 2.0, 2.0, 2.0]  # timeline + 7 signals
        fig      = plt.figure(figsize=FIGSIZE_COMBINED)
        gs       = GridSpec(n_sig + 1, 1, figure=fig,
                            hspace=0.35, height_ratios=heights)

        # ── Title ─────────────────────────────────────────────────────────
        events_str = ", ".join(
            sorted({ev.emotion for ev in self.events})
        ) or "none"
        fig.suptitle(
            "Module 1A  –  Simulated Physiological Signals\n"
            f"Duration: {self.duration:.0f} s  |  "
            f"Events: {len(self.events)}  |  "
            f"Emotions: {events_str}\n"
            f"Noise: {self.result.metadata.get('noise_level','?')}  |  "
            f"Seed: {self.result.metadata.get('seed','?')}",
            fontsize=12, fontweight="bold", y=0.997,
        )

        # ── Row 0: event timeline bar ─────────────────────────────────────
        ax_tl = fig.add_subplot(gs[0])
        _draw_timeline_bar(ax_tl, self.events, self.duration)
        ax_tl.set_title("Target Behaviour / Emotion Timeline",
                         fontsize=10, fontweight="bold", loc="left", pad=4)

        # ── Signal rows ───────────────────────────────────────────────────
        order   = ["EDA", "BVP", "IBI", "ST", "ACC_X", "ACC_Y", "ACC_Z"]
        signals = self._get_plottable_signals()
        ibi_t   = self.result.ibi_times_s
        ibi_v   = self.result.ibi_values_ms

        prev_ax = ax_tl
        for row_idx, sig_name in enumerate(order):
            is_last = (row_idx == n_sig - 1)
            ax = fig.add_subplot(gs[row_idx + 1], sharex=ax_tl)

            if sig_name == "IBI":
                mask = (ibi_t >= 0) & (ibi_t <= self.duration)
                ax.scatter(ibi_t[mask], ibi_v[mask],
                           s=5, color="#C0392B", alpha=0.7, zorder=4)
                ax.plot(ibi_t[mask], ibi_v[mask],
                        color="#C0392B", lw=0.5, alpha=0.35, zorder=3)
                ax.set_ylabel("IBI (ms)", fontsize=9)
                ax.set_xlim(0, self.duration)

                pad  = 0.10 * (np.ptp(ibi_v[mask]) + 1e-6)
                ymin = ibi_v[mask].min() - pad
                ymax = ibi_v[mask].max() + pad * 2.5
                ax.set_ylim(ymin, ymax)
                ax.invert_yaxis()
                ax.tick_params(labelsize=8)
                _shade_events_on_signal(ax, self.events, ymax, ymin)
                ax.set_title("IBI  (inverted — ↑ = faster HR)",
                             fontsize=8, loc="left", pad=2)

            elif sig_name in signals:
                t, sig = signals[sig_name]
                color  = SIGNAL_COLORS[sig_name]
                ylabel = SIGNAL_LABELS[sig_name]
                ds     = self._ds(sig_name)
                _plot_signal_on_ax(ax, t, sig, ylabel, color,
                                   self.events, ds,
                                   xlabel=is_last)
                ax.set_title(sig_name, fontsize=8, loc="left", pad=2)

                if sig_name == "BVP":
                    self._overlay_beats(ax)
            else:
                ax.set_visible(False)
                continue

            if not is_last:
                ax.set_xticklabels([])

        # ── Global legend ─────────────────────────────────────────────────
        patches = _build_legend(self.events)
        if patches:
            fig.legend(handles=patches,
                       loc="lower center",
                       ncol=min(len(patches), 6),
                       fontsize=9, framealpha=0.92,
                       title="Target Behaviour / Emotion  (shaded regions = active event)",
                       title_fontsize=9,
                       bbox_to_anchor=(0.5, -0.002))

        out_path = self.output_dir / "combined_signals.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualizer] Saved {out_path.name}")
        return out_path

    # ── ACC combined plot ─────────────────────────────────────────────────────

    def plot_acc_combined(self) -> Path:
        """3-axis ACC comparison with timeline bar."""
        fig = plt.figure(figsize=(16, 10))
        gs  = GridSpec(4, 1, figure=fig,
                       height_ratios=[1, 3, 3, 3], hspace=0.25)

        ax_tl = fig.add_subplot(gs[0])
        _draw_timeline_bar(ax_tl, self.events, self.duration)
        ax_tl.set_title(
            "Accelerometer (3-axis)  –  Target Behaviour / Emotion Timeline",
            fontsize=11, fontweight="bold", pad=6,
        )

        axes = [fig.add_subplot(gs[i + 1], sharex=ax_tl) for i in range(3)]
        for ax, ax_name in zip(axes, ["ACC_X", "ACC_Y", "ACC_Z"]):
            t   = self.result.time_vectors[ax_name]
            sig = self.result.signals[ax_name]
            _plot_signal_on_ax(ax, t, sig,
                               SIGNAL_LABELS[ax_name],
                               SIGNAL_COLORS[ax_name],
                               self.events, self._ds(ax_name))

        axes[-1].set_xlabel("Time (s)", fontsize=9)
        for ax in axes[:-1]:
            ax.set_xticklabels([])

        patches = _build_legend(self.events)
        if patches:
            fig.legend(handles=patches, loc="upper right",
                       fontsize=8, framealpha=0.90,
                       title="Target Behaviour / Emotion",
                       title_fontsize=8)

        out_path = self.output_dir / "signal_ACC_combined.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualizer] Saved {out_path.name}")
        return out_path

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_plottable_signals(self) -> Dict[str, tuple]:
        out = {}
        for name in ("EDA", "BVP", "ST", "ACC_X", "ACC_Y", "ACC_Z"):
            if name in self.result.signals and name in self.result.time_vectors:
                out[name] = (self.result.time_vectors[name],
                             self.result.signals[name])
        return out

    def _ds(self, name: str) -> int:
        """Display downsample factor for high-rate signals."""
        return {"BVP": 4, "ACC_X": 2, "ACC_Y": 2, "ACC_Z": 2}.get(name, 1)

    def _overlay_beats(self, ax: plt.Axes):
        """Beat tick marks on BVP panel."""
        ibi_t = self.result.ibi_times_s
        mask  = (ibi_t >= 0) & (ibi_t <= self.duration)
        yhi   = ax.get_ylim()[1]
        ax.scatter(ibi_t[mask], np.full(mask.sum(), yhi * 0.93),
                   s=8, color="#6B0000", marker="|",
                   linewidths=0.7, alpha=0.55, zorder=5)

    def _plot_ibi_standalone(self) -> Path:
        """Standalone IBI tachogram with timeline bar."""
        ibi_t = self.result.ibi_times_s
        ibi_v = self.result.ibi_values_ms
        mask  = (ibi_t >= 0) & (ibi_t <= self.duration)

        fig = plt.figure(figsize=FIGSIZE_INDIVIDUAL)
        gs  = GridSpec(2, 1, figure=fig,
                       height_ratios=[1, 5], hspace=0.08)
        ax_tl = fig.add_subplot(gs[0])
        ax_sg = fig.add_subplot(gs[1], sharex=ax_tl)

        _draw_timeline_bar(ax_tl, self.events, self.duration)
        ax_tl.set_title(
            f"Module 1A  –  Simulated IBI  |  Duration = {self.duration:.0f} s  "
            f"|  Noise = {self.result.metadata.get('noise_level','?')}  "
            f"|  Seed = {self.result.metadata.get('seed','?')}",
            fontsize=10, fontweight="bold", pad=6,
        )

        ax_sg.scatter(ibi_t[mask], ibi_v[mask],
                      s=7, color="#C0392B", alpha=0.80, zorder=4)
        ax_sg.plot(ibi_t[mask], ibi_v[mask],
                   color="#C0392B", lw=0.6, alpha=0.35, zorder=3)

        ax_sg.set_ylabel("IBI (ms)", fontsize=9)
        ax_sg.set_xlabel("Time (s)", fontsize=9)
        ax_sg.invert_yaxis()
        ax_sg.set_xlim(0, self.duration)
        ax_sg.tick_params(labelsize=8)

        pad  = 0.10 * (np.ptp(ibi_v[mask]) + 1e-6)
        ymin = ibi_v[mask].min() - pad
        ymax = ibi_v[mask].max() + pad * 2.5
        ax_sg.set_ylim(ymax, ymin)   # inverted
        _shade_events_on_signal(ax_sg, self.events, ymax, ymin)

        patches = _build_legend(self.events)
        if patches:
            ax_sg.legend(handles=patches, loc="upper right",
                         fontsize=8, framealpha=0.90,
                         title="Target Behaviour / Emotion",
                         title_fontsize=8)

        out_path = self.output_dir / "signal_IBI.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualizer] Saved {out_path.name}")
        return out_path

    def save_all(self) -> Dict[str, Path]:
        """Generate and save all plots."""
        print("[Visualizer] Generating individual signal plots ...")
        individual = self.plot_individual_signals()
        print("[Visualizer] Generating combined plot ...")
        combined   = self.plot_combined()
        print("[Visualizer] Generating ACC 3-axis plot ...")
        acc_comb   = self.plot_acc_combined()
        return {**individual, "combined": combined, "acc_combined": acc_comb}
