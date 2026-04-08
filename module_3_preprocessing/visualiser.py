"""
=============================================================================
MODULE 3 – DATA PREPROCESSING  |  visualiser.py
=============================================================================
Two sets of visualisations:

1. Processed signal plots — cleaned, filtered signals with event shading
2. Comparative raw vs processed — side-by-side or overlay showing the
   effect of cleaning and filtering on each signal channel

=============================================================================
"""
from __future__ import annotations
from config import SAMPLING_RATES, PLOT_DPI, PLOT_STYLE
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")


# Signal display config
SIG_COLORS = {
    "EDA": "#2ECC71", "BVP": "#E74C3C", "ST": "#F39C12",
    "ACC_X": "#3498DB", "ACC_Y": "#9B59B6", "ACC_Z": "#1ABC9C",
    "IBI": "#E74C3C", "ACC": "#3498DB",
}
SIG_YLABELS = {
    "EDA": "EDA (µS)", "BVP": "BVP (nT)", "IBI": "IBI (ms)",
    "ST": "ST (°C)", "ACC_X": "ACC X (g)", "ACC_Y": "ACC Y (g)",
    "ACC_Z": "ACC Z (g)", "ACC": "SVM (g)",
}
VALUE_COLS = {
    "EDA": "EDA_uS", "BVP": "BVP_nT", "IBI": "IBI_ms",
    "ST": "ST_degC", "ACC_X": "ACC_X_g", "ACC_Y": "ACC_Y_g",
    "ACC_Z": "ACC_Z_g",
}


def _try_style():
    try:
        plt.style.use(PLOT_STYLE)
    except OSError:
        plt.style.use("ggplot")


def _shade_labels(ax, df, t_col, duration):
    """Shade target_label regions on axes."""
    if "target_label" not in df.columns:
        return
    LABEL_COLORS = {
        "Happy": "#FFD700", "Anger": "#DC143C", "Fear": "#8B008B",
        "Disgust": "#6B8E23", "Sad": "#4169E1", "Surprise": "#FF8C00",
        "Hunger": "#FF6347", "Thirst": "#00BFFF", "Toilet": "#A0522D",
        "Tired": "#708090", "baseline": None,
    }
    labels = df["target_label"].values
    ts = df[t_col].values if t_col in df.columns else np.arange(len(labels))

    i = 0
    while i < len(labels):
        lbl = labels[i]
        if lbl == "baseline" or lbl not in LABEL_COLORS:
            i += 1
            continue
        j = i
        while j < len(labels) and labels[j] == lbl:
            j += 1
        color = LABEL_COLORS.get(lbl, "#AAAAAA")
        ax.axvspan(ts[i], ts[min(j, len(ts) - 1)],
                   alpha=0.22, color=color, linewidth=0)
        mid = (ts[i] + ts[min(j, len(ts) - 1)]) / 2
        yhi = ax.get_ylim()[1]
        ax.text(mid, yhi, lbl, ha="center", va="bottom",
                fontsize=7, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=color, alpha=0.85, linewidth=0.8),
                clip_on=True)
        i = j


class PreprocessingVisualiser:
    """
    Generate processed signal plots and raw-vs-processed comparisons.

    Parameters
    ----------
    output_dir : directory to save figures
    dpi        : figure resolution
    """

    def __init__(self, output_dir: str | Path, dpi: int = PLOT_DPI):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        _try_style()

    # ── 1. Processed signal plots ─────────────────────────────────────────

    def plot_processed_signals(
        self,
        signals: Dict[str, pd.DataFrame],
        session_id: str = "",
        user_id: str = "",
    ) -> Dict[str, Path]:
        """
        One figure per processed signal channel with event shading.
        Also produces a combined 7-panel figure.
        """
        saved = {}

        for sig_name, df in signals.items():
            path = self._plot_one_processed(df, sig_name,
                                            session_id, user_id)
            if path:
                saved[sig_name] = path

        # Combined
        path = self._plot_combined_processed(signals, session_id, user_id)
        if path:
            saved["combined"] = path

        return saved

    def _plot_one_processed(self, df, sig_name, session_id, user_id) -> Optional[Path]:
        val_col = self._get_val_col(df, sig_name)
        if val_col is None:
            return None

        t_col = "timestamp_s" if "timestamp_s" in df.columns else None
        t = df[t_col].values if t_col else np.arange(len(df))
        vals = df[val_col].values

        fig, (ax_tl, ax_sig) = plt.subplots(
            2, 1, figsize=(14, 5.5),
            gridspec_kw={"height_ratios": [1, 5], "hspace": 0.08}
        )

        # Timeline bar
        self._draw_timeline(ax_tl, df, t, session_id, user_id, sig_name)

        # Signal
        color = SIG_COLORS.get(sig_name, "#555555")
        ds = 4 if sig_name == "BVP" else (2 if "ACC" in sig_name else 1)
        ax_sig.plot(t[::ds], vals[::ds], color=color, lw=0.75, alpha=0.9)
        ax_sig.set_ylabel(SIG_YLABELS.get(sig_name, sig_name), fontsize=9)
        ax_sig.set_xlabel("Time (s)", fontsize=9)
        ax_sig.set_xlim(t[0], t[-1])

        pad = 0.10 * (vals.max() - vals.min() + 1e-6)
        ax_sig.set_ylim(vals.min() - pad, vals.max() + pad * 2.5)
        _shade_labels(ax_sig, df, t_col or "idx", t[-1])

        fig.tight_layout()
        out_path = self.output_dir / f"processed_{sig_name}.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualiser] Saved processed_{sig_name}.png")
        return out_path

    def _plot_combined_processed(self, signals, session_id, user_id) -> Optional[Path]:
        order = [s for s in ("EDA", "BVP", "IBI", "ST", "ACC_X", "ACC_Y", "ACC_Z")
                 if s in signals]
        if not order:
            return None

        fig = plt.figure(figsize=(16, 3.5 * len(order) + 2))
        heights = [1] + [3] * len(order)
        gs = GridSpec(len(order) + 1, 1, figure=fig,
                      height_ratios=heights, hspace=0.35)

        # Title
        fig.suptitle(
            f"Processed Physiological Signals — Session: {session_id}  "
            f"User: {user_id}",
            fontsize=12, fontweight="bold", y=0.998,
        )

        # Timeline bar (use first signal for labels)
        ax_tl = fig.add_subplot(gs[0])
        first_df = signals[order[0]]
        t0 = first_df["timestamp_s"].values if "timestamp_s" in first_df.columns \
            else np.arange(len(first_df))
        self._draw_timeline(ax_tl, first_df, t0, session_id, user_id,
                            "All signals")

        for i, sig_name in enumerate(order):
            df = signals[sig_name]
            ax = fig.add_subplot(gs[i + 1])
            val_col = self._get_val_col(df, sig_name)
            if val_col is None:
                continue
            t_col = "timestamp_s" if "timestamp_s" in df.columns else None
            t = df[t_col].values if t_col else np.arange(len(df))
            vals = df[val_col].values
            color = SIG_COLORS.get(sig_name, "#555555")
            ds = 4 if sig_name == "BVP" else 1
            ax.plot(t[::ds], vals[::ds], color=color, lw=0.7)
            ax.set_ylabel(SIG_YLABELS.get(sig_name, sig_name), fontsize=8)
            ax.set_xlim(t[0], t[-1])
            pad = 0.10 * (vals.max() - vals.min() + 1e-6)
            ax.set_ylim(vals.min() - pad, vals.max() + pad * 2)
            _shade_labels(ax, df, t_col or "idx", t[-1])
            if i < len(order) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (s)", fontsize=9)

        out_path = self.output_dir / "processed_combined.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print("  [Visualiser] Saved processed_combined.png")
        return out_path

    # ── 2. Comparative raw vs processed ──────────────────────────────────

    def plot_raw_vs_processed(
        self,
        raw_signals: Dict[str, pd.DataFrame],
        processed_signals: Dict[str, pd.DataFrame],
        session_id: str = "",
        user_id: str = "",
    ) -> Dict[str, Path]:
        """
        Side-by-side (or overlay) comparison of raw and processed signals.
        One figure per signal channel + one combined figure.
        """
        saved = {}
        common = [s for s in raw_signals if s in processed_signals]

        for sig_name in common:
            path = self._plot_comparison_one(
                raw_signals[sig_name],
                processed_signals[sig_name],
                sig_name, session_id, user_id,
            )
            if path:
                saved[sig_name] = path

        # Combined comparison
        path = self._plot_comparison_combined(
            raw_signals, processed_signals, session_id, user_id
        )
        if path:
            saved["comparison_combined"] = path

        return saved

    def _plot_comparison_one(self, raw_df, proc_df, sig_name,
                             session_id, user_id) -> Optional[Path]:
        val_col = self._get_val_col(raw_df, sig_name)
        if val_col is None or val_col not in proc_df.columns:
            return None

        t_col = "timestamp_s"
        t_raw = raw_df[t_col].values if t_col in raw_df.columns else np.arange(len(raw_df))
        t_proc = proc_df[t_col].values if t_col in proc_df.columns else np.arange(len(proc_df))
        raw = raw_df[val_col].values
        proc = proc_df[val_col].values

        fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                                 gridspec_kw={"height_ratios": [1, 3, 3]})

        # Timeline
        self._draw_timeline(axes[0], proc_df, t_proc, session_id, user_id, sig_name)

        # Raw
        color = SIG_COLORS.get(sig_name, "#555555")
        ds = 4 if sig_name == "BVP" else 1
        axes[1].plot(t_raw[::ds], raw[::ds], color="#AAAAAA", lw=0.6,
                     alpha=0.8, label="Raw")
        axes[1].set_ylabel(f"{SIG_YLABELS.get(sig_name, sig_name)}\n(raw)", fontsize=9)
        axes[1].set_xlim(t_raw[0], t_raw[-1])
        axes[1].set_xticklabels([])
        axes[1].legend(fontsize=8, loc="upper right")

        # Processed
        axes[2].plot(t_proc[::ds], proc[::ds], color=color, lw=0.75,
                     alpha=0.9, label="Processed")
        axes[2].set_ylabel(f"{SIG_YLABELS.get(sig_name, sig_name)}\n(processed)", fontsize=9)
        axes[2].set_xlabel("Time (s)", fontsize=9)
        axes[2].set_xlim(t_proc[0], t_proc[-1])
        axes[2].legend(fontsize=8, loc="upper right")
        _shade_labels(axes[2], proc_df, t_col, t_proc[-1])

        # Add noise stats annotation
        if len(raw) > 4 and len(proc) > 4:
            raw_std = np.std(raw)
            proc_std = np.std(proc)
            snr_gain = 20 * np.log10(raw_std / (proc_std + 1e-12))
            axes[1].set_title(
                f"{sig_name} — Raw  (std={raw_std:.4f})",
                fontsize=10, fontweight="bold", loc="left"
            )
            axes[2].set_title(
                f"{sig_name} — Processed  (std={proc_std:.4f}  |  "
                f"SNR gain={snr_gain:.1f} dB)",
                fontsize=10, fontweight="bold", loc="left"
            )

        fig.tight_layout()
        out_path = self.output_dir / f"comparison_{sig_name}.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Visualiser] Saved comparison_{sig_name}.png")
        return out_path

    def _plot_comparison_combined(self, raw_signals, processed_signals,
                                  session_id, user_id) -> Optional[Path]:
        common = [s for s in ("EDA", "BVP", "ST", "ACC_X", "ACC_Y", "ACC_Z")
                  if s in raw_signals and s in processed_signals]
        if not common:
            return None

        ncols = 2
        nrows = len(common)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows),
                                 sharex="row")
        fig.suptitle(
            f"Raw vs Processed — Session: {session_id}  User: {user_id}",
            fontsize=13, fontweight="bold",
        )

        for row, sig_name in enumerate(common):
            val_col = self._get_val_col(raw_signals[sig_name], sig_name)
            if val_col is None:
                continue
            t_col = "timestamp_s"
            t_raw = raw_signals[sig_name][t_col].values  \
                if t_col in raw_signals[sig_name].columns \
                else np.arange(len(raw_signals[sig_name]))
            t_proc = processed_signals[sig_name][t_col].values \
                if t_col in processed_signals[sig_name].columns \
                else np.arange(len(processed_signals[sig_name]))

            raw = raw_signals[sig_name][val_col].values
            proc = processed_signals[sig_name][val_col].values
            color = SIG_COLORS.get(sig_name, "#555555")
            ds = 4 if sig_name == "BVP" else 1
            ylabel = SIG_YLABELS.get(sig_name, sig_name)

            # Left: raw
            ax_r = axes[row][0] if nrows > 1 else axes[0]
            ax_r.plot(t_raw[::ds], raw[::ds], color="#AAAAAA", lw=0.6)
            ax_r.set_ylabel(ylabel, fontsize=8)
            if row == 0:
                ax_r.set_title("Raw Signal", fontsize=10, fontweight="bold")
            if row == nrows - 1:
                ax_r.set_xlabel("Time (s)", fontsize=8)

            # Right: processed
            ax_p = axes[row][1] if nrows > 1 else axes[1]
            ax_p.plot(t_proc[::ds], proc[::ds], color=color, lw=0.7)
            _shade_labels(ax_p, processed_signals[sig_name], t_col, t_proc[-1])
            if row == 0:
                ax_p.set_title("Processed Signal", fontsize=10, fontweight="bold")
            if row == nrows - 1:
                ax_p.set_xlabel("Time (s)", fontsize=8)

        fig.tight_layout()
        out_path = self.output_dir / "comparison_all_signals.png"
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print("  [Visualiser] Saved comparison_all_signals.png")
        return out_path

    # ── Helpers ───────────────────────────────────────────────────────────

    def _draw_timeline(self, ax, df, t, session_id, user_id, sig_name):
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_facecolor("#ECECEC")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.barh(0.5, t[-1] - t[0], left=t[0], height=1.0,
                color="#CCCCCC", alpha=0.5)
        ax.set_title(
            f"Module 3 — Preprocessed {sig_name}  |  "
            f"Session: {session_id}  User: {user_id}",
            fontsize=9, fontweight="bold", pad=4, loc="left"
        )
        ax.tick_params(bottom=False)

        if "target_label" not in df.columns:
            return
        LABEL_COLORS = {
            "Happy": "#FFD700", "Anger": "#DC143C", "Fear": "#8B008B",
            "Disgust": "#6B8E23", "Sad": "#4169E1", "Surprise": "#FF8C00",
            "Hunger": "#FF6347", "Thirst": "#00BFFF", "Toilet": "#A0522D",
            "Tired": "#708090",
        }
        labels = df["target_label"].values
        i = 0
        while i < len(labels):
            lbl = labels[i]
            if lbl == "baseline" or lbl not in LABEL_COLORS:
                i += 1
                continue
            j = i
            while j < len(labels) and labels[j] == lbl:
                j += 1
            t0_lbl = t[i]
            t1_lbl = t[min(j, len(t) - 1)]
            ax.barh(0.5, t1_lbl - t0_lbl, left=t0_lbl, height=1.0,
                    color=LABEL_COLORS[lbl], alpha=0.80)
            ax.text((t0_lbl + t1_lbl) / 2, 0.5, lbl,
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white", clip_on=True)
            i = j

    def _get_val_col(self, df, sig_name) -> Optional[str]:
        """Return the value column name for a signal DataFrame."""
        candidates = {
            "EDA": "EDA_uS", "BVP": "BVP_nT", "IBI": "IBI_ms",
            "ST": "ST_degC", "ACC_X": "ACC_X_g",
            "ACC_Y": "ACC_Y_g", "ACC_Z": "ACC_Z_g",
            "ACC": "ACC_X_g",
        }
        col = candidates.get(sig_name)
        if col and col in df.columns:
            return col
        # Fallback: first numeric non-meta column
        for c in df.columns:
            if c not in ("timestamp_s", "target_label", "event_id",
                         "category", "user_id", "session_id"):
                if pd.api.types.is_numeric_dtype(df[c]):
                    return c
        return None
