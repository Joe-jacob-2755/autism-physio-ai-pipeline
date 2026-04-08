"""
=============================================================================
MODULE 4 – DATA ANALYSER  |  visualiser.py
=============================================================================
All visualisations for Module 4.

2D plots:
  - Descriptive: box plots per target per signal
  - Statistical: significant feature bar chart, effect size heatmap
  - Correlation: feature×target heatmap, feature×feature heatmap
  - Temporal: time-to-peak, subside time, return rate per target
  - Threshold: signal trace with threshold overlaid
  - % change from baseline bar chart
  - Median drift across events

3D plot:
  - Time-synchronised 3D projection of all signals (EDA, BVP, ST, ACC-SVM)
    using Plotly for interactive rotation
=============================================================================
"""
from __future__ import annotations
from config import (
    PLOT_DPI, TARGET_COLORS, SIGNAL_COLORS, SIGNAL_UNITS,
    CATEGORY_COLORS,
)
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


FIGSIZE_WIDE = (16, 6)
FIGSIZE_SQ = (12, 10)
FIGSIZE_TALL = (14, 10)


def _try_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except BaseException:
        plt.style.use("ggplot")


def _save(fig, path: Path, dpi=PLOT_DPI):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Vis] Saved {path.name}")


class AnalysisVisualiser:
    """Generate all Module 4 plots and save to output directory."""

    def __init__(self, output_dir: str | Path, dpi: int = PLOT_DPI):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        _try_style()

    # ── 1. Descriptive: box plots ─────────────────────────────────────────

    def plot_descriptive_boxplots(
        self,
        feature_dfs: Dict[str, pd.DataFrame],
        top_n: int = 6,
    ) -> Dict[str, Path]:
        saved = {}
        for sig, df in feature_dfs.items():
            if "target_label" not in df.columns:
                continue
            feat_cols = [c for c in df.columns
                         if c not in _meta() and pd.api.types.is_numeric_dtype(df[c])]
            if not feat_cols:
                continue
            # Pick top N features by between-group variance
            bv = df.groupby("target_label")[feat_cols].mean().var()
            top = bv.nlargest(top_n).index.tolist()
            labels = sorted(df["target_label"].unique())
            colors = [TARGET_COLORS.get(lbl, "#888888") for lbl in labels]

            fig, axes = plt.subplots(1, len(top), figsize=(3.5 * len(top), 5))
            if len(top) == 1:
                axes = [axes]
            fig.suptitle(f"{sig} — Top {len(top)} Discriminative Features by Target",
                         fontsize=12, fontweight="bold")
            for ax, col in zip(axes, top):
                data = [df.loc[df["target_label"] == lbl, col].dropna().values
                        for lbl in labels]
                bp = ax.boxplot(data, patch_artist=True, notch=False,
                                medianprops={"color": "white", "linewidth": 2})
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.75)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_title(col.replace("_", " "), fontsize=9, fontweight="bold")
                ax.set_ylabel(SIGNAL_UNITS.get(sig, ""), fontsize=8)
            fig.tight_layout()
            p = self.out / f"descriptive_boxplot_{sig}.png"
            _save(fig, p, self.dpi)
            saved[sig] = p
        return saved

    # ── 2. % Change from baseline ─────────────────────────────────────────

    def plot_pct_change(self, pct_df: pd.DataFrame) -> Optional[Path]:
        if pct_df.empty:
            return None
        signals = pct_df["signal"].unique()
        n_sig = len(signals)
        fig, axes = plt.subplots(1, n_sig, figsize=(5 * n_sig, 5), squeeze=False)
        fig.suptitle("Mean Signal % Change from Baseline per Target",
                     fontsize=12, fontweight="bold")
        for ax, sig in zip(axes[0], signals):
            sub = pct_df[pct_df["signal"] == sig].sort_values("mean_change_pct")
            colors = [TARGET_COLORS.get(lbl, "#888") for lbl in sub["target_label"]]
            bars = ax.barh(sub["target_label"], sub["mean_change_pct"],
                           color=colors, edgecolor="white", linewidth=0.5)
            ax.axvline(0, color="#333", linewidth=0.8)
            ax.set_xlabel("Mean % change from baseline", fontsize=9)
            ax.set_title(f"{sig} ({SIGNAL_UNITS.get(sig, '')})",
                         fontsize=10, fontweight="bold")
            # Value labels
            for bar, val in zip(bars, sub["mean_change_pct"]):
                ax.text(val + (0.5 if val >= 0 else -0.5), bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", ha="left" if val >= 0 else "right",
                        fontsize=8)
        fig.tight_layout()
        p = self.out / "pct_change_from_baseline.png"
        _save(fig, p, self.dpi)
        return p

    # ── 3. Time-to-peak and subside ───────────────────────────────────────

    def plot_temporal_dynamics(self, dynamics_df: pd.DataFrame) -> Optional[Path]:
        if dynamics_df.empty:
            return None
        sigs = dynamics_df["signal"].unique()
        metrics = ["peak_delay_s", "subside_time_s", "time_to_return_s"]
        labels_m = ["Time to Peak (s)", "Time to Subside 50% (s)", "Time to Return (s)"]
        n_sig = len(sigs)
        fig, axes = plt.subplots(len(metrics), n_sig,
                                 figsize=(4.5 * n_sig, 4 * len(metrics)))
        if n_sig == 1:
            axes = axes.reshape(-1, 1)
        fig.suptitle("Temporal Dynamics by Target × Signal",
                     fontsize=13, fontweight="bold")
        for row, (metric, ylabel) in enumerate(zip(metrics, labels_m)):
            for col, sig in enumerate(sigs):
                ax = axes[row][col]
                sub = dynamics_df[dynamics_df["signal"] == sig]
                if metric not in sub.columns:
                    continue
                grp = {lbl: g[metric].dropna().values
                       for lbl, g in sub.groupby("target_label")
                       if len(g[metric].dropna()) >= 1}
                if not grp:
                    continue
                labels = sorted(grp.keys())
                data = [grp[lbl] for lbl in labels]
                colors = [TARGET_COLORS.get(lbl, "#888") for lbl in labels]
                bp = ax.boxplot(data, patch_artist=True,
                                medianprops={"color": "white", "linewidth": 2})
                for patch, c in zip(bp["boxes"], colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.75)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
                if col == 0:
                    ax.set_ylabel(ylabel, fontsize=9)
                if row == 0:
                    ax.set_title(sig, fontsize=10, fontweight="bold")
        fig.tight_layout()
        p = self.out / "temporal_dynamics.png"
        _save(fig, p, self.dpi)
        return p

    # ── 4. Return-to-median rate ──────────────────────────────────────────

    def plot_return_rate(self, return_df: pd.DataFrame) -> Optional[Path]:
        if return_df.empty or "pct_returned" not in return_df.columns:
            return None
        sigs = return_df["signal"].unique()
        fig, axes = plt.subplots(1, len(sigs), figsize=(4.5 * len(sigs), 5), squeeze=False)
        fig.suptitle("% Events Returning to Pre-event Median", fontsize=12, fontweight="bold")
        for ax, sig in zip(axes[0], sigs):
            sub = return_df[return_df["signal"] == sig].sort_values("pct_returned")
            colors = [TARGET_COLORS.get(lbl, "#888") for lbl in sub["target_label"]]
            bars = ax.barh(sub["target_label"], sub["pct_returned"],
                           color=colors, edgecolor="white")
            ax.set_xlim(0, 105)
            ax.set_xlabel("% events returned", fontsize=9)
            ax.set_title(sig, fontsize=10, fontweight="bold")
            for bar, val in zip(bars, sub["pct_returned"]):
                ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}%", va="center", fontsize=8)
        fig.tight_layout()
        p = self.out / "return_to_median_rate.png"
        _save(fig, p, self.dpi)
        return p

    # ── 5. Kruskal-Wallis significant features ────────────────────────────

    def plot_significant_features(self, kw_df: pd.DataFrame, top_n=20) -> Optional[Path]:
        if kw_df.empty:
            return None
        sig_df = kw_df[kw_df["significant"]].nsmallest(top_n, "p_value")
        if sig_df.empty:
            return None
        es_colors = {"large": "#C0392B", "medium": "#E67E22", "small": "#27AE60", "negligible": "#95A5A6"}
        colors = [es_colors.get(e, "#888") for e in sig_df["effect_size"]]
        fig, ax = plt.subplots(figsize=(10, max(5, len(sig_df) * 0.4)))
        bars = ax.barh(sig_df["feature"], -np.log10(sig_df["p_value"]),
                       color=colors, edgecolor="white", linewidth=0.5)
        from config import ALPHA as _ALPHA
        ax.axvline(-np.log10(_ALPHA), color="#555", linestyle="--",
                   linewidth=1, label=f"α = {_ALPHA}")
        ax.set_xlabel("-log₁₀(p-value)  [Kruskal-Wallis]", fontsize=10)
        ax.set_title("Most Significant Features by Kruskal-Wallis Test",
                     fontsize=12, fontweight="bold")
        patches = [mpatches.Patch(color=c, label=lbl)
                   for lbl, c in es_colors.items() if lbl != "negligible"]
        ax.legend(handles=patches, title="Effect size", fontsize=8, loc="lower right")
        fig.tight_layout()
        p = self.out / "significant_features_kruskal.png"
        _save(fig, p, self.dpi)
        return p

    # ── 6. Correlation heatmaps ───────────────────────────────────────────

    def plot_correlation_target(self, corr_df: pd.DataFrame, top_n=25) -> Optional[Path]:
        if corr_df.empty:
            return None
        top = corr_df.head(top_n)
        fig, ax = plt.subplots(figsize=(8, max(6, len(top) * 0.35)))
        bar_colors = ["#E74C3C" if r > 0 else "#3498DB" for r in top["spearman_r"]]
        ax.barh(top["feature"], top["spearman_r"], color=bar_colors,
                edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_xlabel("Spearman ρ with target label", fontsize=10)
        ax.set_title("Feature–Target Correlation (Spearman)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = self.out / "correlation_feature_target.png"
        _save(fig, p, self.dpi)
        return p

    def plot_correlation_matrix(self, corr_mat: pd.DataFrame) -> Optional[Path]:
        if corr_mat.empty:
            return None
        fig, ax = plt.subplots(figsize=(max(10, len(corr_mat) * 0.45),
                                        max(8, len(corr_mat) * 0.45)))
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = ax.imshow(corr_mat.values, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(corr_mat.columns)))
        ax.set_yticks(range(len(corr_mat.index)))
        ax.set_xticklabels(corr_mat.columns, rotation=90, fontsize=6)
        ax.set_yticklabels(corr_mat.index, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.03, label="Spearman ρ")
        ax.set_title("Feature–Feature Correlation Matrix", fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = self.out / "correlation_matrix.png"
        _save(fig, p, self.dpi)
        return p

    # ── 7. Threshold overlay ──────────────────────────────────────────────

    def plot_threshold_overlay(
        self,
        median_drift_dfs: List[pd.DataFrame],
        threshold_events: pd.DataFrame,
        mad_factor: float,
    ) -> Dict[str, Path]:
        saved = {}
        for md_df in median_drift_dfs:
            if md_df.empty or "signal" not in md_df.columns:
                continue
            sig = md_df["signal"].iloc[0]
            ts = md_df["timestamp_s"].values
            val = md_df["value"].values
            med = md_df["running_median"].values
            mad = md_df["running_mad"].values
            upper = med + mad_factor * mad
            lower = med - mad_factor * mad

            fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
            c = SIGNAL_COLORS.get(sig, "#555")
            ax.plot(ts, val, color=c, lw=0.7, alpha=0.8, label="Signal")
            ax.plot(ts, med, color="#333", lw=1.5, label="Running median")
            ax.fill_between(ts, lower, upper, alpha=0.15, color="#E74C3C",
                            label=f"±{mad_factor}×MAD threshold")
            ax.plot(ts, upper, color="#E74C3C", lw=0.8, ls="--", alpha=0.6)
            ax.plot(ts, lower, color="#E74C3C", lw=0.8, ls="--", alpha=0.6)

            if not threshold_events.empty:
                thr_sig = threshold_events[threshold_events["signal"] == sig]
                for _, row in thr_sig.iterrows():
                    ax.axvspan(row["start_s"], row["end_s"],
                               color="#FF0000", alpha=0.2)

            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel(f"{sig} ({SIGNAL_UNITS.get(sig, '')})", fontsize=9)
            ax.set_title(f"{sig} — Adaptive Threshold  (window={md_df['timestamp_s'].max():.0f}s)",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, loc="upper right")
            fig.tight_layout()
            p = self.out / f"threshold_{sig}.png"
            _save(fig, p, self.dpi)
            saved[sig] = p
        return saved

    # ── 8. 3D time-synchronised signal projection ─────────────────────────

    def plot_3d_signals(
        self,
        signals: Dict[str, pd.DataFrame],
        out_html: Optional[Path] = None,
        out_png: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Interactive 3D Plotly figure projecting multiple signals into a
        time × signal_axis × amplitude space.

        X axis: Time (s)
        Y axis: Signal channel (0=EDA, 1=BVP, 2=ST, 3=ACC-SVM)
        Z axis: Normalised amplitude (z-scored per signal so scales are comparable)

        Traces are colour-coded by target_label annotation.
        """
        if not HAS_PLOTLY:
            print("  [Vis] Plotly not available — skipping 3D plot")
            return None

        SIGNAL_ORDER = {"EDA": 0, "BVP": 1, "IBI": 2, "ST": 3, "ACC": 4}
        VALUE_MAP = {"EDA": "EDA_uS", "BVP": "BVP_nT",
                     "IBI": "IBI_ms", "ST": "ST_degC", "ACC": "ACC_X_g"}

        fig = go.Figure()
        added_labels = set()

        for sig_name, y_pos in sorted(SIGNAL_ORDER.items(), key=lambda x: x[1]):
            if sig_name not in signals:
                continue
            df = signals[sig_name]
            col = VALUE_MAP.get(sig_name)
            if col is None or col not in df.columns:
                continue
            if "timestamp_s" not in df.columns:
                continue

            ts = df["timestamp_s"].values
            vals = df[col].values.astype(float)
            # Z-score normalise so all signals share the same amplitude axis
            mu, sd = np.nanmean(vals), np.nanstd(vals) + 1e-9
            z_vals = (vals - mu) / sd

            has_label = "target_label" in df.columns
            labels = df["target_label"].values if has_label else ["signal"] * len(ts)

            # Segment by label for colouring
            i = 0
            while i < len(ts):
                lbl = labels[i]
                j = i
                while j < len(ts) and labels[j] == lbl:
                    j += 1
                color = TARGET_COLORS.get(lbl, "#AAAAAA")
                show_legend = (lbl not in added_labels)
                fig.add_trace(go.Scatter3d(
                    x=ts[i:j],
                    y=np.full(j - i, y_pos),
                    z=z_vals[i:j],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=lbl,
                    legendgroup=lbl,
                    showlegend=show_legend,
                ))
                if show_legend:
                    added_labels.add(lbl)
                i = j

        tick_text = [s for s in sorted(SIGNAL_ORDER, key=SIGNAL_ORDER.get)
                     if s in signals]
        tick_vals = [SIGNAL_ORDER[s] for s in tick_text]

        fig.update_layout(
            title=dict(
                text="3D Time-Synchronised Signal Projection — All Channels",
                font=dict(size=14),
            ),
            scene=dict(
                xaxis_title="Time (s)",
                yaxis=dict(
                    title="Signal channel",
                    tickvals=tick_vals,
                    ticktext=tick_text,
                ),
                zaxis_title="Normalised amplitude (z-score)",
                camera=dict(eye=dict(x=1.8, y=-1.6, z=0.8)),
            ),
            legend=dict(title="Target label", font=dict(size=10)),
            height=700,
        )

        # Save interactive HTML
        html_path = out_html or (self.out / "3d_signals.html")
        fig.write_html(str(html_path))
        print(f"  [Vis] Saved {html_path.name}  (interactive 3D)")

        # Save static PNG if kaleido available
        png_path = out_png or (self.out / "3d_signals.png")
        try:
            fig.write_image(str(png_path), scale=1.5)
            print(f"  [Vis] Saved {png_path.name}")
        except Exception:
            png_path = None

        return html_path

    # ── 9. Median drift across events ─────────────────────────────────────

    def plot_median_drift(self, dynamics_df: pd.DataFrame) -> Optional[Path]:
        if dynamics_df.empty or "median_drift" not in dynamics_df.columns:
            return None
        sigs = dynamics_df["signal"].unique()
        fig, axes = plt.subplots(1, len(sigs), figsize=(4.5 * len(sigs), 5), squeeze=False)
        fig.suptitle("Median Drift (Post-event − Pre-event Median)", fontsize=12, fontweight="bold")
        for ax, sig in zip(axes[0], sigs):
            sub = dynamics_df[dynamics_df["signal"] == sig]
            grp = sub.groupby("target_label")["median_drift"].mean().sort_values()
            colors = [TARGET_COLORS.get(lbl, "#888") for lbl in grp.index]
            bars = ax.barh(grp.index, grp.values, color=colors, edgecolor="white")
            ax.axvline(0, color="#333", lw=0.8)
            ax.set_xlabel("Mean median drift", fontsize=9)
            ax.set_title(sig, fontsize=10, fontweight="bold")
        fig.tight_layout()
        p = self.out / "median_drift.png"
        _save(fig, p, self.dpi)
        return p


def _meta():
    from config import META_COLS
    return META_COLS
