"""
=============================================================================
MODULE 4 - EXPLORATORY DATA ANALYSIS  |  visualiser.py
=============================================================================
All 25 visualisation functions for the combined multi-user EDA report.

V1-V4:   Data Understanding & Quality (Section 2-3)
V5:      Distribution (Section 4)
V6-V9:   Univariate (Section 5)
V10-V12: Bivariate (Section 6)
V13-V14: Correlational (Section 7)
V15-V22: Temporal Event Dynamics (Section 8)
V23-V25: 3D Visualisations (Section 9)
=============================================================================
"""
from __future__ import annotations

import warnings
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

from config import (
    PLOT_DPI, TARGET_COLORS, SIGNAL_COLORS, SIGNAL_UNITS,
    CATEGORY_COLORS, VALUE_COLS, LABEL_TO_CATEGORY,
    MAX_SCATTER_POINTS,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


FIGSIZE_WIDE = (16, 6)
FIGSIZE_SQ = (12, 10)
FIGSIZE_TALL = (14, 10)


def _try_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        try:
            plt.style.use("ggplot")
        except Exception:
            pass


def _save(fig, path: Path, dpi=PLOT_DPI):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Vis] Saved {path.name}")


def _target_color(label: str) -> str:
    return TARGET_COLORS.get(label, "#888888")


def _cat_color(cat: str) -> str:
    return CATEGORY_COLORS.get(cat, "#888888")


class EDAVisualiser:
    """Generate all Module 4 EDA charts and save to output directory."""

    def __init__(self, output_dir: Path, dpi: int = PLOT_DPI):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        _try_style()

    # =====================================================================
    # V1 — Target distribution (dual panel: log scale + non-baseline)
    # =====================================================================
    def plot_target_distribution(self, dist_df: pd.DataFrame) -> Optional[Path]:
        """Dual-panel bar chart: log-scale all labels + linear non-baseline."""
        if dist_df is None or dist_df.empty:
            return None

        labels = dist_df["target_label"].values
        counts = dist_df["n_samples"].values
        colors = [_target_color(l) for l in labels]

        # Non-baseline subset
        non_bl = dist_df[dist_df["target_label"] != "baseline"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6),
                                        gridspec_kw={"width_ratios": [1, 1.2]})
        fig.suptitle("Target Label Distribution (Combined Training Data)",
                     fontsize=14, fontweight="bold")

        # Left: all labels, log scale
        bars1 = ax1.bar(range(len(labels)), counts, color=colors, edgecolor="white")
        ax1.set_yscale("log")
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax1.set_ylabel("Sample count (log scale)", fontsize=10)
        ax1.set_title("All Labels (Log Scale)", fontsize=11, fontweight="bold")
        for bar, c in zip(bars1, counts):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.15,
                     f"{c:,}", ha="center", va="bottom", fontsize=7)

        # Right: non-baseline only, linear scale
        if not non_bl.empty:
            nb_labels = non_bl["target_label"].values
            nb_counts = non_bl["n_samples"].values
            nb_colors = [_target_color(l) for l in nb_labels]
            bars2 = ax2.bar(range(len(nb_labels)), nb_counts,
                            color=nb_colors, edgecolor="white")
            ax2.set_xticks(range(len(nb_labels)))
            ax2.set_xticklabels(nb_labels, rotation=45, ha="right", fontsize=9)
            ax2.set_ylabel("Sample count", fontsize=10)
            ax2.set_title("Event Labels Only (Baseline Excluded)",
                          fontsize=11, fontweight="bold")
            for bar, c in zip(bars2, nb_counts):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(nb_counts) * 0.01,
                         f"{c:,}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        p = self.out / "V01_target_distribution.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V2 — User-label presence heatmap
    # =====================================================================
    def plot_user_label_heatmap(self, matrix_df: pd.DataFrame) -> Optional[Path]:
        """Dual-panel heatmap: log-scale all labels + linear event-only."""
        if matrix_df is None or matrix_df.empty:
            return None
        from matplotlib.colors import LogNorm

        has_baseline = "baseline" in matrix_df.columns
        n_users = matrix_df.shape[0]

        if has_baseline:
            # --- Dual panel: log all + linear events-only ---
            event_cols = [c for c in matrix_df.columns if c != "baseline"]
            event_df = matrix_df[event_cols]

            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(22, max(6, n_users * 0.45)),
                gridspec_kw={"width_ratios": [1, 1.2]},
            )

            # Left: all labels, log scale
            all_data = matrix_df.values.astype(float)
            vmin = max(1, all_data[all_data > 0].min()) if (all_data > 0).any() else 1
            vmax = all_data.max()
            im1 = ax1.imshow(
                np.where(all_data > 0, all_data, np.nan),
                cmap="YlOrRd", aspect="auto",
                norm=LogNorm(vmin=vmin, vmax=vmax),
            )
            ax1.set_xticks(range(matrix_df.shape[1]))
            ax1.set_xticklabels(matrix_df.columns, rotation=45, ha="right",
                                fontsize=8)
            ax1.set_yticks(range(n_users))
            ax1.set_yticklabels(matrix_df.index, fontsize=8)
            # Annotate cells
            for i in range(n_users):
                for j in range(matrix_df.shape[1]):
                    v = int(all_data[i, j])
                    if v > 0:
                        txt = f"{v:,}" if v < 10_000 else f"{v / 1000:.0f}k"
                        clr = "white" if v > vmax * 0.3 else "black"
                        ax1.text(j, i, txt, ha="center", va="center",
                                 fontsize=6, color=clr)
            plt.colorbar(im1, ax=ax1, fraction=0.03, label="Sample count (log)")
            ax1.set_title("All Labels (log scale)", fontsize=11,
                          fontweight="bold")

            # Right: event labels only, linear scale
            ev_data = event_df.values.astype(float)
            im2 = ax2.imshow(ev_data, cmap="YlGnBu", aspect="auto")
            ax2.set_xticks(range(event_df.shape[1]))
            ax2.set_xticklabels(event_df.columns, rotation=45, ha="right",
                                fontsize=8)
            ax2.set_yticks(range(n_users))
            ax2.set_yticklabels(matrix_df.index, fontsize=8)
            # Annotate cells — highlight zeros
            for i in range(n_users):
                for j in range(event_df.shape[1]):
                    v = int(ev_data[i, j])
                    if v == 0:
                        ax2.text(j, i, "—", ha="center", va="center",
                                 fontsize=7, color="#999999")
                    else:
                        ev_max = ev_data.max() if ev_data.max() > 0 else 1
                        clr = "white" if v > ev_max * 0.6 else "black"
                        ax2.text(j, i, f"{v:,}", ha="center", va="center",
                                 fontsize=7, color=clr)
            plt.colorbar(im2, ax=ax2, fraction=0.03, label="Sample count")
            ax2.set_title("Event Labels Only (linear scale — baseline excluded)",
                          fontsize=11, fontweight="bold")

            fig.suptitle("User x Target Label Sample Counts",
                         fontsize=13, fontweight="bold", y=1.01)
        else:
            # No baseline column — single heatmap is fine
            fig, ax1 = plt.subplots(
                figsize=(max(10, matrix_df.shape[1] * 0.8),
                         max(6, n_users * 0.4))
            )
            data = matrix_df.values.astype(float)
            im = ax1.imshow(data, cmap="YlOrRd", aspect="auto")
            ax1.set_xticks(range(matrix_df.shape[1]))
            ax1.set_xticklabels(matrix_df.columns, rotation=45, ha="right",
                                fontsize=8)
            ax1.set_yticks(range(n_users))
            ax1.set_yticklabels(matrix_df.index, fontsize=8)
            plt.colorbar(im, ax=ax1, fraction=0.03, label="Sample count")
            ax1.set_title("User x Target Label Sample Counts",
                          fontsize=12, fontweight="bold")

        fig.tight_layout()
        p = self.out / "V02_user_label_heatmap.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V3 — Missing data heatmap
    # =====================================================================
    def plot_missing_heatmap(self, missing_df: pd.DataFrame) -> Optional[Path]:
        """Heatmap of missing value percentages per signal."""
        if missing_df is None or missing_df.empty:
            return None
        fig, ax = plt.subplots(figsize=(8, max(4, len(missing_df) * 0.5)))
        pivot = missing_df[["value_col", "pct_missing"]].copy()
        vals = pivot["pct_missing"].values.reshape(-1, 1)
        labels = pivot["value_col"].values
        im = ax.imshow(vals, cmap="Reds", aspect="auto", vmin=0)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xticks([0])
        ax.set_xticklabels(["% Missing"], fontsize=9)
        for i, v in enumerate(vals.flatten()):
            ax.text(0, i, f"{v:.2f}%", ha="center", va="center", fontsize=9,
                    color="white" if v > 5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.03, label="% Missing")
        ax.set_title("Missing Values by Signal", fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = self.out / "V03_missing_heatmap.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V4 — Outlier box plots
    # =====================================================================
    def plot_outlier_boxplots(
        self, signals: Dict[str, pd.DataFrame],
    ) -> Optional[Path]:
        """Box plots showing outlier distribution per signal value column."""
        val_data = {}
        for sig_name, df in sorted(signals.items()):
            if df is None:
                continue
            for vc in VALUE_COLS.get(sig_name, []):
                if vc in df.columns:
                    val_data[vc] = df[vc].dropna().values

        if not val_data:
            return None
        names = list(val_data.keys())
        fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 5))
        if len(names) == 1:
            axes = [axes]
        fig.suptitle("Signal Value Distributions with Outliers (IQR Method)",
                     fontsize=12, fontweight="bold")
        for ax, name in zip(axes, names):
            vals = val_data[name]
            bp = ax.boxplot(vals, patch_artist=True, notch=False,
                            medianprops={"color": "white", "linewidth": 2})
            sig_key = name.split("_")[0]
            bp["boxes"][0].set_facecolor(SIGNAL_COLORS.get(sig_key, "#888"))
            bp["boxes"][0].set_alpha(0.7)
            ax.set_title(name, fontsize=9, fontweight="bold")
            ax.set_ylabel(SIGNAL_UNITS.get(sig_key, ""), fontsize=8)
        fig.tight_layout()
        p = self.out / "V04_outlier_boxplots.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V5 — Histograms + KDE per signal
    # =====================================================================
    def plot_histograms_kde(
        self, signals: Dict[str, pd.DataFrame],
    ) -> Optional[Path]:
        """Histograms with KDE overlay for each signal value column."""
        val_data = {}
        for sig_name, df in sorted(signals.items()):
            if df is None:
                continue
            for vc in VALUE_COLS.get(sig_name, []):
                if vc in df.columns:
                    val_data[vc] = df[vc].dropna().values

        if not val_data:
            return None
        names = list(val_data.keys())
        n = len(names)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        fig.suptitle("Signal Value Distributions (Histogram + KDE)",
                     fontsize=13, fontweight="bold")
        for i, name in enumerate(names):
            ax = axes[i]
            vals = val_data[name]
            # Subsample for performance
            if len(vals) > 100_000:
                vals = np.random.default_rng(42).choice(vals, 100_000, replace=False)
            sig_key = name.split("_")[0]
            ax.hist(vals, bins=80, density=True, alpha=0.6,
                    color=SIGNAL_COLORS.get(sig_key, "#888"), edgecolor="white", lw=0.3)
            if HAS_SEABORN:
                try:
                    sns.kdeplot(vals, ax=ax, color="black", lw=1.5)
                except Exception:
                    pass
            ax.set_title(name, fontsize=9, fontweight="bold")
            ax.set_xlabel(SIGNAL_UNITS.get(sig_key, ""), fontsize=8)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()
        p = self.out / "V05_histograms_kde.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V6 — Violin plots by target
    # =====================================================================
    def plot_violin_by_target(
        self, signals: Dict[str, pd.DataFrame],
    ) -> Dict[str, Path]:
        """Violin plots of each signal's primary value column by target."""
        saved = {}
        for sig_name, df in sorted(signals.items()):
            if df is None or "target_label" not in df.columns:
                continue
            vc = VALUE_COLS.get(sig_name, [None])[0] if VALUE_COLS.get(sig_name) else None
            if vc is None or vc not in df.columns:
                continue
            labels = sorted(df["target_label"].unique())
            data = [df.loc[df["target_label"] == l, vc].dropna().values for l in labels]
            data = [d for d, l in zip(data, labels) if len(d) >= 2]
            valid_labels = [l for l, d in zip(labels, [df.loc[df["target_label"] == l, vc].dropna() for l in labels]) if len(d) >= 2]
            if len(data) < 2:
                continue

            fig, ax = plt.subplots(figsize=(max(10, len(valid_labels) * 0.8), 6))
            vp = ax.violinplot(data, showmeans=True, showmedians=True)
            for i, body in enumerate(vp["bodies"]):
                body.set_facecolor(_target_color(valid_labels[i]))
                body.set_alpha(0.7)
            ax.set_xticks(range(1, len(valid_labels) + 1))
            ax.set_xticklabels(valid_labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(f"{vc} ({SIGNAL_UNITS.get(sig_name, '')})", fontsize=10)
            ax.set_title(f"{sig_name} — Distribution by Target Label (Violin)",
                         fontsize=12, fontweight="bold")
            fig.tight_layout()
            p = self.out / f"V06_violin_{sig_name}.png"
            _save(fig, p, self.dpi)
            saved[sig_name] = p
        return saved

    # =====================================================================
    # V7 — Box plots by target
    # =====================================================================
    def plot_box_by_target(
        self, signals: Dict[str, pd.DataFrame],
    ) -> Dict[str, Path]:
        """Box plots of each signal's primary value column by target."""
        saved = {}
        for sig_name, df in sorted(signals.items()):
            if df is None or "target_label" not in df.columns:
                continue
            vc = VALUE_COLS.get(sig_name, [None])[0] if VALUE_COLS.get(sig_name) else None
            if vc is None or vc not in df.columns:
                continue
            labels = sorted(df["target_label"].unique())
            data = [df.loc[df["target_label"] == l, vc].dropna().values for l in labels]
            colors = [_target_color(l) for l in labels]

            fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
            bp = ax.boxplot(data, patch_artist=True, notch=False,
                            medianprops={"color": "white", "linewidth": 2})
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(f"{vc} ({SIGNAL_UNITS.get(sig_name, '')})", fontsize=10)
            ax.set_title(f"{sig_name} — Distribution by Target Label (Box)",
                         fontsize=12, fontweight="bold")
            fig.tight_layout()
            p = self.out / f"V07_boxplot_{sig_name}.png"
            _save(fig, p, self.dpi)
            saved[sig_name] = p
        return saved

    # =====================================================================
    # V8 — Category grouped bar
    # =====================================================================
    def plot_category_bar(self, cat_desc_df: pd.DataFrame) -> Optional[Path]:
        """Grouped bar chart of mean values by category."""
        if cat_desc_df is None or cat_desc_df.empty:
            return None
        # Pivot: rows=value_col, cols=category
        pivot = cat_desc_df.pivot_table(index="value_col", columns="category",
                                        values="mean", aggfunc="first")
        if pivot.empty:
            return None
        fig, ax = plt.subplots(figsize=(max(10, len(pivot) * 0.6), 6))
        x = np.arange(len(pivot.index))
        width = 0.8 / len(pivot.columns)
        for i, cat in enumerate(pivot.columns):
            vals = pivot[cat].values
            ax.bar(x + i * width, vals, width, label=cat,
                   color=_cat_color(cat), alpha=0.8, edgecolor="white")
        ax.set_xticks(x + width * len(pivot.columns) / 2)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean value", fontsize=10)
        ax.set_title("Mean Signal Values by Category",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()
        p = self.out / "V08_category_bar.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V9 — Demographic box plots
    # =====================================================================
    def plot_demographic_boxplots(
        self, signals: Dict[str, pd.DataFrame],
        demographics: pd.DataFrame,
        field: str = "autism_severity",
    ) -> Optional[Path]:
        """Box plots of signal values grouped by demographic field."""
        if demographics is None or demographics.empty or field not in demographics.columns:
            return None
        demo_sub = demographics[["user_id", field]].drop_duplicates()
        plots_data = {}
        for sig_name, df in sorted(signals.items()):
            if df is None or "user_id" not in df.columns:
                continue
            vc = VALUE_COLS.get(sig_name, [None])[0] if VALUE_COLS.get(sig_name) else None
            if vc is None or vc not in df.columns:
                continue
            merged = df.merge(demo_sub, on="user_id", how="left")
            plots_data[sig_name] = (vc, merged)

        if not plots_data:
            return None
        n = len(plots_data)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]
        fig.suptitle(f"Signal Distributions by {field.replace('_', ' ').title()}",
                     fontsize=13, fontweight="bold")
        for ax, (sig_name, (vc, merged)) in zip(axes, plots_data.items()):
            groups = sorted(merged[field].dropna().unique())
            data = [merged.loc[merged[field] == g, vc].dropna().values for g in groups]
            bp = ax.boxplot(data, patch_artist=True,
                            medianprops={"color": "white", "linewidth": 2})
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(groups)))
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{sig_name}", fontsize=10, fontweight="bold")
            ax.set_ylabel(SIGNAL_UNITS.get(sig_name, ""), fontsize=8)
        fig.tight_layout()
        p = self.out / f"V09_demographic_{field}.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V10 — Inter-signal correlation heatmap
    # =====================================================================
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> Optional[Path]:
        """Heatmap of inter-signal Spearman correlation matrix."""
        if corr_matrix is None or corr_matrix.empty:
            return None
        n = len(corr_matrix)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.6)))
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        im = ax.imshow(corr_matrix.values, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(corr_matrix.index, fontsize=7)
        # Annotate cells
        for i in range(n):
            for j in range(n):
                v = corr_matrix.iloc[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if abs(v) > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.03, label="Spearman rho")
        ax.set_title("Inter-Signal Spearman Correlation Matrix",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = self.out / "V10_correlation_heatmap.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V11 — KW significance bar
    # =====================================================================
    def plot_kw_significance(self, kw_df: pd.DataFrame, top_n: int = 15) -> Optional[Path]:
        """Bar chart of -log10(p) for Kruskal-Wallis results."""
        if kw_df is None or kw_df.empty:
            return None
        sig_df = kw_df[kw_df["significant"]].nsmallest(top_n, "p_value")
        if sig_df.empty:
            sig_df = kw_df.nsmallest(top_n, "p_value")
        es_colors = {"large": "#C0392B", "medium": "#E67E22",
                     "small": "#27AE60", "negligible": "#95A5A6"}
        colors = [es_colors.get(e, "#888") for e in sig_df["effect_size"]]

        fig, ax = plt.subplots(figsize=(10, max(5, len(sig_df) * 0.4)))
        labels = sig_df["signal"] + " / " + sig_df["value_col"]
        ax.barh(labels, -np.log10(sig_df["p_value"].clip(lower=1e-300)),
                color=colors, edgecolor="white", linewidth=0.5)
        from config import ALPHA as _ALPHA
        ax.axvline(-np.log10(_ALPHA), color="#555", ls="--", lw=1,
                   label=f"alpha = {_ALPHA}")
        ax.set_xlabel("-log10(p-value)  [Kruskal-Wallis]", fontsize=10)
        ax.set_title("Most Significant Signals by Kruskal-Wallis H-Test",
                     fontsize=12, fontweight="bold")
        patches = [mpatches.Patch(color=c, label=l)
                   for l, c in es_colors.items()]
        ax.legend(handles=patches, title="Effect size", fontsize=8, loc="lower right")
        fig.tight_layout()
        p = self.out / "V11_kw_significance.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V12 — Effect size heatmap
    # =====================================================================
    def plot_effect_size_heatmap(self, kw_df: pd.DataFrame) -> Optional[Path]:
        """Heatmap of eta-squared effect sizes from KW test."""
        if kw_df is None or kw_df.empty or "eta_squared" not in kw_df.columns:
            return None
        pivot_data = kw_df[["signal", "value_col", "eta_squared"]].copy()
        pivot_data["label"] = pivot_data["signal"] + " / " + pivot_data["value_col"]
        fig, ax = plt.subplots(figsize=(6, max(4, len(pivot_data) * 0.4)))
        y = range(len(pivot_data))
        colors = ["#C0392B" if e >= 0.14 else "#E67E22" if e >= 0.06
                  else "#27AE60" if e >= 0.01 else "#95A5A6"
                  for e in pivot_data["eta_squared"]]
        ax.barh(list(y), pivot_data["eta_squared"].values, color=colors, edgecolor="white")
        ax.set_yticks(list(y))
        ax.set_yticklabels(pivot_data["label"].values, fontsize=8)
        ax.set_xlabel("eta-squared (effect size)", fontsize=10)
        ax.set_title("Kruskal-Wallis Effect Sizes (eta-squared)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = self.out / "V12_effect_size.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V13 — Scatter plot matrix (top signals)
    # =====================================================================
    def plot_scatter_matrix(
        self, signals: Dict[str, pd.DataFrame],
        max_points: int = MAX_SCATTER_POINTS,
    ) -> Optional[Path]:
        """Scatter plot matrix of primary signal value columns."""
        aligned = {}
        for sig_name, df in sorted(signals.items()):
            if df is None or "timestamp_s" not in df.columns:
                continue
            vc = VALUE_COLS.get(sig_name, [None])[0] if VALUE_COLS.get(sig_name) else None
            if vc is None or vc not in df.columns:
                continue
            aligned[vc] = df[["timestamp_s", vc]].dropna().sort_values("timestamp_s")

        if len(aligned) < 2:
            return None

        # Merge all on timestamp
        names = list(aligned.keys())
        base = aligned[names[0]]
        for n in names[1:]:
            base = pd.merge_asof(base.sort_values("timestamp_s"),
                                 aligned[n].sort_values("timestamp_s"),
                                 on="timestamp_s", tolerance=0.5, direction="nearest")
        base = base.drop(columns=["timestamp_s"]).dropna()
        if len(base) > max_points:
            base = base.sample(max_points, random_state=42)

        n = len(names)
        fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
        fig.suptitle("Signal Scatter Plot Matrix", fontsize=14, fontweight="bold")
        for i in range(n):
            for j in range(n):
                ax = axes[i][j]
                if i == j:
                    ax.hist(base[names[i]].values, bins=50, alpha=0.7,
                            color=SIGNAL_COLORS.get(names[i].split("_")[0], "#888"))
                    if j == 0:
                        ax.set_ylabel(names[i], fontsize=7)
                else:
                    ax.scatter(base[names[j]].values, base[names[i]].values,
                               alpha=0.15, s=1, color="#2C3E50")
                if i == n - 1:
                    ax.set_xlabel(names[j], fontsize=7)
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=7)
                ax.tick_params(labelsize=6)
        fig.tight_layout()
        p = self.out / "V13_scatter_matrix.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V14 — Point-biserial heatmap
    # =====================================================================
    def plot_point_biserial_heatmap(self, pb_matrix: pd.DataFrame) -> Optional[Path]:
        """Heatmap of signal vs target point-biserial correlations."""
        if pb_matrix is None or pb_matrix.empty:
            return None
        fig, ax = plt.subplots(figsize=(max(10, pb_matrix.shape[1] * 0.8),
                                        max(5, pb_matrix.shape[0] * 0.6)))
        norm = TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)
        data = pb_matrix.values.astype(float)
        im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(pb_matrix.shape[1]))
        ax.set_xticklabels(pb_matrix.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(pb_matrix.shape[0]))
        ax.set_yticklabels(pb_matrix.index, fontsize=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if abs(v) > 0.15 else "black")
        plt.colorbar(im, ax=ax, fraction=0.03, label="Point-biserial r")
        ax.set_title("Signal-Target Point-Biserial Correlations (one-vs-rest)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = self.out / "V14_point_biserial_heatmap.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V15 — Event duration box plot
    # =====================================================================
    def plot_event_duration(self, dur_df: pd.DataFrame) -> Optional[Path]:
        """Box plot of event durations per target label."""
        if dur_df is None or dur_df.empty:
            return None
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = dur_df["target_label"].values
        data = dur_df[["mean_duration_s", "std_duration_s"]].values
        colors = [_target_color(l) for l in labels]
        ax.bar(range(len(labels)), dur_df["mean_duration_s"],
               yerr=dur_df["std_duration_s"].fillna(0),
               color=colors, edgecolor="white", capsize=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Duration (seconds)", fontsize=10)
        ax.set_title("Event Duration by Target Label",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = self.out / "V15_event_duration.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V16 — % Change grouped bar
    # =====================================================================
    def plot_pct_change(self, pct_df: pd.DataFrame) -> Optional[Path]:
        """Grouped bar chart of mean % change from baseline per signal x target."""
        if pct_df is None or pct_df.empty:
            return None
        signals = pct_df["signal_name"].unique()
        n_sig = len(signals)
        fig, axes = plt.subplots(1, n_sig, figsize=(5 * n_sig, 5), squeeze=False)
        fig.suptitle("Mean Signal % Change from Baseline per Target",
                     fontsize=12, fontweight="bold")
        for ax, sig in zip(axes[0], signals):
            sub = pct_df[pct_df["signal_name"] == sig].sort_values("mean_change_pct")
            colors = [_target_color(l) for l in sub["target_label"]]
            bars = ax.barh(sub["target_label"], sub["mean_change_pct"],
                           color=colors, edgecolor="white", linewidth=0.5)
            ax.axvline(0, color="#333", lw=0.8)
            ax.set_xlabel("Mean % change", fontsize=9)
            ax.set_title(f"{sig}", fontsize=10, fontweight="bold")
        fig.tight_layout()
        p = self.out / "V16_pct_change.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V17 — Time-to-peak box plots
    # =====================================================================
    def plot_time_to_peak(self, ttp_df: pd.DataFrame) -> Optional[Path]:
        """Box plots of time-to-peak per signal x target."""
        if ttp_df is None or ttp_df.empty:
            return None
        signals = ttp_df["signal_name"].unique()
        fig, axes = plt.subplots(1, len(signals), figsize=(5 * len(signals), 5), squeeze=False)
        fig.suptitle("Time to Peak by Signal and Target",
                     fontsize=12, fontweight="bold")
        for ax, sig in zip(axes[0], signals):
            sub = ttp_df[ttp_df["signal_name"] == sig]
            labels = sub["target_label"].values
            colors = [_target_color(l) for l in labels]
            ax.bar(range(len(labels)), sub["mean_peak_delay_s"],
                   yerr=sub["std_peak_delay_s"].fillna(0),
                   color=colors, edgecolor="white", capsize=3)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Seconds", fontsize=9)
            ax.set_title(sig, fontsize=10, fontweight="bold")
        fig.tight_layout()
        p = self.out / "V17_time_to_peak.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V18 — Return-to-median stacked bar
    # =====================================================================
    def plot_return_rate(self, return_df: pd.DataFrame) -> Optional[Path]:
        """Stacked bar: returned vs not returned per signal x target."""
        if return_df is None or return_df.empty or "pct_returned" not in return_df.columns:
            return None
        signals = return_df["signal_name"].unique()
        fig, axes = plt.subplots(1, len(signals), figsize=(5 * len(signals), 5), squeeze=False)
        fig.suptitle("% Events Returning to Pre-event Median",
                     fontsize=12, fontweight="bold")
        for ax, sig in zip(axes[0], signals):
            sub = return_df[return_df["signal_name"] == sig].sort_values("pct_returned")
            colors = [_target_color(l) for l in sub["target_label"]]
            bars = ax.barh(sub["target_label"], sub["pct_returned"],
                           color=colors, edgecolor="white")
            ax.set_xlim(0, 105)
            ax.set_xlabel("% returned", fontsize=9)
            ax.set_title(sig, fontsize=10, fontweight="bold")
            for bar, val in zip(bars, sub["pct_returned"]):
                ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.0f}%", va="center", fontsize=8)
        fig.tight_layout()
        p = self.out / "V18_return_rate.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V19 — Return time heatmap
    # =====================================================================
    def plot_return_time_heatmap(self, rt_df: pd.DataFrame) -> Optional[Path]:
        """Heatmap: mean return time (signal x target)."""
        if rt_df is None or rt_df.empty:
            return None
        pivot = rt_df.pivot_table(index="signal_name", columns="target_label",
                                  values="mean_return_s", aggfunc="first")
        if pivot.empty:
            return None
        fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.8),
                                        max(4, pivot.shape[0] * 0.8)))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}s", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.03, label="Seconds")
        ax.set_title("Mean Time to Return to Median (Signal x Target)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = self.out / "V19_return_time_heatmap.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V20 — Return count bar chart
    # =====================================================================
    def plot_return_counts(self, rc_df: pd.DataFrame) -> Optional[Path]:
        """Grouped bar: returned vs not-returned counts per signal x target."""
        if rc_df is None or rc_df.empty:
            return None
        signals = rc_df["signal_name"].unique()
        fig, axes = plt.subplots(1, len(signals), figsize=(5 * len(signals), 5), squeeze=False)
        fig.suptitle("Return Count by Signal and Target",
                     fontsize=12, fontweight="bold")
        for ax, sig in zip(axes[0], signals):
            sub = rc_df[rc_df["signal_name"] == sig]
            x = np.arange(len(sub))
            w = 0.35
            ax.bar(x - w / 2, sub["n_returned"], w, label="Returned", color="#27AE60", alpha=0.8)
            ax.bar(x + w / 2, sub["n_not_returned"], w, label="Not returned", color="#E74C3C", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(sub["target_label"], rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Count", fontsize=9)
            ax.set_title(sig, fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
        fig.tight_layout()
        p = self.out / "V20_return_counts.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V21 — Median drift bar chart
    # =====================================================================
    def plot_median_drift(self, drift_df: pd.DataFrame) -> Optional[Path]:
        """Bar chart of mean median drift per signal x target."""
        if drift_df is None or drift_df.empty:
            return None
        signals = drift_df["signal_name"].unique()
        fig, axes = plt.subplots(1, len(signals), figsize=(5 * len(signals), 5), squeeze=False)
        fig.suptitle("Median Drift (Post-event - Pre-event Median)",
                     fontsize=12, fontweight="bold")
        for ax, sig in zip(axes[0], signals):
            sub = drift_df[drift_df["signal_name"] == sig].sort_values("mean_drift")
            colors = [_target_color(l) for l in sub["target_label"]]
            ax.barh(sub["target_label"], sub["mean_drift"],
                    color=colors, edgecolor="white")
            ax.axvline(0, color="#333", lw=0.8)
            ax.set_xlabel("Mean median drift", fontsize=9)
            ax.set_title(sig, fontsize=10, fontweight="bold")
        fig.tight_layout()
        p = self.out / "V21_median_drift.png"
        _save(fig, p, self.dpi)
        return p

    # =====================================================================
    # V22 — Threshold overlay traces (per signal)
    # =====================================================================
    def plot_threshold_traces(
        self,
        signals: Dict[str, pd.DataFrame],
        threshold_df: pd.DataFrame,
        mad_factor: float = 3.0,
    ) -> Dict[str, Path]:
        """Signal trace with adaptive threshold overlay for each signal."""
        saved = {}
        for sig_name, df in sorted(signals.items()):
            if df is None or "timestamp_s" not in df.columns:
                continue
            vc = VALUE_COLS.get(sig_name, [None])[0] if VALUE_COLS.get(sig_name) else None
            if vc is None or vc not in df.columns:
                continue

            # Subsample for plotting
            ts = df["timestamp_s"].values
            vals = df[vc].values.astype(float)
            step = max(1, len(ts) // 50_000)
            ts_s, vals_s = ts[::step], vals[::step]

            # Rolling median and MAD
            ser = pd.Series(vals_s)
            win = max(3, int(300 * (1.0 / (np.median(np.diff(ts_s)) + 1e-9))))
            med = ser.rolling(win, min_periods=3, center=True).median().values
            abs_dev = np.abs(vals_s - med)
            mad = pd.Series(abs_dev).rolling(win, min_periods=3, center=True).median().values
            upper = med + mad_factor * mad
            lower = med - mad_factor * mad

            fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
            c = SIGNAL_COLORS.get(sig_name, "#555")
            ax.plot(ts_s, vals_s, color=c, lw=0.5, alpha=0.7, label="Signal")
            ax.plot(ts_s, med, color="#333", lw=1.2, label="Running median")
            ax.fill_between(ts_s, lower, upper, alpha=0.12, color="#E74C3C",
                            label=f"+/-{mad_factor}xMAD")

            # Highlight crossings
            if threshold_df is not None and not threshold_df.empty:
                thr_sig = threshold_df[threshold_df["signal"] == sig_name]
                for _, row in thr_sig.iterrows():
                    ax.axvspan(row["start_s"], row["end_s"], color="#FF0000", alpha=0.15)

            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel(f"{vc} ({SIGNAL_UNITS.get(sig_name, '')})", fontsize=9)
            ax.set_title(f"{sig_name} — Adaptive Threshold Detection",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, loc="upper right")
            fig.tight_layout()
            p = self.out / f"V22_threshold_{sig_name}.png"
            _save(fig, p, self.dpi)
            saved[sig_name] = p
        return saved

    # =====================================================================
    # V23 — 3D time-synchronised signal projection
    # =====================================================================
    def plot_3d_signals(
        self, signals: Dict[str, pd.DataFrame],
    ) -> Optional[Path]:
        """3D Plotly: Time x Channel x Normalised Amplitude, coloured by target.

        Baseline is subsampled aggressively (1/50) to let event segments
        stand out visually; event data keeps every Nth sample where N is
        chosen to cap total points at ~20k per signal.
        """
        if not HAS_PLOTLY:
            print("  [Vis] Plotly not available — skipping 3D signal plot")
            return None

        SIG_ORDER = {"EDA": 0, "BVP": 1, "IBI": 2, "ST": 3, "ACC": 4}
        VAL_MAP = {"EDA": "EDA_uS", "BVP": "BVP_nT", "IBI": "IBI_ms",
                   "ST": "ST_degC", "ACC": "ACC_X_g"}

        fig = go.Figure()
        added = set()
        for sn, yp in sorted(SIG_ORDER.items(), key=lambda x: x[1]):
            if sn not in signals:
                continue
            df = signals[sn]
            col = VAL_MAP.get(sn)
            if col is None or col not in df.columns or "timestamp_s" not in df.columns:
                continue

            has_label = "target_label" in df.columns

            # Smart subsample: keep more event data, less baseline
            if has_label:
                is_event = df["target_label"].values != "baseline"
                n_event = is_event.sum()
                n_baseline = len(df) - n_event
                # Event: subsample to ~10k points
                event_step = max(1, n_event // 10_000)
                # Baseline: subsample much more aggressively (~5k points)
                baseline_step = max(1, n_baseline // 5_000)

                idx_event = np.where(is_event)[0][::event_step]
                idx_baseline = np.where(~is_event)[0][::baseline_step]
                idx = np.sort(np.concatenate([idx_event, idx_baseline]))
            else:
                step = max(1, len(df) // 15_000)
                idx = np.arange(0, len(df), step)

            ts = df["timestamp_s"].values[idx]
            vals = df[col].values[idx].astype(float)
            mu, sd = np.nanmean(vals), np.nanstd(vals) + 1e-9
            z = (vals - mu) / sd
            labels = df["target_label"].values[idx] if has_label else np.array(["signal"] * len(ts))

            # Segment by label
            i = 0
            while i < len(ts):
                lbl = str(labels[i])
                j = i
                while j < len(ts) and str(labels[j]) == lbl:
                    j += 1
                show = lbl not in added
                # Thinner line for baseline so events pop
                lw = 1 if lbl == "baseline" else 3
                opacity = 0.3 if lbl == "baseline" else 1.0
                fig.add_trace(go.Scatter3d(
                    x=ts[i:j], y=np.full(j - i, yp), z=z[i:j],
                    mode="lines",
                    line=dict(color=_target_color(lbl), width=lw),
                    opacity=opacity,
                    name=lbl, legendgroup=lbl, showlegend=show,
                ))
                if show:
                    added.add(lbl)
                i = j

        tick_text = [s for s in sorted(SIG_ORDER, key=SIG_ORDER.get) if s in signals]
        tick_vals = [SIG_ORDER[s] for s in tick_text]
        fig.update_layout(
            title="3D Time-Synchronised Signal Projection (events emphasised)",
            scene=dict(
                xaxis_title="Time (s)",
                yaxis=dict(title="Channel", tickvals=tick_vals, ticktext=tick_text),
                zaxis_title="Z-score amplitude",
                camera=dict(eye=dict(x=1.8, y=-1.6, z=0.8)),
            ),
            height=700,
        )
        png_path = self.out / "V23_3d_signals.png"
        try:
            fig.write_image(str(png_path), scale=1.5)
            print(f"  [Vis] Saved V23_3d_signals.png")
        except Exception:
            fig.write_html(str(self.out / "V23_3d_signals.html"))
            print(f"  [Vis] Saved V23_3d_signals.html (kaleido not available)")
            png_path = self.out / "V23_3d_signals.html"
        return png_path

    # =====================================================================
    # V24 — PCA 3D projection coloured by target
    # =====================================================================
    def plot_pca_3d(
        self, signals: Dict[str, pd.DataFrame],
    ) -> Optional[Path]:
        """PCA 3D projection of aligned signal values, coloured by target."""
        if not HAS_PLOTLY:
            return None
        from sklearn.decomposition import PCA

        # Build aligned matrix
        aligned = {}
        for sn, df in sorted(signals.items()):
            if df is None or "timestamp_s" not in df.columns:
                continue
            vc = VALUE_COLS.get(sn, [None])[0] if VALUE_COLS.get(sn) else None
            if vc is None or vc not in df.columns:
                continue
            aligned[vc] = df[["timestamp_s", vc]].dropna().sort_values("timestamp_s")

        if len(aligned) < 3:
            return None
        names = list(aligned.keys())
        base = aligned[names[0]]
        for n in names[1:]:
            base = pd.merge_asof(base.sort_values("timestamp_s"),
                                 aligned[n].sort_values("timestamp_s"),
                                 on="timestamp_s", tolerance=0.5, direction="nearest")

        # Add target labels
        ref_df = None
        for df in signals.values():
            if df is not None and "target_label" in df.columns:
                ref_df = df
                break
        if ref_df is not None:
            label_df = ref_df[["timestamp_s", "target_label"]].drop_duplicates("timestamp_s").sort_values("timestamp_s")
            base = pd.merge_asof(base.sort_values("timestamp_s"),
                                 label_df, on="timestamp_s", tolerance=0.5, direction="nearest")

        base = base.dropna(subset=names)
        if len(base) > MAX_SCATTER_POINTS:
            base = base.sample(MAX_SCATTER_POINTS, random_state=42)
        if len(base) < 50:
            return None

        X = base[names].values
        pca = PCA(n_components=3)
        coords = pca.fit_transform(X)
        labels = base["target_label"].values if "target_label" in base.columns else ["unknown"] * len(base)

        fig = go.Figure()
        added = set()
        for lbl in sorted(set(labels)):
            mask = labels == lbl
            show = lbl not in added
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0], y=coords[mask, 1], z=coords[mask, 2],
                mode="markers", marker=dict(size=2, color=_target_color(str(lbl)), opacity=0.6),
                name=str(lbl), legendgroup=str(lbl), showlegend=show,
            ))
            added.add(lbl)

        var_exp = pca.explained_variance_ratio_
        fig.update_layout(
            title=f"PCA 3D Projection (var explained: {var_exp[0]:.1%}, {var_exp[1]:.1%}, {var_exp[2]:.1%})",
            scene=dict(
                xaxis_title=f"PC1 ({var_exp[0]:.1%})",
                yaxis_title=f"PC2 ({var_exp[1]:.1%})",
                zaxis_title=f"PC3 ({var_exp[2]:.1%})",
            ),
            height=700,
        )
        png_path = self.out / "V24_pca_3d.png"
        try:
            fig.write_image(str(png_path), scale=1.5)
            print(f"  [Vis] Saved V24_pca_3d.png")
        except Exception:
            fig.write_html(str(self.out / "V24_pca_3d.html"))
            print(f"  [Vis] Saved V24_pca_3d.html")
            png_path = self.out / "V24_pca_3d.html"
        return png_path

    # =====================================================================
    # V25 — 3D demographic-signal interaction
    # =====================================================================
    def plot_demographic_3d(
        self, signals: Dict[str, pd.DataFrame],
        demographics: pd.DataFrame,
    ) -> Optional[Path]:
        """3D scatter: severity x signal_mean x signal_std per user."""
        if not HAS_PLOTLY or demographics is None or demographics.empty:
            return None
        if "autism_severity" not in demographics.columns:
            return None

        # Compute per-user signal means
        rows = []
        for sig_name, df in sorted(signals.items()):
            if df is None or "user_id" not in df.columns:
                continue
            vc = VALUE_COLS.get(sig_name, [None])[0] if VALUE_COLS.get(sig_name) else None
            if vc is None or vc not in df.columns:
                continue
            for uid, udf in df.groupby("user_id"):
                vals = udf[vc].dropna()
                if len(vals) < 10:
                    continue
                rows.append({
                    "user_id": uid, "signal": sig_name,
                    "mean": float(vals.mean()), "std": float(vals.std()),
                })
        if not rows:
            return None

        user_stats = pd.DataFrame(rows)
        user_stats = user_stats.merge(
            demographics[["user_id", "autism_severity"]].drop_duplicates(),
            on="user_id", how="left",
        )
        sev_map = {"Low": 1, "Medium": 2, "Severe": 3}
        user_stats["severity_num"] = user_stats["autism_severity"].map(sev_map).fillna(0)

        fig = go.Figure()
        for sig in user_stats["signal"].unique():
            sub = user_stats[user_stats["signal"] == sig]
            fig.add_trace(go.Scatter3d(
                x=sub["severity_num"], y=sub["mean"], z=sub["std"],
                mode="markers",
                marker=dict(size=5, color=SIGNAL_COLORS.get(sig, "#888"), opacity=0.7),
                name=sig, text=sub["user_id"],
            ))
        fig.update_layout(
            title="Demographic-Signal 3D Interaction (Severity x Mean x Std)",
            scene=dict(
                xaxis=dict(title="Severity", tickvals=[1, 2, 3],
                           ticktext=["Low", "Medium", "Severe"]),
                yaxis_title="Mean signal value",
                zaxis_title="Std signal value",
            ),
            height=700,
        )
        png_path = self.out / "V25_demographic_3d.png"
        try:
            fig.write_image(str(png_path), scale=1.5)
            print(f"  [Vis] Saved V25_demographic_3d.png")
        except Exception:
            fig.write_html(str(self.out / "V25_demographic_3d.html"))
            print(f"  [Vis] Saved V25_demographic_3d.html")
            png_path = self.out / "V25_demographic_3d.html"
        return png_path
