"""
=============================================================================
MODULE 8 - MODEL EVALUATION  |  visualizer.py
=============================================================================
32 chart functions for comprehensive model evaluation visualisation.
All functions return the saved file path for report embedding.
=============================================================================
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm

from config import (
    PLOT_DPI, PLOT_FIGSIZE, PLOT_FIGSIZE_WIDE,
    HIGH_PRIORITY_STATES, COMPARISON_METRICS, METRIC_DISPLAY,
)

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=0.9)

# Colour palette
MODEL_PALETTE = sns.color_palette("Set2", 12)
CLASS_PALETTE = sns.color_palette("tab10", 12)


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 -- Test Set Performance
# ═══════════════════════════════════════════════════════════════════════════

def v01_confusion_matrices(
    eval_results: list, class_names: List[str], out_dir: Path
) -> Path:
    """V01: Multi-panel confusion matrices for all models."""
    valid = [r for r in eval_results if not r.skipped and "confusion_matrix" in r.test_metrics]
    n = len(valid)
    if n == 0:
        return _empty_plot(out_dir / "v01_confusion_matrices.png")

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, r in enumerate(valid):
        cm = np.array(r.test_metrics["confusion_matrix"])
        ax = axes[i]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar=False,
        )
        ax.set_title(r.model_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Test Set Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir / "v01_confusion_matrices.png")


def v02_roc_curves(
    eval_results: list, y_test: np.ndarray, class_names: List[str], out_dir: Path
) -> Path:
    """V02: ROC curves overlaid per class for each model."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    valid = [r for r in eval_results if not r.skipped and r.y_prob is not None]
    if not valid:
        return _empty_plot(out_dir / "v02_roc_curves.png")

    n = len(valid)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    y_bin = label_binarize(y_test, classes=list(range(len(class_names))))

    for i, r in enumerate(valid):
        ax = axes[i]
        for c_idx, cls in enumerate(class_names):
            if r.y_prob.shape[1] <= c_idx or y_bin[:, c_idx].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, c_idx], r.y_prob[:, c_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{cls} ({roc_auc:.2f})",
                    color=CLASS_PALETTE[c_idx % len(CLASS_PALETTE)], linewidth=1.2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
        ax.set_title(r.model_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=6, loc="lower right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Test Set ROC Curves (per class)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir / "v02_roc_curves.png")


def v03_precision_recall_curves(
    eval_results: list, y_test: np.ndarray, class_names: List[str], out_dir: Path
) -> Path:
    """V03: Precision-recall curves per class per model."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    valid = [r for r in eval_results if not r.skipped and r.y_prob is not None]
    if not valid:
        return _empty_plot(out_dir / "v03_pr_curves.png")

    n = len(valid)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    y_bin = label_binarize(y_test, classes=list(range(len(class_names))))

    for i, r in enumerate(valid):
        ax = axes[i]
        for c_idx, cls in enumerate(class_names):
            if r.y_prob.shape[1] <= c_idx or y_bin[:, c_idx].sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(y_bin[:, c_idx], r.y_prob[:, c_idx])
            ap = average_precision_score(y_bin[:, c_idx], r.y_prob[:, c_idx])
            ax.plot(rec, prec, label=f"{cls} (AP={ap:.2f})",
                    color=CLASS_PALETTE[c_idx % len(CLASS_PALETTE)], linewidth=1.2)

        ax.set_title(r.model_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=6, loc="lower left")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Test Set Precision-Recall Curves", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir / "v03_pr_curves.png")


def v04_metric_comparison_bar(
    comparison_df: pd.DataFrame, out_dir: Path
) -> Path:
    """V04: Grouped bar chart comparing all models across metrics."""
    if comparison_df.empty:
        return _empty_plot(out_dir / "v04_metric_comparison.png")

    metrics = [m for m in COMPARISON_METRICS if m in comparison_df.columns]
    display_names = [METRIC_DISPLAY.get(m, m) for m in metrics]

    df_melt = comparison_df.melt(
        id_vars=["model"], value_vars=metrics,
        var_name="metric", value_name="value",
    )
    df_melt["metric"] = df_melt["metric"].map(
        lambda m: METRIC_DISPLAY.get(m, m)
    )

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    sns.barplot(data=df_melt, x="metric", y="value", hue="model",
                palette=MODEL_PALETTE, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Test Set Metric Comparison", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return _save(fig, out_dir / "v04_metric_comparison.png")


def v05_per_class_f1_heatmap(
    eval_results: list, class_names: List[str], out_dir: Path
) -> Path:
    """V05: Heatmap of per-class F1 scores (models x classes)."""
    valid = [r for r in eval_results if not r.skipped and "per_class" in r.test_metrics]
    if not valid:
        return _empty_plot(out_dir / "v05_per_class_f1.png")

    data = {}
    for r in valid:
        pc = r.test_metrics["per_class"]
        data[r.model_name] = {cls: pc.get(cls, {}).get("f1", 0) for cls in class_names}

    df = pd.DataFrame(data).T
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), max(4, len(valid) * 0.6)))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1,
                ax=ax, linewidths=0.5)
    ax.set_title("Per-Class F1 Score (Models x Classes)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model")
    ax.set_xlabel("Class")
    fig.tight_layout()
    return _save(fig, out_dir / "v05_per_class_f1.png")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4 -- Generalization Gap
# ═══════════════════════════════════════════════════════════════════════════

def v06_train_val_test_comparison(
    gap_table: pd.DataFrame, out_dir: Path
) -> Path:
    """V06: Grouped bar of val vs test metrics per model."""
    if gap_table.empty:
        return _empty_plot(out_dir / "v06_val_test_comparison.png")

    models = gap_table["model"].tolist()
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)

    metric = "f1_weighted"
    val_col = f"val_{metric}"
    test_col = f"test_{metric}"

    if val_col in gap_table.columns and test_col in gap_table.columns:
        x = np.arange(len(models))
        w = 0.35
        ax.bar(x - w / 2, gap_table[val_col], w, label="Validation", color="#5B9BD5")
        ax.bar(x + w / 2, gap_table[test_col], w, label="Test", color="#ED7D31")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel("F1 (weighted)")
        ax.set_title("Validation vs Test F1", fontsize=13, fontweight="bold")
        ax.legend()
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    return _save(fig, out_dir / "v06_val_test_comparison.png")


def v07_overfitting_heatmap(
    gap_matrix: pd.DataFrame, out_dir: Path
) -> Path:
    """V07: Heatmap of generalisation gaps (models x metrics)."""
    if gap_matrix.empty:
        return _empty_plot(out_dir / "v07_overfitting_heatmap.png")

    display_cols = {m: METRIC_DISPLAY.get(m, m) for m in gap_matrix.columns}
    df = gap_matrix.rename(columns=display_cols)

    vmax = max(abs(df.values[np.isfinite(df.values)].max()),
               abs(df.values[np.isfinite(df.values)].min()), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.2), max(4, len(df) * 0.6)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn_r", norm=norm,
                ax=ax, linewidths=0.5, center=0)
    ax.set_title("Generalisation Gap (Val − Test): Positive = Overfit",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir / "v07_overfitting_heatmap.png")


def v08_generalization_waterfall(
    gen_results: dict, out_dir: Path
) -> Path:
    """V08: Waterfall chart showing val->test metric change per model."""
    if not gen_results:
        return _empty_plot(out_dir / "v08_gen_waterfall.png")

    metric = "f1_weighted"
    models = []
    vals = []
    drops = []
    for name, gr in gen_results.items():
        if metric in gr.val_metrics and metric in gr.test_metrics:
            models.append(name)
            vals.append(gr.val_metrics[metric])
            drops.append(gr.gaps.get(metric, 0))

    if not models:
        return _empty_plot(out_dir / "v08_gen_waterfall.png")

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    x = np.arange(len(models))
    colors = ["#E74C3C" if d > 0.05 else "#F39C12" if d > 0 else "#27AE60" for d in drops]

    ax.bar(x, [-d for d in drops], bottom=[v for v in vals], color=colors, alpha=0.8)
    ax.scatter(x, vals, marker="_", s=200, color="blue", zorder=5, label="Val F1")
    ax.scatter(x, [v - d for v, d in zip(vals, drops)], marker="_", s=200,
               color="green", zorder=5, label="Test F1")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("F1 (weighted)")
    ax.set_title("Generalisation Waterfall: Val -> Test", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir / "v08_gen_waterfall.png")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5 -- Calibration
# ═══════════════════════════════════════════════════════════════════════════

def v09_reliability_diagrams(
    cal_results: dict, out_dir: Path
) -> Path:
    """V09: Reliability diagrams (multi-panel)."""
    valid = {k: v for k, v in cal_results.items() if not v.skipped and v.ece is not None}
    if not valid:
        return _empty_plot(out_dir / "v09_reliability_diagrams.png")

    n = len(valid)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (name, cr) in enumerate(valid.items()):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
        ax.bar(cr.bin_midpoints, cr.bin_accuracies, width=1.0 / len(cr.bin_midpoints),
               alpha=0.6, color="#5B9BD5", edgecolor="white")
        ax.plot(cr.bin_midpoints, cr.bin_confidences, "r-o", markersize=3,
                label="Mean conf.", linewidth=1.2)
        ax.set_title(f"{name}\nECE={cr.ece:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Reliability Diagrams", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir / "v09_reliability_diagrams.png")


def v10_ece_comparison(cal_results: dict, out_dir: Path) -> Path:
    """V10: ECE comparison bar chart."""
    valid = {k: v for k, v in cal_results.items() if not v.skipped and v.ece is not None}
    if not valid:
        return _empty_plot(out_dir / "v10_ece_comparison.png")

    names = list(valid.keys())
    eces = [valid[n].ece for n in names]
    briers = [valid[n].brier_score for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_FIGSIZE_WIDE)

    ax1.barh(names, eces, color="#5B9BD5")
    ax1.axvline(0.05, color="green", linestyle="--", alpha=0.5, label="Good (<5%)")
    ax1.set_xlabel("ECE")
    ax1.set_title("Expected Calibration Error", fontweight="bold")
    ax1.legend(fontsize=8)

    ax2.barh(names, briers, color="#ED7D31")
    ax2.set_xlabel("Brier Score")
    ax2.set_title("Brier Score (lower = better)", fontweight="bold")

    fig.tight_layout()
    return _save(fig, out_dir / "v10_ece_comparison.png")


def v11_confidence_histogram(
    eval_results: list, out_dir: Path
) -> Path:
    """V11: Confidence distribution histogram per model."""
    valid = [r for r in eval_results if not r.skipped and r.y_prob is not None]
    if not valid:
        return _empty_plot(out_dir / "v11_confidence_hist.png")

    n = len(valid)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, r in enumerate(valid):
        ax = axes[i]
        max_probs = r.y_prob.max(axis=1)
        ax.hist(max_probs, bins=30, color="#5B9BD5", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(max_probs), color="red", linestyle="--",
                   label=f"Mean={np.mean(max_probs):.2f}")
        ax.set_title(r.model_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Max Probability")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Prediction Confidence Distribution", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir / "v11_confidence_hist.png")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6 -- Clinical Utility
# ═══════════════════════════════════════════════════════════════════════════

def v12_hp_fnr_comparison(
    clinical_results: dict, out_dir: Path
) -> Path:
    """V12: High-priority FNR comparison grouped bar."""
    if not clinical_results:
        return _empty_plot(out_dir / "v12_hp_fnr.png")

    rows = []
    for name, cr in clinical_results.items():
        for state, fnr in cr.fnr_high_priority.items():
            rows.append({"model": name, "state": state, "FNR": fnr})

    if not rows:
        return _empty_plot(out_dir / "v12_hp_fnr.png")

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    sns.barplot(data=df, x="state", y="FNR", hue="model", palette=MODEL_PALETTE, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("False Negative Rate")
    ax.set_title("High-Priority State FNR (lower = better)", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir / "v12_hp_fnr.png")


def v13_clinical_cost_heatmap(
    clinical_results: dict, class_names: List[str], out_dir: Path
) -> Path:
    """V13: Clinical cost heatmap (models x classes)."""
    if not clinical_results:
        return _empty_plot(out_dir / "v13_clinical_cost.png")

    data = {}
    for name, cr in clinical_results.items():
        data[name] = {cls: cr.cost_by_class.get(cls, {}).get("total_cost", 0)
                      for cls in class_names}

    df = pd.DataFrame(data).T
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), max(4, len(data) * 0.6)))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title("Clinical Cost by Class (lower = better)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir / "v13_clinical_cost.png")


def v14_nns_bar(clinical_results: dict, out_dir: Path) -> Path:
    """V14: Number Needed to Screen bar chart."""
    if not clinical_results:
        return _empty_plot(out_dir / "v14_nns.png")

    rows = []
    for name, cr in clinical_results.items():
        for state in sorted(HIGH_PRIORITY_STATES):
            nns = cr.nns_by_class.get(state, float("inf"))
            if np.isfinite(nns):
                rows.append({"model": name, "state": state, "NNS": nns})

    if not rows:
        return _empty_plot(out_dir / "v14_nns.png")

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    sns.barplot(data=df, x="state", y="NNS", hue="model", palette=MODEL_PALETTE, ax=ax)
    ax.set_ylabel("Number Needed to Screen")
    ax.set_title("NNS for High-Priority States (lower = better)", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir / "v14_nns.png")


def v15_sensitivity_specificity(
    clinical_results: dict, class_names: List[str], out_dir: Path
) -> Path:
    """V15: Sensitivity vs Specificity scatter plot."""
    if not clinical_results:
        return _empty_plot(out_dir / "v15_sens_spec.png")

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    for i, (name, cr) in enumerate(clinical_results.items()):
        for cls in class_names:
            sens = cr.sensitivity_by_class.get(cls, 0)
            spec = cr.specificity_by_class.get(cls, 0)
            hp = cls in HIGH_PRIORITY_STATES
            ax.scatter(1 - spec, sens, color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                       marker="*" if hp else "o", s=100 if hp else 40,
                       alpha=0.7)

    # Legend for models
    for i, name in enumerate(clinical_results.keys()):
        ax.scatter([], [], color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                   label=name, s=40)
    ax.scatter([], [], marker="*", color="gray", s=100, label="High-priority")
    ax.plot([0, 1], [1, 0], "k--", alpha=0.2)
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("Sensitivity vs Specificity", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    return _save(fig, out_dir / "v15_sens_spec.png")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7+8 -- Demographics & Equity
# ═══════════════════════════════════════════════════════════════════════════

def v16_f1_by_severity(
    subgroup_df: pd.DataFrame, out_dir: Path
) -> Path:
    """V16: F1 by autism severity grouped bar."""
    return _demographic_bar(subgroup_df, "Autism Severity", out_dir, "v16_f1_severity.png")


def v17_f1_by_verbal(
    subgroup_df: pd.DataFrame, out_dir: Path
) -> Path:
    """V17: F1 by verbal status grouped bar."""
    return _demographic_bar(subgroup_df, "Verbal Status", out_dir, "v17_f1_verbal.png")


def v18_equity_radar(
    demo_results: dict, out_dir: Path
) -> Path:
    """V18: Equity radar chart showing F1 disparity across dimensions."""
    if not demo_results:
        return _empty_plot(out_dir / "v18_equity_radar.png")

    # Use the best model's subgroup metrics
    best_model = list(demo_results.keys())[0]
    dr = demo_results[best_model]

    dimensions = {}
    for sm in dr.subgroup_metrics:
        if sm.dimension not in dimensions:
            dimensions[sm.dimension] = {}
        dimensions[sm.dimension][sm.subgroup] = sm.f1_weighted

    if not dimensions:
        return _empty_plot(out_dir / "v18_equity_radar.png")

    # Build radar: one line per dimension, subgroups as spokes
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, subplot_kw=dict(polar=True))
    all_subgroups = []
    all_values = []
    for dim, subs in dimensions.items():
        for sg, val in subs.items():
            all_subgroups.append(f"{dim}\n{sg}")
            all_values.append(val)

    if not all_subgroups:
        return _empty_plot(out_dir / "v18_equity_radar.png")

    angles = np.linspace(0, 2 * np.pi, len(all_subgroups), endpoint=False).tolist()
    all_values_plot = all_values + [all_values[0]]
    angles_plot = angles + [angles[0]]

    ax.plot(angles_plot, all_values_plot, "o-", linewidth=1.5, color="#5B9BD5")
    ax.fill(angles_plot, all_values_plot, alpha=0.15, color="#5B9BD5")
    ax.set_xticks(angles)
    ax.set_xticklabels(all_subgroups, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title(f"Equity Radar -- {best_model}", fontsize=12, fontweight="bold", pad=20)

    fig.tight_layout()
    return _save(fig, out_dir / "v18_equity_radar.png")


def v19_disparity_heatmap(
    equity_df: pd.DataFrame, out_dir: Path
) -> Path:
    """V19: Disparity significance heatmap (models x dimensions)."""
    if equity_df.empty:
        return _empty_plot(out_dir / "v19_disparity_heatmap.png")

    pivot = equity_df.pivot_table(
        index="model", columns="dimension", values="p_value", aggfunc="min"
    )

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=0.1,
                ax=ax, linewidths=0.5)
    ax.set_title("Equity Test p-values (green = no disparity)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir / "v19_disparity_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 9 -- Robustness
# ═══════════════════════════════════════════════════════════════════════════

def v20_noise_degradation(
    rob_results: dict, out_dir: Path
) -> Path:
    """V20: Noise degradation curves."""
    if not rob_results:
        return _empty_plot(out_dir / "v20_noise_degradation.png")

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    for i, (name, rr) in enumerate(rob_results.items()):
        levels = [0.0] + [nr.noise_level for nr in rr.noise_results]
        f1s = [rr.clean_f1] + [nr.f1_weighted for nr in rr.noise_results]
        ax.plot(levels, f1s, "o-", label=name,
                color=MODEL_PALETTE[i % len(MODEL_PALETTE)], linewidth=1.5)

    ax.set_xlabel("Noise Level (σ x feature std)")
    ax.set_ylabel("F1 (weighted)")
    ax.set_title("Performance Under Gaussian Noise", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir / "v20_noise_degradation.png")


def v21_channel_dropout(
    rob_results: dict, out_dir: Path
) -> Path:
    """V21: Channel dropout impact bar chart."""
    if not rob_results:
        return _empty_plot(out_dir / "v21_channel_dropout.png")

    rows = []
    for name, rr in rob_results.items():
        for cr in rr.channel_results:
            rows.append({
                "model": name,
                "dropped": "+".join(cr.dropped_channels),
                "f1_drop": cr.f1_drop,
            })

    if not rows:
        return _empty_plot(out_dir / "v21_channel_dropout.png")

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    sns.barplot(data=df, x="dropped", y="f1_drop", hue="model",
                palette=MODEL_PALETTE, ax=ax)
    ax.set_xlabel("Dropped Channels")
    ax.set_ylabel("F1 Drop")
    ax.set_title("Impact of Channel Dropout", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return _save(fig, out_dir / "v21_channel_dropout.png")


def v22_missing_feature_impact(
    rob_results: dict, out_dir: Path
) -> Path:
    """V22: Missing feature degradation curves."""
    if not rob_results:
        return _empty_plot(out_dir / "v22_missing_features.png")

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    for i, (name, rr) in enumerate(rob_results.items()):
        fracs = [0.0] + [mr.missing_fraction for mr in rr.missing_results]
        f1s = [rr.clean_f1] + [mr.f1_weighted for mr in rr.missing_results]
        ax.plot(fracs, f1s, "o-", label=name,
                color=MODEL_PALETTE[i % len(MODEL_PALETTE)], linewidth=1.5)

    ax.set_xlabel("Fraction of Features Missing")
    ax.set_ylabel("F1 (weighted)")
    ax.set_title("Performance Under Missing Features", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir / "v22_missing_features.png")


def v23_robustness_ranking(
    rob_results: dict, out_dir: Path
) -> Path:
    """V23: Robustness ranking table visualisation."""
    if not rob_results:
        return _empty_plot(out_dir / "v23_robustness_ranking.png")

    data = []
    for name, rr in rob_results.items():
        data.append({
            "Model": name,
            "Clean F1": f"{rr.clean_f1:.3f}",
            "Noise AUC": f"{rr.noise_auc:.4f}",
            "Missing AUC": f"{rr.missing_auc:.4f}",
            "Worst Ch. Drop": f"{rr.worst_channel_drop:.3f}",
            "Score": f"{rr.robustness_score:.3f}",
        })

    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(10, max(2, len(data) * 0.5 + 1)))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("Robustness Ranking", fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    return _save(fig, out_dir / "v23_robustness_ranking.png")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 10 -- Statistical Comparison
# ═══════════════════════════════════════════════════════════════════════════

def v24_mcnemar_heatmap(
    mcnemar_matrix: pd.DataFrame, out_dir: Path
) -> Path:
    """V24: McNemar's p-value heatmap."""
    if mcnemar_matrix.empty:
        return _empty_plot(out_dir / "v24_mcnemar_heatmap.png")

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    mask = np.eye(len(mcnemar_matrix), dtype=bool)
    sns.heatmap(mcnemar_matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=0.1, mask=mask, ax=ax, linewidths=0.5)
    ax.set_title("McNemar's Test p-values (green = same errors)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir / "v24_mcnemar_heatmap.png")


def v25_delong_forest(
    comparison: Any, out_dir: Path
) -> Path:
    """V25: DeLong AUC comparison forest plot."""
    tests = comparison.delong_tests if comparison else []
    if not tests:
        return _empty_plot(out_dir / "v25_delong_forest.png")

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    labels = []
    diffs = []
    for t in tests:
        labels.append(f"{t.model_a} vs {t.model_b}")
        diffs.append(t.auc_a - t.auc_b)

    y_pos = range(len(labels))
    colors = ["#E74C3C" if abs(d) > 0.05 else "#3498DB" for d in diffs]
    ax.barh(y_pos, diffs, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("AUC Difference")
    ax.set_title("AUC-ROC Comparison (DeLong/Bootstrap)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir / "v25_delong_forest.png")


def v26_cochran_q(comparison: Any, out_dir: Path) -> Path:
    """V26: Cochran's Q result visualisation."""
    if not comparison or not comparison.cochran_q:
        return _empty_plot(out_dir / "v26_cochran_q.png")

    q = comparison.cochran_q
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    text = (
        f"Cochran's Q Test\n\n"
        f"Q statistic: {q.q_statistic:.2f}\n"
        f"p-value: {q.p_value:.4f}\n"
        f"Degrees of freedom: {q.df}\n"
        f"N models: {q.n_models}\n\n"
        f"Verdict: {'Models differ significantly' if q.significant else 'No significant difference'}"
    )
    ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=12,
            verticalalignment="center", horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#E8F0FE", alpha=0.8))
    fig.tight_layout()
    return _save(fig, out_dir / "v26_cochran_q.png")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 11 -- Error Analysis
# ═══════════════════════════════════════════════════════════════════════════

def v27_confused_pairs(
    error_results: dict, out_dir: Path
) -> Path:
    """V27: Most confused class pairs bar chart (best model)."""
    if not error_results:
        return _empty_plot(out_dir / "v27_confused_pairs.png")

    best = list(error_results.values())[0]
    pairs = best.confused_pairs[:10]
    if not pairs:
        return _empty_plot(out_dir / "v27_confused_pairs.png")

    labels = [f"{cp.true_class}->{cp.pred_class}" for cp in pairs]
    counts = [cp.count for cp in pairs]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.barh(range(len(labels)), counts, color="#E74C3C", alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title(f"Most Confused Pairs -- {best.model_name}", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, out_dir / "v27_confused_pairs.png")


def v28_per_user_error(
    error_results: dict, out_dir: Path
) -> Path:
    """V28: Per-user error rate scatter plot."""
    if not error_results:
        return _empty_plot(out_dir / "v28_user_errors.png")

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    for i, (name, ea) in enumerate(error_results.items()):
        if not ea.user_profiles:
            continue
        users = [up.user_id for up in ea.user_profiles]
        rates = [up.error_rate for up in ea.user_profiles]
        sizes = [up.n_samples for up in ea.user_profiles]
        ax.scatter(range(len(users)), rates,
                   s=[max(20, s / 5) for s in sizes],
                   label=name, alpha=0.6,
                   color=MODEL_PALETTE[i % len(MODEL_PALETTE)])

    ax.set_xlabel("User Index")
    ax.set_ylabel("Error Rate")
    ax.set_title("Per-User Error Rates", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir / "v28_user_errors.png")


def v29_misclassification_flow(
    error_results: dict, class_names: List[str], out_dir: Path
) -> Path:
    """V29: Misclassification flow (chord-style heatmap, since Sankey needs plotly)."""
    if not error_results:
        return _empty_plot(out_dir / "v29_misclass_flow.png")

    best = list(error_results.values())[0]
    if not best.misclassification_flow:
        return _empty_plot(out_dir / "v29_misclass_flow.png")

    # Build flow matrix
    n = len(class_names)
    flow = np.zeros((n, n))
    cls_idx = {c: i for i, c in enumerate(class_names)}
    for true_cls, pred_cls, count in best.misclassification_flow:
        if true_cls in cls_idx and pred_cls in cls_idx:
            flow[cls_idx[true_cls], cls_idx[pred_cls]] = count

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    sns.heatmap(flow, annot=True, fmt=".0f", cmap="Reds",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Misclassification Flow -- {best.model_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir / "v29_misclass_flow.png")


def v30_failure_tsne(
    eval_results: list, y_test: np.ndarray, X_test: np.ndarray,
    class_names: List[str], out_dir: Path
) -> Path:
    """V30: t-SNE of test set coloured by correct/incorrect (best model)."""
    valid = [r for r in eval_results if not r.skipped and r.y_pred is not None]
    if not valid or X_test.shape[0] > 10000:
        # Skip for very large test sets (t-SNE is slow)
        return _empty_plot(out_dir / "v30_failure_tsne.png")

    try:
        from sklearn.manifold import TSNE

        best = valid[0]
        correct = best.y_pred == y_test

        # Subsample if needed
        n = min(3000, X_test.shape[0])
        rng = np.random.default_rng(42)
        idx = rng.choice(X_test.shape[0], size=n, replace=False)

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n - 1))
        embedding = tsne.fit_transform(X_test[idx])

        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        correct_sub = correct[idx]
        ax.scatter(embedding[correct_sub, 0], embedding[correct_sub, 1],
                   c="#27AE60", alpha=0.4, s=15, label="Correct")
        ax.scatter(embedding[~correct_sub, 0], embedding[~correct_sub, 1],
                   c="#E74C3C", alpha=0.6, s=25, label="Error", marker="x")
        ax.set_title(f"t-SNE -- {best.model_name} Errors", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.tight_layout()
        return _save(fig, out_dir / "v30_failure_tsne.png")
    except Exception:
        return _empty_plot(out_dir / "v30_failure_tsne.png")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 12 -- Inference Profiling
# ═══════════════════════════════════════════════════════════════════════════

def v31_latency_comparison(
    prof_results: dict, out_dir: Path
) -> Path:
    """V31: Latency comparison bar chart (single sample)."""
    if not prof_results:
        return _empty_plot(out_dir / "v31_latency.png")

    names = list(prof_results.keys())
    latencies = [prof_results[n].single_sample_ms for n in names]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    colors = ["#27AE60" if l < 100 else "#E74C3C" for l in latencies]
    ax.barh(names, latencies, color=colors, alpha=0.8)
    ax.axvline(100, color="red", linestyle="--", alpha=0.5, label="Real-time threshold (100ms)")
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Single-Sample Inference Latency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir / "v31_latency.png")


def v32_pareto_frontier(
    eval_results: list, prof_results: dict, out_dir: Path
) -> Path:
    """V32: Pareto frontier -- F1 vs latency."""
    if not prof_results:
        return _empty_plot(out_dir / "v32_pareto.png")

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    for i, er in enumerate(eval_results):
        if er.skipped or er.model_name not in prof_results:
            continue
        f1 = er.test_metrics.get("f1_weighted", 0)
        lat = prof_results[er.model_name].single_sample_ms
        ax.scatter(lat, f1, s=100, color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                   zorder=5)
        ax.annotate(er.model_name, (lat, f1), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")

    ax.axhline(0.8, color="green", linestyle="--", alpha=0.3, label="F1=0.8 target")
    ax.axvline(100, color="red", linestyle="--", alpha=0.3, label="100ms threshold")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("F1 (weighted)")
    ax.set_title("Pareto Frontier: F1 vs Latency", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir / "v32_pareto.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _empty_plot(path: Path) -> Path:
    """Create an empty placeholder plot."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, "No data available", ha="center", va="center",
            fontsize=14, color="gray")
    ax.axis("off")
    return _save(fig, path)


def _demographic_bar(
    subgroup_df: pd.DataFrame, dimension: str, out_dir: Path, filename: str
) -> Path:
    """Generic demographic grouped bar chart."""
    if subgroup_df.empty:
        return _empty_plot(out_dir / filename)

    subset = subgroup_df[subgroup_df["dimension"] == dimension]
    if subset.empty:
        return _empty_plot(out_dir / filename)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_WIDE)
    sns.barplot(data=subset, x="subgroup", y="f1_weighted", hue="model",
                palette=MODEL_PALETTE, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(dimension)
    ax.set_ylabel("F1 (weighted)")
    ax.set_title(f"F1 by {dimension}", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir / filename)
