"""
=============================================================================
MODULE 7 -- MODEL TRAINING  |  exporter.py
=============================================================================
Export trained models, metrics, and comparison reports. Generates an HTML
comparison report with ranked tables and radar charts.
=============================================================================
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import PRIMARY_METRIC, PLOT_DPI, MAX_INFERENCE_LATENCY_MS, MAX_MODEL_SIZE_MB


class ModelExporter:
    """Export model artefacts, metrics, and comparison reports."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Exporter] {msg}")

    def export_comparison_html(
        self,
        comparison_df: pd.DataFrame,
        results: list,
        output_dir: Path,
    ) -> Path:
        """Generate an HTML comparison report with tables and radar chart."""
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "comparison_report.html"

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<meta charset='utf-8'>",
            "<title>Module 7 — Model Comparison Report</title>",
            "<style>",
            "body { font-family: 'Segoe UI', Arial, sans-serif; "
            "margin: 2em; background: #fafafa; color: #333; }",
            "h1 { color: #1a365d; }",
            "h2 { color: #2c5282; margin-top: 2em; }",
            "table { border-collapse: collapse; width: 100%; "
            "margin: 1em 0; background: white; box-shadow: "
            "0 1px 3px rgba(0,0,0,0.1); }",
            "th, td { padding: 8px 12px; text-align: left; "
            "border-bottom: 1px solid #e2e8f0; }",
            "th { background: #2c5282; color: white; font-weight: 600; }",
            "tr:hover { background: #edf2f7; }",
            ".best { background: #c6f6d5 !important; font-weight: bold; }",
            ".skipped { color: #a0aec0; font-style: italic; }",
            ".metric-good { color: #276749; }",
            ".metric-warn { color: #c05621; }",
            ".metric-bad { color: #c53030; }",
            ".summary-box { background: white; padding: 1.5em; "
            "border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); "
            "margin: 1em 0; }",
            "</style>",
            "</head><body>",
            "<h1>Module 7 — Model Training Comparison Report</h1>",
        ]

        # Summary
        n_ok = sum(1 for r in results if not r.skipped)
        n_skip = sum(1 for r in results if r.skipped)
        best = None
        for r in results:
            if (not r.skipped and r.model_type == "supervised"
                    and PRIMARY_METRIC in r.metrics):
                if best is None or (r.metrics[PRIMARY_METRIC]
                                     > best.metrics[PRIMARY_METRIC]):
                    best = r

        html_parts.append("<div class='summary-box'>")
        html_parts.append(f"<p><strong>Models trained:</strong> {n_ok} | "
                           f"<strong>Skipped:</strong> {n_skip}</p>")
        if best:
            html_parts.append(
                f"<p><strong>Best model:</strong> {best.model_name} "
                f"({PRIMARY_METRIC} = "
                f"{best.metrics[PRIMARY_METRIC]:.4f})</p>")
        html_parts.append("</div>")

        # Supervised table
        html_parts.append("<h2>Supervised Models</h2>")
        html_parts.append(self._results_to_html_table(
            [r for r in results if r.model_type == "supervised"],
            best_name=best.model_name if best else None))

        # Unsupervised table
        unsup = [r for r in results if r.model_type == "unsupervised"]
        if unsup:
            html_parts.append("<h2>Unsupervised Models</h2>")
            html_parts.append(self._results_to_html_table(unsup))

        # Deployment feasibility
        html_parts.append("<h2>Deployment Feasibility</h2>")
        html_parts.append(self._deployment_table(results))

        # Per-model details
        html_parts.append("<h2>Per-Model Details</h2>")
        for r in results:
            if r.skipped:
                continue
            html_parts.append(f"<h3>{r.model_name}</h3>")
            html_parts.append(f"<p>Framework: {r.framework} | "
                               f"Time: {r.training_time_s:.1f}s | "
                               f"Size: {r.model_size_kb:.1f} KB</p>")
            if "per_class" in r.metrics:
                html_parts.append("<table><tr><th>Class</th>"
                                   "<th>Sensitivity</th>"
                                   "<th>Specificity</th>"
                                   "<th>F1</th><th>Support</th></tr>")
                for cls, vals in r.metrics["per_class"].items():
                    html_parts.append(
                        f"<tr><td>{cls}</td>"
                        f"<td>{vals['sensitivity']:.3f}</td>"
                        f"<td>{vals['specificity']:.3f}</td>"
                        f"<td>{vals['f1']:.3f}</td>"
                        f"<td>{vals['support']}</td></tr>")
                html_parts.append("</table>")

        html_parts.extend(["</body></html>"])

        html_path.write_text("\n".join(html_parts), encoding="utf-8")
        self._log(f"HTML report: {html_path.name}")
        return html_path

    def _results_to_html_table(self, results, best_name=None):
        """Convert results list to HTML table."""
        cols = ["Model", "Status", "F1 Weighted", "F1 Macro",
                "Accuracy", "Kappa", "Time (s)", "Size (KB)"]
        rows = ["<table>", "<tr>" + "".join(
            f"<th>{c}</th>" for c in cols) + "</tr>"]

        for r in results:
            cls = ""
            if r.skipped:
                cls = " class='skipped'"
                rows.append(
                    f"<tr{cls}><td>{r.model_name}</td>"
                    f"<td>SKIPPED</td>"
                    f"<td colspan='6'>{r.skip_reason}</td></tr>")
                continue

            if r.model_name == best_name:
                cls = " class='best'"

            m = r.metrics
            rows.append(
                f"<tr{cls}>"
                f"<td>{r.model_name}</td>"
                f"<td>OK</td>"
                f"<td>{m.get('f1_weighted', 0):.4f}</td>"
                f"<td>{m.get('f1_macro', 0):.4f}</td>"
                f"<td>{m.get('accuracy', 0):.4f}</td>"
                f"<td>{m.get('cohens_kappa', 0):.4f}</td>"
                f"<td>{r.training_time_s:.1f}</td>"
                f"<td>{r.model_size_kb:.1f}</td>"
                f"</tr>")

        rows.append("</table>")
        return "\n".join(rows)

    def _deployment_table(self, results):
        """Table showing deployment feasibility per model."""
        rows = [
            "<table><tr><th>Model</th><th>Latency (ms)</th>"
            "<th>Size (KB)</th><th>Mobile Feasible</th></tr>"
        ]
        for r in results:
            if r.skipped or r.model_type != "supervised":
                continue
            lat = r.inference_latency_ms
            size = r.model_size_kb
            feasible = (lat <= MAX_INFERENCE_LATENCY_MS
                        and size <= MAX_MODEL_SIZE_MB * 1024)
            cls = "metric-good" if feasible else "metric-warn"
            rows.append(
                f"<tr><td>{r.model_name}</td>"
                f"<td>{lat:.1f}</td>"
                f"<td>{size:.1f}</td>"
                f"<td class='{cls}'>{'Yes' if feasible else 'No'}</td>"
                f"</tr>")
        rows.append("</table>")
        return "\n".join(rows)

    def export_radar_chart(self, results: list, output_dir: Path) -> Path:
        """Save radar chart comparing top supervised models."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Filter non-skipped supervised
        valid = [r for r in results
                 if not r.skipped and r.model_type == "supervised"
                 and "f1_weighted" in r.metrics]
        if len(valid) < 2:
            return output_dir

        metrics_keys = ["f1_weighted", "f1_macro", "accuracy",
                        "cohens_kappa", "precision_weighted",
                        "recall_weighted"]
        labels = ["F1 Weighted", "F1 Macro", "Accuracy",
                  "Kappa", "Precision", "Recall"]

        n_metrics = len(metrics_keys)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])

        fig, ax = plt.subplots(figsize=(8, 8),
                                subplot_kw=dict(projection="polar"))

        for r in valid[:6]:  # Top 6
            values = [r.metrics.get(k, 0) for k in metrics_keys]
            values.append(values[0])  # Close polygon
            ax.plot(angles, values, "-o", label=r.model_name,
                    markersize=4, linewidth=1.5)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("Model Comparison — Radar Chart", pad=20,
                      fontsize=13)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                  fontsize=8)
        fig.tight_layout()

        path = output_dir / "radar_chart.png"
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        self._log(f"Radar chart: {path.name}")
        return path
