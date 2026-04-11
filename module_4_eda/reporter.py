"""
=============================================================================
MODULE 4 – EXPLORATORY DATA ANALYSIS  |  reporter.py
=============================================================================
Generates a comprehensive HTML report + summary CSVs.

Output:
  analysis_report.html   — full human-readable report with embedded plots
  summary_tables/        — all analysis results as CSV
=============================================================================
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

from config import MODULE_VERSION, MODULE_LABEL


class AnalysisReporter:
    """Generate comprehensive report from all analysis results."""

    def __init__(self, output_dir: str | Path):
        self.out = Path(output_dir)
        self.tables_dir = self.out / "summary_tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        session_id: str,
        user_id: str,
        metadata: dict,
        descriptive_df: pd.DataFrame,
        kruskal_df: pd.DataFrame,
        pairwise_df: pd.DataFrame,
        corr_target_df: pd.DataFrame,
        corr_matrix_df: pd.DataFrame,
        dynamics_df: pd.DataFrame,
        return_df: pd.DataFrame,
        pct_change_df: pd.DataFrame,
        threshold_df: pd.DataFrame,
        plot_paths: Dict[str, Path],
    ) -> Path:
        """Write all tables as CSVs and produce the HTML report."""

        # ── Save CSV tables ───────────────────────────────────────────────
        self._save_csv(descriptive_df, "descriptive_statistics.csv")
        self._save_csv(kruskal_df, "kruskal_wallis_results.csv")
        self._save_csv(pairwise_df, "pairwise_mannwhitney.csv")
        self._save_csv(corr_target_df, "correlation_feature_target.csv")
        self._save_csv(corr_matrix_df, "correlation_matrix.csv")
        self._save_csv(dynamics_df, "temporal_dynamics.csv")
        self._save_csv(return_df, "return_to_median_summary.csv")
        self._save_csv(pct_change_df, "pct_change_summary.csv")
        self._save_csv(threshold_df, "threshold_events.csv")

        # ── Build HTML ────────────────────────────────────────────────────
        html = self._build_html(
            session_id, user_id, metadata,
            descriptive_df, kruskal_df, corr_target_df,
            dynamics_df, return_df, pct_change_df,
            threshold_df, plot_paths,
        )
        report_path = self.out / "analysis_report.html"
        report_path.write_text(html, encoding="utf-8")
        print(f"  [Reporter] Saved analysis_report.html")
        return report_path

    def _save_csv(self, df: pd.DataFrame, name: str):
        if df is None or (hasattr(df, "empty") and df.empty):
            return
        df.to_csv(self.tables_dir / name, index=False, float_format="%.4f")
        print(f"  [Reporter] Saved summary_tables/{name}  ({len(df)} rows)")

    def _build_html(self, sid, uid, meta, desc, kw, corr_t,
                    dyn, ret, pct, thr, plots) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        targets = sorted(desc["target_label"].unique()) if not desc.empty and "target_label" in desc.columns else []

        # Key stats
        n_sig_features = int(kw["significant"].sum()) if not kw.empty and "significant" in kw.columns else 0
        n_thr_events = len(thr) if not thr.empty else 0
        top_features = corr_t.head(5)["feature"].tolist() if not corr_t.empty else []

        def _df_to_html(df, max_rows=20, classes="table"):
            if df is None or (hasattr(df, "empty") and df.empty):
                return "<p><em>No data available.</em></p>"
            return df.head(max_rows).to_html(
                index=False, classes=classes,
                border=0, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x
            )

        def _img(path: Path, caption="", width="100%") -> str:
            if path is None or not Path(path).exists():
                return ""
            rel = Path(path).name
            return (f'<figure><img src="{rel}" style="width:{width};'
                    f'border-radius:6px;margin:12px 0;">'
                    f'<figcaption style="font-size:12px;color:#666;">{caption}</figcaption>'
                    f'</figure>')

        def _iframe(path: Path, caption="") -> str:
            if path is None or not Path(path).exists():
                return ""
            return (f'<p><a href="{Path(path).name}" target="_blank" '
                    f'style="padding:8px 18px;background:#3498DB;color:#fff;'
                    f'border-radius:4px;text-decoration:none;font-size:13px;">'
                    f'🌐 Open Interactive 3D Plot</a>'
                    f'<span style="margin-left:12px;font-size:12px;color:#666;">{caption}</span></p>')

        css = """
<style>
body{font-family:Arial,sans-serif;margin:0;padding:0;background:#f5f6fa;color:#333}
.header{background:linear-gradient(135deg,#1B3A5C,#2E7D92);color:#fff;padding:32px 40px}
.header h1{margin:0;font-size:26px;font-weight:700}
.header p{margin:4px 0;font-size:13px;opacity:0.85}
.content{max-width:1400px;margin:0 auto;padding:24px 40px}
.section{background:#fff;border-radius:10px;padding:24px;margin-bottom:24px;
         box-shadow:0 1px 6px rgba(0,0,0,0.08)}
h2{font-size:18px;font-weight:700;color:#1B3A5C;border-left:4px solid #3498DB;
   padding-left:12px;margin-top:0}
h3{font-size:14px;font-weight:700;color:#2E7D92;margin-top:20px}
.stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:16px 0}
.stat-card{background:#EBF4FB;border-radius:8px;padding:16px;text-align:center}
.stat-val{font-size:28px;font-weight:700;color:#1B3A5C}
.stat-lbl{font-size:11px;color:#666;margin-top:4px}
.table{width:100%;border-collapse:collapse;font-size:12px;margin:12px 0}
.table th{background:#1B3A5C;color:#fff;padding:8px 10px;text-align:left}
.table td{padding:7px 10px;border-bottom:1px solid #eee}
.table tr:hover td{background:#F0F4F8}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;
       font-weight:600;margin:2px}
.badge-large{background:#FADBD8;color:#C0392B}
.badge-medium{background:#FAE5D3;color:#E67E22}
.badge-small{background:#D5F5E3;color:#27AE60}
.badge-neg{background:#EBF5FB;color:#2E86C1}
.insight{background:#EBF5FB;border-left:4px solid #3498DB;padding:12px 16px;
         border-radius:4px;margin:12px 0;font-size:13px}
.target-chip{display:inline-block;padding:3px 10px;border-radius:12px;
             font-size:11px;font-weight:600;color:#fff;margin:3px}
@media print{.header{background:#1B3A5C!important;-webkit-print-color-adjust:exact}}
</style>"""

        target_chips = "".join(
            f'<span class="target-chip" style="background:{_tcolor(t)}">{t}</span>'
            for t in targets
        )

        # Pct change table snippet
        pct_html = _df_to_html(pct.head(15)) if not pct.empty else "<p>No data</p>"
        # Top KW features
        kw_top = kw[kw["significant"]].head(10) if not kw.empty and "significant" in kw.columns else pd.DataFrame()
        kw_html = _df_to_html(kw_top)
        corr_html = _df_to_html(corr_t.head(10))
        ret_html = _df_to_html(ret)
        dyn_html = _df_to_html(dyn.head(15)) if not dyn.empty else "<p>No data</p>"
        thr_html = _df_to_html(thr) if not thr.empty else "<p>No threshold events detected.</p>"

        # Qualitative insights
        insights = self._generate_insights(desc, kw, corr_t, dyn, ret, pct, thr)

        threshold_imgs = "".join(
            _img(plots.get(f"threshold_{s}"),
                 f"{s} with adaptive threshold overlay")
            for s in ("EDA", "BVP", "ST", "ACC", "IBI")
        )
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Module 5 Analysis Report — {sid}</title>
{css}
</head>
<body>
<div class="header">
  <h1>🧠 Autism Physio-AI Pipeline — Data Analysis Report</h1>
  <p><strong>Module {MODULE_LABEL} v{MODULE_VERSION}</strong> &nbsp;|&nbsp;
     Session: <strong>{sid}</strong> &nbsp;|&nbsp;
     User: <strong>{uid}</strong> &nbsp;|&nbsp;
     Generated: {now}</p>
</div>
<div class="content">

<!-- KEY STATS -->
<div class="section">
  <h2>Summary</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-val">{len(targets)}</div><div class="stat-lbl">Target classes</div></div>
    <div class="stat-card"><div class="stat-val">{n_sig_features}</div><div class="stat-lbl">Significant features (KW)</div></div>  # noqa: E501
    <div class="stat-card"><div class="stat-val">{len(dyn) if not dyn.empty else 0}</div><div class="stat-lbl">Events analysed</div></div>  # noqa: E501
    <div class="stat-card"><div class="stat-val">{n_thr_events}</div><div class="stat-lbl">Threshold crossings</div></div>  # noqa: E501
  </div>
  <p><strong>Target labels in this session:</strong><br>{target_chips}</p>
  <p><strong>Top correlated features:</strong> {", ".join(top_features) or "—"}</p>
</div>

<!-- QUALITATIVE INSIGHTS -->
<div class="section">
  <h2>Qualitative & Quantitative Insights</h2>
  {"".join(f'<div class="insight">{i}</div>' for i in insights)}
</div>

<!-- DESCRIPTIVE -->
<div class="section">
  <h2>1. Descriptive Analysis</h2>
  <h3>Signal % Change from Baseline per Target</h3>
  {pct_html}
  {_img(plots.get("pct_change"), "Mean % change from pre-event baseline for each signal and target")}
  {_img(plots.get("descriptive_EDA"), "EDA feature distributions per target")}
  {_img(plots.get("descriptive_BVP"), "BVP feature distributions per target")}
  {_img(plots.get("descriptive_IBI"), "IBI/HRV feature distributions per target")}
  {_img(plots.get("descriptive_ST"), "ST feature distributions per target")}
  {_img(plots.get("descriptive_ACC"), "ACC feature distributions per target")}
</div>

<!-- STATISTICAL -->
<div class="section">
  <h2>2. Statistical Analysis</h2>
  <h3>Kruskal-Wallis H-test (most significant features)</h3>
  {kw_html}
  {_img(plots.get("kruskal"), "Significant features ranked by p-value")}
</div>

<!-- CORRELATION -->
<div class="section">
  <h2>3. Correlation Analysis</h2>
  <h3>Feature–Target Correlation (Spearman ρ)</h3>
  {corr_html}
  {_img(plots.get("corr_target"), "Spearman correlation between features and target label")}
  {_img(plots.get("corr_matrix"), "Feature–feature correlation matrix (top features by variance)")}
</div>

<!-- TEMPORAL DYNAMICS -->
<div class="section">
  <h2>4. Temporal Dynamics</h2>
  <h3>Event-level dynamics (time to peak, subside, return)</h3>
  {dyn_html}
  {_img(plots.get("temporal"), "Time to peak, subside, and return distributions by signal and target")}
  <h3>Return-to-median rate</h3>
  {ret_html}
  {_img(plots.get("return_rate"), "Percentage of events where signal returned to pre-event median")}
  <h3>Median drift across events</h3>
  {_img(plots.get("median_drift"), "How much the running median shifts after events")}
</div>

<!-- THRESHOLD -->
<div class="section">
  <h2>5. Adaptive Threshold Analysis</h2>
  <p>Threshold: signal deviates &gt; {meta.get("mad_factor", 3.0)}×MAD from running median
     for &gt; {meta.get("sustain_s", 30)}s within a {meta.get("window_s", 300)}s window.</p>
  {thr_html}
  {threshold_imgs}
</div>

<!-- 3D VISUALISATION -->
<div class="section">
  <h2>6. 3D Time-Synchronised Signal Projection</h2>
  <p>All signals projected into a single 3D space (Time × Channel × Normalised Amplitude).
     Each trace is coloured by its annotated target label. Rotate and zoom interactively.</p>
  {_iframe(plots.get("3d_html"), "All signals — time-synchronised and colour-coded by target")}
  {_img(plots.get("3d_png"), "3D projection (static preview)")}
</div>

</div><!-- end content -->
</body>
</html>"""
        return html

    def _generate_insights(self, desc, kw, corr_t, dyn, ret, pct, thr) -> list:
        insights = []

        # Significant features
        if not kw.empty and "significant" in kw.columns:
            n_sig = kw["significant"].sum()
            n_tot = len(kw)
            top3 = kw[kw["significant"]].head(3)["feature"].tolist() if n_sig > 0 else []
            insights.append(
                f"<strong>Statistical discrimination:</strong> {n_sig} of {n_tot} features "
                f"show significant differences across target states (Kruskal-Wallis, α=0.05). "
                + (f"Top features: {', '.join(top3)}." if top3 else "")
            )

        # Pct change
        if not pct.empty and "mean_change_pct" in pct.columns:
            max_row = pct.loc[pct["mean_change_pct"].abs().idxmax()]
            insights.append(
                f"<strong>Largest signal response:</strong> {max_row['signal']} shows "
                f"{max_row['mean_change_pct']:+.1f}% mean change from baseline during "
                f"<em>{max_row['target_label']}</em> events."
            )

        # Return to median
        if not ret.empty and "pct_returned" in ret.columns:
            low_return = ret[ret["pct_returned"] < 50]
            if not low_return.empty:
                lr = low_return.iloc[0]
                insights.append(
                    f"<strong>Sustained activation:</strong> {lr['signal']} returns to baseline "
                    f"in only {lr['pct_returned']:.0f}% of <em>{lr['target_label']}</em> events — "
                    f"suggesting prolonged autonomic activation."
                )

        # Threshold crossings
        if not thr.empty:
            insights.append(
                f"<strong>Threshold alerts:</strong> {len(thr)} sustained threshold crossings "
                f"detected across signals. These represent periods of significant physiological "
                f"deviation from the running baseline."
            )

        # Temporal dynamics
        if not dyn.empty and "peak_delay_s" in dyn.columns:
            fast = dyn[dyn["peak_delay_s"] == dyn["peak_delay_s"].min()].iloc[0]
            insights.append(
                f"<strong>Fastest signal response:</strong> {fast['signal']} peaks "
                f"{fast['peak_delay_s']:.1f}s after event onset for "
                f"<em>{fast['target_label']}</em> — "
                f"{'consistent with rapid sympathetic activation.' if fast['peak_delay_s'] < 5 else 'reflecting a slower physiological response.'}"  # noqa: E501
            )

        if not insights:
            insights.append("Insufficient data for automatic qualitative insights. Review tables and plots above.")
        return insights


def _tcolor(target: str) -> str:
    from config import TARGET_COLORS
    return TARGET_COLORS.get(target, "#888888")
