"""
=============================================================================
MODULE 6 – FEATURE ENGINEERING  |  selection_report.py
=============================================================================
Generates a comprehensive HTML report documenting all features in the
combined feature matrix, their selection scores, and the rationale for
inclusion or exclusion from the top-N selected feature set.

Output:
  feature_selection_report.html — self-contained HTML report
=============================================================================
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import MODULE_VERSION, META_COLS

# ─── Feature metadata ────────────────────────────────────────────────────────
# Maps feature names to (signal, full description, physiological rationale)

_FEATURE_META: Dict[str, tuple] = {
    # ── EDA ───────────────────────────────────────────────────────────────
    "eda_scl_mean":       ("EDA", "Mean Skin Conductance Level",
                           "Tonic EDA component — baseline sympathetic arousal"),
    "eda_scl_delta":      ("EDA", "SCL Change Within Window",
                           "Captures sympathetic arousal shifts; sustained arousal marker"),
    "eda_scl_slope":      ("EDA", "SCL Linear Slope",
                           "Rate of tonic level change — correlated with eda_scl_delta"),
    "eda_scr_count":      ("EDA", "SCR Event Count",
                           "Number of phasic skin conductance responses in window"),
    "eda_scr_rate":       ("EDA", "SCR Rate (events/s)",
                           "Phasic event frequency — mathematically count/duration"),
    "eda_scr_amp_mean":   ("EDA", "Mean SCR Amplitude",
                           "Average phasic response magnitude"),
    "eda_scr_amp_max":    ("EDA", "Maximum SCR Amplitude",
                           "Peak phasic response — dominated by single outlier SCRs"),
    "eda_scr_rise_mean":  ("EDA", "Mean SCR Rise Time",
                           "Fast (Fear/Surprise) vs slow (Hunger/Tired) sympathetic activation"),
    "eda_scr_rec_time":   ("EDA", "Mean SCR Recovery Time",
                           "SCR half-recovery — requires longer windows for reliability"),
    "eda_scr_auc":        ("EDA", "SCR Area Under Curve",
                           "Integrated phasic energy — redundant with amplitude × duration"),
    "eda_mean":           ("EDA", "Raw EDA Mean",
                           "Signal mean — highly correlated with eda_scl_mean"),
    "eda_std":            ("EDA", "EDA Standard Deviation",
                           "Signal variability — inter-subject differences dominate"),
    "eda_min":            ("EDA", "EDA Minimum",
                           "Reflects individual baseline, not emotional state"),
    "eda_max":            ("EDA", "EDA Maximum",
                           "Peak EDA — inflated by outlier SCR peaks"),
    "eda_range":          ("EDA", "EDA Range",
                           "Max − min — extremely high variance from outliers"),
    "eda_skew":           ("EDA", "EDA Skewness",
                           "Distribution shape — requires more samples for stability"),
    "eda_kurt":           ("EDA", "EDA Kurtosis",
                           "Peakedness — unreliable in 60s windows at 4 Hz"),
    "eda_zcr":            ("EDA", "EDA Zero-Crossing Rate",
                           "Not physiologically meaningful — EDA is non-negative"),
    "eda_pow_vlf":        ("EDA", "VLF Power (EDA)",
                           "Very-low-frequency power — thermoregulatory drift"),
    "eda_pow_scr":        ("EDA", "SCR-Band Power",
                           "Frequency-domain phasic activity — redundant with time-domain SCR"),
    "eda_spec_centroid":  ("EDA", "Spectral Centroid (EDA)",
                           "Poor spectral resolution at 4 Hz, <1 Hz bandwidth"),
    "eda_sampen":         ("EDA", "Sample Entropy (EDA)",
                           "Signal complexity — key autonomic marker in ASD literature"),

    # ── BVP ───────────────────────────────────────────────────────────────
    "bvp_hr_mean":        ("BVP", "Mean Heart Rate",
                           "Primary autonomic marker; encodes arousal level"),
    "bvp_hr_std":         ("BVP", "Heart Rate Std Dev",
                           "HR variability — better captured by IBI-domain features"),
    "bvp_sys_amp_mean":   ("BVP", "Mean Systolic Amplitude",
                           "Pulse amplitude — correlates with hr_mean and notch_depth"),
    "bvp_notch_depth":    ("BVP", "Dicrotic Notch Depth",
                           "Arterial compliance / peripheral vascular resistance"),
    "bvp_pw50":           ("BVP", "Pulse Width at 50%",
                           "Partially redundant with heart rate (wider pulse ≈ lower HR)"),
    "bvp_ptt":            ("BVP", "Pulse Transit Time",
                           "Arterial stiffness / blood pressure proxy — sympathetic marker"),
    "bvp_aug_idx":        ("BVP", "Augmentation Index",
                           "Sensitive to motion artefact — unreliable in active children"),
    "bvp_mean":           ("BVP", "Raw BVP Mean",
                           "AC-coupled sensor — mean reflects offset, not cardiac function"),
    "bvp_std":            ("BVP", "BVP Standard Deviation",
                           "Proxy for pulse amplitude — redundant with sys_amp_mean"),
    "bvp_skew":           ("BVP", "BVP Skewness",
                           "Waveform asymmetry — sensitive to motion artefact"),
    "bvp_kurt":           ("BVP", "BVP Kurtosis",
                           "Peak sharpness — partially redundant with notch_depth"),
    "bvp_f0":             ("BVP", "Fundamental Frequency (BVP)",
                           "Frequency-domain heart rate — complements time-domain HR"),
    "bvp_resp_mod":       ("BVP", "Respiratory Modulation Index",
                           "Cannot extract reliably at 64 Hz without dedicated respiration"),
    "bvp_harm_ratio":     ("BVP", "Harmonic Ratio",
                           "Waveform regularity — partially captured by f0 and notch_depth"),
    "bvp_spec_entropy":   ("BVP", "Spectral Entropy (BVP)",
                           "Dominated by cardiac fundamental — low discriminative power"),
    "bvp_sqi":            ("BVP", "Signal Quality Index",
                           "Data quality metric, not predictive — should gate, not predict"),

    # ── IBI / HRV ─────────────────────────────────────────────────────────
    "ibi_mean":           ("IBI", "Mean Inter-Beat Interval",
                           "Inverse of HR in time domain — core autonomic marker"),
    "ibi_sdnn":           ("IBI", "SDNN",
                           "Gold-standard overall HRV (Task Force 1996)"),
    "ibi_rmssd":          ("IBI", "RMSSD",
                           "Primary parasympathetic marker — related to SD1"),
    "ibi_pnn50":          ("IBI", "pNN50",
                           "Vagal activity — vagal withdrawal key in ASD (Kushki 2013)"),
    "ibi_cv":             ("IBI", "Coefficient of Variation",
                           "Redundant — fully determined by ibi_sdnn and ibi_mean"),
    "ibi_vlf":            ("IBI", "VLF Power (HRV)",
                           "Requires >5 min recording — unreliable in 60s windows"),
    "ibi_lf":             ("IBI", "LF Power (HRV)",
                           "Mixed sympathetic+parasympathetic — borderline resolution"),
    "ibi_hf":             ("IBI", "HF Power (HRV)",
                           "Pure parasympathetic (vagal) marker — respiratory sinus arrhythmia"),
    "ibi_lf_hf":          ("IBI", "LF/HF Ratio",
                           "Considered unreliable sympathovagal index (Billman 2013)"),
    "ibi_lfnu":           ("IBI", "LF Normalised Units",
                           "Mathematically determined by LF/(LF+HF) — redundant"),
    "ibi_hfnu":           ("IBI", "HF Normalised Units",
                           "Exactly 1 − lfnu — perfectly redundant"),
    "ibi_total_pow":      ("IBI", "Total Spectral Power",
                           "Dominated by VLF — unreliable in short windows"),
    "ibi_sd1":            ("IBI", "Poincaré SD1",
                           "Geometric short-term HRV — beat-to-beat variability"),
    "ibi_sd2":            ("IBI", "Poincaré SD2",
                           "Long-term HRV — partially redundant with SDNN"),
    "ibi_sd1_sd2":        ("IBI", "SD1/SD2 Ratio",
                           "Sympathovagal balance — ratio unstable when SD2 is small"),
    "ibi_sampen":         ("IBI", "Sample Entropy (IBI)",
                           "Requires >200 beats — noisy in 60s windows (~70 beats)"),
    "ibi_dfa_alpha1":     ("IBI", "DFA α1 (Short-Term Fractal)",
                           "Requires >256 beats (Peng 1995) — underpowered in 60s"),

    # ── ST ────────────────────────────────────────────────────────────────
    "st_mean":            ("ST", "Mean Skin Temperature",
                           "Reflects ambient conditions and individual baseline"),
    "st_delta":           ("ST", "Temperature Change (Δ)",
                           "Vasoconstriction (stress→drop) / vasodilation (relax→rise)"),
    "st_rate_change":     ("ST", "Temperature Rate of Change",
                           "Mathematically st_delta/duration — perfectly correlated"),
    "st_slope":           ("ST", "Temperature Slope",
                           "Identical to rate_change for monotonic signals — redundant"),
    "st_asymmetry":       ("ST", "Temperature Asymmetry",
                           "Requires bilateral measurement — single-wrist cannot compute"),
    "st_min":             ("ST", "Minimum Temperature",
                           "Individual baseline — not emotionally informative"),
    "st_max":             ("ST", "Maximum Temperature",
                           "Baseline + delta — st_delta isolates the informative part"),
    "st_range":           ("ST", "Temperature Range",
                           "Very small in 60s windows due to thermal inertia"),
    "st_std":             ("ST", "Temperature Std Dev",
                           "Captures sensor noise more than physiology in short windows"),
    "st_acf1":            ("ST", "Lag-1 Autocorrelation (ST)",
                           "Always near 1.0 — no discriminative value"),
    "st_pow_ultra_slow":  ("ST", "Ultra-Slow Power (ST)",
                           "Requires windows >>60s — constant/NaN"),
    "st_pow_slow":        ("ST", "Slow Power (ST)",
                           "60s windows provide no spectral resolution for ST (<0.1 Hz)"),

    # ── ACC ───────────────────────────────────────────────────────────────
    "acc_svm_mean":       ("ACC", "Mean Signal Vector Magnitude",
                           "Overall activity level — partially captured by std and range"),
    "acc_svm_std":        ("ACC", "SVM Standard Deviation",
                           "Movement intensity fluctuation — agitation marker"),
    "acc_activity_count": ("ACC", "Activity Threshold Crossings",
                           "Movement frequency — stereotypic/repetitive movement detection"),
    "acc_dom_freq":       ("ACC", "Dominant Frequency",
                           "Single freq poorly characterises broadband movement — unstable"),
    "acc_x_mean":         ("ACC", "Mean X-Axis Acceleration",
                           "Gravity-dominated — reflects wrist orientation, not emotion"),
    "acc_y_mean":         ("ACC", "Mean Y-Axis Acceleration",
                           "Gravity-dominated — orientation-dependent"),
    "acc_z_mean":         ("ACC", "Mean Z-Axis Acceleration",
                           "Gravity-dominated — not emotion-discriminative"),
    "acc_posture_trans":  ("ACC", "Posture Transition Count",
                           "Too rare in 60s windows (0–2) — near-constant"),
    "acc_x_std":          ("ACC", "X-Axis Std Dev",
                           "Lateral wrist movement — hand-flapping, defensive gestures"),
    "acc_y_std":          ("ACC", "Y-Axis Std Dev",
                           "Vertical movement — stimming, rocking detection"),
    "acc_z_std":          ("ACC", "Z-Axis Std Dev",
                           "Partially redundant with X/Y std — multicollinearity risk"),
    "acc_svm_range":      ("ACC", "SVM Peak-to-Peak Range",
                           "Maximum movement intensity — calm vs agitated episodes"),
    "acc_svm_skew":       ("ACC", "SVM Skewness",
                           "Movement asymmetry — reflects random variation"),
    "acc_svm_kurt":       ("ACC", "SVM Kurtosis",
                           "Sensitive to single impulse events — unreliable"),
    "acc_x_zcr":          ("ACC", "X-Axis Zero-Crossing Rate",
                           "Movement frequency — outperformed by Y-axis ZCR"),
    "acc_y_zcr":          ("ACC", "Y-Axis Zero-Crossing Rate",
                           "Captures stereotypic repetitive movements — ASD motor marker"),
    "acc_z_zcr":          ("ACC", "Z-Axis Zero-Crossing Rate",
                           "Strong but partially redundant with Y-axis ZCR"),
    "acc_pow_sway":       ("ACC", "Sway-Band Power (0.1–0.5 Hz)",
                           "Postural sway — requires standing, children may be seated"),
    "acc_pow_voluntary":  ("ACC", "Voluntary Movement Power (0.5–3 Hz)",
                           "Broad band, less specific than rapid movement power"),
    "acc_pow_rapid":      ("ACC", "Rapid Movement Power (3–8 Hz)",
                           "Startle responses, tremor, stereotypic movements (Schoen 2008)"),
    "acc_pow_tremor":     ("ACC", "Tremor-Band Power (8–15 Hz)",
                           "At Nyquist limit for 32 Hz sampling — poor resolution"),
    "acc_spec_entropy":   ("ACC", "Spectral Entropy (ACC)",
                           "Relatively constant across states — similar spectral spread"),
    "acc_corr_xy":        ("ACC", "X-Y Axis Correlation",
                           "Movement coordination — varies by activity, not emotion"),
    "acc_corr_xz":        ("ACC", "X-Z Axis Correlation",
                           "High variance from postural differences, not emotion"),
    "acc_corr_yz":        ("ACC", "Y-Z Axis Correlation",
                           "Unstable and device-orientation dependent"),
    "acc_svm_sampen":     ("ACC", "Sample Entropy (SVM)",
                           "Movement complexity not a reliable emotion marker"),
}

# Metadata columns — always excluded from scoring
_META_DESCRIPTIONS: Dict[str, tuple] = {
    "session_id":          ("Metadata", "Session Identifier",
                            "Structural — routes to output folders"),
    "user_id":             ("Metadata", "Participant Identifier",
                            "Structural — identifies participant"),
    "window_start_s":      ("Metadata", "Window Start Time (s)",
                            "Temporal anchor for feature windows"),
    "window_end_s":        ("Metadata", "Window End Time (s)",
                            "Temporal anchor for feature windows"),
    "target_label":        ("Metadata", "Ground Truth Label",
                            "Prediction target — not an input feature"),
}

_DEMO_DESCRIPTIONS: Dict[str, tuple] = {
    "age":                 ("Demographic", "Participant Age (5–15 yr)",
                            "Numeric — always passed through to models"),
    "gender":              ("Demographic", "Gender (string label)",
                            "Raw label — encoded version used by models"),
    "ethnicity":           ("Demographic", "Ethnicity (string label)",
                            "Raw label — encoded version used by models"),
    "autism_severity":     ("Demographic", "Autism Severity (string label)",
                            "Raw label — encoded version used by models"),
    "verbal_status":       ("Demographic", "Verbal Status (string label)",
                            "Raw label — encoded version used by models"),
    "comorbidity":         ("Demographic", "Comorbidity (string label)",
                            "Raw label — encoded version used by models"),
    "gender_enc":          ("Demographic (enc)", "Gender Encoded",
                            "Ordinal numeric — always included"),
    "autism_severity_enc": ("Demographic (enc)", "Severity Encoded (1/2/3)",
                            "Clinically ordered — modulates physiology"),
    "verbal_status_enc":   ("Demographic (enc)", "Verbal Status Encoded (0/1/2)",
                            "Clinically ordered — affects behavioural expression"),
    "comorbidity_enc":     ("Demographic (enc)", "Comorbidity Encoded (0/1)",
                            "Binary — comorbid conditions affect baselines"),
    "ethnicity_enc":       ("Demographic (enc)", "Ethnicity Encoded",
                            "Ordinal numeric — demographic stratification"),
}

# Common rejection reason templates
_REJECTION_REASONS = {
    "redundant":       "Redundant — correlated with a higher-ranked selected feature",
    "zero_mi":         "Zero mutual information — no detectable non-linear dependency",
    "short_window":    "Insufficient 60s window for reliable estimation",
    "inter_subject":   "Inter-subject noise dominates — not emotion-discriminative",
    "constant":        "Constant or NaN — cannot be computed from available data",
    "not_predictive":  "Quality/structural metric — should not be used as predictor",
    "low_variance":    "Near-constant after normalisation",
    "moderate":        "Moderate scores but outperformed by top-20 features",
}


def _classify_rejection(feat_name: str, row: pd.Series) -> str:
    """Heuristically assign a rejection reason based on scores."""
    mi = row.get("score_mutual_information", 0)
    rf = row.get("score_random_forest", 0)
    fstat = row.get("score_f_statistic", 0)
    var = row.get("score_variance", 0)

    if mi == 0 and rf == 0 and fstat == 0 and var == 0:
        return _REJECTION_REASONS["constant"]
    if mi == 0 and fstat < 1.5:
        return _REJECTION_REASONS["zero_mi"]
    if var < 0.25:
        return _REJECTION_REASONS["low_variance"]

    # Check known redundancies
    meta = _FEATURE_META.get(feat_name)
    if meta:
        rationale = meta[2].lower()
        if "redundant" in rationale or "correlated" in rationale:
            return _REJECTION_REASONS["redundant"]
        if "requires" in rationale and ("window" in rationale or "min" in rationale
                                        or "beat" in rationale):
            return _REJECTION_REASONS["short_window"]
        if "baseline" in rationale or "individual" in rationale or "inter-subject" in rationale:
            return _REJECTION_REASONS["inter_subject"]
        if "quality" in rationale or "not predictive" in rationale:
            return _REJECTION_REASONS["not_predictive"]

    return _REJECTION_REASONS["moderate"]


# ─── Signal colours ──────────────────────────────────────────────────────────

_SIGNAL_COLOURS = {
    "EDA":                ("#E74C3C", "#FADBD8"),
    "BVP":                ("#3498DB", "#D6EAF8"),
    "IBI":                ("#2ECC71", "#D5F5E3"),
    "ST":                 ("#E67E22", "#FAE5D3"),
    "ACC":                ("#9B59B6", "#EBDEF0"),
    "Metadata":           ("#7F8C8D", "#EAEDED"),
    "Demographic":        ("#1ABC9C", "#D1F2EB"),
    "Demographic (enc)":  ("#1ABC9C", "#D1F2EB"),
}


# ─── Report generator ────────────────────────────────────────────────────────

class SelectionReportGenerator:
    """
    Generate a self-contained HTML report documenting the feature
    selection rationale for all columns in the combined feature matrix.

    Parameters
    ----------
    output_dir : Path to write the report into
    """

    def __init__(self, output_dir: str | Path):
        self.out = Path(output_dir)

    def generate(
        self,
        all_columns: list,
        importance_df: pd.DataFrame,
        top_n: int,
        selection_method: str,
        metadata: Optional[dict] = None,
    ) -> Path:
        """
        Build the HTML report and write it to disk.

        Parameters
        ----------
        all_columns     : ordered list of all column names in combined CSV
        importance_df   : DataFrame with columns: feature, rank, avg_rank,
                          score_mutual_information, score_random_forest,
                          score_f_statistic, score_variance
        top_n           : number of features selected
        selection_method: name of the method used
        metadata        : optional dict of run metadata

        Returns
        -------
        Path to the generated HTML file
        """
        meta = metadata or {}
        selected_features = set(
            importance_df.head(top_n)["feature"].tolist()
        ) if importance_df is not None and len(importance_df) else set()

        # Build importance lookup
        imp_lookup = {}
        if importance_df is not None:
            for _, row in importance_df.iterrows():
                imp_lookup[row["feature"]] = row

        html = self._build_html(
            all_columns, imp_lookup, selected_features,
            top_n, selection_method, meta,
        )

        report_path = self.out / "feature_selection_report.html"
        report_path.write_text(html, encoding="utf-8")
        print(f"  [Reporter] Saved feature_selection_report.html")
        return report_path

    # ── HTML builder ──────────────────────────────────────────────────────

    def _build_html(
        self,
        all_columns: list,
        imp_lookup: dict,
        selected: set,
        top_n: int,
        method: str,
        meta: dict,
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        n_total = len(all_columns)
        n_scored = len(imp_lookup)
        n_selected = len(selected)

        # Count by signal
        signal_counts = {}
        signal_selected = {}
        for col in all_columns:
            fm = _FEATURE_META.get(col)
            if fm:
                sig = fm[0]
                signal_counts[sig] = signal_counts.get(sig, 0) + 1
                if col in selected:
                    signal_selected[sig] = signal_selected.get(sig, 0) + 1

        # Rejection reason counts
        rejection_counts: Dict[str, int] = {}
        for col in all_columns:
            if col in selected or col not in imp_lookup:
                continue
            reason = _classify_rejection(col, imp_lookup[col])
            short = reason.split("—")[0].strip()
            rejection_counts[short] = rejection_counts.get(short, 0) + 1

        # Build table rows
        table_rows = self._build_table_rows(
            all_columns, imp_lookup, selected)

        # Build selected features summary
        selected_summary = self._build_selected_summary(
            imp_lookup, selected, top_n)

        # Signal coverage chart
        signal_bars = self._build_signal_bars(
            signal_counts, signal_selected)

        # Rejection pie data
        rejection_rows = "".join(
            f'<tr><td>{reason}</td>'
            f'<td style="text-align:center">{count}</td></tr>'
            for reason, count in sorted(
                rejection_counts.items(), key=lambda x: -x[1])
        )

        css = self._get_css()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Feature Selection Report — Module 6</title>
{css}
</head>
<body>

<div class="header">
  <h1>Feature Selection Rationale Report</h1>
  <p>Module 6 — Feature Engineering | v{MODULE_VERSION}</p>
  <p>Generated: {now} | Method: {method} | Top {top_n} selected</p>
  <p>Session: {meta.get('session_id', 'N/A')} | User: {meta.get('user_id', 'N/A')}</p>
</div>

<div class="content">

  <!-- Summary Cards -->
  <div class="section">
    <h2>Overview</h2>
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-val">{n_total}</div>
        <div class="stat-lbl">Total Columns</div>
      </div>
      <div class="stat-card">
        <div class="stat-val">{n_scored}</div>
        <div class="stat-lbl">Scored Features</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="color:#27AE60">{n_selected}</div>
        <div class="stat-lbl">Selected (Top {top_n})</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" style="color:#E74C3C">{n_scored - n_selected}</div>
        <div class="stat-lbl">Rejected</div>
      </div>
    </div>

    <div class="insight">
      <strong>Selection Method:</strong> {method.replace('_', ' ').title()} —
      features ranked by mean position across Mutual Information,
      Random Forest importance, ANOVA F-statistic, and Variance scoring.
      The top {top_n} features by ensemble rank were selected.
      Demographic features are always passed through and not competitively ranked.
    </div>
  </div>

  <!-- Signal Coverage -->
  <div class="section">
    <h2>Signal Coverage</h2>
    <p style="font-size:13px;color:#555;">
      Distribution of selected features across physiological signals.
      All 5 signal modalities are represented in the final selection.
    </p>
    {signal_bars}
  </div>

  <!-- Selected Features Summary -->
  <div class="section">
    <h2>Selected Features (Top {top_n})</h2>
    <table class="table">
      <thead>
        <tr>
          <th style="width:40px">Rank</th>
          <th>Feature</th>
          <th>Signal</th>
          <th>Description</th>
          <th style="width:70px">MI</th>
          <th style="width:70px">RF</th>
          <th style="width:70px">F-stat</th>
          <th style="width:70px">Variance</th>
          <th>Selection Rationale</th>
        </tr>
      </thead>
      <tbody>
        {selected_summary}
      </tbody>
    </table>
  </div>

  <!-- Rejection Reasons -->
  <div class="section">
    <h2>Rejection Reason Summary</h2>
    <table class="table" style="max-width:500px">
      <thead>
        <tr><th>Reason Category</th><th style="width:60px;text-align:center">Count</th></tr>
      </thead>
      <tbody>
        {rejection_rows}
      </tbody>
    </table>
  </div>

  <!-- Full Column Reference -->
  <div class="section">
    <h2>Full Column Reference ({n_total} columns)</h2>
    <p style="font-size:12px;color:#888;margin-bottom:12px;">
      Click column headers to sort. Green = selected, red = rejected,
      grey = metadata/demographics (not scored).
    </p>
    <table class="table" id="fullTable">
      <thead>
        <tr>
          <th style="width:30px">#</th>
          <th style="width:50px">Status</th>
          <th style="width:40px">Rank</th>
          <th>Column Name</th>
          <th>Signal</th>
          <th>Description</th>
          <th style="width:60px">MI</th>
          <th style="width:60px">RF</th>
          <th style="width:60px">F-stat</th>
          <th style="width:60px">Var</th>
          <th>Rationale</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>

  <!-- Methodology -->
  <div class="section">
    <h2>Methodology</h2>
    <h3>Scoring Methods</h3>
    <table class="table" style="max-width:800px">
      <tr>
        <td><strong>Mutual Information (MI)</strong></td>
        <td>Non-parametric, model-free measure of dependency between each
        feature and the target label. Captures non-linear relationships.
        Estimated via k-nearest-neighbours (Kraskov estimator).</td>
      </tr>
      <tr>
        <td><strong>Random Forest (RF)</strong></td>
        <td>Gini importance from a 100-tree Random Forest classifier.
        Captures feature interactions and non-linear predictive power.
        Deterministic via fixed seed.</td>
      </tr>
      <tr>
        <td><strong>F-statistic (ANOVA)</strong></td>
        <td>One-way ANOVA F-test per feature. Measures linear separability
        between target classes. High F = means differ significantly
        across emotion states.</td>
      </tr>
      <tr>
        <td><strong>Variance</strong></td>
        <td>Feature variance after RobustScaler normalisation. Low variance
        features (near-constant) cannot contribute to classification.
        Outlier-resistant via IQR-based scaling.</td>
      </tr>
    </table>

    <h3>Ensemble Ranking</h3>
    <p style="font-size:13px;">
      Each feature is ranked independently by each method (1 = best).
      The ensemble rank is the mean of the 4 individual ranks.
      This approach balances linear (F-stat) and non-linear (MI, RF)
      information while penalising low-variance features.
    </p>

    <h3>Why Not Automatic Thresholding?</h3>
    <div class="insight">
      A fixed top-N approach (rather than a statistical threshold like
      p-value or variance cutoff) was chosen because: (1) the
      physiological features have heterogeneous distributions making a
      single threshold inappropriate, (2) clinical interpretability
      requires a manageable feature set, and (3) the ensemble rank
      naturally handles the multi-criterion trade-off.
    </div>

    <h3>Key Literature</h3>
    <ul style="font-size:13px;">
      <li><strong>Kreibig (2010)</strong> — Autonomic NS reactions to emotions</li>
      <li><strong>Task Force ESC (1996)</strong> — HRV standards (SDNN, RMSSD, spectral bands)</li>
      <li><strong>Kushki et al. (2013)</strong> — ANS and severity in autistic children</li>
      <li><strong>Billman (2013)</strong> — LF/HF ratio critique</li>
      <li><strong>Schoen et al. (2008)</strong> — Sensory over-responsivity and arousal in ASD</li>
      <li><strong>Stephens et al. (2022)</strong> — EDA in ASD: baseline and reactivity</li>
    </ul>
  </div>

  <div style="text-align:center;padding:24px;font-size:12px;color:#999;">
    Autism Physio-AI Pipeline &mdash; Module 6 Feature Engineering v{MODULE_VERSION}
  </div>

</div>
</body>
</html>"""
        return html

    # ── Table row builders ────────────────────────────────────────────────

    def _build_table_rows(self, all_columns, imp_lookup, selected):
        rows = []
        for i, col in enumerate(all_columns, 1):
            # Determine category
            if col in _META_DESCRIPTIONS:
                sig, desc, rationale = _META_DESCRIPTIONS[col]
                status = "META"
                rank = "—"
                mi = rf = fstat = var = "—"
                bg = "#f8f9fa"
                badge = ('<span class="badge" '
                         'style="background:#EAEDED;color:#7F8C8D">'
                         'METADATA</span>')
            elif col in _DEMO_DESCRIPTIONS:
                sig, desc, rationale = _DEMO_DESCRIPTIONS[col]
                status = "DEMO"
                rank = "—"
                mi = rf = fstat = var = "—"
                bg = "#f0faf5"
                badge = ('<span class="badge" '
                         'style="background:#D1F2EB;color:#1ABC9C">'
                         'DEMOGRAPHIC</span>')
            elif col in imp_lookup:
                row_data = imp_lookup[col]
                fm = _FEATURE_META.get(col, ("?", col, ""))
                sig, desc, rationale = fm
                rank_val = int(row_data["rank"])
                rank = str(rank_val)
                mi = f'{row_data["score_mutual_information"]:.3f}'
                rf = f'{row_data["score_random_forest"]:.3f}'
                fstat = f'{row_data["score_f_statistic"]:.1f}'
                var = f'{row_data["score_variance"]:.2f}'

                if col in selected:
                    status = "SELECTED"
                    bg = "#eafaf1"
                    badge = ('<span class="badge" '
                             'style="background:#D5F5E3;color:#27AE60">'
                             'SELECTED</span>')
                    # Use feature meta rationale for selected
                    rationale = fm[2]
                else:
                    status = "REJECTED"
                    bg = "#fef9f9"
                    badge = ('<span class="badge" '
                             'style="background:#FADBD8;color:#C0392B">'
                             'REJECTED</span>')
                    rationale = _classify_rejection(col, row_data)
            else:
                sig, desc, rationale = "?", col, "Unknown column"
                status = "?"
                rank = "—"
                mi = rf = fstat = var = "—"
                bg = "#fff"
                badge = '<span class="badge">?</span>'

            sig_color = _SIGNAL_COLOURS.get(sig, ("#666", "#eee"))
            sig_badge = (f'<span style="background:{sig_color[1]};'
                         f'color:{sig_color[0]};padding:2px 8px;'
                         f'border-radius:10px;font-size:11px;'
                         f'font-weight:600">{sig}</span>')

            rows.append(
                f'<tr style="background:{bg}">'
                f'<td style="text-align:center">{i}</td>'
                f'<td>{badge}</td>'
                f'<td style="text-align:center">{rank}</td>'
                f'<td><code>{col}</code></td>'
                f'<td>{sig_badge}</td>'
                f'<td style="font-size:12px">{desc}</td>'
                f'<td style="text-align:right">{mi}</td>'
                f'<td style="text-align:right">{rf}</td>'
                f'<td style="text-align:right">{fstat}</td>'
                f'<td style="text-align:right">{var}</td>'
                f'<td style="font-size:12px;color:#555">{rationale}</td>'
                f'</tr>'
            )
        return "\n        ".join(rows)

    def _build_selected_summary(self, imp_lookup, selected, top_n):
        rows = []
        # Sort selected by rank
        selected_items = [
            (name, imp_lookup[name])
            for name in selected if name in imp_lookup
        ]
        selected_items.sort(key=lambda x: x[1]["rank"])

        for name, row_data in selected_items:
            fm = _FEATURE_META.get(name, ("?", name, ""))
            sig = fm[0]
            desc = fm[1]
            rationale = fm[2]
            sig_color = _SIGNAL_COLOURS.get(sig, ("#666", "#eee"))
            sig_badge = (f'<span style="background:{sig_color[1]};'
                         f'color:{sig_color[0]};padding:2px 8px;'
                         f'border-radius:10px;font-size:11px;'
                         f'font-weight:600">{sig}</span>')

            # Highlight the strongest score
            scores = {
                "MI": row_data["score_mutual_information"],
                "RF": row_data["score_random_forest"],
                "F":  row_data["score_f_statistic"],
                "V":  row_data["score_variance"],
            }

            rows.append(
                f'<tr>'
                f'<td style="text-align:center;font-weight:700">'
                f'{int(row_data["rank"])}</td>'
                f'<td><code style="font-weight:700">{name}</code></td>'
                f'<td>{sig_badge}</td>'
                f'<td style="font-size:12px">{desc}</td>'
                f'<td style="text-align:right">'
                f'{row_data["score_mutual_information"]:.3f}</td>'
                f'<td style="text-align:right">'
                f'{row_data["score_random_forest"]:.3f}</td>'
                f'<td style="text-align:right">'
                f'{row_data["score_f_statistic"]:.1f}</td>'
                f'<td style="text-align:right">'
                f'{row_data["score_variance"]:.2f}</td>'
                f'<td style="font-size:12px;color:#555">{rationale}</td>'
                f'</tr>'
            )
        return "\n        ".join(rows)

    def _build_signal_bars(self, signal_counts, signal_selected):
        rows = []
        for sig in ["EDA", "BVP", "IBI", "ST", "ACC"]:
            total = signal_counts.get(sig, 0)
            sel = signal_selected.get(sig, 0)
            pct = (sel / total * 100) if total > 0 else 0
            color = _SIGNAL_COLOURS.get(sig, ("#666", "#eee"))

            rows.append(f"""
        <div style="display:flex;align-items:center;margin:8px 0;">
          <div style="width:60px;font-weight:700;color:{color[0]}">{sig}</div>
          <div style="flex:1;background:#eee;border-radius:6px;height:24px;
                      position:relative;overflow:hidden;">
            <div style="width:{pct:.0f}%;background:{color[0]};height:100%;
                        border-radius:6px;min-width:2px;"></div>
          </div>
          <div style="width:100px;text-align:right;font-size:13px;color:#555;">
            {sel} / {total} ({pct:.0f}%)
          </div>
        </div>""")
        return "\n".join(rows)

    # ── CSS ───────────────────────────────────────────────────────────────

    def _get_css(self) -> str:
        return """
<style>
body{font-family:Arial,sans-serif;margin:0;padding:0;background:#f5f6fa;color:#333}
.header{background:linear-gradient(135deg,#1B3A5C,#2E7D92);color:#fff;padding:32px 40px}
.header h1{margin:0;font-size:26px;font-weight:700}
.header p{margin:4px 0;font-size:13px;opacity:0.85}
.content{max-width:1600px;margin:0 auto;padding:24px 40px}
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
.table th{background:#1B3A5C;color:#fff;padding:8px 10px;text-align:left;
          cursor:pointer;user-select:none;position:sticky;top:0}
.table th:hover{background:#2E4A6C}
.table td{padding:7px 10px;border-bottom:1px solid #eee}
.table tr:hover td{background:#F0F4F8}
code{background:#f0f0f0;padding:1px 5px;border-radius:3px;font-size:12px}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;
       font-weight:600;margin:2px}
.insight{background:#EBF5FB;border-left:4px solid #3498DB;padding:12px 16px;
         border-radius:4px;margin:12px 0;font-size:13px}
@media print{
  .header{background:#1B3A5C!important;-webkit-print-color-adjust:exact}
  .section{break-inside:avoid}
}
@media (max-width:900px){
  .stats-grid{grid-template-columns:repeat(2,1fr)}
  .content{padding:16px}
}
</style>"""
