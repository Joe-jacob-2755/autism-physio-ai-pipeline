# Module 4: Data Analyser — Comprehensive Analysis Report

**Autism Physio-AI Pipeline | Module 4 v1.0.0**
**Analysis of physiological signals for emotion and behaviour classification in autistic children**

---

## 1. Overview

Module 4 applies a seven-component analytical framework to the preprocessed physiological signals and extracted features produced by Module 3. Each analysis targets a distinct aspect of the relationship between autonomic nervous system (ANS) activity and the 13 target states (6 affective emotions, 4 physiological need states, 3 behavioural targets).

### Target States Analysed

| Category | Labels | Physiological Signature |
|---|---|---|
| **Affective** | Happy, Anger, Fear, Disgust, Sad, Surprise | Sympathetic activation; varying EDA/HR elevation |
| **Physiological need** | Hunger, Thirst, Toilet, Tired | Moderate arousal; parasympathetic shift in Tired |
| **Behavioural** | SIB, ATO, GAB | High-amplitude ACC; extreme EDA/HR |

### Data Dimensions (sample session)

- Duration: 300 seconds | 7 annotated events
- Feature windows: 19 windows (30 s, 50% overlap)
- Feature columns: 86 numeric features + 11 demographic columns = 97 total
- Target labels present: Anger, Fear, Sad, Thirst, Tired, baseline

---

## 2. Analysis Components

### 2.1 Descriptive Statistics

**What it computes:** Per-feature descriptive statistics grouped by target label.

**Outcome variables per feature per target:**

| Variable | Definition | Why it matters |
|---|---|---|
| `mean` | Arithmetic mean across windows | Central tendency of physiological state |
| `std` | Standard deviation | Intra-state variability |
| `median` | 50th percentile | Robust centre, unaffected by outliers |
| `iqr` | Q3 − Q1 (interquartile range) | Spread without outlier distortion |
| `q25`, `q75` | 25th and 75th percentiles | Distribution shape |
| `min`, `max` | Observed extremes | Range of physiological response |
| `cv` | Coefficient of variation = std/mean | Relative variability — normalises across scales |
| `n` | Number of feature windows | Sample size for that target |

**Test selection rationale:** No statistical test is applied here — this component characterises the data before hypothesis testing. Descriptive statistics reveal whether the dataset has sufficient spread and inter-class separation to support classification models.

**Parameters to check:**
- **cv > 0.5** for a feature suggests high within-class variability — the feature may be noisy
- **min/max spread**: large range relative to IQR indicates outlier contamination
- **n < 3** for a target label means that class is under-represented and KW test will exclude it

**Sample values (EDA features, Anger vs Tired):**

| Feature | Target | Mean | Std | Median | IQR | CV |
|---|---|---|---|---|---|---|
| `eda_scl_mean` | Anger | 19.09 µS | 5.43 | 19.09 | 3.84 | 0.28 |
| `eda_scl_mean` | Tired | — | — | ~3.1 µS | — | — |
| `eda_scr_count` | Anger | 6.00 | 0.00 | 6.00 | 0.00 | 0.00 |
| `eda_scr_rise_mean` | Anger | 1.65 s | 0.15 | 1.65 | 0.10 | 0.09 |

**Interpretation:** `eda_scl_mean` is 19.09 µS during Anger — approximately 6× the resting baseline of ~3.1 µS. The CV of 0.28 indicates moderate within-class consistency.

---

### 2.2 Signal % Change from Baseline

**What it computes:** For each event, the percentage change from the pre-event window median to the peak value during the event.

**Formula:** `change_pct = (peak_value − baseline_median) / |baseline_median| × 100`

**Outcome variables:**

| Variable | Definition |
|---|---|
| `mean_change_pct` | Mean % change across all events of this target |
| `std_change_pct` | Standard deviation of % changes |
| `median_change_pct` | Median % change |
| `min_change_pct` | Smallest observed change |
| `max_change_pct` | Largest observed change |
| `n_events` | Number of events contributing |

**Test selection rationale:** Percentage change normalises across participants with different physiological baselines (EDA ranges 0.3–18 µS; HR ranges 60–115 bpm). It answers the clinical question: *"How much does this signal move during this target state, relative to where the child was beforehand?"*

**Parameters to check:**
- **|change_pct| > 20%** is clinically meaningful for EDA and ACC
- **Negative change_pct** indicates parasympathetic activity (HR slowing, EDA decrease — expected in Tired)
- **Consistency** (low std_change_pct): reliable biomarker candidate

**Sample values:**

| Signal | Target | Mean % Change | Interpretation |
|---|---|---|---|
| ACC | Anger | **+1609%** | Explosive motor burst — strongest discriminator |
| BVP | Anger | **+865%** | Rapid heart rate surge |
| EDA | Anger | **+652%** | Strong sympathetic activation |
| ACC | Fear | **+575%** | Defensive/startle movement |
| EDA | Fear | **+562%** | Fear arousal signature |
| BVP | Thirst | **−692%** | Parasympathetic shift — reduced cardiac output |
| BVP | Tired | **−545%** | Heart rate decrease; drowsiness |
| ACC | Tired | **−15%** | Near-stillness; reduced movement |

**Key insight:** ACC and BVP distinguish high-arousal states (Anger, Fear) from low-arousal states (Tired, Thirst) with 10× larger percentage changes, making them strong candidates for the first-stage classifier.

---

### 2.3 Temporal Dynamics

**What it computes:** Time-course analysis of each event × signal combination.

**Outcome variables:**

| Variable | Definition | Clinical significance |
|---|---|---|
| `baseline_median` | Median of pre-event window (30 s) | Resting physiological level |
| `baseline_std` | Std of pre-event window | Pre-event stability |
| `peak_value` | Maximum (or minimum) value during event | Extreme physiological response |
| `peak_delay_s` | Seconds from event onset to peak | Response latency |
| `change_pct` | (peak − baseline) / \|baseline\| × 100 | Magnitude of response |
| `peak_in_event` | Boolean — did peak occur within event window? | Annotation alignment |
| `subside_time_s` | Seconds from peak to 50% return toward baseline | Recovery rate |
| `returned_to_median` | Boolean — did signal return to ±10% of baseline? | Full recovery flag |
| `time_to_return_s` | Seconds after event end to full return | Recovery duration |
| `post_event_median` | Median of 60 s window after event | Residual activation |
| `median_drift` | post_event_median − baseline_median | Tonic level shift |

**Test selection rationale:** Temporal dynamics answer four specific clinical questions that descriptive statistics cannot:
1. *Does the ANS response begin immediately or is there a latency?* (onset delay)
2. *Does the response track the labelled event or lead/lag it?* (peak alignment)
3. *How long does the child's body take to recover?* (subside + return time)
4. *Does repeated arousal cause a persistent shift in the resting level?* (median drift)

These are foundational to deployment — a caregiver alert system needs to know whether the signal will peak 5 seconds or 60 seconds after an event begins.

**Parameters to check:**
- **`peak_in_event = False`**: peak occurred outside the labelled window — suggests label timing error or lagged physiological response
- **`peak_delay_s > event_duration_s`**: response is slower than the event itself (common in EDA and ST due to thermal/sudomotor inertia)
- **`returned_to_median = False`**: signal did not recover — indicates sustained arousal (critical for SIB/ATO)
- **`median_drift > 0.5 × baseline_std`**: tonic shift after events — baseline recalibration needed for next event

**Sample values (Event 6: Anger, 30 s duration):**

| Signal | Baseline Median | Peak | % Change | Time to Peak | Returned | Drift |
|---|---|---|---|---|---|---|
| EDA | 3.14 µS | 23.61 µS | **+652%** | 15.0 s | No | +13.59 µS |
| BVP | −0.04 nT | +8.65 nT | **+865%** | 15.0 s | Yes | +0.07 |
| ST | 31.5 °C | 31.73 °C | +0.7% | 15.0 s | Yes | +0.08 °C |
| ACC | 0.12 g | 2.07 g | **+1609%** | 15.0 s | No | +0.32 g |
| IBI | 837 ms | — | — | — | — | — |

**Signal-averaged time to peak (all targets):**

| Signal | Mean Time to Peak | Implication |
|---|---|---|
| ST | 7.5 s | Fastest — skin temperature responds immediately |
| ACC, IBI, EDA | 10.0 s | Mid-range — sympathetic response latency |
| BVP | 12.5 s | Slowest in this dataset — cardiac response |

---

### 2.4 Return-to-Median Analysis

**What it computes:** For each signal × target combination, counts how many events returned to within 10% of the pre-event baseline, and how long that took.

**Outcome variables:**

| Variable | Definition |
|---|---|
| `total` | Total events of this target–signal pair |
| `returned` | Number where `returned_to_median = True` |
| `pct_returned` | `returned / total × 100` |
| `mean_return_time_s` | Mean seconds from event end to return (for those that returned) |

**Test selection rationale:** Non-return to median is the physiological signature of *sustained arousal* — a state more dangerous for self-injurious behaviour and aggression than transient peaks. Identifying which target states cause the body to "stay elevated" is essential for caregiver alert prioritisation.

**Parameters to check:**
- **`pct_returned < 50%`**: the signal is chronically elevated for this target — caregiver should be alerted even after the visible behaviour ends
- **`mean_return_time_s > 60`**: recovery takes more than 1 minute — supports extended monitoring windows in deployment
- **Discordance between signals**: EDA non-return + ACC return = physiological arousal without physical activity — may indicate covert distress

**Sample values:**

| Signal | Target | Events | Returned | % Returned | Mean Return (s) |
|---|---|---|---|---|---|
| EDA | Anger | 1 | 0 | **0%** | — |
| EDA | Fear | 1 | 0 | **0%** | — |
| EDA | Thirst | 1 | 0 | **0%** | — |
| EDA | Tired | 2 | 1 | 50% | 60.0 s |
| BVP | Anger | 1 | 1 | **100%** | 45.0 s |
| BVP | Sad | 1 | 1 | **100%** | 105.0 s |
| ACC | Sad | 1 | 1 | **100%** | 15.0 s |
| ACC | Tired | 2 | 1 | 50% | 15.0 s |

**Key insight:** EDA did not return to baseline after Anger, Fear, or Thirst events — indicating persistent sympathetic activation lasting beyond the event window. BVP (cardiac) recovered fully, suggesting the heart adapted faster than the sweat glands. This dissociation is clinically meaningful: the child may appear calm to an observer (normalised movement/HR) while still physiologically aroused (EDA elevated).

---

### 2.5 Median Drift Analysis

**What it computes:** The shift in the running median of each signal after a sequence of events compared to the pre-session baseline.

**Formula:** `median_drift = post_event_median − pre_event_baseline_median`

**Outcome variables:**

| Signal | Mean Drift (sample session) |
|---|---|
| EDA | **+4.90 µS** (large upward shift — allostatic load) |
| IBI | **−128.7 ms** (HR acceleration — sympathetic dominance) |
| ACC | +0.32 g (mild residual movement) |
| ST | +0.10 °C (minor thermal elevation) |
| BVP | +0.07 nT (negligible cardiac drift) |

**Test selection rationale:** A single event's effect on the baseline can confound subsequent event analysis. Detecting median drift is essential for:
1. **Recalibrating adaptive thresholds** during a session (baseline shifts if child becomes progressively more distressed)
2. **Modelling allostatic load** — repeated emotional events may cause a tonic EDA elevation that reflects accumulated stress rather than any single event
3. **Informing session length** — large drift suggests the child is nearing physiological capacity

**Parameters to check:**
- **EDA drift > 2 µS**: significant allostatic loading — session should be flagged
- **IBI drift < −50 ms**: sustained tachycardia — physiological distress indicator
- **Drift increasing monotonically** across events: escalating arousal pattern requiring intervention

---

### 2.6 Statistical Analysis

#### 2.6.1 Kruskal-Wallis H-Test

**Purpose:** Non-parametric one-way ANOVA — tests whether the distribution of each feature differs significantly across target state groups.

**Why Kruskal-Wallis (not one-way ANOVA)?**
- Physiological features (EDA, HRV spectral powers) are **not normally distributed** — they follow log-normal or heavy-tailed distributions
- Small group sizes (2–5 windows per target in short sessions) violate ANOVA's n ≥ 20 assumption
- KW tests rank-order distributions — robust to the outliers characteristic of SCR peaks and ACC impulses

**Outcome variables:**

| Variable | Definition | Threshold |
|---|---|---|
| `H_stat` | Kruskal-Wallis H statistic (larger = more separation) | Increases with group separation |
| `p_value` | Probability under H₀ of equal distributions | < 0.05 → reject H₀ |
| `significant` | Boolean: p < α (α = 0.05) | True → feature discriminates target states |
| `eta_squared` | η² = (H − k + 1) / (N − k) — proportion of variance explained | > 0.14 = large, > 0.06 = medium |
| `effect_size` | Interpreted label: negligible / small / medium / large | |

**Parameters to check:**
- **H_stat > 7.0** with **p < 0.05**: strong evidence of between-group differences
- **η² > 0.14**: large effect — feature explains >14% of variance across targets
- **p < 0.001**: feature survives Bonferroni correction even at individual-test level
- **Significant features per signal**: guides which signal to prioritise in unimodal models

**Sample results (86 features tested, 22 significant):**

| Feature | H-stat | p-value | η² | Effect |
|---|---|---|---|---|
| `acc_svm_skew` | **10.68** | 0.0048 | **0.868** | Large |
| `bvp_kurt` | 9.17 | 0.0102 | 0.717 | Medium |
| `acc_svm_sampen` | 9.17 | 0.0102 | 0.717 | Medium |
| `bvp_notch_depth` | 9.17 | 0.0102 | 0.717 | Medium |
| `bvp_std` | 9.05 | 0.0108 | 0.705 | Medium |
| `ibi_mean` | 8.56 | 0.0139 | 0.656 | Medium |
| `eda_scr_auc` | 7.21 | 0.0270 | 0.535 | Medium |

**Key insight:** `acc_svm_skew` is the single most discriminative feature (η² = 0.87, large effect) — the asymmetry of the ACC signal vector magnitude distribution reliably separates target states. This reflects the difference between sustained rhythmic movement (SIB), explosive burst movement (ATO/Anger), and near-stillness (Tired). BVP morphology features (kurtosis, notch depth, standard deviation) form a cluster of medium-effect discriminators.

---

#### 2.6.2 Pairwise Mann-Whitney U Test

**Purpose:** Post-hoc pairwise comparison of each feature between all pairs of target labels, corrected for multiple comparisons.

**Why Mann-Whitney U (not Student's t-test)?**
- Same rationale as KW — non-normal distributions, small n
- Mann-Whitney U tests the probability that a randomly selected value from group A exceeds one from group B — directly interpretable clinically
- More powerful than KW for detecting specific pairwise differences after an omnibus KW test

**Correction method:** Bonferroni — α_corrected = 0.05 / n_tests. Conservative but appropriate for a clinical application where false positives (incorrectly flagging a state) have caregiver safety implications.

**Outcome variables:**

| Variable | Definition |
|---|---|
| `U_stat` | Mann-Whitney U statistic |
| `p_value` | Raw p-value |
| `p_bonf` | Bonferroni-corrected p-value |
| `significant` | p_bonf < 0.05 |
| `effect_r` | Rank-biserial correlation = 1 − (2U / n₁n₂) — ranges −1 to +1 |
| `effect_size` | small / medium / large based on \|r\| thresholds |
| `n_a`, `n_b` | Sample sizes for each group |

**Parameters to check:**
- **effect_r → ±1.0**: perfect separation between groups — biomarker candidate
- **p_bonf < 0.05**: significant after full multiple-test correction — reliable discriminator
- **Consistent direction across signals**: feature that separates Anger vs baseline in EDA AND BVP AND ACC is a robust multimodal discriminator

**Sample values:**

| Feature | Group A | Group B | U | p (raw) | p (Bonferroni) | effect_r |
|---|---|---|---|---|---|---|
| `bvp_notch_depth` | Tired | baseline | 0.0 | 0.016 | 4.76 | **1.00** |
| `bvp_std` | Tired | baseline | 0.0 | 0.016 | 4.76 | **1.00** |
| `acc_svm_sampen` | Tired | baseline | 20.0 | 0.016 | 4.76 | **1.00** |
| `bvp_hr_mean` | Sad | baseline | 0.0 | 0.016 | 4.76 | **1.00** |
| `ibi_mean` | Tired | baseline | 20.0 | 0.016 | 4.76 | **1.00** |

**Note on Bonferroni p-values > 1.0:** With 300 comparisons and α = 0.05, the corrected threshold is 0.000167. The corrected p-values exceed 1.0 because they represent p × n_tests — this is expected in small-sample exploratory analysis and indicates these specific features do not survive strict correction in isolation. For model training, features should be selected by combining KW significance, effect size, and cross-validation performance.

---

### 2.7 Correlation Analysis

#### 2.7.1 Feature–Target Correlation (Spearman ρ)

**Purpose:** Measure the monotonic association between each feature and the ordinally encoded target label.

**Why Spearman (not Pearson)?**
- Target labels are ordinal (encoded by severity/arousal level), not truly continuous
- Spearman captures monotonic relationships in non-normally distributed features
- Robust to outliers (SCR peaks, ACC impulses)

**Outcome variables:**

| Variable | Definition | Range |
|---|---|---|
| `spearman_r` | Spearman rank correlation coefficient | −1 to +1 |
| `abs_r` | Absolute value of ρ | 0 to 1 |
| `p_value` | Significance of correlation | < 0.05 → reject H₀ |
| `significant` | Boolean: p < 0.05 | |
| `effect_size` | small (≥0.2) / medium (≥0.5) / large (≥0.8) | |

**Parameters to check:**
- **\|ρ\| > 0.5**: feature has medium-to-large correlation with target — likely a useful model feature
- **Positive ρ**: feature increases as target arousal increases (EDA, HR, ACC with high-arousal targets)
- **Negative ρ**: feature decreases with arousal (IBI_mean decreases as HR increases; HF power decreases with sympathetic activation)
- **p < 0.05 with \|ρ\| > 0.3**: statistically significant and practically meaningful

**Sample results (top 8 by \|ρ\|):**

| Feature | ρ | |ρ| | p-value | Effect | Direction |
|---|---|---|---|---|---|
| `acc_posture_trans` | +0.647 | 0.647 | 0.003 | Medium | ↑ more transitions in high-arousal |
| `acc_svm_skew` | +0.626 | 0.626 | 0.004 | Medium | ↑ asymmetric movement in high-arousal |
| `acc_svm_kurt` | +0.621 | 0.621 | 0.005 | Medium | ↑ impulsive ACC peaks with arousal |
| `bvp_pw50` | +0.597 | 0.597 | 0.007 | Medium | ↑ wider pulse with sympathetic activation |
| `ibi_hfnu` | −0.529 | 0.529 | 0.043 | Medium | ↓ vagal withdrawal during arousal |
| `ibi_lf_hf` | +0.529 | 0.529 | 0.043 | Medium | ↑ sympathovagal balance shifts with arousal |
| `ibi_lfnu` | +0.529 | 0.529 | 0.043 | Medium | ↑ LF power during sympathetic activation |
| `acc_pow_voluntary` | −0.522 | 0.522 | 0.022 | Medium | ↓ organised movement in high-arousal |

**Key insight:** The negative correlation of `ibi_hfnu` (HF power normalised unit, ρ = −0.529) and positive correlation of `ibi_lf_hf` (LF/HF ratio, ρ = +0.529) are physiologically coherent — they reflect vagal withdrawal during sympathetic arousal. This is one of the most established findings in the autonomic literature (Task Force 1996) and its presence in simulated data validates the physiological fidelity of Module 2A.

#### 2.7.2 Feature–Feature Correlation Matrix (Spearman)

**Purpose:** Identify redundant feature pairs and feature clusters that should inform dimensionality reduction or feature selection before model training.

**Parameters to check:**
- **\|ρ\| > 0.85 between features**: near-perfect collinearity — one feature should be dropped from any linear model (keeps both for non-linear models like RF/XGBoost)
- **Feature clusters** (groups of mutually highly correlated features): candidate for PCA compression in Module 4 feature engineering
- **IBI features cluster**: `ibi_sdnn`, `ibi_sd2`, `ibi_lf`, `ibi_rmssd` typically form a HRV cluster (all measure similar aspects of HRV)

**Sample high-correlation pairs (|ρ| > 0.8):**

| Feature A | Feature B | ρ | Implication |
|---|---|---|---|
| `ibi_sdnn` | `ibi_sd2` | 0.957 | Both overall HRV — use one |
| `ibi_mean` | `bvp_ptt` | 0.828 | Both reflect R-R interval |
| `ibi_mean` | `bvp_sys_amp_mean` | −0.882 | HR↑ → systolic amplitude changes |
| `acc_activity_count` | `ibi_mean` | −0.771 | Movement suppresses HRV |

---

### 2.8 Adaptive Threshold Analysis

**Purpose:** Detect sustained physiological deviations from the running median that exceed a configurable threshold, flagging periods of significant arousal independent of event labels.

**Algorithm:**
1. Compute rolling median and rolling MAD over a configurable window (default 300 s / 5 min)
2. Compute deviation score: `deviation = |signal − rolling_median| / (N × MAD)`
3. Flag samples where `deviation ≥ 1.0` (signal exceeds N×MAD from median)
4. Threshold event fired only if deviation sustained for ≥ T seconds (default 30 s)

**Outcome variables:**

| Variable | Definition |
|---|---|
| `signal` | Which channel exceeded threshold |
| `start_s`, `end_s` | Event time window |
| `duration_s` | How long signal was elevated |
| `deviation_mad` | Peak deviation in MAD units (larger = more extreme) |
| `peak_value` | Signal value at peak deviation |
| `running_median` | Baseline at time of event |

**Parameters to check:**
- **`deviation_mad > 5`**: extreme deviation — severe physiological event
- **Events on EDA + ACC simultaneously**: suggests genuine emotional episode (not artefact)
- **Events without corresponding annotation**: behavioural event the researcher may have missed
- **Threshold window setting**: 300 s (5 min) smooths short spikes; reduce to 60 s for more sensitive real-time detection

**Sample threshold events:**

| Signal | Time (s) | Duration | Deviation (MAD) | Peak Value | Running Median |
|---|---|---|---|---|---|
| IBI | 15.0 | — | **11.19** | 858.2 ms | 837.8 ms |
| ACC | 90.0 | — | 2.71 | 0.135 g | 0.119 g |
| EDA | 210.0 | — | 2.52 | 23.61 µS | 15.23 µS |
| BVP | 150.0 | — | 2.26 | 0.198 nT | 0.002 nT |
| ST | 210.0 | — | 1.93 | 31.73 °C | 31.57 °C |

**Key insight:** The IBI threshold crossing at t=15 s shows the highest deviation (11.19 MAD) — an extremely large heart rate fluctuation at session start. EDA at t=210 s (Event 6: Anger) shows a 2.52 MAD deviation at a running median of 15.23 µS, consistent with a sustained high EDA tonic level after prior events (allostatic loading).

---

## 3. Output Files Reference

All outputs written to: `outputs/run_NNN/module_4_analysis/user_NNN/`

### CSV Summary Tables

| File | Rows | Description |
|---|---|---|
| `descriptive_statistics.csv` | 512 | Mean/std/median/IQR/CV per feature per target |
| `kruskal_wallis_results.csv` | 86 | H-stat, p-value, η², effect size per feature |
| `pairwise_mannwhitney.csv` | 300 | U-stat, Bonferroni p, rank-biserial r per pair |
| `correlation_feature_target.csv` | 86 | Spearman ρ between each feature and target |
| `correlation_matrix.csv` | 30×30 | Feature–feature Spearman correlation matrix |
| `temporal_dynamics.csv` | 30 | Per-event temporal metrics (30 events × 5 signals) |
| `return_to_median_summary.csv` | 25 | % events returning to baseline per signal × target |
| `pct_change_summary.csv` | 25 | Mean/std/median % change from baseline |
| `threshold_events.csv` | 10 | Detected adaptive threshold crossings |

### Visualisation Outputs (17 PNGs + 2 HTMLs)

| File | Analysis |
|---|---|
| `descriptive_boxplot_{EDA,BVP,IBI,ST,ACC}.png` | Feature distributions per target |
| `pct_change_from_baseline.png` | % change bar chart per signal × target |
| `temporal_dynamics.png` | Time-to-peak, subside, return box plots |
| `return_to_median_rate.png` | % events returned per signal × target |
| `significant_features_kruskal.png` | −log₁₀(p) bar chart coloured by effect size |
| `correlation_feature_target.png` | Spearman ρ bar chart (top 25 features) |
| `correlation_matrix.png` | Feature–feature heatmap |
| `threshold_{EDA,BVP,ST,ACC,IBI}.png` | Signal trace + MAD threshold overlay |
| `median_drift.png` | Post-event median shift per signal × target |
| `analysis_report.html` | Full report with all tables and plots embedded |
| `3d_signals.html` | Interactive 3D signal projection (rotate in browser) |

---

## 4. Clinical Interpretation Framework

### Selecting Features for Model Training

Use the following decision hierarchy from Module 4 outputs:

```
Step 1: Kruskal-Wallis filter
  Keep features where: significant=True AND eta_squared ≥ 0.06
  → Typically 20–30 of 86 features in a short session

Step 2: Correlation filter
  From step 1 survivors, check correlation_matrix.csv
  For any pair with |ρ| > 0.85: drop the one with lower KW H-stat

Step 3: Effect size filter
  Prioritise features where pairwise effect_r ≥ 0.5 for the most
  clinically important pairs (e.g. SIB vs baseline, ATO vs Anger)

Step 4: Signal-level weighting
  Rank signals by number of surviving features:
  Guides unimodal model investment and decision fusion weights
```

### Interpreting Results for Clinical Practice

| Finding | Clinical Implication |
|---|---|
| EDA pct_returned = 0% for Anger/Fear | Child remains physiologically distressed after behaviour ends — extend monitoring window |
| IBI median drift < −50 ms | Progressive tachycardia — session may need to end |
| acc_svm_skew η² = 0.87 | ACC asymmetry is the most reliable single biomarker in this dataset |
| ST fastest time to peak (7.5 s) | Skin temperature responds earliest — may be useful for very early warning |
| LF/HF ratio ρ = +0.53 | Sympathovagal balance tracks arousal — established HRV biomarker |
| Threshold event without annotation | Possible missed behavioural event — review video footage |

---

## 5. Limitations and Recommendations

1. **Small sample sizes per target**: With 2–5 windows per target label in a 300 s session, statistical power is limited. Bonferroni-corrected pairwise tests will rarely reach significance. Use η² from KW as the primary effect size for feature selection.

2. **Proxy signals**: When raw signal CSVs are unavailable, Module 4 uses feature means as window-level proxy signals for temporal analysis. This limits temporal resolution to the window granularity (30 s) rather than the native sampling rate. Run M1 → M2A → M3 → M4 as a full pipeline to enable native-rate temporal analysis.

3. **Session duration**: A minimum of 600 s (10 min) with ≥ 3 events per target state is recommended for statistically reliable KW and pairwise tests. Short sessions produce descriptive results only.

4. **Single-participant analysis**: Module 4 currently analyses one participant at a time. Multi-participant pooled analysis (required for Leave-One-Subject-Out cross-validation in Module 5) will be implemented in Module 5.

5. **Behavioural targets (SIB, ATO, GAB)**: These require real annotated data — the simulator provides physiologically plausible waveforms but real SIB/ATO data will show higher inter-individual variability in ACC patterns. Threshold and feature parameters should be validated against real clinical recordings.

---

*Report generated for: Autism Physio-AI Pipeline, Module 4 v1.0.0*
*Sample data: 300 s session, 7 events, 5 signals, 86 features, 19 windows*
*For queries contact the research team.*
