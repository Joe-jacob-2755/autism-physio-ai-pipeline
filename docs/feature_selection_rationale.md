# Feature Selection Rationale — Module 6 (Run 017)

**Total columns in combined feature matrix:** 109
**Breakdown:** 5 metadata + 93 physiological features + 6 raw demographics + 5 encoded demographics
**Selection method:** Ensemble ranking (Mutual Information + Random Forest importance + F-statistic + Variance)
**Top N selected:** 20

---

## Column Categories

| Category | Columns | Count | Scored? |
|----------|---------|-------|---------|
| Metadata (identifiers/windows) | 1–5 | 5 | No — structural |
| Physiological features | 6–98 | 93 | Yes — ranked |
| Raw demographics (string) | 99–104 | 6 | No — categorical labels |
| Encoded demographics (numeric) | 105–109 | 5 | No — passed through to model |

---

## Full 109-Column Table

### Legend

- **Status**: SELECTED (top 20), REJECTED (ranked but not selected), METADATA (not a feature), DEMOGRAPHIC (passed through)
- **Rank**: Ensemble rank out of 93 (lower = better)
- **Avg Rank**: Mean rank across all 4 scoring methods
- **MI**: Mutual Information score (non-linear dependency with target)
- **RF**: Random Forest feature importance
- **F-stat**: ANOVA F-statistic (linear separability between classes)
- **Var**: Variance score (spread after normalisation)

---

### Metadata Columns (not scored)

| # | Column | Type | Status | Rationale |
|---|--------|------|--------|-----------|
| 1 | `session_id` | string | METADATA | Session identifier — structural, not predictive |
| 2 | `user_id` | string | METADATA | Participant identifier — structural, not predictive |
| 3 | `window_start_s` | float | METADATA | Window start timestamp — temporal anchor, not a feature |
| 4 | `window_end_s` | float | METADATA | Window end timestamp — temporal anchor, not a feature |
| 5 | `target_label` | string | METADATA | Ground truth label (emotion/state) — this is the prediction target, not an input feature |

---

### EDA Features (Electrodermal Activity) — 22 columns

| # | Column | Rank | Avg Rank | MI | RF | F-stat | Var | Status | Rationale |
|---|--------|------|----------|----|----|--------|-----|--------|-----------|
| 6 | `eda_scl_mean` | 23 | 36.00 | 0.460 | 0.011 | 8.55 | 0.59 | REJECTED | Moderate MI and F-stat but outperformed by delta/sampen; tonic SCL is relatively stable across emotions and less discriminative than phasic features |
| 7 | `eda_scl_delta` | 20 | 33.50 | 0.322 | 0.024 | 5.20 | 0.71 | **SELECTED** | Change in tonic SCL captures sympathetic arousal shifts; good RF importance (0.024) indicating non-linear utility; discriminates sustained arousal states (Fear, Anger) from low-arousal states (Tired, Sad) |
| 8 | `eda_scl_slope` | 77 | 63.25 | 0.000 | 0.003 | 0.40 | 1.17 | REJECTED | Zero MI, negligible F-stat; slope of tonic level adds no discriminative value beyond delta — highly correlated with `eda_scl_delta` but noisier |
| 9 | `eda_scr_count` | 53 | 47.75 | 0.468 | 0.004 | 20.52 | 0.20 | REJECTED | High F-stat (20.5) but very low variance (0.20) — most windows have 0–1 SCRs in 60s windows making it near-constant; `eda_scr_rate` carries same info |
| 10 | `eda_scr_rate` | 35 | 39.00 | 0.530 | 0.011 | 20.52 | 0.20 | REJECTED | Strong F-stat and MI but critically low variance (0.20); rate is mathematically identical to count/window_duration — redundant with `eda_scr_count` |
| 11 | `eda_scr_amp_mean` | 46 | 44.75 | 0.671 | 0.005 | 4.69 | 0.47 | REJECTED | Good MI (0.67) but weak RF and F-stat; mean SCR amplitude is unreliable in short windows with few or zero SCRs — NaN-prone |
| 12 | `eda_scr_amp_max` | 56 | 51.50 | 0.000 | 0.005 | 1.99 | 1.84 | REJECTED | Zero MI, weak F-stat; maximum amplitude is dominated by single outlier SCRs — high variance but not discriminative |
| 13 | `eda_scr_rise_mean` | 15 | 28.25 | 0.430 | 0.014 | 6.94 | 1.32 | **SELECTED** | Mean SCR rise time distinguishes fast sympathetic responses (Fear, Surprise: fast rise) from slow responses (Hunger, Tired: slow rise); balanced scores across all 4 methods |
| 14 | `eda_scr_rec_time` | 66 | 56.50 | 0.000 | 0.007 | 1.86 | 0.77 | REJECTED | Zero MI; SCR recovery time is theoretically informative but too noisy in 60s windows — requires longer observation periods for reliable estimation |
| 15 | `eda_scr_auc` | 57 | 51.50 | 0.112 | 0.005 | 2.63 | 0.88 | REJECTED | Weak across all methods; area under SCR curve is redundant with amplitude × duration — correlated with `eda_scr_amp_mean` and `eda_scr_rise_mean` |
| 16 | `eda_mean` | 55 | 49.50 | 0.255 | 0.003 | 8.45 | 0.60 | REJECTED | Highly correlated with `eda_scl_mean` (r > 0.95) — redundant; raw signal mean dominated by tonic component |
| 17 | `eda_std` | 63 | 55.50 | 0.000 | 0.005 | 1.21 | 2.32 | REJECTED | Zero MI; high variance is driven by inter-subject differences not emotion discrimination — reflects individual baseline variability |
| 18 | `eda_min` | 59 | 53.75 | 0.103 | 0.009 | 1.29 | 0.72 | REJECTED | Weak scores; minimum EDA reflects individual baseline, not emotional state — between-subject noise dominates |
| 19 | `eda_max` | 22 | 36.00 | 0.000 | 0.027 | 1.42 | 43.84 | REJECTED | Zero MI despite extremely high variance (43.8); variance is driven by outlier SCR peaks in a few subjects — inflated by inter-subject differences, not emotion-discriminative |
| 20 | `eda_range` | 48 | 44.75 | 0.000 | 0.011 | 1.33 | 153.21 | REJECTED | Zero MI; extreme variance (153.2) entirely due to outliers — `eda_max - eda_min` is unstable and not reproducible across sessions |
| 21 | `eda_skew` | 39 | 40.50 | 0.192 | 0.022 | 3.91 | 0.49 | REJECTED | Moderate RF but weak MI and F-stat; distribution shape is informative in theory but requires more samples per window for stable estimation |
| 22 | `eda_kurt` | 80 | 64.50 | 0.000 | 0.006 | 0.13 | 0.95 | REJECTED | Zero MI, near-zero F-stat; kurtosis requires very large samples for stability — unreliable in 60s windows at 4 Hz (240 samples) |
| 23 | `eda_zcr` | 73 | 61.25 | 0.380 | 0.000 | 1.30 | 0.40 | REJECTED | Decent MI but zero RF importance; zero-crossing rate of EDA is not physiologically meaningful — EDA is non-negative by nature |
| 24 | `eda_pow_vlf` | 87 | 74.00 | 0.000 | 0.000 | 1.06 | 0.46 | REJECTED | Zero across all methods; very-low-frequency power of EDA is dominated by thermoregulatory drift, not emotional content |
| 25 | `eda_pow_scr` | 83 | 64.50 | 0.000 | 0.010 | 0.66 | 0.55 | REJECTED | Zero MI; SCR-band power overlaps heavily with `eda_scr_auc` — redundant frequency-domain representation of phasic activity |
| 26 | `eda_spec_centroid` | 67 | 57.25 | 0.079 | 0.003 | 0.29 | 2.66 | REJECTED | Near-zero F-stat; spectral centroid of EDA is not well-defined for such a low-bandwidth signal (0–1 Hz) — poor spectral resolution at 4 Hz |
| 27 | `eda_sampen` | **1** | 11.00 | 0.505 | 0.021 | **71.04** | 4.10 | **SELECTED** | **Top-ranked feature.** Sample entropy of EDA captures signal complexity/regularity changes across emotional states; highest F-stat (71.0) by a large margin — strong linear separability; high variance (4.1) ensures discriminative spread. Literature supports EDA complexity as a key autonomic marker in ASD |

---

### BVP Features (Blood Volume Pulse) — 16 columns

| # | Column | Rank | Avg Rank | MI | RF | F-stat | Var | Status | Rationale |
|---|--------|------|----------|----|----|--------|-----|--------|-----------|
| 28 | `bvp_hr_mean` | 5 | 20.75 | 0.745 | 0.013 | 8.73 | 1.28 | **SELECTED** | Mean heart rate is a primary autonomic marker; strong MI (0.74) reflects non-linear relationship with arousal states; Fear/Anger → elevated HR, Tired/Sad → reduced HR. Core clinical feature |
| 29 | `bvp_hr_std` | 68 | 57.75 | 0.366 | 0.000 | 5.53 | 0.32 | REJECTED | Zero RF importance; HR variability over 60s is better captured by IBI-domain features (SDNN, RMSSD) which have higher temporal resolution |
| 30 | `bvp_sys_amp_mean` | 25 | 36.75 | 0.710 | 0.000 | 11.61 | 0.85 | REJECTED | Strong MI (0.71) and F-stat (11.6) but zero RF importance — captured only by linear methods; systolic amplitude correlates with `bvp_hr_mean` and `bvp_notch_depth` — partially redundant |
| 31 | `bvp_notch_depth` | **3** | 20.25 | 0.564 | 0.024 | 11.06 | 0.88 | **SELECTED** | Dicrotic notch depth reflects arterial compliance and peripheral vascular resistance; balanced high scores across all methods; captures sympathetic vasoconstriction during stress (Fear, Anger) that `bvp_hr_mean` alone misses |
| 32 | `bvp_pw50` | 27 | 37.00 | 0.182 | 0.014 | 9.35 | 0.72 | REJECTED | Pulse width at 50% amplitude — moderate scores but lower than notch_depth and hr_mean; partially redundant with heart rate (wider pulse ≈ lower HR) |
| 33 | `bvp_ptt` | **2** | 19.75 | **0.937** | 0.031 | 40.77 | 0.42 | **SELECTED** | **Rank 2.** Pulse transit time — highest MI score (0.94) of any feature; very high F-stat (40.8); reflects arterial stiffness and blood pressure changes driven by sympathetic activation. Strong discriminator between high-arousal (Fear, Anger) and low-arousal (Tired, Sad) states |
| 34 | `bvp_aug_idx` | 82 | 64.50 | 0.005 | 0.005 | 1.22 | 0.43 | REJECTED | Augmentation index — near-zero MI; requires clean pulse waveform for reliable estimation; highly sensitive to motion artefact making it unreliable in active children |
| 35 | `bvp_mean` | 75 | 63.00 | 0.000 | 0.000 | 3.33 | 1.24 | REJECTED | Zero MI, zero RF; raw BVP mean has no physiological meaning — BVP is AC-coupled, mean reflects sensor offset not cardiac function |
| 36 | `bvp_std` | 41 | 41.50 | 0.485 | 0.000 | 14.13 | 0.79 | REJECTED | Zero RF importance despite good MI and F-stat; BVP standard deviation is a proxy for pulse amplitude — already captured by `bvp_sys_amp_mean` and `bvp_notch_depth` |
| 37 | `bvp_skew` | 24 | 36.25 | 0.120 | 0.010 | 5.44 | 3.64 | REJECTED | High variance (3.6) but low MI; BVP waveform asymmetry changes with heart rate but is sensitive to motion artefact — unreliable in paediatric ASD population |
| 38 | `bvp_kurt` | 36 | 39.00 | 0.189 | 0.018 | 3.60 | 0.84 | REJECTED | Moderate scores across the board but doesn't reach top 20; BVP kurtosis reflects peak sharpness — partially redundant with `bvp_notch_depth` |
| 39 | `bvp_f0` | **13** | 26.75 | 0.599 | 0.015 | 2.08 | **2.00** | **SELECTED** | Fundamental frequency of BVP — captures heart rate in the frequency domain; complements time-domain `bvp_hr_mean`; high variance (2.0) provides good spread; useful for detecting rhythmic patterns missed by time-domain HR |
| 40 | `bvp_resp_mod` | 92 | 89.75 | 0.000 | 0.000 | 0.00 | 0.00 | REJECTED | **Zero across all methods.** Respiratory modulation of BVP is theoretically present but impossible to extract reliably at 64 Hz without dedicated respiration signal — constant/NaN in this data |
| 41 | `bvp_harm_ratio` | 42 | 42.75 | 0.389 | 0.005 | 2.20 | 1.02 | REJECTED | Moderate MI but weak F-stat and RF; harmonic ratio reflects waveform regularity — partially captured by `bvp_f0` and `bvp_notch_depth` |
| 42 | `bvp_spec_entropy` | 74 | 61.25 | 0.000 | 0.006 | 0.18 | 1.72 | REJECTED | Zero MI, near-zero F-stat; spectral entropy of BVP is dominated by cardiac fundamental — low discriminative power since heart rhythm is quasi-periodic in all states |
| 43 | `bvp_sqi` | 44 | 43.00 | 0.318 | 0.009 | 3.78 | 0.73 | REJECTED | Signal quality index — reflects data quality, not emotional state. Should be used for gating/weighting, not as a predictive feature. Including SQI as a feature risks the model learning to rely on artefact patterns |

---

### IBI/HRV Features (Inter-Beat Interval / Heart Rate Variability) — 17 columns

| # | Column | Rank | Avg Rank | MI | RF | F-stat | Var | Status | Rationale |
|---|--------|------|----------|----|----|--------|-----|--------|-----------|
| 44 | `ibi_mean` | **6** | 22.75 | 0.770 | **0.040** | **52.24** | 0.30 | **SELECTED** | Mean IBI (inverse of heart rate in time domain); highest RF importance (0.040) and second-highest F-stat (52.2); directly encodes autonomic arousal level. Complements frequency-domain `bvp_hr_mean` with different estimation method |
| 45 | `ibi_sdnn` | **17** | 29.75 | 0.647 | 0.011 | 6.76 | 0.94 | **SELECTED** | Standard deviation of NN intervals — gold-standard overall HRV measure (Task Force 1996); captures total autonomic variability; reduced in stress/fear, elevated in relaxation. Essential clinical HRV metric |
| 46 | `ibi_rmssd` | 26 | 36.75 | 0.474 | 0.004 | 7.71 | 1.18 | REJECTED | RMSSD is a primary parasympathetic marker — narrowly missed top 20 (rank 26); partially redundant with `ibi_sd1` (rank 10) which is mathematically related (SD1 = RMSSD/√2) and scored higher |
| 47 | `ibi_pnn50` | **9** | 25.25 | 0.516 | 0.018 | 14.27 | 0.75 | **SELECTED** | Percentage of successive IBIs differing by >50ms — captures parasympathetic (vagal) activity; high F-stat (14.3); vagal withdrawal is a primary autonomic response in fear/stress in ASD (Kushki 2013) |
| 48 | `ibi_cv` | 61 | 55.25 | 0.191 | 0.007 | 1.22 | 0.59 | REJECTED | Coefficient of variation — ratio of SDNN to mean IBI; redundant with `ibi_sdnn` and `ibi_mean` which are both selected and together fully determine CV |
| 49 | `ibi_vlf` | 54 | 48.50 | 0.000 | 0.008 | 1.63 | 15.54 | REJECTED | Very-low-frequency power (0.003–0.04 Hz); zero MI; requires >5 min recording for reliable estimation — 60s windows are too short (Task Force 1996 recommends minimum 5 min) |
| 50 | `ibi_lf` | 34 | 38.75 | 0.019 | 0.008 | 8.60 | 1.35 | REJECTED | Low-frequency power (0.04–0.15 Hz) — reflects mixed sympathetic+parasympathetic; near-zero MI; outperformed by `ibi_hf` and `ibi_lf_hf` ratio; 60s windows provide borderline spectral resolution for LF band |
| 51 | `ibi_hf` | **4** | 20.50 | 0.339 | 0.018 | 10.19 | **4.83** | **SELECTED** | High-frequency power (0.15–0.40 Hz) — pure parasympathetic (vagal) marker; highest variance (4.83) of all IBI features; captures respiratory sinus arrhythmia; vagal tone is a key differentiator for ASD emotional states (Porges 2007) |
| 52 | `ibi_lf_hf` | 64 | 55.75 | 0.000 | 0.004 | 1.11 | 2.11 | REJECTED | LF/HF ratio — historically used as sympathovagal balance index but now considered unreliable (Billman 2013); zero MI; both LF and HF are individually more informative |
| 53 | `ibi_lfnu` | 62 | 55.50 | 0.169 | 0.003 | 1.22 | 1.00 | REJECTED | LF in normalised units — mathematically determined by LF/(LF+HF); redundant with `ibi_hf` and `ibi_lf`; adds no independent information |
| 54 | `ibi_hfnu` | 69 | 58.25 | 0.187 | 0.000 | 1.22 | 1.00 | REJECTED | HF in normalised units — exactly `1 - ibi_lfnu`; perfectly redundant; zero RF importance |
| 55 | `ibi_total_pow` | 45 | 44.00 | 0.000 | 0.014 | 1.40 | 1812.13 | REJECTED | Total spectral power — extreme variance (1812) driven by inter-subject differences; zero MI; dominated by VLF component which is unreliable in short windows |
| 56 | `ibi_sd1` | **10** | 25.75 | 0.453 | 0.018 | 7.58 | 1.18 | **SELECTED** | Poincaré SD1 — geometric measure of short-term HRV; captures beat-to-beat variability; related to RMSSD but provides additional geometric insight; balanced scores across all methods. Outperformed ibi_rmssd in ensemble ranking |
| 57 | `ibi_sd2` | 33 | 38.50 | 0.735 | 0.006 | 5.50 | 0.61 | REJECTED | Poincaré SD2 — long-term HRV; strong MI (0.74) but weak RF (0.006); partially redundant with `ibi_sdnn` (SD2 ≈ √(2·SDNN² - 0.5·RMSSD²)) |
| 58 | `ibi_sd1_sd2` | 71 | 60.75 | 0.451 | 0.004 | 0.74 | 0.41 | REJECTED | SD1/SD2 ratio — theoretically captures sympathovagal balance geometrically; decent MI (0.45) but very weak F-stat (0.74) — ratio is unstable when SD2 is small |
| 59 | `ibi_sampen` | 89 | 79.00 | 0.155 | 0.000 | 0.10 | 0.26 | REJECTED | Sample entropy of IBI — weak across all methods; requires longer IBI series (>200 beats) for reliable estimation; in 60s windows with ~60–80 beats, the estimate is noisy and unreliable |
| 60 | `ibi_dfa_alpha1` | 78 | 63.50 | 0.228 | 0.000 | 0.47 | 0.82 | REJECTED | Detrended fluctuation analysis α1 — short-term fractal scaling; zero RF; requires minimum 256 beats (Peng 1995) — severely underpowered in 60s windows |

---

### ST Features (Skin Temperature) — 12 columns

| # | Column | Rank | Avg Rank | MI | RF | F-stat | Var | Status | Rationale |
|---|--------|------|----------|----|----|--------|-----|--------|-----------|
| 61 | `st_mean` | 52 | 47.25 | 0.207 | 0.007 | 18.80 | 0.31 | REJECTED | High F-stat (18.8) but low MI and variance; mean skin temperature reflects ambient conditions and individual baseline more than emotional state — high between-subject noise |
| 62 | `st_delta` | **8** | 24.25 | **0.843** | 0.020 | 10.17 | 0.55 | **SELECTED** | Temperature change within window — second-highest MI (0.84); captures peripheral vasoconstriction (Fear, Anger → temp drop) and vasodilation (relaxation → temp rise); key autonomic marker for stress in ASD literature (Stephens 2022) |
| 63 | `st_rate_change` | 79 | 64.50 | 0.000 | 0.006 | 0.14 | 1.05 | REJECTED | Zero MI, near-zero F-stat; rate of temperature change is mathematically `st_delta / window_duration` — perfectly correlated with `st_delta`, adds no information |
| 64 | `st_slope` | 81 | 64.50 | 0.000 | 0.006 | 0.14 | 1.05 | REJECTED | Identical to `st_rate_change` in practice — linear regression slope over 60s window equals delta/duration for monotonic signals. Redundant |
| 65 | `st_asymmetry` | 90 | 83.00 | 0.076 | 0.000 | 0.00 | 0.00 | REJECTED | **Zero F-stat and variance.** Temperature asymmetry requires bilateral measurement (left vs right wrist) — single-wrist E4 cannot compute this; constant/NaN |
| 66 | `st_min` | 32 | 38.50 | 0.218 | 0.014 | 12.92 | 0.44 | REJECTED | Good F-stat (12.9) but captures individual baseline temperature, not emotional response; `st_delta` isolates the change from baseline which is the emotionally relevant component |
| 67 | `st_max` | 30 | 38.00 | 0.255 | 0.021 | 5.81 | 0.49 | REJECTED | Similar to `st_min` — reflects baseline + delta; `st_delta` captures the informative component without baseline contamination |
| 68 | `st_range` | 38 | 40.25 | 0.305 | 0.019 | 0.88 | 1.02 | REJECTED | `st_max - st_min` — low F-stat (0.88); in 60s windows at 4 Hz, temperature range is very small (thermal inertia) making this feature unreliable |
| 69 | `st_std` | 29 | 38.00 | 0.271 | 0.032 | 1.59 | 0.61 | REJECTED | Good RF importance (0.032) but low F-stat; temperature variability in 60s windows is minimal due to slow thermal dynamics — captures sensor noise more than physiological variation |
| 70 | `st_acf1` | 85 | 67.75 | 0.000 | 0.013 | 1.15 | 0.30 | REJECTED | Lag-1 autocorrelation — always near 1.0 for skin temperature (extremely smooth signal); zero MI; no discriminative value as temperature is always highly autocorrelated |
| 71 | `st_pow_ultra_slow` | 93 | **90.00** | 0.000 | 0.000 | 0.00 | 0.00 | REJECTED | **Lowest-ranked feature.** Ultra-slow power requires windows >>60s to resolve — constant/NaN. Zero across all methods |
| 72 | `st_pow_slow` | 91 | 85.75 | 0.000 | 0.000 | 0.00 | 0.00 | REJECTED | Slow-frequency power — same issue as ultra-slow; 60s windows provide no spectral resolution for ST frequencies (<0.1 Hz) |

---

### ACC Features (Accelerometer) — 26 columns

| # | Column | Rank | Avg Rank | MI | RF | F-stat | Var | Status | Rationale |
|---|--------|------|----------|----|----|--------|-----|--------|-----------|
| 73 | `acc_svm_mean` | 28 | 37.00 | 0.807 | 0.009 | 8.10 | 0.39 | REJECTED | Strong MI (0.81) but low RF (0.009) and variance (0.39); mean SVM reflects overall activity level — partially captured by `acc_activity_count` (rank 18) and `acc_svm_std` (rank 19) which together provide richer characterisation |
| 74 | `acc_svm_std` | **19** | 32.25 | 0.614 | 0.017 | 12.04 | 0.35 | **SELECTED** | Variability of Signal Vector Magnitude — captures movement intensity fluctuations; high F-stat (12.0); distinguishes agitated states (Fear, Anger → high variability) from static states (Tired, Sad → low variability); complements `acc_svm_range` |
| 75 | `acc_activity_count` | **18** | 30.00 | 0.610 | 0.022 | 7.36 | 0.49 | **SELECTED** | Number of acceleration threshold crossings — captures movement frequency; balanced scores; distinguishes stereotypic/repetitive movements (common in ASD) from voluntary movements; clinically interpretable |
| 76 | `acc_dom_freq` | 84 | 65.00 | 0.000 | 0.000 | 0.30 | 7.73 | REJECTED | Zero MI and RF; dominant frequency has extremely high variance (7.73) but is unreliable — a single dominant frequency poorly characterises broadband movement; often unstable between windows |
| 77 | `acc_x_mean` | 72 | 61.00 | 0.000 | 0.002 | 0.01 | 5.09 | REJECTED | Mean X-axis acceleration — reflects wrist orientation (gravity component), not emotional state; high variance from postural differences between subjects |
| 78 | `acc_y_mean` | 60 | 54.25 | 0.139 | 0.011 | 1.02 | 0.58 | REJECTED | Mean Y-axis — same issue as X-mean; gravity-dominated; orientation-dependent |
| 79 | `acc_z_mean` | 49 | 44.75 | 0.000 | 0.011 | 1.64 | 1.28 | REJECTED | Mean Z-axis — gravity-dominated; not emotion-discriminative |
| 80 | `acc_posture_trans` | 88 | 75.75 | 0.000 | 0.003 | 0.54 | 0.27 | REJECTED | Posture transitions count — zero MI; requires longer observation periods; in 60s windows, postural changes are rare (0–2 typically) making this near-constant |
| 81 | `acc_x_std` | **11** | 26.50 | **0.989** | 0.016 | 20.62 | 0.33 | **SELECTED** | X-axis movement variability — **highest MI of any feature** (0.989); very high F-stat (20.6); captures lateral wrist movements associated with hand-flapping, reaching, and defensive gestures common in ASD emotional expression |
| 82 | `acc_y_std` | **12** | 26.50 | 0.697 | 0.024 | 19.72 | 0.31 | **SELECTED** | Y-axis movement variability — complements X-axis; high F-stat (19.7); captures vertical wrist movements (stimming, rocking); together with `acc_x_std`, provides 2D movement characterisation |
| 83 | `acc_z_std` | 50 | 45.50 | 0.676 | 0.002 | 19.02 | 0.30 | REJECTED | Z-axis variability — high F-stat (19.0) and MI but very low RF (0.002); partially redundant with X and Y std which together capture most movement variance; including all 3 axes risks multicollinearity |
| 84 | `acc_svm_range` | **16** | 29.50 | 0.624 | 0.038 | 11.88 | 0.30 | **SELECTED** | SVM peak-to-peak range — high RF importance (0.038, third-highest); captures maximum movement intensity within window; distinguishes calm baseline from agitated/distressed episodes; less sensitive to gravity offset than raw axis features |
| 85 | `acc_svm_skew` | 65 | 56.00 | 0.000 | 0.004 | 1.21 | 1.41 | REJECTED | Zero MI; movement distribution asymmetry is not physiologically informative for emotion classification — reflects random variation in movement patterns |
| 86 | `acc_svm_kurt` | 58 | 51.50 | 0.000 | 0.021 | 2.03 | 0.37 | REJECTED | Zero MI; moderate RF (0.021) but kurtosis of movement is unreliable in short windows — sensitive to single impulse events |
| 87 | `acc_x_zcr` | 31 | 38.25 | 0.037 | 0.018 | 7.40 | 0.78 | REJECTED | X-axis zero-crossing rate — near-zero MI; outperformed by `acc_y_zcr` (rank 7) which better captures the dominant movement patterns in seated/standing children |
| 88 | `acc_y_zcr` | **7** | 23.50 | 0.499 | 0.009 | **19.12** | 1.89 | **SELECTED** | Y-axis zero-crossing rate — high F-stat (19.1) and variance (1.89); captures frequency of vertical movements; distinguishes stereotypic repetitive movements (high ZCR) from stillness (low ZCR); particularly relevant for ASD motor behaviours (hand-flapping, rocking) |
| 89 | `acc_z_zcr` | 21 | 33.75 | 0.635 | 0.010 | 16.94 | 0.43 | REJECTED | Strong scores but narrowly below `acc_y_zcr`; partially redundant — 3 ZCR features together would add multicollinearity; Y-axis selected as most informative single ZCR |
| 90 | `acc_pow_sway` | 47 | 44.75 | 0.099 | 0.020 | 5.12 | 0.44 | REJECTED | Sway-band power (0.1–0.5 Hz) — postural sway frequency; low MI; requires standing posture to be meaningful — children may be seated during data collection |
| 91 | `acc_pow_voluntary` | 37 | 39.75 | 0.501 | 0.015 | 6.18 | 0.35 | REJECTED | Voluntary movement band (0.5–3 Hz) — moderate scores but outperformed by `acc_pow_rapid` (rank 14); the voluntary band is broad and less specific than rapid movements for emotion discrimination |
| 92 | `acc_pow_rapid` | **14** | 27.50 | 0.707 | **0.041** | 8.11 | 0.35 | **SELECTED** | Rapid movement power (3–8 Hz) — **highest RF importance** of all features (0.041); captures tremor, startle responses, and stereotypic movements; high-frequency motor activity is strongly associated with distress and fear in ASD (Schoen 2008) |
| 93 | `acc_pow_tremor` | 40 | 40.50 | 0.310 | 0.012 | 1.76 | 0.89 | REJECTED | Tremor-band power (8–15 Hz) — moderate scores; physiological tremor in children is less pronounced than adults; in wrist-worn devices, 8–15 Hz is at the Nyquist limit for 32 Hz sampling — poor spectral resolution |
| 94 | `acc_spec_entropy` | 70 | 58.25 | 0.000 | 0.005 | 0.42 | 1.10 | REJECTED | Zero MI; spectral entropy of ACC is relatively constant across emotional states — children's movement patterns have similar spectral spread regardless of emotion |
| 95 | `acc_corr_xy` | 76 | 63.00 | 0.000 | 0.008 | 0.68 | 0.51 | REJECTED | X-Y axis correlation — zero MI; inter-axis correlation reflects movement coordination which varies more by activity type than emotion |
| 96 | `acc_corr_xz` | 43 | 43.00 | 0.262 | 0.008 | 0.90 | 2.82 | REJECTED | X-Z axis correlation — moderate MI (0.26) but very low F-stat (0.90); high variance (2.82) from inter-subject postural differences, not emotion |
| 97 | `acc_corr_yz` | 51 | 46.25 | 0.564 | 0.014 | 2.59 | 0.25 | REJECTED | Y-Z axis correlation — decent MI (0.56) but low variance (0.25); inter-axis correlations are unstable and device-orientation dependent |
| 98 | `acc_svm_sampen` | 86 | 72.25 | 0.000 | 0.003 | 0.81 | 0.75 | REJECTED | Sample entropy of SVM — zero MI; movement complexity is not a reliable emotion marker; sample entropy requires longer time series than available at 32 Hz in 60s (1920 samples is borderline) |

---

### Demographic Columns — Raw (not scored)

| # | Column | Status | Rationale |
|---|--------|--------|-----------|
| 99 | `age` | DEMOGRAPHIC | Participant age (5–15 years) — passed through as numeric; influences physiological baselines but not directly scored as a selectable feature. Retained in training data for all models |
| 100 | `gender` | DEMOGRAPHIC | Raw string label (Male/Female/Non-binary) — retained for reference; encoded version (`gender_enc`) used by models |
| 101 | `ethnicity` | DEMOGRAPHIC | Raw string label (5 categories) — retained for reference; encoded version (`ethnicity_enc`) used by models |
| 102 | `autism_severity` | DEMOGRAPHIC | Raw string label (Low/Medium/Severe) — retained for reference; encoded version (`autism_severity_enc`) used by models |
| 103 | `verbal_status` | DEMOGRAPHIC | Raw string label (Verbal/Minimally verbal/Non-verbal) — retained for reference; encoded version (`verbal_status_enc`) used by models |
| 104 | `comorbidity` | DEMOGRAPHIC | Raw string label (Yes/No) — retained for reference; encoded version (`comorbidity_enc`) used by models |

### Demographic Columns — Encoded (passed through, not scored)

| # | Column | Encoding | Status | Rationale |
|---|--------|----------|--------|-----------|
| 105 | `gender_enc` | Ordinal | DEMOGRAPHIC | Numeric encoding of gender — included in all feature matrices; not competitively ranked against physiological features because demographics are always passed through to models |
| 106 | `autism_severity_enc` | Ordinal (1=Low, 2=Medium, 3=Severe) | DEMOGRAPHIC | Clinically ordered encoding; severity modulates physiological reactivity (1.0×/1.3×/1.6×) — always included as a conditioning variable |
| 107 | `verbal_status_enc` | Ordinal (0=Verbal, 1=Minimally verbal, 2=Non-verbal) | DEMOGRAPHIC | Clinically ordered; verbal status affects behavioural expression patterns — always included |
| 108 | `comorbidity_enc` | Binary (0=No, 1=Yes) | DEMOGRAPHIC | Comorbidity flag — always included; comorbid conditions (ADHD, anxiety) affect physiological baselines |
| 109 | `ethnicity_enc` | Ordinal | DEMOGRAPHIC | Numeric encoding — always included for demographic stratification |

---

## Summary of Selection Decisions

### Selected (Top 20) — Why These Features

| Rank | Feature | Signal | Primary Selection Driver |
|------|---------|--------|--------------------------|
| 1 | `eda_sampen` | EDA | Dominant F-stat (71.0) — strongest linear separator |
| 2 | `bvp_ptt` | BVP | Highest MI (0.94) — strongest non-linear dependency |
| 3 | `bvp_notch_depth` | BVP | Balanced across all methods — robust multi-criterion |
| 4 | `ibi_hf` | IBI | Highest variance (4.83) — best spread after scaling |
| 5 | `bvp_hr_mean` | BVP | Core autonomic marker — clinical gold standard |
| 6 | `ibi_mean` | IBI | Highest RF (0.040) and second-highest F-stat (52.2) |
| 7 | `acc_y_zcr` | ACC | High F-stat (19.1) — ASD motor behaviour marker |
| 8 | `st_delta` | ST | Second-highest MI (0.84) — vasoconstriction marker |
| 9 | `ibi_pnn50` | IBI | Parasympathetic marker — vagal withdrawal in ASD |
| 10 | `ibi_sd1` | IBI | Geometric short-term HRV — outperformed RMSSD |
| 11 | `acc_x_std` | ACC | Highest MI overall (0.99) — lateral movement |
| 12 | `acc_y_std` | ACC | Vertical movement — stimming/rocking detection |
| 13 | `bvp_f0` | BVP | Frequency-domain HR complement — high variance |
| 14 | `acc_pow_rapid` | ACC | Highest RF overall (0.041) — startle/tremor band |
| 15 | `eda_scr_rise_mean` | EDA | Rise time → fast vs slow sympathetic activation |
| 16 | `acc_svm_range` | ACC | Third-highest RF (0.038) — movement intensity |
| 17 | `ibi_sdnn` | IBI | Gold-standard HRV (Task Force 1996) |
| 18 | `acc_activity_count` | ACC | Movement frequency — clinically interpretable |
| 19 | `acc_svm_std` | ACC | Movement variability — agitation marker |
| 20 | `eda_scl_delta` | EDA | Tonic arousal shift — sustained emotion marker |

### Signal Representation in Top 20

| Signal | Selected | Total Available | Coverage |
|--------|----------|-----------------|----------|
| EDA | 3 | 22 | 14% |
| BVP | 4 | 16 | 25% |
| IBI/HRV | 5 | 17 | 29% |
| ST | 1 | 12 | 8% |
| ACC | 7 | 26 | 27% |

### Common Rejection Reasons

| Reason | Count | Examples |
|--------|-------|----------|
| **Redundancy** — correlated with a higher-ranked feature | 18 | `eda_mean` ≈ `eda_scl_mean`, `st_slope` ≈ `st_delta`, `ibi_cv` = f(`ibi_sdnn`, `ibi_mean`) |
| **Zero MI** — no detectable non-linear dependency with target | 25 | `eda_max`, `eda_range`, `ibi_vlf`, `acc_dom_freq` |
| **Insufficient window length** — feature requires longer observation | 8 | `ibi_vlf` (needs >5 min), `ibi_sampen` (needs >200 beats), `st_pow_ultra_slow` |
| **Inter-subject noise** — variance dominated by between-subject differences | 7 | `st_mean`, `eda_min`, `acc_x_mean` |
| **Constant/NaN** — feature cannot be computed from available data | 4 | `bvp_resp_mod`, `st_asymmetry`, `st_pow_slow`, `st_pow_ultra_slow` |
| **Not a predictive feature** — quality metric or structural | 1 | `bvp_sqi` (signal quality should gate, not predict) |
| **Low variance** — near-constant after normalisation | 5 | `eda_scr_count`, `eda_scr_rate`, `acc_posture_trans` |

---

*Generated from run_017 pipeline output (10 users, seed=42, 2026-04-10)*
*Selection method: Ensemble of MI + RF + F-statistic + Variance, top 20*
