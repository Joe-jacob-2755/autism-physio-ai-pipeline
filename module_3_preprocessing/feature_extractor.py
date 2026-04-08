"""
=============================================================================
MODULE 3 – DATA PREPROCESSING  |  feature_extractor.py
=============================================================================
Extract all 80 physiological features from cleaned, filtered signals.

Feature extraction is window-based:
  - Default window: 60 s with 50% overlap
  - Each window yields one feature vector per signal
  - Feature vectors are then averaged across windows per session
    (or kept windowed for time-series models)

Feature coverage (80 total):
  EDA:     18 features  (9 derived + 9 time/freq/non-linear)
  BVP:     13 features  (7 derived + 6 time/freq)
  IBI/HRV: 17 features  (5 derived + 12 freq/non-linear)
  ST:      10 features  (4 derived + 6 time/freq)
  ACC:     14 features  (6 derived + 8 time/freq/non-linear)
  Total:   72 core + 8 ACC per-axis variants = 80

=============================================================================
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis, pearsonr
from typing import Dict, List, Optional, Tuple

try:
    import antropy as ant
    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False

from config import (
    SAMPLING_RATES, WINDOW_SIZE_S, WINDOW_OVERLAP, FREQ_BANDS,
    MIN_WINDOW_SAMPLES,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe(val, default=np.nan):
    """Return val or default if val is nan/inf."""
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _band_power(sig: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Power spectral density integrated over [lo, hi] Hz using Welch."""
    nperseg = min(256, len(sig) // 2)
    if nperseg < 8:
        return np.nan
    f, pxx = sp_signal.welch(sig, fs=fs, nperseg=nperseg)
    mask = (f >= lo) & (f <= hi)
    return float(np.trapezoid(pxx[mask], f[mask])) if mask.sum() > 1 else np.nan


def _dominant_freq(sig: np.ndarray, fs: float,
                   fmin: float = 0.0, fmax: float = None) -> float:
    """Dominant (peak power) frequency in [fmin, fmax] Hz."""
    nperseg = min(256, len(sig) // 2)
    if nperseg < 8:
        return np.nan
    f, pxx = sp_signal.welch(sig, fs=fs, nperseg=nperseg)
    fmax = fmax or fs / 2
    mask = (f >= fmin) & (f <= fmax)
    if not mask.any():
        return np.nan
    return float(f[mask][np.argmax(pxx[mask])])


def _spectral_centroid(sig: np.ndarray, fs: float) -> float:
    """Frequency centre of mass of the power spectrum."""
    nperseg = min(256, len(sig) // 2)
    if nperseg < 8:
        return np.nan
    f, pxx = sp_signal.welch(sig, fs=fs, nperseg=nperseg)
    tot = pxx.sum()
    return float(np.dot(f, pxx) / tot) if tot > 0 else np.nan


def _spectral_entropy(sig: np.ndarray, fs: float) -> float:
    """Shannon entropy of normalised PSD."""
    nperseg = min(256, len(sig) // 2)
    if nperseg < 8:
        return np.nan
    _, pxx = sp_signal.welch(sig, fs=fs, nperseg=nperseg)
    pxx /= (pxx.sum() + 1e-12)
    return float(-np.sum(pxx * np.log(pxx + 1e-12)))


def _sample_entropy(sig: np.ndarray, m: int = 2, r: float = None) -> float:
    """Sample entropy (SampEn) — approximate implementation."""
    if HAS_ANTROPY:
        try:
            r_val = r if r is not None else 0.2 * np.std(sig)
            return float(ant.sample_entropy(sig, order=m, metric="chebyshev"))
        except Exception:
            pass

    # Fallback: simplified SampEn
    n = len(sig)
    if n < 10:
        return np.nan
    r_val = r if r is not None else 0.2 * float(np.std(sig))
    if r_val == 0:
        return np.nan

    def _count_matches(sig, m, r_val):
        count = 0
        for i in range(n - m):
            template = sig[i: i + m]
            for j in range(i + 1, n - m):
                if np.max(np.abs(sig[j: j + m] - template)) < r_val:
                    count += 1
        return count

    A = _count_matches(sig, m + 1, r_val)
    B = _count_matches(sig, m, r_val)
    if B == 0:
        return np.nan
    return float(-np.log((A + 1e-12) / (B + 1e-12)))


def _zero_crossing_rate(sig: np.ndarray) -> float:
    """Zero-crossing rate (crossings per sample)."""
    if len(sig) < 2:
        return np.nan
    demeaned = sig - np.mean(sig)
    crossings = np.diff(np.sign(demeaned))
    return float(np.sum(np.abs(crossings) > 0) / len(sig))


def _detect_scr_peaks(phasic: np.ndarray, fs: float,
                      min_amp: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """Detect SCR peaks in phasic EDA component."""
    if len(phasic) < 4:
        return np.array([]), np.array([])
    min_dist = max(1, int(0.5 * fs))  # minimum 0.5 s between peaks
    peaks, props = sp_signal.find_peaks(
        phasic, height=min_amp, distance=min_dist)
    amps = props.get("peak_heights", phasic[peaks]) if len(peaks) > 0 else np.array([])
    return peaks, amps


def _decompose_eda(eda: np.ndarray, fs: float):
    """
    Simple tonic/phasic EDA decomposition using low-pass filter.
    Tonic (SCL): low-pass at 0.05 Hz
    Phasic (SCR): residual after tonic removal
    """
    from signal_filters import butterworth_lowpass
    if len(eda) < 20:
        return eda, np.zeros_like(eda)
    tonic = butterworth_lowpass(eda, fs, cutoff=0.05, order=2)
    phasic = eda - tonic
    phasic = np.maximum(phasic, 0)   # SCRs are positive deflections
    return tonic, phasic


def _detect_bvp_peaks(bvp: np.ndarray, fs: float):
    """Detect systolic peaks in BVP/PPG signal."""
    min_dist = max(1, int(0.4 * fs))   # minimum 0.4 s (150 bpm max)
    height = np.percentile(bvp, 60)  # peaks above 60th percentile
    peaks, _ = sp_signal.find_peaks(bvp, distance=min_dist, height=height)
    return peaks


def _windows(n: int, win_n: int, step_n: int):
    """Generate (start, end) window indices."""
    start = 0
    while start + win_n <= n:
        yield start, start + win_n
        start += step_n


def _poincare(rr: np.ndarray) -> Tuple[float, float]:
    """Compute Poincaré SD1 and SD2 from RR series."""
    if len(rr) < 4:
        return np.nan, np.nan
    rr1 = rr[:-1]
    rr2 = rr[1:]
    sd1 = float(np.std((rr2 - rr1) / np.sqrt(2)))
    sd2 = float(np.std((rr2 + rr1) / np.sqrt(2)))
    return sd1, sd2


def _dfa_alpha1(rr: np.ndarray) -> float:
    """Detrended Fluctuation Analysis short-range α1 (4–16 beats)."""
    if HAS_ANTROPY:
        try:
            return float(ant.dfa(rr, overlap=True)[0])
        except Exception:
            pass

    # Simplified DFA
    n = len(rr)
    if n < 20:
        return np.nan
    scales = [4, 6, 8, 12, 16]
    flucs = []
    for s in scales:
        if n < 2 * s:
            continue
        n_seg = n // s
        rms_list = []
        cumsum = np.cumsum(rr - np.mean(rr))
        for i in range(n_seg):
            seg = cumsum[i * s: (i + 1) * s]
            t = np.arange(len(seg))
            poly = np.polyfit(t, seg, 1)
            trend = np.polyval(poly, t)
            rms_list.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if rms_list:
            flucs.append((np.log(s), np.log(np.mean(rms_list))))

    if len(flucs) < 2:
        return np.nan
    xs, ys = zip(*flucs)
    poly = np.polyfit(xs, ys, 1)
    return float(poly[0])


# ─────────────────────────────────────────────────────────────────────────────
# PER-SIGNAL FEATURE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def extract_eda_features(sig: np.ndarray, fs: float,
                         baseline_mean: float = None) -> dict:
    """Extract all 18 EDA features."""
    f = {}
    if len(sig) < MIN_WINDOW_SAMPLES:
        return f

    tonic, phasic = _decompose_eda(sig, fs)
    peaks, amps = _detect_scr_peaks(phasic, fs)
    dur_s = len(sig) / fs
    base = baseline_mean if baseline_mean else np.median(tonic)

    # ── Level 1: Derived ─────────────────────────────────────────────
    f["eda_scl_mean"] = _safe(np.mean(tonic))
    f["eda_scl_slope"] = _safe(np.polyfit(np.arange(len(tonic)), tonic, 1)[0])
    f["eda_scr_count"] = _safe(len(peaks))
    f["eda_scr_rate"] = _safe(len(peaks) / dur_s if dur_s > 0 else 0)
    f["eda_scr_amp_mean"] = _safe(np.mean(amps) if len(amps) > 0 else 0)
    f["eda_scr_amp_max"] = _safe(np.max(amps) if len(amps) > 0 else 0)
    # Rise time: estimate as time from trough before peak to peak
    if len(peaks) > 0:
        rise_times = []
        for pk in peaks:
            start = max(0, pk - int(2 * fs))
            seg = phasic[start:pk]
            if len(seg) > 0:
                trough_idx = np.argmin(seg)
                rise_times.append((pk - (start + trough_idx)) / fs)
        f["eda_scr_rise_mean"] = _safe(np.mean(rise_times) if rise_times else np.nan)
        # Recovery: time to fall to 50% amplitude
        rec_times = []
        for pk in peaks:
            end = min(len(phasic), pk + int(10 * fs))
            seg = phasic[pk:end]
            if len(seg) > 1:
                half = phasic[pk] * 0.5
                below = np.where(seg < half)[0]
                rec_times.append(below[0] / fs if len(below) > 0 else np.nan)
        f["eda_scr_rec_time"] = _safe(np.nanmean(rec_times) if rec_times else np.nan)
    else:
        f["eda_scr_rise_mean"] = 0.0
        f["eda_scr_rec_time"] = 0.0
    f["eda_scr_auc"] = _safe(np.trapezoid(np.maximum(phasic, 0)) / fs)

    # ── Level 2: Time domain ─────────────────────────────────────────
    f["eda_mean"] = _safe(np.mean(sig))
    f["eda_std"] = _safe(np.std(sig))
    f["eda_min"] = _safe(np.min(sig))
    f["eda_max"] = _safe(np.max(sig))
    f["eda_range"] = _safe(np.max(sig) - np.min(sig))
    f["eda_skew"] = _safe(skew(sig))
    f["eda_kurt"] = _safe(kurtosis(sig))
    f["eda_zcr"] = _safe(_zero_crossing_rate(sig) * fs)

    # ── Level 2: Frequency domain ────────────────────────────────────
    bands = FREQ_BANDS.get("EDA", {})
    f["eda_pow_vlf"] = _safe(_band_power(sig, fs, *bands.get("vlf", (0, 0.05))))
    f["eda_pow_scr"] = _safe(_band_power(sig, fs, *bands.get("scr", (0.05, 0.25))))
    f["eda_spec_centroid"] = _safe(_spectral_centroid(sig, fs))

    # ── Level 2: Non-linear ──────────────────────────────────────────
    f["eda_sampen"] = _safe(_sample_entropy(sig))

    return f


def extract_bvp_features(sig: np.ndarray, fs: float) -> dict:
    """Extract all 13 BVP features."""
    f = {}
    if len(sig) < MIN_WINDOW_SAMPLES:
        return f

    peaks = _detect_bvp_peaks(sig, fs)

    # ── Level 1: Derived (beat morphology) ──────────────────────────
    if len(peaks) >= 2:
        rr_ms = np.diff(peaks) / fs * 1000
        hr_inst = 60_000 / rr_ms
        f["bvp_hr_mean"] = _safe(np.mean(hr_inst))
        f["bvp_hr_std"] = _safe(np.std(hr_inst))

        # Systolic amplitude
        amp_vals = sig[peaks]
        f["bvp_sys_amp_mean"] = _safe(np.mean(amp_vals))

        # Dicrotic notch: look for local minimum between consecutive peaks
        notch_depths = []
        for i in range(len(peaks) - 1):
            seg = sig[peaks[i]:peaks[i + 1]]
            if len(seg) > 3:
                mid = len(seg) // 2
                notch = np.min(seg[mid // 2: mid + mid // 2])
                notch_depths.append(sig[peaks[i]] - notch)
        f["bvp_notch_depth"] = _safe(np.mean(notch_depths) if notch_depths else np.nan)

        # Pulse width at 50% (per beat)
        pw50_list = []
        for pk in peaks:
            lo_bound = max(0, pk - int(0.5 * fs))
            hi_bound = min(len(sig), pk + int(0.5 * fs))
            beat_seg = sig[lo_bound:hi_bound]
            peak_amp = sig[pk]
            half_amp = peak_amp * 0.5
            above = beat_seg > half_amp
            if above.sum() > 0:
                pw50_list.append(above.sum() / fs * 1000)
        f["bvp_pw50"] = _safe(np.mean(pw50_list) if pw50_list else np.nan)

        # PTT proxy: 40–60% of median RR (rough estimate without ECG)
        f["bvp_ptt"] = _safe(np.median(rr_ms) * 0.50)

        # Augmentation index: ratio of secondary to primary peak
        aug_list = []
        for pk in peaks:
            seg_start = max(0, pk - int(0.1 * fs))
            seg_end = min(len(sig), pk + int(0.7 * fs))
            seg = sig[seg_start:seg_end]
            if len(seg) > 4:
                second_half = seg[len(seg) // 2:]
                if second_half.max() > 0 and sig[pk] > 0:
                    aug_list.append(second_half.max() / sig[pk] * 100)
        f["bvp_aug_idx"] = _safe(np.mean(aug_list) if aug_list else np.nan)
    else:
        for k in ("bvp_hr_mean", "bvp_hr_std", "bvp_sys_amp_mean",
                  "bvp_notch_depth", "bvp_pw50", "bvp_ptt", "bvp_aug_idx"):
            f[k] = np.nan

    # ── Level 2: Time domain ─────────────────────────────────────────
    f["bvp_mean"] = _safe(np.mean(sig))
    f["bvp_std"] = _safe(np.std(sig))
    f["bvp_skew"] = _safe(skew(sig))
    f["bvp_kurt"] = _safe(kurtosis(sig))

    # ── Level 2: Frequency domain ────────────────────────────────────
    f["bvp_f0"] = _safe(_dominant_freq(sig, fs, 0.5, 3.5))
    f["bvp_resp_mod"] = _safe(_band_power(sig, fs, 0.15, 0.4))
    f["bvp_harm_ratio"] = _safe(_spectral_entropy(sig, fs))   # spectral regularity proxy
    f["bvp_spec_entropy"] = _safe(_spectral_entropy(sig, fs))
    # Signal quality index: peak regularity
    if len(peaks) >= 3:
        rr_var = np.std(np.diff(peaks)) / (np.mean(np.diff(peaks)) + 1e-12)
        f["bvp_sqi"] = float(np.clip(1.0 - rr_var, 0, 1))
    else:
        f["bvp_sqi"] = 0.0

    return f


def extract_ibi_features(
    ibi_times: np.ndarray, ibi_values: np.ndarray
) -> dict:
    """Extract all 17 IBI/HRV features."""
    f = {}
    rr = ibi_values.astype(float)
    times_in = ibi_times.astype(float)
    if len(rr) < 4:
        return f

    # Remove ectopic beats (>20% deviation from local median)
    rr_med = np.median(rr)
    valid = np.abs(rr - rr_med) / (rr_med + 1e-12) < 0.20
    times_in = times_in[valid]
    rr = rr[valid]
    if len(rr) < 4:
        return f

    # ── Level 1: Derived ─────────────────────────────────────────────
    f["ibi_mean"] = _safe(np.mean(rr))
    f["ibi_sdnn"] = _safe(np.std(rr, ddof=1))
    diff_rr = np.diff(rr)
    f["ibi_rmssd"] = _safe(np.sqrt(np.mean(diff_rr ** 2)))
    f["ibi_pnn50"] = _safe(np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100
                           if len(diff_rr) > 0 else 0)
    f["ibi_cv"] = _safe(f["ibi_sdnn"] / (f["ibi_mean"] + 1e-12) * 100)

    # ── Level 2: Frequency domain (interpolate RR to uniform grid) ──
    if len(rr) >= 8 and len(times_in) >= 8:
        t_rr = times_in
        if len(t_rr) != len(rr):
            t_rr = np.cumsum(rr / 1000)
        dur = t_rr[-1] - t_rr[0]
        fs_rr = 4.0   # interpolation rate for HRV
        t_uni = np.arange(t_rr[0], t_rr[-1], 1 / fs_rr)
        if len(t_uni) >= 8:
            rr_uni = np.interp(t_uni, t_rr, rr)
            bands = FREQ_BANDS.get("IBI", {})
            vlf = _band_power(rr_uni, fs_rr, *bands.get("vlf", (0.0, 0.04)))
            lf = _band_power(rr_uni, fs_rr, *bands.get("lf", (0.04, 0.15)))
            hf = _band_power(rr_uni, fs_rr, *bands.get("hf", (0.15, 0.40)))
            f["ibi_vlf"] = _safe(vlf)
            f["ibi_lf"] = _safe(lf)
            f["ibi_hf"] = _safe(hf)
            tot = (lf or 0) + (hf or 0) + 1e-12
            f["ibi_lf_hf"] = _safe((lf or 0) / (hf + 1e-12))
            f["ibi_lfnu"] = _safe((lf or 0) / tot * 100)
            f["ibi_hfnu"] = _safe((hf or 0) / tot * 100)
            f["ibi_total_pow"] = _safe((vlf or 0) + (lf or 0) + (hf or 0))
        else:
            for k in ("ibi_vlf", "ibi_lf", "ibi_hf", "ibi_lf_hf",
                      "ibi_lfnu", "ibi_hfnu", "ibi_total_pow"):
                f[k] = np.nan
    else:
        for k in ("ibi_vlf", "ibi_lf", "ibi_hf", "ibi_lf_hf",
                  "ibi_lfnu", "ibi_hfnu", "ibi_total_pow"):
            f[k] = np.nan

    # ── Level 2: Non-linear ──────────────────────────────────────────
    sd1, sd2 = _poincare(rr)
    f["ibi_sd1"] = _safe(sd1)
    f["ibi_sd2"] = _safe(sd2)
    f["ibi_sd1_sd2"] = _safe(sd1 / (sd2 + 1e-12) if sd2 else np.nan)
    f["ibi_sampen"] = _safe(_sample_entropy(rr))
    f["ibi_dfa_alpha1"] = _safe(_dfa_alpha1(rr))

    return f


def extract_st_features(sig: np.ndarray, fs: float,
                        baseline_mean: float = None) -> dict:
    """Extract all 10 ST features."""
    f = {}
    if len(sig) < MIN_WINDOW_SAMPLES:
        return f

    base = baseline_mean if baseline_mean else np.median(sig)

    # ── Level 1: Derived ─────────────────────────────────────────────
    f["st_mean"] = _safe(np.mean(sig))
    f["st_delta"] = _safe(np.mean(sig) - base)
    t = np.arange(len(sig)) / fs / 60   # minutes
    slope = np.polyfit(t, sig, 1)[0]
    f["st_rate_change"] = _safe(slope / 60)   # convert /min to /s
    f["st_slope"] = _safe(slope)
    f["st_asymmetry"] = np.nan  # requires dual-wrist device

    # ── Level 2: Time domain ─────────────────────────────────────────
    f["st_min"] = _safe(np.min(sig))
    f["st_max"] = _safe(np.max(sig))
    f["st_range"] = _safe(np.max(sig) - np.min(sig))
    f["st_std"] = _safe(np.std(sig))

    # Autocorrelation lag-1
    if len(sig) > 2:
        try:
            acf1 = pearsonr(sig[:-1], sig[1:])[0]
            f["st_acf1"] = _safe(acf1)
        except Exception:
            f["st_acf1"] = np.nan
    else:
        f["st_acf1"] = np.nan

    # ── Level 2: Frequency domain ────────────────────────────────────
    bands = FREQ_BANDS.get("ST", {})
    f["st_pow_ultra_slow"] = _safe(_band_power(sig, fs, *bands.get("ultra_slow", (0.0, 0.003))))
    f["st_pow_slow"] = _safe(_band_power(sig, fs, *bands.get("slow", (0.003, 0.04))))

    return f


def extract_acc_features(
    acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray, fs: float
) -> dict:
    """Extract all 14 ACC features."""
    f = {}
    if len(acc_x) < MIN_WINDOW_SAMPLES:
        return f

    # Vector magnitude (SVM)
    svm = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    # Remove gravity DC
    svm_dyn = svm - np.mean(svm)

    # ── Level 1: Derived ─────────────────────────────────────────────
    f["acc_svm_mean"] = _safe(np.mean(svm))
    f["acc_svm_std"] = _safe(np.std(svm))
    f["acc_activity_count"] = _safe(np.sum(np.abs(np.diff(svm))))
    f["acc_dom_freq"] = _safe(_dominant_freq(svm_dyn, fs, 0.1, 15.0))

    # Orientation (static component = mean per axis)
    f["acc_x_mean"] = _safe(np.mean(acc_x))
    f["acc_y_mean"] = _safe(np.mean(acc_y))
    f["acc_z_mean"] = _safe(np.mean(acc_z))

    # Postural transitions: large SVM derivative spikes
    svm_diff = np.abs(np.diff(svm))
    thresh = np.mean(svm_diff) + 3 * np.std(svm_diff)
    f["acc_posture_trans"] = _safe(np.sum(svm_diff > thresh))

    # ── Level 2: Time domain ─────────────────────────────────────────
    f["acc_x_std"] = _safe(np.std(acc_x))
    f["acc_y_std"] = _safe(np.std(acc_y))
    f["acc_z_std"] = _safe(np.std(acc_z))
    f["acc_svm_range"] = _safe(np.max(svm) - np.min(svm))
    f["acc_svm_skew"] = _safe(skew(svm))
    f["acc_svm_kurt"] = _safe(kurtosis(svm))

    # Zero-crossing rate per axis (detrended)
    f["acc_x_zcr"] = _safe(_zero_crossing_rate(acc_x - np.mean(acc_x)) * fs)
    f["acc_y_zcr"] = _safe(_zero_crossing_rate(acc_y - np.mean(acc_y)) * fs)
    f["acc_z_zcr"] = _safe(_zero_crossing_rate(acc_z - np.mean(acc_z)) * fs)

    # ── Level 2: Frequency domain ────────────────────────────────────
    bands = FREQ_BANDS.get("ACC", {})
    f["acc_pow_sway"] = _safe(_band_power(svm_dyn, fs, *bands.get("sway", (0.0, 1.0))))
    f["acc_pow_voluntary"] = _safe(_band_power(svm_dyn, fs, *bands.get("voluntary", (1.0, 3.0))))
    f["acc_pow_rapid"] = _safe(_band_power(svm_dyn, fs, *bands.get("rapid", (3.0, 8.0))))
    f["acc_pow_tremor"] = _safe(_band_power(svm_dyn, fs, *bands.get("tremor", (8.0, 16.0))))
    f["acc_spec_entropy"] = _safe(_spectral_entropy(svm_dyn, fs))

    # Cross-axis correlations
    try:
        f["acc_corr_xy"] = _safe(pearsonr(acc_x, acc_y)[0])
        f["acc_corr_xz"] = _safe(pearsonr(acc_x, acc_z)[0])
        f["acc_corr_yz"] = _safe(pearsonr(acc_y, acc_z)[0])
    except Exception:
        f["acc_corr_xy"] = f["acc_corr_xz"] = f["acc_corr_yz"] = np.nan

    # Non-linear
    f["acc_svm_sampen"] = _safe(_sample_entropy(svm))

    return f


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR  (orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extract all 80 physiological features from cleaned, filtered signals.

    Parameters
    ----------
    window_s    : feature extraction window in seconds (default 60)
    overlap     : window overlap fraction (default 0.50)
    verbose     : print progress
    """

    def __init__(
        self,
        window_s: float = WINDOW_SIZE_S,
        overlap: float = WINDOW_OVERLAP,
        verbose: bool = True,
    ):
        self.window_s = window_s
        self.overlap = overlap
        self.verbose = verbose

    def extract_all(
        self,
        signals: Dict[str, pd.DataFrame],
        session_id: str = "unknown",
        user_id: str = "unknown",
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract features from all signal channels.

        Returns
        -------
        {signal_name: DataFrame} — one row per window
        Each DataFrame has columns: session_id, user_id, window_start_s,
        window_end_s, label (if annotated), + all signal-specific features
        """
        all_feats = {}

        import time as _t

        # ── EDA ─────────────────────────────────────────────────────
        if "EDA" in signals:
            self._log("  EDA  — extracting ...")
            _t0 = _t.time()
            all_feats["EDA"] = self._extract_windowed(
                signals["EDA"], "EDA_uS", "EDA",
                self._extract_eda_window, session_id, user_id
            )
            self._log(f"  EDA: {len(all_feats['EDA'])} windows, "
                      f"{all_feats['EDA'].shape[1]} cols  ({_t.time() - _t0:.1f}s)")

        # ── BVP ─────────────────────────────────────────────────────
        if "BVP" in signals:
            self._log("  BVP  — extracting ...")
            _t0 = _t.time()
            all_feats["BVP"] = self._extract_windowed(
                signals["BVP"], "BVP_nT", "BVP",
                self._extract_bvp_window, session_id, user_id
            )
            self._log(f"  BVP: {len(all_feats['BVP'])} windows, "
                      f"{all_feats['BVP'].shape[1]} cols  ({_t.time() - _t0:.1f}s)")

        # ── IBI ─────────────────────────────────────────────────────
        if "IBI" in signals:
            self._log("  IBI  — extracting ...")
            _t0 = _t.time()
            all_feats["IBI"] = self._extract_ibi(
                signals["IBI"], session_id, user_id)
            self._log(f"  IBI: {len(all_feats['IBI'])} windows, "
                      f"{all_feats['IBI'].shape[1]} cols  ({_t.time() - _t0:.1f}s)")

        # ── ST ──────────────────────────────────────────────────────
        if "ST" in signals:
            self._log("  ST   — extracting ...")
            _t0 = _t.time()
            all_feats["ST"] = self._extract_windowed(
                signals["ST"], "ST_degC", "ST",
                self._extract_st_window, session_id, user_id
            )
            self._log(f"  ST: {len(all_feats['ST'])} windows, "
                      f"{all_feats['ST'].shape[1]} cols  ({_t.time() - _t0:.1f}s)")

        # ── ACC ─────────────────────────────────────────────────────
        if "ACC" in signals:
            self._log("  ACC  — extracting ...")
            _t0 = _t.time()
            all_feats["ACC"] = self._extract_acc(
                signals["ACC"], session_id, user_id)
            self._log(f"  ACC: {len(all_feats['ACC'])} windows, "
                      f"{all_feats['ACC'].shape[1]} cols  ({_t.time() - _t0:.1f}s)")

        return all_feats

    # ── Internal extraction helpers ──────────────────────────────────

    def _extract_windowed(self, df, val_col, sig_name, fn,
                          session_id, user_id) -> pd.DataFrame:
        """Generic windowed feature extraction for single-column signals."""
        if val_col not in df.columns:
            return pd.DataFrame()
        fs = float(SAMPLING_RATES.get(sig_name, 4) or 4)
        win_n = int(self.window_s * fs)
        step_n = int(win_n * (1 - self.overlap))
        arr = df[val_col].values.astype(float)
        ts = df["timestamp_s"].values if "timestamp_s" in df.columns else np.arange(len(arr)) / fs

        has_label = "target_label" in df.columns
        rows = []
        for s, e in _windows(len(arr), win_n, step_n):
            seg = arr[s:e]
            label = None
            if has_label:
                labels_seg = df["target_label"].iloc[s:e]
                # Most frequent label in window
                label = labels_seg.mode()[0] if len(labels_seg) > 0 else "unknown"

            base = float(arr[:s].mean()) if s > 0 else float(np.median(arr))
            feats = fn(seg, fs, base)

            row = {
                "session_id": session_id,
                "user_id": user_id,
                "window_start_s": _safe(ts[s]),
                "window_end_s": _safe(ts[min(e - 1, len(ts) - 1)]),
                "target_label": label if has_label else "unknown",
                **feats,
            }
            rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _extract_eda_window(self, seg, fs, base):
        return extract_eda_features(seg, fs, baseline_mean=base)

    def _extract_bvp_window(self, seg, fs, base):
        return extract_bvp_features(seg, fs)

    def _extract_st_window(self, seg, fs, base):
        return extract_st_features(seg, fs, baseline_mean=base)

    def _extract_ibi(self, df, session_id, user_id) -> pd.DataFrame:
        """IBI is event-based — use all values in window."""
        if "IBI_ms" not in df.columns:
            return pd.DataFrame()
        times = df["timestamp_s"].values if "timestamp_s" in df.columns else np.arange(len(df))
        values = df["IBI_ms"].values.astype(float)
        has_label = "target_label" in df.columns
        dur_s = times[-1] - times[0] if len(times) > 1 else 0
        win_s = self.window_s
        step_s = win_s * (1 - self.overlap)

        rows = []
        t0 = times[0]
        while t0 + win_s <= times[-1] + 1:
            mask = (times >= t0) & (times < t0 + win_s)
            if mask.sum() >= 4:
                seg_t = times[mask]
                seg_v = values[mask]
                label = None
                if has_label:
                    labels = df["target_label"].values[mask]
                    label = pd.Series(labels).mode()[0] if len(labels) > 0 else "unknown"
                feats = extract_ibi_features(seg_t, seg_v)
                rows.append({
                    "session_id": session_id, "user_id": user_id,
                    "window_start_s": _safe(t0),
                    "window_end_s": _safe(t0 + win_s),
                    "target_label": label if has_label else "unknown",
                    **feats,
                })
            t0 += step_s

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _extract_acc(self, df, session_id, user_id) -> pd.DataFrame:
        """Extract ACC features from 3-axis data."""
        if not all(c in df.columns for c in ("ACC_X_g", "ACC_Y_g", "ACC_Z_g")):
            return pd.DataFrame()
        fs = float(SAMPLING_RATES.get("ACC", 32))
        win_n = int(self.window_s * fs)
        step_n = int(win_n * (1 - self.overlap))
        ax = df["ACC_X_g"].values.astype(float)
        ay = df["ACC_Y_g"].values.astype(float)
        az = df["ACC_Z_g"].values.astype(float)
        ts = df["timestamp_s"].values if "timestamp_s" in df.columns else np.arange(len(ax)) / fs
        has_label = "target_label" in df.columns

        rows = []
        for s, e in _windows(len(ax), win_n, step_n):
            label = None
            if has_label:
                labels = df["target_label"].iloc[s:e]
                label = labels.mode()[0] if len(labels) > 0 else "unknown"
            feats = extract_acc_features(ax[s:e], ay[s:e], az[s:e], fs)
            rows.append({
                "session_id": session_id, "user_id": user_id,
                "window_start_s": _safe(ts[s]),
                "window_end_s": _safe(ts[min(e - 1, len(ts) - 1)]),
                "target_label": label if has_label else "unknown",
                **feats,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _log(self, msg):
        if self.verbose:
            print(msg)
