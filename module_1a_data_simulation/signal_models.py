"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  signal_models.py
=============================================================================
Low-level physiological signal generation primitives.

Implements physically realistic waveform models for each sensor modality:
  • EDA  – Skin Conductance Level (SCL) + Skin Conductance Response (SCR)
  • BVP  – PPG waveform via IPFM-based beat placement
  • IBI  – RR-interval series derived from beat timing
  • ST   – Skin Temperature slow-drift model
  • ACC  – 3-axis accelerometer with breathing artifact and activity model

=============================================================================
"""

from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def smooth_onset_offset_envelope(
    n_samples: int,
    onset_samples: int,
    offset_samples: int,
) -> np.ndarray:
    """
    Create a trapezoidal envelope with sigmoid-smoothed edges.

    Parameters
    ----------
    n_samples      : total samples in the window
    onset_samples  : ramp-up duration in samples
    offset_samples : ramp-down duration in samples

    Returns
    -------
    envelope : ndarray of shape (n_samples,) in [0, 1]
    """
    env = np.ones(n_samples)
    # Sigmoid-shaped rise
    for i in range(onset_samples):
        x = (i / onset_samples) * 6 - 3          # –3 … +3 for tanh
        env[i] = 0.5 * (1 + np.tanh(x))
    # Sigmoid-shaped fall
    for i in range(offset_samples):
        x = ((offset_samples - i) / offset_samples) * 6 - 3
        env[n_samples - offset_samples + i] = 0.5 * (1 + np.tanh(x))
    return env


def lowpass_filter(sig: np.ndarray, cutoff_hz: float, fs: int, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter."""
    nyq = fs / 2.0
    wn  = min(cutoff_hz / nyq, 0.99)
    b, a = sp_signal.butter(order, wn, btype="low")
    return sp_signal.filtfilt(b, a, sig)


def clip_to_range(sig: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip signal to physiological range."""
    return np.clip(sig, lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# EDA – ELECTRODERMAL ACTIVITY
# ─────────────────────────────────────────────────────────────────────────────

def eda_scr_kernel(
    fs: int,
    amplitude: float,
    tau_rise_s: float  = 0.50,
    tau_decay_s: float = 5.00,
    max_dur_s:  float  = 20.0,
) -> np.ndarray:
    """
    Generate a single SCR (Skin Conductance Response) kernel.

    Model: A * (1 – exp(–t/τ_rise)) * exp(–t/τ_decay)
    This produces a biphasic response: fast rise, slow recovery.

    Parameters
    ----------
    fs            : sampling rate (Hz)
    amplitude     : peak amplitude in µS
    tau_rise_s    : rise time constant (s)
    tau_decay_s   : decay time constant (s)
    max_dur_s     : kernel duration cap (s)

    Returns
    -------
    scr : 1D ndarray
    """
    n = int(max_dur_s * fs)
    t = np.arange(n) / fs
    scr = amplitude * (1 - np.exp(-t / tau_rise_s)) * np.exp(-t / tau_decay_s)
    return scr


def generate_eda_baseline(
    duration_s: float,
    fs: int,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate EDA tonic baseline (Skin Conductance Level).

    Comprises:
      1. Mean tonic level  ± subject variability
      2. Slow sinusoidal ultradian drift
      3. Low-amplitude spontaneous SCRs
    """
    n      = int(duration_s * fs)
    t      = np.arange(n) / fs

    # (1) Tonic baseline
    tonic_base = params["tonic_mean"] + rng.normal(0, params["tonic_std"])

    # (2) Slow drift
    drift_phase = rng.uniform(0, 2 * np.pi)
    drift = params["drift_amp"] * np.sin(2 * np.pi * params["drift_freq"] * t + drift_phase)

    # (3) Spontaneous SCRs
    spont = np.zeros(n)
    interval_s = 1.0 / max(params["scr_rate"], 1e-6)
    t_next = rng.exponential(interval_s)
    while t_next < duration_s:
        amp   = params["scr_amp_mean"] * rng.lognormal(0, 0.3)
        onset = int(t_next * fs)
        kernel = eda_scr_kernel(fs, amp)
        end   = min(onset + len(kernel), n)
        spont[onset:end] += kernel[:end - onset]
        t_next += rng.exponential(interval_s)

    return tonic_base + drift + spont


def generate_eda_event_signal(
    duration_s: float,
    fs: int,
    profile: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate EDA signal perturbation during an emotion/behaviour event.

    Returns additive component to be superimposed on the baseline.
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # ── Tonic elevation with smooth onset ─────────────────────────────────
    rise_tau = profile["rise_time_s"]
    tonic_rise = profile["tonic_delta"] * (1 - np.exp(-t / rise_tau))

    # ── Recovery at end: partial exponential decay ─────────────────────
    recovery_start = int(0.80 * n)
    recovery = np.zeros(n)
    if recovery_start < n:
        recover_t = np.arange(n - recovery_start) / fs
        recover_tau = max(duration_s * 0.25, 2.0)
        target = profile["tonic_delta"] * (1 - profile["recovery_factor"])
        for i, v in enumerate(
            target * (1 - np.exp(-recover_t / recover_tau)) - target
        ):
            recovery[recovery_start + i] = v

    tonic_env = tonic_rise + recovery

    # ── Phasic SCRs ──────────────────────────────────────────────────────
    phasic = np.zeros(n)
    scr_interval = 1.0 / max(profile["scr_rate_hz"], 1e-3)
    t_next = rng.exponential(0.5 * scr_interval)  # start with shorter first interval

    while t_next < duration_s * 0.90:
        amp = abs(rng.normal(
            profile["scr_amplitude_mean"],
            profile["scr_amplitude_std"],
        ))
        amp  = max(amp, 0.05)
        onset = int(t_next * fs)
        kernel = eda_scr_kernel(fs, amp)
        end   = min(onset + len(kernel), n)
        phasic[onset:end] += kernel[:end - onset]
        t_next += rng.exponential(scr_interval)

    return tonic_env + phasic


# ─────────────────────────────────────────────────────────────────────────────
# BVP – BLOOD VOLUME PULSE (PPG)
# ─────────────────────────────────────────────────────────────────────────────

def ppg_beat_template(fs: int, rr_s: float, amplitude: float) -> np.ndarray:
    """
    Physiologically-inspired single PPG beat template.

    Three-Gaussian model:
      G1 – systolic peak      (largest, ~18% into beat)
      G2 – dicrotic notch     (small negative deflection, ~45%)
      G3 – diastolic peak     (moderate, ~55%)
    """
    n   = int(rr_s * fs)
    t01 = np.linspace(0, 1, n)  # normalised time [0,1] per beat

    # Systolic peak
    g1 = amplitude       * np.exp(-((t01 - 0.18) ** 2) / (2 * 0.06 ** 2))
    # Dicrotic notch (dip)
    g2 = -0.12*amplitude * np.exp(-((t01 - 0.42) ** 2) / (2 * 0.03 ** 2))
    # Diastolic peak
    g3 = 0.30*amplitude  * np.exp(-((t01 - 0.54) ** 2) / (2 * 0.08 ** 2))

    return g1 + g2 + g3


def generate_ibi_sequence(
    duration_s: float,
    mean_hr_bpm: float,
    hrv_std_ms: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a beat-by-beat IBI (RR interval) sequence.

    Uses an AR(1) model to produce correlated HRV — physiologically more
    realistic than independent Gaussian draws.

    Returns
    -------
    beat_times_s : ndarray – absolute beat onset times (seconds)
    ibi_ms       : ndarray – inter-beat intervals in milliseconds
    """
    mean_rr_s  = 60.0 / mean_hr_bpm
    hrv_std_s  = hrv_std_ms / 1000.0
    phi        = 0.55          # AR(1) coefficient
    prev_rr    = mean_rr_s

    beat_times = []
    ibi_vals   = []
    t_now      = 0.0

    while t_now < duration_s + 2.0:   # small buffer
        noise  = rng.normal(0, hrv_std_s)
        rr     = mean_rr_s + phi * (prev_rr - mean_rr_s) + noise
        rr     = float(np.clip(rr, 0.30, 2.00))   # 30–200 bpm bounds
        beat_times.append(t_now)
        ibi_vals.append(rr * 1000.0)
        t_now  += rr
        prev_rr = rr

    return np.array(beat_times), np.array(ibi_vals)


def generate_bvp_from_beats(
    duration_s:   float,
    fs:           int,
    beat_times_s: np.ndarray,
    ibi_ms:       np.ndarray,
    amplitude:    float,
) -> np.ndarray:
    """
    Construct continuous BVP signal by superimposing beat templates.

    Parameters
    ----------
    duration_s    : total signal length in seconds
    fs            : BVP sampling rate
    beat_times_s  : array of beat onset times (s)
    ibi_ms        : corresponding IBI values (ms)
    amplitude     : nominal systolic peak amplitude (nT)
    """
    n   = int(duration_s * fs)
    bvp = np.zeros(n)

    for i, (bt, ibi) in enumerate(zip(beat_times_s, ibi_ms)):
        idx = int(bt * fs)
        if idx >= n:
            break
        rr_s  = ibi / 1000.0
        template = ppg_beat_template(fs, rr_s, amplitude)
        end   = min(idx + len(template), n)
        bvp[idx:end] += template[:end - idx]

    return bvp


def build_hr_track(
    duration_s:  float,
    fs_out:      int,
    events:      list,          # list of EventConfig
    baseline_hr: float,
    rng:         np.random.Generator,
) -> np.ndarray:
    """
    Build a smooth instantaneous HR track for the full recording.

    Baseline HR with slow drift; event periods push HR toward their target
    using an exponential approach curve.

    Returns
    -------
    hr_track : ndarray of length int(duration_s * fs_out), unit: bpm
    """
    n  = int(duration_s * fs_out)
    hr = np.full(n, baseline_hr)

    for ev in events:
        start_idx = int(ev.start_s * fs_out)
        end_idx   = int((ev.start_s + ev.duration_s) * fs_out)
        end_idx   = min(end_idx, n)

        delta = ev.bvp_hr_delta
        tau_s = max(ev.bvp_hr_rise_s, 0.5)
        tau_n = int(tau_s * fs_out)

        for i in range(start_idx, end_idx):
            t_in  = (i - start_idx) / fs_out
            hr[i] = baseline_hr + delta * (1 - np.exp(-t_in / tau_s))

        # Recovery after event
        recovery_end = min(end_idx + int(20 * fs_out), n)
        tau_rec = max(ev.duration_s * 0.4, 5.0)
        for i in range(end_idx, recovery_end):
            t_out = (i - end_idx) / fs_out
            hr_at_end = baseline_hr + delta * (1 - np.exp(-ev.duration_s / tau_s))
            hr[i] = baseline_hr + (hr_at_end - baseline_hr) * np.exp(-t_out / tau_rec)

    # Slow baseline drift ± a few bpm
    drift_phase = rng.uniform(0, 2 * np.pi)
    slow_drift  = 2.0 * np.sin(2 * np.pi * 0.002 * np.arange(n) / fs_out + drift_phase)
    hr = hr + slow_drift

    return np.clip(hr, 30.0, 200.0)


# ─────────────────────────────────────────────────────────────────────────────
# ST – SKIN TEMPERATURE
# ─────────────────────────────────────────────────────────────────────────────

def generate_st_baseline(
    duration_s: float,
    fs: int,
    params: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate skin temperature baseline.

    Very slow signal:
      • Mean level with subject variability
      • Ultra-low-frequency sinusoidal drift (vascular/circadian)
      • Smoothed random walk (thermoregulation)
    """
    n     = int(duration_s * fs)
    t     = np.arange(n) / fs

    base  = params["mean_celsius"] + rng.normal(0, 0.4)  # subject variability
    phase = rng.uniform(0, 2 * np.pi)
    drift = params["slow_drift_amp"] * np.sin(
        2 * np.pi * params["slow_drift_freq"] * t + phase
    )

    # Random walk → low-pass filter to simulate very slow regulation
    rw = np.cumsum(rng.normal(0, 0.01, n))
    rw = rw - rw.mean()
    rw = lowpass_filter(rw, cutoff_hz=0.008, fs=fs)

    noise = rng.normal(0, params["noise_std"], n)

    return base + drift + rw * 0.08 + noise


def generate_st_event_signal(
    duration_s: float,
    fs: int,
    profile: dict,
) -> np.ndarray:
    """
    Generate ST perturbation during event (additive).

    ST changes very slowly – modelled as exponential approach to target delta.
    """
    n     = int(duration_s * fs)
    t     = np.arange(n) / fs
    delta = profile["delta_celsius"]
    rate  = max(profile["change_rate"], 0.005)

    # Approach target temperature change
    st_delta = delta * (1 - np.exp(-rate * t))
    return st_delta


# ─────────────────────────────────────────────────────────────────────────────
# ACC – ACCELEROMETER (3-AXIS)
# ─────────────────────────────────────────────────────────────────────────────

def generate_acc_baseline(
    duration_s: float,
    fs: int,
    params: dict,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3-axis accelerometer baseline.

    Gravity dominant on Z; breathing artifact on all three axes;
    white noise sensor floor.
    """
    n     = int(duration_s * fs)
    t     = np.arange(n) / fs
    bf    = params["breathing_freq"]
    ba    = params["breathing_amp"]
    ns    = params["noise_std"]

    phase_x = rng.uniform(0, 2 * np.pi)
    phase_y = rng.uniform(0, 2 * np.pi)
    phase_z = rng.uniform(0, 2 * np.pi)

    acc_x = (params["x_mean"]
             + ba * np.sin(2 * np.pi * bf * t + phase_x)
             + rng.normal(0, ns, n))
    acc_y = (params["y_mean"]
             + ba * 0.7 * np.cos(2 * np.pi * bf * t + phase_y)
             + rng.normal(0, ns, n))
    acc_z = (params["z_mean"]                           # gravity
             + ba * 0.5 * np.sin(2 * np.pi * bf * t + phase_z)
             + rng.normal(0, ns, n))

    return acc_x, acc_y, acc_z


def generate_acc_event_signal(
    duration_s: float,
    fs: int,
    profile: dict,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate ACC perturbation during event (additive delta from baseline).

    Amplitude, dominant frequency, and irregular component are
    emotion-specific.
    """
    n     = int(duration_s * fs)
    t     = np.arange(n) / fs
    amp   = profile["activity_amp"]
    freq  = profile["dominant_freq_hz"]
    jit   = profile["jitter"]

    # Smooth onset / offset envelope
    onset_n  = int(min(1.5 * fs, n * 0.20))
    offset_n = int(min(2.0 * fs, n * 0.20))
    env      = smooth_onset_offset_envelope(n, onset_n, offset_n)

    px, py, pz = (rng.uniform(0, 2 * np.pi) for _ in range(3))

    ax_delta = env * (amp       * np.sin(2 * np.pi * freq * t + px)
                      + jit     * rng.standard_normal(n))
    ay_delta = env * (amp * 0.8 * np.sin(2 * np.pi * freq * t + py)
                      + jit     * rng.standard_normal(n))
    az_delta = env * (amp * 0.5 * np.sin(2 * np.pi * freq * t + pz)
                      + jit * 0.5 * rng.standard_normal(n))

    return ax_delta, ay_delta, az_delta
