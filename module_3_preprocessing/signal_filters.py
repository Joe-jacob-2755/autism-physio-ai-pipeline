"""
=============================================================================
MODULE 3 – DATA PREPROCESSING  |  signal_filters.py
=============================================================================
Three noise-reduction filter implementations for all physiological signals.

Filters
-------
1. Butterworth Low-pass / Band-pass
   - Configurable cutoff(s) and order
   - Applied per signal using zero-phase filtfilt (no phase distortion)
   - Default cutoffs are signal-specific (see config.py)

2. Hampel Filter (median absolute deviation outlier removal)
   - Sliding window median filter with adaptive threshold
   - Window size and sigma (threshold multiplier) are configurable
   - Best for impulsive artefacts (motion spikes, electrode pops)

3. Kalman Filter (1D linear state-space smoother)
   - Models signal as random walk (constant velocity model)
   - Observation and process noise configurable per signal
   - Best for Gaussian sensor noise in ST and EDA tonic component

The preprocessing pipeline applies filters sequentially:
   Raw → Hampel (outlier removal) → Butterworth / Kalman (noise reduction)

=============================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from typing import Optional, Tuple

from config import FILTER_DEFAULTS, KALMAN_DEFAULTS, SAMPLING_RATES


# ─────────────────────────────────────────────────────────────────────────────
# 1. BUTTERWORTH LOW-PASS / BAND-PASS FILTER
# ─────────────────────────────────────────────────────────────────────────────

def butterworth_lowpass(
    sig:      np.ndarray,
    fs:       float,
    cutoff:   float,
    order:    int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter.

    Uses scipy.signal.filtfilt for zero phase distortion — essential for
    preserving the temporal alignment of physiological events.

    Parameters
    ----------
    sig    : 1D signal array
    fs     : sampling rate (Hz)
    cutoff : -3 dB cutoff frequency (Hz)
    order  : filter order (4 gives 80 dB/decade roll-off)
    """
    nyq = fs / 2.0
    wn  = np.clip(cutoff / nyq, 1e-4, 0.9999)
    b, a = sp_signal.butter(order, wn, btype="low")
    return sp_signal.filtfilt(b, a, sig, padlen=min(3 * max(len(b), len(a)), len(sig) - 1))


def butterworth_bandpass(
    sig:     np.ndarray,
    fs:      float,
    lowcut:  float,
    highcut: float,
    order:   int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    sig     : 1D signal array
    fs      : sampling rate (Hz)
    lowcut  : lower -3 dB cutoff (Hz)
    highcut : upper -3 dB cutoff (Hz)
    order   : filter order
    """
    nyq = fs / 2.0
    lo  = np.clip(lowcut  / nyq, 1e-4, 0.9999)
    hi  = np.clip(highcut / nyq, 1e-4, 0.9999)
    if lo >= hi:
        raise ValueError(f"lowcut ({lowcut}) must be < highcut ({highcut})")
    b, a = sp_signal.butter(order, [lo, hi], btype="band")
    return sp_signal.filtfilt(b, a, sig, padlen=min(3 * max(len(b), len(a)), len(sig) - 1))


def butterworth_highpass(
    sig:    np.ndarray,
    fs:     float,
    cutoff: float,
    order:  int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter."""
    nyq = fs / 2.0
    wn  = np.clip(cutoff / nyq, 1e-4, 0.9999)
    b, a = sp_signal.butter(order, wn, btype="high")
    return sp_signal.filtfilt(b, a, sig, padlen=min(3 * max(len(b), len(a)), len(sig) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# 2. HAMPEL FILTER
# ─────────────────────────────────────────────────────────────────────────────

def hampel_filter(
    sig:    np.ndarray,
    window: int   = 7,
    sigma:  float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hampel identifier: detect and replace impulsive outliers using a
    sliding window median absolute deviation (MAD) estimator.

    Algorithm
    ---------
    For each sample x[i]:
      1. Compute local median m over a window of width (2k+1) centred on i
      2. Compute MAD = median(|x[j] - m|) for j in window
      3. Estimate local σ = 1.4826 · MAD  (consistent estimator for Gaussian)
      4. If |x[i] - m| > sigma · σ → replace x[i] with m (outlier)

    The 1.4826 scaling factor makes MAD consistent with the standard
    deviation for normally distributed data.

    Parameters
    ----------
    sig    : 1D signal array
    window : half-window size k (total window = 2k+1 samples)
    sigma  : detection threshold in units of estimated local SD

    Returns
    -------
    filtered_sig : outlier-replaced signal
    outlier_mask : boolean array True where outliers were detected
    """
    n       = len(sig)
    out     = sig.copy().astype(float)
    mask    = np.zeros(n, dtype=bool)
    k       = window // 2
    SCALE   = 1.4826   # MAD → sigma scaling constant

    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        window_vals = sig[lo:hi]

        if len(window_vals) < 3:
            continue

        med     = np.median(window_vals)
        mad     = np.median(np.abs(window_vals - med))
        local_s = SCALE * mad

        if local_s > 0 and abs(sig[i] - med) > sigma * local_s:
            out[i]  = med
            mask[i] = True

    return out, mask


# ─────────────────────────────────────────────────────────────────────────────
# 3. KALMAN FILTER (1D Optimal Linear Smoother)
# ─────────────────────────────────────────────────────────────────────────────

def kalman_filter_1d(
    sig:               np.ndarray,
    observation_noise: float = 1.0,
    process_noise:     float = 0.1,
) -> np.ndarray:
    """
    1D Kalman filter / smoother for a scalar time series.

    State-space model
    -----------------
    State transition : x[t] = x[t-1] + w[t]      w ~ N(0, Q)  (random walk)
    Observation      : z[t] = x[t]   + v[t]      v ~ N(0, R)

    The filter performs two passes (forward predict-update, backward smooth)
    using the Rauch-Tung-Striebel smoother for bidirectional smoothing.

    Parameters
    ----------
    sig                : 1D observed signal
    observation_noise  : measurement noise variance R (larger = smoother)
    process_noise      : process noise variance Q (larger = faster response)

    Returns
    -------
    smoothed : Kalman-smoothed signal (same length as input)

    Rationale for physiological signals
    ------------------------------------
    Unlike simple moving average, the Kalman filter is optimal (minimum
    variance unbiased) when noise is Gaussian. For EDA tonic level and ST,
    both the signal dynamics and sensor noise are approximately Gaussian,
    making Kalman the preferred smoother over fixed-window averages.
    """
    n    = len(sig)
    Q    = float(process_noise)
    R    = float(observation_noise)

    # ── Forward pass: Kalman filter ────────────────────────────────────
    x_pred = np.zeros(n)
    P_pred = np.zeros(n)
    x_upd  = np.zeros(n)
    P_upd  = np.zeros(n)
    K      = np.zeros(n)

    # Initialise
    x_upd[0] = sig[0]
    P_upd[0] = R

    for t in range(1, n):
        # Predict
        x_pred[t] = x_upd[t - 1]
        P_pred[t] = P_upd[t - 1] + Q

        # Update (Kalman gain)
        K[t]      = P_pred[t] / (P_pred[t] + R)
        x_upd[t]  = x_pred[t] + K[t] * (sig[t] - x_pred[t])
        P_upd[t]  = (1 - K[t]) * P_pred[t]

    # ── Backward pass: RTS smoother ─────────────────────────────────
    x_smooth = x_upd.copy()
    P_smooth = P_upd.copy()

    for t in range(n - 2, -1, -1):
        if P_pred[t + 1] > 0:
            G          = P_upd[t] / P_pred[t + 1]
            x_smooth[t]= x_upd[t] + G * (x_smooth[t + 1] - x_pred[t + 1])
            P_smooth[t]= P_upd[t] + G ** 2 * (P_smooth[t + 1] - P_pred[t + 1])

    return x_smooth


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL FILTER MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class SignalFilterManager:
    """
    Apply a configurable filter pipeline to all cleaned signal DataFrames.

    For each signal:
      1. Hampel filter  → remove impulsive artefacts
      2. Butterworth / Kalman → reduce broadband / Gaussian noise

    Parameters
    ----------
    filter_type     : 'butterworth' | 'kalman' | 'hampel_only' | 'none'
    signal_overrides: dict to override default filter params per signal
    verbose         : print per-signal filter summary
    """

    def __init__(
        self,
        filter_type:      str  = "butterworth",
        apply_hampel:     bool = True,
        signal_overrides: dict = None,
        verbose:          bool = True,
    ):
        self.filter_type      = filter_type
        self.apply_hampel     = apply_hampel
        self.signal_overrides = signal_overrides or {}
        self.verbose          = verbose

    # ── Public API ─────────────────────────────────────────────────────────

    def filter_all(
        self, signals: dict
    ) -> dict:
        """
        Filter all signal DataFrames.

        Parameters
        ----------
        signals : {signal_name: DataFrame} — cleaned signals from SignalCleaner

        Returns
        -------
        filtered signals dict (same structure)
        """
        filtered = {}
        for name, df in signals.items():
            filtered[name] = self.filter_signal(df, name)
        return filtered

    def filter_signal(
        self, df: pd.DataFrame, signal_name: str
    ) -> pd.DataFrame:
        """Apply filter pipeline to a single signal DataFrame."""
        df_out  = df.copy()
        params  = {**FILTER_DEFAULTS.get(signal_name, {}),
                   **self.signal_overrides.get(signal_name, {})}
        fs      = SAMPLING_RATES.get(signal_name, 4) or 4.0

        val_cols = [c for c in df_out.columns
                    if c not in ("timestamp_s", "target_label",
                                 "event_id", "category", "user_id")
                    and df_out[c].dtype in (float, "float64", "float32")]

        hampel_stats = {}
        for col in val_cols:
            arr = df_out[col].values.astype(float)
            if len(arr) < 4:
                continue

            # ── Stage 1: Hampel outlier removal ──────────────────────
            n_outliers = 0
            if self.apply_hampel:
                h_win   = params.get("hampel_window", 7)
                h_sigma = params.get("hampel_sigma",  3.0)
                arr, mask = hampel_filter(arr, window=h_win, sigma=h_sigma)
                n_outliers = int(mask.sum())

            # ── Stage 2: Frequency filter or Kalman ──────────────────
            if self.filter_type == "butterworth":
                ftype = params.get("type", "lowpass")
                try:
                    if ftype == "lowpass":
                        cutoff = params.get("cutoff_hz", 1.0)
                        arr = butterworth_lowpass(arr, fs, cutoff,
                                                  params.get("order", 4))
                    elif ftype == "bandpass":
                        arr = butterworth_bandpass(
                            arr, fs,
                            params.get("lowcut_hz",  0.5),
                            params.get("highcut_hz", 8.0),
                            params.get("order", 4),
                        )
                    elif ftype == "highpass":
                        arr = butterworth_highpass(
                            arr, fs, params.get("cutoff_hz", 0.5),
                            params.get("order", 4))
                except Exception as e:
                    self._log(f"  [Filter] WARNING {signal_name}.{col}: "
                              f"Butterworth failed ({e}) — skipping")

            elif self.filter_type == "kalman":
                kp  = KALMAN_DEFAULTS.get(signal_name, {})
                R   = kp.get("observation_noise", 1.0)
                Q   = kp.get("process_noise", 0.1)
                arr = kalman_filter_1d(arr, observation_noise=R,
                                       process_noise=Q)

            df_out[col] = arr
            hampel_stats[col] = n_outliers

        out_str = ", ".join(f"{c}:{v} outliers"
                            for c, v in hampel_stats.items() if v > 0)
        self._log(
            f"  [Filter] {signal_name}: {self.filter_type}"
            f"{' + Hampel' if self.apply_hampel else ''}"
            f"{' | removed: ' + out_str if out_str else ''}"
        )
        return df_out

    def _log(self, msg):
        if self.verbose:
            print(msg)
