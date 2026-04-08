"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  noise_injector.py
=============================================================================
Realistic noise injection for all physiological signal channels.

Noise types modelled:
  EDA  – Gaussian sensor noise, slow electrode drift, powerline interference
  BVP  – Gaussian noise, motion artifact, baseline wander
  ST   – Gaussian thermal noise
  ACC  – Gaussian noise (all axes), correlated cross-axis spill

Noise levels: 'low', 'medium', 'high'  (see config.NOISE_PROFILES)
=============================================================================
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional
from scipy import signal as sp_signal

from config import NOISE_PROFILES, POWERLINE_FREQ_HZ, SAMPLING_RATES


class NoiseInjector:
    """
    Add realistic noise to a dictionary of simulated physiological signals.

    Parameters
    ----------
    level  : 'low' | 'medium' | 'high'
    seed   : optional random seed for reproducibility
    """

    def __init__(self, level: str = "medium", seed: Optional[int] = None):
        if level not in NOISE_PROFILES:
            raise ValueError(f"level must be one of {list(NOISE_PROFILES)}.")
        self.level = level
        self.profile = NOISE_PROFILES[level]
        self.rng = np.random.default_rng(seed)

    # ── Master entry point ──────────────────────────────────────────────────

    def inject(self, signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Add noise to every signal in the dictionary in-place.

        Parameters
        ----------
        signals : dict keyed by signal name (e.g. 'EDA', 'BVP', 'ACC_X', …)

        Returns
        -------
        Same dict with noise added (also modified in-place for efficiency).
        """
        noisy = {}

        # ── EDA ──────────────────────────────────────────────────────────
        if "EDA" in signals:
            noisy["EDA"] = self._noise_eda(signals["EDA"])

        # ── BVP ──────────────────────────────────────────────────────────
        if "BVP" in signals:
            noisy["BVP"] = self._noise_bvp(signals["BVP"])

        # ── ST ───────────────────────────────────────────────────────────
        if "ST" in signals:
            noisy["ST"] = self._noise_st(signals["ST"])

        # ── ACC (all three axes, correlated) ─────────────────────────────
        if all(k in signals for k in ("ACC_X", "ACC_Y", "ACC_Z")):
            nx, ny, nz = self._noise_acc(
                signals["ACC_X"], signals["ACC_Y"], signals["ACC_Z"]
            )
            noisy["ACC_X"], noisy["ACC_Y"], noisy["ACC_Z"] = nx, ny, nz
        else:
            for ax in ("ACC_X", "ACC_Y", "ACC_Z"):
                if ax in signals:
                    noisy[ax] = signals[ax] + self.rng.normal(
                        0, self.profile["ACC"]["gaussian_std"], len(signals[ax])
                    )

        # Pass IBI through unchanged (timing noise handled in generator)
        if "IBI_TIMES" in signals:
            noisy["IBI_TIMES"] = signals["IBI_TIMES"]
        if "IBI_VALUES" in signals:
            noisy["IBI_VALUES"] = self._noise_ibi(signals["IBI_VALUES"])

        return noisy

    # ── EDA noise ────────────────────────────────────────────────────────────

    def _noise_eda(self, sig: np.ndarray) -> np.ndarray:
        n = len(sig)
        fs = SAMPLING_RATES["EDA"]
        p = self.profile["EDA"]
        result = sig.copy().astype(float)

        # Gaussian sensor noise
        result += self.rng.normal(0, p["gaussian_std"], n)

        # Slow electrode drift (low-frequency random walk, bandlimited)
        if p["drift_amp"] > 0:
            drift_raw = np.cumsum(self.rng.normal(0, 0.02, n))
            drift_raw -= drift_raw.mean()
            # Low-pass at 0.05 Hz to keep it ultra-slow
            b, a = sp_signal.butter(2, 0.05 / (fs / 2), btype="low")
            drift = sp_signal.filtfilt(b, a, drift_raw)
            drift *= p["drift_amp"] / (np.std(drift) + 1e-9)
            result += drift

        # Power-line interference (50 Hz / 60 Hz) – typically very weak on EDA
        if p.get("powerline_amp", 0) > 0:
            t = np.arange(n) / fs
            result += p["powerline_amp"] * np.sin(
                2 * np.pi * POWERLINE_FREQ_HZ * t
                + self.rng.uniform(0, 2 * np.pi)
            )

        return result

    # ── BVP noise ────────────────────────────────────────────────────────────

    def _noise_bvp(self, sig: np.ndarray) -> np.ndarray:
        n = len(sig)
        fs = SAMPLING_RATES["BVP"]
        p = self.profile["BVP"]
        result = sig.copy().astype(float)

        # Gaussian noise
        result += self.rng.normal(0, p["gaussian_std"], n)

        # Power-line interference (clearly visible in poorly-shielded devices)
        if p.get("powerline_amp", 0) > 0:
            t = np.arange(n) / fs
            result += p["powerline_amp"] * np.sin(
                2 * np.pi * POWERLINE_FREQ_HZ * t
                + self.rng.uniform(0, 2 * np.pi)
            )

        # Motion artifact: short burst of broadband noise
        if p.get("motion_amp", 0) > 0:
            n_bursts = max(1, int(n / (fs * 30)))   # ~1 burst per 30 s
            for _ in range(n_bursts):
                burst_start = int(self.rng.integers(0, max(1, n - fs * 3)))
                burst_len = int(self.rng.uniform(0.5, 3.0) * fs)
                burst_end = min(burst_start + burst_len, n)
                artifact = p["motion_amp"] * self.rng.standard_normal(
                    burst_end - burst_start
                )
                # Taper edges
                taper = np.hanning(burst_end - burst_start)
                result[burst_start:burst_end] += artifact * taper

        # Baseline wander (very slow drift from electrode movement)
        wander_raw = np.cumsum(self.rng.normal(0, 0.3, n))
        wander_raw -= wander_raw.mean()
        b, a = sp_signal.butter(2, 0.5 / (fs / 2), btype="low")
        wander = sp_signal.filtfilt(b, a, wander_raw)
        wander *= (p["gaussian_std"] * 0.5) / (np.std(wander) + 1e-9)
        result += wander

        return result

    # ── ST noise ─────────────────────────────────────────────────────────────

    def _noise_st(self, sig: np.ndarray) -> np.ndarray:
        p = self.profile["ST"]
        return sig.copy() + self.rng.normal(0, p["gaussian_std"], len(sig))

    # ── ACC noise ────────────────────────────────────────────────────────────

    def _noise_acc(
        self,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
    ):
        """
        Add Gaussian noise with slight correlated component across axes
        (simulates common-mode vibration artefact).
        """
        p = self.profile["ACC"]
        std = p["gaussian_std"]
        n = len(ax)

        # Shared vibration component (10 % weight)
        common = self.rng.normal(0, std * 0.10, n)

        nx = ax.copy() + self.rng.normal(0, std * 0.90, n) + common
        ny = ay.copy() + self.rng.normal(0, std * 0.90, n) + common
        nz = az.copy() + self.rng.normal(0, std * 0.90, n) + common

        return nx, ny, nz

    # ── IBI noise ────────────────────────────────────────────────────────────

    def _noise_ibi(self, ibi_vals: np.ndarray) -> np.ndarray:
        """
        Add tiny measurement uncertainty to IBI values (±1–3 ms quantisation).
        """
        quant_ms = {"low": 0.5, "medium": 1.5, "high": 3.0}[self.level]
        return ibi_vals + self.rng.normal(0, quant_ms, len(ibi_vals))
