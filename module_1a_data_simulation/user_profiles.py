"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  user_profiles.py
=============================================================================
Per-user physiological parameter generation.

Each simulated user is assigned a unique set of baseline physiological
parameters drawn from a paediatric population distribution. This creates
realistic inter-subject variability and supports both:
  - User-dependent models (trained on a single user's data)
  - Global models (trained across all users)

UserProfile contains two types of parameters:
  1. Baseline offsets    — resting levels (EDA tonic, HR, ST, etc.)
  2. Reactivity factors  — how strongly the user responds to emotion events

The population distributions are grounded in:
  - Paediatric autonomic physiology (children age 5-18)
  - Autism-specific literature on EDA and HRV variability
  - Empatica E4 normative data

=============================================================================
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from config import POPULATION, BASELINE


# ─────────────────────────────────────────────────────────────────────────────
# USER PROFILE DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    """
    Physiological parameter profile for a single simulated participant.

    All values are drawn from population distributions defined in
    config.POPULATION. They modulate the baseline signal generators and
    the event-response magnitudes in DataSimulator.

    Attributes
    ----------
    user_id         : zero-padded string identifier, e.g. 'user_003'
    user_index      : 1-based integer index
    user_seed       : individual random seed (derived from master seed)

    --- Baseline parameters ---
    eda_tonic_mean  : resting skin conductance level (µS)
    hr_resting      : resting heart rate (bpm)
    hrv_sdnn        : heart rate variability SDNN (ms)
    st_baseline     : resting wrist skin temperature (°C)
    bvp_amplitude   : BVP peak amplitude scaling factor

    --- Reactivity factors (multipliers on event response magnitude) ---
    eda_reactivity  : EDA response multiplier  (1.0 = population average)
    hr_reactivity   : HR delta multiplier       (1.0 = population average)
    movement_scale  : ACC activity multiplier   (1.0 = population average)
    noise_contact   : EDA electrode contact quality (1.0 = perfect)

    --- Derived display fields ---
    reactivity_class : 'hypo-reactive' | 'typical' | 'hyper-reactive'
    """

    user_id:          str
    user_index:       int
    user_seed:        int

    # Baseline levels
    eda_tonic_mean:   float
    hr_resting:       float
    hrv_sdnn:         float
    st_baseline:      float
    bvp_amplitude:    float

    # Reactivity multipliers
    eda_reactivity:   float
    hr_reactivity:    float
    movement_scale:   float
    noise_contact:    float

    @property
    def reactivity_class(self) -> str:
        """Classify EDA reactivity for reporting."""
        if self.eda_reactivity < 0.60:
            return "hypo-reactive"
        elif self.eda_reactivity > 1.60:
            return "hyper-reactive"
        return "typical"

    def summary(self) -> str:
        return (
            f"  {self.user_id}  |  "
            f"EDA={self.eda_tonic_mean:.2f}µS  "
            f"HR={self.hr_resting:.0f}bpm  "
            f"HRV={self.hrv_sdnn:.0f}ms  "
            f"ST={self.st_baseline:.1f}°C  |  "
            f"EDA-react={self.eda_reactivity:.2f}x  "
            f"HR-react={self.hr_reactivity:.2f}x  "
            f"Move={self.movement_scale:.2f}x  "
            f"[{self.reactivity_class}]"
        )

    def to_dict(self) -> dict:
        return {
            "user_id":          self.user_id,
            "user_index":       self.user_index,
            "user_seed":        self.user_seed,
            "eda_tonic_mean":   round(self.eda_tonic_mean,  4),
            "hr_resting":       round(self.hr_resting,      2),
            "hrv_sdnn":         round(self.hrv_sdnn,        2),
            "st_baseline":      round(self.st_baseline,     3),
            "bvp_amplitude":    round(self.bvp_amplitude,   2),
            "eda_reactivity":   round(self.eda_reactivity,  4),
            "hr_reactivity":    round(self.hr_reactivity,   4),
            "movement_scale":   round(self.movement_scale,  4),
            "noise_contact":    round(self.noise_contact,   4),
            "reactivity_class": self.reactivity_class,
        }


# ─────────────────────────────────────────────────────────────────────────────
# USER PROFILE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class UserProfileGenerator:
    """
    Generate a cohort of UserProfile objects with realistic inter-subject
    variability drawn from paediatric population distributions.

    Parameters
    ----------
    n_users      : number of users to generate
    master_seed  : controls all random draws — same seed = same cohort
    """

    def __init__(self, n_users: int = 1, master_seed: int = 42):
        self.n_users     = n_users
        self.master_seed = master_seed
        self._rng        = np.random.default_rng(master_seed)

    def generate(self) -> List[UserProfile]:
        """
        Generate and return a list of n_users UserProfile objects.

        Each user gets a unique user_seed derived from the master_seed,
        so individual users are reproducible and adding more users does
        not change the profiles of existing users.
        """
        profiles = []
        p        = POPULATION

        for i in range(1, self.n_users + 1):
            # Each user gets a reproducible child seed
            user_seed = int(self._rng.integers(1, 999_999))

            # ── Draw baseline physiological parameters ────────────────────
            eda_tonic = self._clipped_normal(
                p["EDA"]["tonic_mean_mu"],
                p["EDA"]["tonic_mean_sigma"],
                p["EDA"]["tonic_mean_min"],
                p["EDA"]["tonic_mean_max"],
            )

            hr_resting = self._clipped_normal(
                p["BVP"]["hr_bpm_mu"],
                p["BVP"]["hr_bpm_sigma"],
                p["BVP"]["hr_bpm_min"],
                p["BVP"]["hr_bpm_max"],
            )

            hrv_sdnn = self._clipped_normal(
                p["IBI"]["hrv_sdnn_mu"],
                p["IBI"]["hrv_sdnn_sigma"],
                p["IBI"]["hrv_sdnn_min"],
                p["IBI"]["hrv_sdnn_max"],
            )

            st_baseline = self._clipped_normal(
                p["ST"]["mean_celsius_mu"],
                p["ST"]["mean_celsius_sigma"],
                p["ST"]["mean_celsius_min"],
                p["ST"]["mean_celsius_max"],
            )

            bvp_amplitude = self._clipped_normal(
                p["BVP"]["amplitude_mu"],
                p["BVP"]["amplitude_sigma"],
                40.0,
                200.0,
            )

            # ── Draw reactivity factors ───────────────────────────────────
            eda_reactivity = self._clipped_normal(
                p["EDA"]["reactivity_mu"],
                p["EDA"]["reactivity_sigma"],
                p["EDA"]["reactivity_min"],
                p["EDA"]["reactivity_max"],
            )

            hr_reactivity = self._clipped_normal(
                p["BVP"]["reactivity_mu"],
                p["BVP"]["reactivity_sigma"],
                p["BVP"]["reactivity_min"],
                p["BVP"]["reactivity_max"],
            )

            movement_scale = self._clipped_normal(
                p["ACC"]["movement_scale_mu"],
                p["ACC"]["movement_scale_sigma"],
                p["ACC"]["movement_scale_min"],
                p["ACC"]["movement_scale_max"],
            )

            noise_contact = self._clipped_normal(
                p["NOISE"]["eda_contact_mu"],
                p["NOISE"]["eda_contact_sigma"],
                p["NOISE"]["eda_contact_min"],
                p["NOISE"]["eda_contact_max"],
            )

            profiles.append(UserProfile(
                user_id        = f"user_{i:03d}",
                user_index     = i,
                user_seed      = user_seed,
                eda_tonic_mean = eda_tonic,
                hr_resting     = hr_resting,
                hrv_sdnn       = hrv_sdnn,
                st_baseline    = st_baseline,
                bvp_amplitude  = bvp_amplitude,
                eda_reactivity = eda_reactivity,
                hr_reactivity  = hr_reactivity,
                movement_scale = movement_scale,
                noise_contact  = noise_contact,
            ))

        return profiles

    def _clipped_normal(self, mu: float, sigma: float,
                        lo: float, hi: float) -> float:
        """Draw from N(mu, sigma) clipped to [lo, hi]."""
        val = self._rng.normal(mu, sigma)
        return float(np.clip(val, lo, hi))
