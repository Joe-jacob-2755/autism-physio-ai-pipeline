"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  user_profiles.py  (v1.2.0)
=============================================================================
Per-user physiological AND demographic parameter generation.

Now includes participant demographics required by Module 3:
  age (5–15), gender, ethnicity, autism severity, verbal status, comorbidity

Autism severity modulates physiological reactivity — higher severity
→ stronger EDA and HR responses — grounded in:
  Schoen et al. (2008) — sensory over-responsivity and arousal in ASD
  Kushki et al. (2013) — ANS and severity in ASD children
=============================================================================
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from config import POPULATION, BASELINE, PARTICIPANT_DEMOGRAPHICS


@dataclass
class UserProfile:
    """
    Full participant profile: physiology + demographics.

    Physiological parameters modulate signal generation.
    Demographics are passed through to output CSVs as features for Module 3.
    """
    # Identity
    user_id:    str
    user_index: int
    user_seed:  int

    # ── Physiological baseline ──────────────────────────────────────────
    eda_tonic_mean: float   # µS
    hr_resting:     float   # bpm
    hrv_sdnn:       float   # ms
    st_baseline:    float   # °C
    bvp_amplitude:  float   # nT

    # ── Reactivity factors ──────────────────────────────────────────────
    eda_reactivity:  float   # multiplier on EDA event magnitude
    hr_reactivity:   float   # multiplier on HR delta
    movement_scale:  float   # multiplier on ACC activity
    noise_contact:   float   # EDA electrode quality

    # ── Demographics ────────────────────────────────────────────────────
    age:             int     # years (5–15)
    gender:          str     # Male / Female / Non-binary
    ethnicity:       str
    autism_severity: str     # Low / Medium / Severe
    verbal_status:   str     # Verbal / Minimally verbal / Non-verbal
    comorbidity:     str     # Yes / No

    # ── Derived ─────────────────────────────────────────────────────────
    @property
    def reactivity_class(self) -> str:
        if self.eda_reactivity < 0.60:
            return "hypo-reactive"
        elif self.eda_reactivity > 1.60:
            return "hyper-reactive"
        return "typical"

    def summary(self) -> str:
        return (
            f"  {self.user_id}  |  Age={self.age}  Gender={self.gender}  "
            f"Severity={self.autism_severity}  Verbal={self.verbal_status}  "
            f"Comorbidity={self.comorbidity}  |  "
            f"EDA={self.eda_tonic_mean:.2f}µS  HR={self.hr_resting:.0f}bpm  "
            f"HRV={self.hrv_sdnn:.0f}ms  ST={self.st_baseline:.1f}°C  |  "
            f"EDA-react={self.eda_reactivity:.2f}x  [{self.reactivity_class}]"
        )

    def to_dict(self) -> dict:
        return {
            "user_id":         self.user_id,
            "user_index":      self.user_index,
            "user_seed":       self.user_seed,
            # Physiology
            "eda_tonic_mean":  round(self.eda_tonic_mean,  4),
            "hr_resting":      round(self.hr_resting,      2),
            "hrv_sdnn":        round(self.hrv_sdnn,        2),
            "st_baseline":     round(self.st_baseline,     3),
            "bvp_amplitude":   round(self.bvp_amplitude,   2),
            "eda_reactivity":  round(self.eda_reactivity,  4),
            "hr_reactivity":   round(self.hr_reactivity,   4),
            "movement_scale":  round(self.movement_scale,  4),
            "noise_contact":   round(self.noise_contact,   4),
            "reactivity_class": self.reactivity_class,
            # Demographics
            "age":             self.age,
            "gender":          self.gender,
            "ethnicity":       self.ethnicity,
            "autism_severity": self.autism_severity,
            "verbal_status":   self.verbal_status,
            "comorbidity":     self.comorbidity,
        }

    def demographic_dict(self) -> dict:
        """Return only demographic fields — for Module 3 feature fusion."""
        return {
            "age":             self.age,
            "gender":          self.gender,
            "ethnicity":       self.ethnicity,
            "autism_severity": self.autism_severity,
            "verbal_status":   self.verbal_status,
            "comorbidity":     self.comorbidity,
        }


class UserProfileGenerator:
    """
    Generate a cohort of UserProfile objects with physiological AND
    demographic variability from paediatric population distributions.

    Parameters
    ----------
    n_users      : number of users to generate
    master_seed  : controls all random draws reproducibly
    """

    def __init__(self, n_users: int = 1, master_seed: int = 42):
        self.n_users     = n_users
        self.master_seed = master_seed
        self._rng        = np.random.default_rng(master_seed)

    def generate(self) -> List[UserProfile]:
        profiles = []
        p   = POPULATION
        dem = PARTICIPANT_DEMOGRAPHICS

        for i in range(1, self.n_users + 1):
            user_seed = int(self._rng.integers(1, 999_999))

            # ── Demographics ────────────────────────────────────────────
            age = int(self._rng.integers(
                dem["age"]["min"], dem["age"]["max"] + 1
            ))
            gender = self._rng.choice(
                dem["gender"]["options"],
                p=dem["gender"]["probabilities"]
            )
            ethnicity = self._rng.choice(
                dem["ethnicity"]["options"],
                p=dem["ethnicity"]["probabilities"]
            )
            autism_severity = self._rng.choice(
                dem["autism_severity"]["options"],
                p=dem["autism_severity"]["probabilities"]
            )
            verbal_status = self._rng.choice(
                dem["verbal_status"]["options"],
                p=dem["verbal_status"]["probabilities"]
            )
            comorbidity = self._rng.choice(
                dem["comorbidity"]["options"],
                p=dem["comorbidity"]["probabilities"]
            )

            # Severity modifier — higher severity → higher EDA reactivity
            sev_mod = dem["autism_severity"]["reactivity_modifiers"][autism_severity]

            # ── Physiological baseline ──────────────────────────────────
            # Age effect: younger children have higher HR, slightly lower HRV
            age_hr_offset  = max(0, (10 - age) * 1.2)   # +12 bpm for 5 yr
            age_hrv_offset = max(0, (age - 5)  * 1.0)   # +10 ms for 15 yr

            eda_tonic = self._clipped_normal(
                p["EDA"]["tonic_mean_mu"],
                p["EDA"]["tonic_mean_sigma"],
                p["EDA"]["tonic_mean_min"],
                p["EDA"]["tonic_mean_max"],
            )
            hr_resting = self._clipped_normal(
                p["BVP"]["hr_bpm_mu"]    + age_hr_offset,
                p["BVP"]["hr_bpm_sigma"],
                p["BVP"]["hr_bpm_min"],
                p["BVP"]["hr_bpm_max"],
            )
            hrv_sdnn = self._clipped_normal(
                p["IBI"]["hrv_sdnn_mu"]  + age_hrv_offset,
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
                40.0, 200.0,
            )

            # ── Reactivity — modulated by autism severity ───────────────
            eda_reactivity = self._clipped_normal(
                p["EDA"]["reactivity_mu"]  * sev_mod,
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
                age            = age,
                gender         = gender,
                ethnicity      = ethnicity,
                autism_severity= autism_severity,
                verbal_status  = verbal_status,
                comorbidity    = comorbidity,
            ))

        return profiles

    def _clipped_normal(self, mu, sigma, lo, hi) -> float:
        return float(np.clip(self._rng.normal(mu, sigma), lo, hi))
