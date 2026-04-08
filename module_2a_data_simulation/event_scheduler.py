"""
=============================================================================
MODULE 1A – DATA SIMULATION  |  event_scheduler.py
=============================================================================
Event scheduling engine.

Responsibilities:
  • Accept user-defined scheduling parameters (count, duration, emotion mode)
  • Generate a validated list of non-overlapping EventConfig objects
  • Support three emotion modes: specific, multiple (list), random

=============================================================================
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Union

from config import (
    EMOTIONS_ALL, EMOTION_PROFILES,
    DEFAULT_EVENT_DUR_S,
)


# ─────────────────────────────────────────────────────────────────────────────
# EVENT DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EventConfig:
    """
    Represents a single labelled physiological event window.

    Attributes
    ----------
    emotion      : emotion / behaviour label (must be in EMOTIONS_ALL)
    start_s      : start time in seconds from recording onset
    duration_s   : event duration in seconds
    """
    emotion: str
    start_s: float
    duration_s: float

    # ── Convenience property accessors (pulled from EMOTION_PROFILES) ─────
    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    @property
    def profile(self) -> dict:
        return EMOTION_PROFILES[self.emotion]

    @property
    def color(self) -> str:
        return EMOTION_PROFILES[self.emotion]["color"]

    # BVP helpers
    @property
    def bvp_hr_delta(self) -> float:
        return EMOTION_PROFILES[self.emotion]["BVP"]["hr_delta_bpm"]

    @property
    def bvp_hr_rise_s(self) -> float:
        return EMOTION_PROFILES[self.emotion]["EDA"]["rise_time_s"]

    def __repr__(self) -> str:
        return (
            f"Event('{self.emotion}', "
            f"t={self.start_s:.1f}–{self.end_s:.1f}s, "
            f"dur={self.duration_s:.1f}s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class EventScheduler:
    """
    Generate a schedule of emotion events for a simulation run.

    Parameters
    ----------
    duration_s        : total recording duration (s)
    n_events          : number of events ('random' → sampled from [1, max_events])
    event_duration_s  : per-event duration in s ('random' → sampled uniformly)
    emotions          : one of:
                          str  → single emotion label ("Anger")
                          list → cycle or random from this subset
                          None → random from EMOTIONS_ALL
    rng               : numpy random Generator for reproducibility
    min_gap_s         : minimum quiet gap between consecutive events (s)
    min_lead_s        : quiet period at recording start before first event (s)
    max_events        : upper bound when n_events='random'
    event_dur_min_s   : lower bound for random event duration
    event_dur_max_s   : upper bound for random event duration
    """

    def __init__(
        self,
        duration_s: float,
        n_events: Union[int, str] = 5,
        event_duration_s: Union[float, str] = DEFAULT_EVENT_DUR_S,
        emotions: Optional[Union[str, list]] = None,
        rng: Optional[np.random.Generator] = None,
        min_gap_s: float = 15.0,
        min_lead_s: float = 10.0,
        max_events: int = 10,
        event_dur_min_s: float = 10.0,
        event_dur_max_s: float = 60.0,
    ):
        self.duration_s = float(duration_s)
        self.n_events_param = n_events
        self.event_dur_param = event_duration_s
        self.emotions_param = emotions
        self.rng = rng if rng is not None else np.random.default_rng()
        self.min_gap_s = float(min_gap_s)
        self.min_lead_s = float(min_lead_s)
        self.max_events = int(max_events)
        self.event_dur_min_s = float(event_dur_min_s)
        self.event_dur_max_s = float(event_dur_max_s)

    # ── Public API ──────────────────────────────────────────────────────────

    def schedule(self) -> List[EventConfig]:
        """
        Generate and return the list of EventConfig objects.

        Returns empty list if events do not fit within the recording.
        """
        n_events = self._resolve_n_events()
        emotion_seq = self._resolve_emotions(n_events)
        durations = self._resolve_durations(n_events)
        start_times = self._place_events(n_events, durations)

        events = []
        for i in range(len(start_times)):
            events.append(EventConfig(
                emotion=emotion_seq[i],
                start_s=start_times[i],
                duration_s=durations[i],
            ))

        events.sort(key=lambda e: e.start_s)
        return events

    # ── Private helpers ─────────────────────────────────────────────────────

    def _resolve_n_events(self) -> int:
        if self.n_events_param == "random":
            return int(self.rng.integers(1, self.max_events + 1))
        n = int(self.n_events_param)
        if n < 1:
            raise ValueError("n_events must be ≥ 1.")
        return n

    def _resolve_emotions(self, n: int) -> List[str]:
        """Resolve emotion label(s) for each of the n events."""
        if self.emotions_param is None:
            # Fully random – sample without immediate repeat
            pool = EMOTIONS_ALL.copy()
            result = []
            last = None
            for _ in range(n):
                choices = [e for e in pool if e != last] or pool
                chosen = self.rng.choice(choices)
                result.append(chosen)
                last = chosen
            return result

        if isinstance(self.emotions_param, str):
            # Single emotion → repeat for all events
            if self.emotions_param not in EMOTIONS_ALL:
                raise ValueError(
                    f"Unknown emotion '{self.emotions_param}'. "
                    f"Valid: {EMOTIONS_ALL}"
                )
            return [self.emotions_param] * n

        if isinstance(self.emotions_param, list):
            for e in self.emotions_param:
                if e not in EMOTIONS_ALL:
                    raise ValueError(f"Unknown emotion '{e}' in list.")
            # Randomly sample from the provided subset
            indices = self.rng.integers(0, len(self.emotions_param), size=n)
            return [self.emotions_param[i] for i in indices]

        raise TypeError(
            "emotions must be None, a str, or a list of str. "
            f"Got: {type(self.emotions_param)}"
        )

    def _resolve_durations(self, n: int) -> List[float]:
        """Return per-event duration list."""
        if self.event_dur_param == "random":
            return list(
                self.rng.uniform(self.event_dur_min_s, self.event_dur_max_s, size=n)
            )
        d = float(self.event_dur_param)
        if d < 5.0:
            raise ValueError("Event duration must be ≥ 5 s.")
        return [d] * n

    def _place_events(
        self, n: int, durations: List[float]
    ) -> List[float]:
        """
        Place n events sequentially with guaranteed gaps.

        Uses a greedy placement strategy:
          available_window → lead gap → event → minimum gap → event → …

        Silently truncates the event list if they cannot all fit.
        """
        starts: List[float] = []
        cursor = self.min_lead_s

        for i in range(n):
            dur = durations[i]
            end = cursor + dur
            tail = self.min_gap_s if i < n - 1 else 5.0

            if end + tail > self.duration_s:
                print(
                    f"[EventScheduler] WARNING: Only {len(starts)}/{n} events "
                    f"fit within the {self.duration_s:.0f}s recording. "
                    f"Increase duration or reduce n_events."
                )
                break

            starts.append(cursor)
            cursor = end + self.min_gap_s

        return starts


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FACTORY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def make_events_specific(
    emotion: str,
    n_events: int,
    duration_s: float,
    total_dur_s: float,
    rng: Optional[np.random.Generator] = None,
) -> List[EventConfig]:
    """Shorthand: schedule n identical single-emotion events."""
    return EventScheduler(
        duration_s=total_dur_s,
        n_events=n_events,
        event_duration_s=duration_s,
        emotions=emotion,
        rng=rng,
    ).schedule()


def make_events_multiple(
    emotions: List[str],
    n_events: int,
    duration_s: float,
    total_dur_s: float,
    rng: Optional[np.random.Generator] = None,
) -> List[EventConfig]:
    """Shorthand: schedule n events drawn from an emotion subset."""
    return EventScheduler(
        duration_s=total_dur_s,
        n_events=n_events,
        event_duration_s=duration_s,
        emotions=emotions,
        rng=rng,
    ).schedule()


def make_events_random(
    total_dur_s: float,
    max_events: int = 8,
    event_dur_min_s: float = 15.0,
    event_dur_max_s: float = 45.0,
    rng: Optional[np.random.Generator] = None,
) -> List[EventConfig]:
    """Shorthand: fully random schedule (count, duration, emotion)."""
    return EventScheduler(
        duration_s=total_dur_s,
        n_events="random",
        event_duration_s="random",
        emotions=None,
        max_events=max_events,
        event_dur_min_s=event_dur_min_s,
        event_dur_max_s=event_dur_max_s,
        rng=rng,
    ).schedule()
