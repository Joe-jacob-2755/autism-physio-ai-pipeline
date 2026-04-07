"""
Module 1A – Data Simulation
Autism Physio-AI Pipeline

Public API
----------
from module_1a_data_simulation import DataSimulator, SignalVisualizer, DataExporter
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from simulator  import DataSimulator, SimulationResult
from visualizer import SignalVisualizer
from exporter   import DataExporter
from event_scheduler import (
    EventConfig, EventScheduler,
    make_events_specific, make_events_multiple, make_events_random,
)
from config import EMOTIONS_ALL, EMOTION_PROFILES, SAMPLING_RATES

__all__ = [
    "DataSimulator",
    "SimulationResult",
    "SignalVisualizer",
    "DataExporter",
    "EventConfig",
    "EventScheduler",
    "make_events_specific",
    "make_events_multiple",
    "make_events_random",
    "EMOTIONS_ALL",
    "EMOTION_PROFILES",
    "SAMPLING_RATES",
]

__version__ = "1.0.0"
__author__  = "Autism Physio-AI Pipeline"
