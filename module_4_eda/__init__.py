"""Module 4 — Exploratory Data Analysis, Autism Physio-AI Pipeline (v2.0)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from analyser import CombinedEDA  # noqa: E402, F401
__version__ = "2.0.0"
