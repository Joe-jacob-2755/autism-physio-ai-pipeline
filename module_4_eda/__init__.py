"""Module 4 — Exploratory Data Analysis (IBM EDA), Autism Physio-AI Pipeline"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from analyser import ExploratoryDataAnalyser, DataAnalyser  # noqa: E402, F401
__version__ = "1.0.0"
