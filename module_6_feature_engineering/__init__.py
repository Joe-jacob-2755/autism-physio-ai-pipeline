"""Module 6 — Feature Engineering, Autism Physio-AI Pipeline"""
import sys
from pathlib import Path

# Ensure module directory is on sys.path for sibling imports
_MODULE_DIR = str(Path(__file__).resolve().parent)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from feature_engineer import FeatureEngineer  # noqa: E402, F401

__version__ = "1.0.0"
