"""
Module 3 — Data Splitting

User-level, stratified train/val/test splitting for physiological
signal pipelines.  Runs IMMEDIATELY after data acquisition (M1 + M2A)
to split raw signals BEFORE any preprocessing or feature engineering.
Test data is kept untouched until final model evaluation.
"""
import sys
from pathlib import Path

_MODULE_DIR = str(Path(__file__).resolve().parent)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from splitter import DataSplitter, SplitResult   # noqa: E402, F401
from exporter import SplitExporter               # noqa: E402, F401
