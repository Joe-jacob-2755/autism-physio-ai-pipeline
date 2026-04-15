"""
Module 2B -- Data Annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Manual annotation of physiological signals with emotion/behaviour labels.

Public API
----------
ManualAnnotator : Core annotation engine
run_annotation  : Pipeline-callable entry point
"""

from annotator import ManualAnnotator
from main import run_annotation

__all__ = ["ManualAnnotator", "run_annotation"]
