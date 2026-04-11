"""Module 5B — Data Augmentation (Optional), Autism Physio-AI Pipeline"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from augmenter import DataAugmenter, AugmentationResult  # noqa: E402, F401
from imbalance_detector import ImbalanceDetector, ImbalanceReport  # noqa: E402, F401
__version__ = "1.0.0"
