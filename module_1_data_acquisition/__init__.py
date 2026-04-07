"""
Module 2 – Data Acquisition
Autism Physio-AI Pipeline

Public API
----------
from module_1_data_acquisition import DataAcquisitionModule, PipelinePacket
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from acquisition_module import DataAcquisitionModule
from pipeline_packet import (
    PipelinePacket,
    build_packet_from_dataframes,
    SOURCE_IMPORTED, SOURCE_SIMULATED, SOURCE_LIVE, SOURCE_DEPLOYMENT,
)

__version__ = "1.0.0"
__all__ = [
    "DataAcquisitionModule",
    "PipelinePacket",
    "build_packet_from_dataframes",
    "SOURCE_IMPORTED", "SOURCE_SIMULATED", "SOURCE_LIVE", "SOURCE_DEPLOYMENT",
]
