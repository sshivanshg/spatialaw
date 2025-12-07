"""
Utilities for preparing WiAR and related CSI datasets.

Currently exposes helpers for listing and loading CSI recordings.
"""

from .csi_loader import load_csi_file, list_recordings
from .features import (
    extract_csi_features,
    extract_fusion_features,
    extract_rss_features,
    features_to_vector,
)

__all__ = [
    "load_csi_file",
    "list_recordings",
    "extract_csi_features",
    "extract_rss_features",
    "extract_fusion_features",
    "features_to_vector",
]

