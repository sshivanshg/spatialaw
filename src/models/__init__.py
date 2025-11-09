# Unified model (recommended)
from .spatial_model import (
    SpatialModel, 
    ModelSize, 
    BaselineSpatialModel, 
    MediumSpatialModel, 
    LargeSpatialModel,
    CSIEncoder
)

__all__ = [
    'SpatialModel', 'ModelSize',  # Unified model
    'BaselineSpatialModel', 'MediumSpatialModel', 'LargeSpatialModel',  # Convenience functions
    'CSIEncoder'  # CSI encoder utility
]

