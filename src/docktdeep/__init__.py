"""
docktdeep: A deep learning model for protein-ligand binding affinity prediction.
"""

__version__ = "0.1.0"

# Import modules to make them available at package level
from . import models

# Import main functions
try:
    from .dataset import VoxelDataset
    from .inference import predict_binding_affinity

    __all__ = ["predict_binding_affinity", "VoxelDataset", "models"]
except ImportError:
    # Graceful fallback if optional dependencies are missing
    __all__ = ["models"]
