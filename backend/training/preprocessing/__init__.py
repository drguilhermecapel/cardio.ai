"""
Módulo de pré-processamento para sinais ECG
"""

from .filters import ECGFilters
from .normalization import ECGNormalizer
from .augmentation import ECGAugmentation

__all__ = [
    "ECGFilters",
    "ECGNormalizer",
    "ECGAugmentation"
]


