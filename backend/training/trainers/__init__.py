"""
Módulo de treinadores para modelos ECG
"""

from .base_trainer import BaseTrainer
from .classification_trainer import ClassificationTrainer

__all__ = [
    "BaseTrainer",
    "ClassificationTrainer"
]


