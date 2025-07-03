"""
CardioAI Pro Training Platform
Sistema avançado de treinamento de modelos de deep learning para análise de ECG
"""

__version__ = "1.0.0"
__author__ = "CardioAI Pro Team"

from .config.training_config import TrainingConfig
from .models.model_factory import ModelFactory
from .datasets.dataset_factory import DatasetFactory
from .trainers.base_trainer import BaseTrainer

__all__ = [
    "TrainingConfig",
    "ModelFactory", 
    "DatasetFactory",
    "BaseTrainer"
]


