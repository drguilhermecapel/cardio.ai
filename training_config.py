"""
Configuração principal da plataforma de treinamento ECG AI
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import Field
from pydantic_settings import BaseSettings
import torch
from pydantic_settings import BaseSettings
import torch
import os

# Importa configurações do sistema existente
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.core.config import settings as app_settings


class TrainingConfig(BaseSettings):
    """Configuração principal para treinamento de modelos ECG"""
    
    # Caminhos base
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent
    TRAINING_ROOT: Path = PROJECT_ROOT / "training"
    DATA_ROOT: Path = TRAINING_ROOT / "data"
    CHECKPOINT_ROOT: Path = TRAINING_ROOT / "checkpoints"
    LOG_ROOT: Path = TRAINING_ROOT / "logs"
    EXPORT_ROOT: Path = TRAINING_ROOT / "exported_models"
    
    # Configurações de hardware
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = Field(default=4, description="Workers para DataLoader")
    PIN_MEMORY: bool = Field(default=True, description="Pin memory para GPU")
    MIXED_PRECISION: bool = Field(default=True, description="Usar AMP")
    
    # Configurações de treinamento
    BATCH_SIZE: int = Field(default=32, description="Batch size padrão")
    LEARNING_RATE: float = Field(default=1e-4, description="Learning rate inicial")
    EPOCHS: int = Field(default=100, description="Número máximo de épocas")
    EARLY_STOPPING_PATIENCE: int = Field(default=10)
    GRADIENT_CLIP_VAL: float = Field(default=1.0)
    
    # Configurações de validação
    VAL_SPLIT: float = Field(default=0.2, description="Proporção para validação")
    TEST_SPLIT: float = Field(default=0.1, description="Proporção para teste")
    CROSS_VALIDATION_FOLDS: int = Field(default=5)
    
    # Configurações de modelo
    MODEL_TYPE: str = Field(default="heartbeit", description="Tipo de modelo")
    PRETRAINED: bool = Field(default=True, description="Usar pré-treinamento")
    FREEZE_BACKBONE: bool = Field(default=False)
    
    # Configurações de dados
    SAMPLING_RATE: int = Field(default=500, description="Taxa de amostragem Hz")
    SIGNAL_LENGTH: int = Field(default=5000, description="Comprimento do sinal")
    NUM_LEADS: int = Field(default=12, description="Número de derivações")
    
    # Configurações de augmentation
    AUGMENTATION_PROB: float = Field(default=0.5)
    NOISE_LEVEL: float = Field(default=0.01)
    BASELINE_WANDER: bool = Field(default=True)
    
    # Configurações de logging
    LOG_EVERY_N_STEPS: int = Field(default=50)
    SAVE_TOP_K: int = Field(default=3, description="Salvar K melhores modelos")
    TENSORBOARD: bool = Field(default=True)
    WANDB_PROJECT: Optional[str] = Field(default=None)
    
    class Config:
        env_file = ".env"
        env_prefix = "TRAINING_"


# Instância global de configuração
training_config = TrainingConfig()
