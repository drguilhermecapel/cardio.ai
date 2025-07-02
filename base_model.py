
"""
Classe base para todos os modelos de ECG
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModel(nn.Module, ABC):
    """Classe base abstrata para modelos de deep learning de ECG."""
    
    def __init__(self, num_classes: int, input_channels: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass do modelo."""
        raise NotImplementedError("O método forward deve ser implementado pela subclasse.")
        
    def load_pretrained(self, path: str):
        """Carrega pesos pré-treinados no modelo."""
        try:
            self.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Pesos pré-treinados carregados de {path}")
        except Exception as e:
            print(f"Erro ao carregar pesos pré-treinados de {path}: {e}")
            
    @property
    def device(self) -> torch.device:
        """Retorna o dispositivo atual do modelo."""
        return next(self.parameters()).device


