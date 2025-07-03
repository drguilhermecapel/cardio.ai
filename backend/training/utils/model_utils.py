
"""
Utilitários para manipulação de modelos
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """Conta o número de parâmetros treináveis em um modelo."""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Número de parâmetros treináveis: {num_params}")
    return num_params


def get_device() -> torch.device:
    """Retorna o dispositivo disponível (GPU se houver, senão CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")
    return device


