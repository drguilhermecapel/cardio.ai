
"""
Utilitários para manipulação de dados
"""

from torch.utils.data import Dataset, random_split
from typing import Tuple
import logging
import torch

logger = logging.getLogger(__name__)


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Divide um dataset em conjuntos de treino, validação e teste."""
    if not (train_ratio + val_ratio + test_ratio == 1.0):
        raise ValueError("A soma dos ratios de treino, validação e teste deve ser 1.0")

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # Garante que a soma seja total_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    logger.info(
        f"Dataset dividido: Treino={len(train_dataset)}, Validação={len(val_dataset)}, Teste={len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset


