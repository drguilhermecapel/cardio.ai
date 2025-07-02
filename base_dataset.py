
"""
Dataset base para todos os datasets de ECG
"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from abc import ABC, abstractmethod

from ..preprocessing.filters import ECGFilters
from ..preprocessing.normalization import ECGNormalizer
from ..preprocessing.augmentation import ECGAugmentation
from ..config.training_config import training_config

logger = logging.getLogger(__name__)


class BaseECGDataset(Dataset, ABC):
    """Classe base abstrata para datasets de ECG"""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        transform: Optional[object] = None,
        target_length: int = 5000,
        sampling_rate: int = 500,
        num_leads: int = 12,
        normalize: bool = True,
        augment: bool = True,
        filter_noise: bool = True,
        cache_data: bool = False
    ):
        """
        Args:
            data_path: Caminho para os dados
            split: 'train', 'val' ou 'test'
            transform: Transformações adicionais
            target_length: Comprimento alvo do sinal
            sampling_rate: Taxa de amostragem alvo
            num_leads: Número de derivações
            normalize: Se deve normalizar os sinais
            augment: Se deve aplicar augmentation (apenas no treino)
            filter_noise: Se deve filtrar ruído
            cache_data: Se deve cachear dados na memória
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.target_length = target_length
        self.sampling_rate = sampling_rate
        self.num_leads = num_leads
        self.normalize = normalize
        self.augment = augment and split == "train"
        self.filter_noise = filter_noise
        self.cache_data = cache_data
        
        # Inicializa preprocessadores
        self.filters = ECGFilters(sampling_rate=sampling_rate)
        self.normalizer = ECGNormalizer()
        self.augmenter = ECGAugmentation() if self.augment else None
        
        # Cache para dados
        self._cache = {} if cache_data else None
        
        # Carrega metadados
        self.samples = []
        self.labels = []
        self.load_metadata()
        
        logger.info(f"Dataset {self.__class__.__name__} inicializado: "
                   f"{len(self.samples)} amostras no split '{split}'")
    
    def load_npy_signal(self, file_path: Union[str, Path]) -> np.ndarray:
        """Carrega um sinal ECG de um arquivo .npy"""
        if os.path.exists(file_path):
            return np.load(file_path)
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    @abstractmethod
    def load_metadata(self):
        """Carrega metadados do dataset (paths, labels, etc)"""
        pass
    
    @abstractmethod
    def load_signal(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """Carrega sinal ECG e metadados"""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retorna amostra processada"""
        # Verifica cache
        if self.cache_data and idx in self._cache:
            return self._cache[idx]
        
        # Carrega sinal
        signal, metadata = self.load_signal(idx)
        
        # Preprocessamento
        signal = self.preprocess_signal(signal, metadata)
        
        # Prepara output
        sample = {
            "signal": torch.FloatTensor(signal),
            "label": torch.LongTensor([self.labels[idx]]),
            "metadata": metadata
        }
        
        # Aplica transformações adicionais
        if self.transform:
            sample = self.transform(sample)
        
        # Cacheia se necessário
        if self.cache_data:
            self._cache[idx] = sample
            
        return sample
    
    def preprocess_signal(self, signal: np.ndarray, metadata: Dict) -> np.ndarray:
        """Pipeline de preprocessamento"""
        # Resample se necessário
        if metadata.get("sampling_rate") != self.sampling_rate:
            signal = self.filters.resample(signal, metadata["sampling_rate"], self.sampling_rate)
            metadata["sampling_rate"] = self.sampling_rate
            
        # Filtra ruído
        if self.filter_noise:
            signal = self.filters.bandpass_filter(signal)
            
        # Normaliza
        if self.normalize:
            signal = self.normalizer.normalize(signal)
            
        # Augmentation (apenas para treino)
        if self.augment and self.augmenter:
            signal = self.augmenter.apply_augmentation(signal)
            
        # Pad/Truncate para target_length
        if signal.shape[-1] < self.target_length:
            padding = self.target_length - signal.shape[-1]
            signal = np.pad(signal, ((0, 0), (0, padding)), 'constant')
        elif signal.shape[-1] > self.target_length:
            signal = signal[..., :self.target_length]
            
        return signal
    
    def _check_data_path(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Caminho de dados não encontrado: {self.data_path}")

    def _get_label_map(self) -> Dict[str, int]:
        """Retorna um mapeamento de labels para inteiros"""
        unique_labels = sorted(list(set(self.labels)))
        return {label: i for i, label in enumerate(unique_labels)}

    def _get_num_classes(self) -> int:
        """Retorna o número de classes únicas"""
        return len(set(self.labels))


