
"""
Funções de augmentation para sinais ECG
"""

import numpy as np
import logging
from ..config.training_config import training_config

logger = logging.getLogger(__name__)


class ECGAugmentation:
    """Coleção de métodos de augmentation para sinais ECG"""
    
    def __init__(self):
        self.augmentation_prob = training_config.AUGMENTATION_PROB
        self.noise_level = training_config.NOISE_LEVEL
        self.baseline_wander = training_config.BASELINE_WANDER
        
    def apply_augmentation(self, signal: np.ndarray) -> np.ndarray:
        """Aplica augmentations ao sinal ECG com uma dada probabilidade."""
        if np.random.rand() < self.augmentation_prob:
            if self.noise_level > 0:
                signal = self._add_noise(signal, self.noise_level)
            if self.baseline_wander:
                signal = self._add_baseline_wander(signal)
            # Adicionar outras augmentations aqui (e.g., scaling, time warping)
            logger.debug("Augmentation aplicada ao sinal.")
        return signal
        
    def _add_noise(self, signal: np.ndarray, noise_level: float) -> np.ndarray:
        """Adiciona ruído gaussiano ao sinal."""
        noise = np.random.normal(0, noise_level, signal.shape)
        return signal + noise
        
    def _add_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Adiciona desvio da linha de base simulado ao sinal."""
        # Simula um desvio de linha de base lento e aleatório
        baseline = np.cumsum(np.random.normal(0, 0.001, signal.shape), axis=-1)
        return signal + baseline


