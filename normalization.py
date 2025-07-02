
"""
Funções de normalização para sinais ECG
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class ECGNormalizer:
    """Coleção de métodos de normalização para sinais ECG"""
    
    def __init__(self):
        pass
        
    def normalize(self, signal: np.ndarray, method: str = "z_score") -> np.ndarray:
        """Normaliza o sinal ECG usando o método especificado."""
        if method == "z_score":
            return self._z_score_normalize(signal)
        elif method == "min_max":
            return self._min_max_normalize(signal)
        else:
            raise ValueError(f"Método de normalização {method} não suportado.")
            
    def _z_score_normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalização Z-score (média 0, desvio padrão 1)."""
        mean = np.mean(signal, axis=-1, keepdims=True)
        std = np.std(signal, axis=-1, keepdims=True)
        # Evita divisão por zero
        std[std == 0] = 1.0
        normalized_signal = (signal - mean) / std
        logger.debug("Sinal normalizado com Z-score.")
        return normalized_signal
        
    def _min_max_normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalização Min-Max (escala para [0, 1] ou [-1, 1])."""
        min_val = np.min(signal, axis=-1, keepdims=True)
        max_val = np.max(signal, axis=-1, keepdims=True)
        
        # Evita divisão por zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0
        
        normalized_signal = (signal - min_val) / range_val
        logger.debug("Sinal normalizado com Min-Max.")
        return normalized_signal


