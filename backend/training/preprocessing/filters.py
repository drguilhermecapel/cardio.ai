
"""
Funções de filtragem para sinais ECG
"""

import numpy as np
from scipy.signal import butter, lfilter, resample
import logging

logger = logging.getLogger(__name__)


class ECGFilters:
    """Coleção de filtros para sinais ECG"""
    
    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate
        
    def butter_bandpass(self, lowcut, highcut, order=5):
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut=0.5, highcut=40.0, order=5):
        """Aplica um filtro passa-banda ao sinal ECG."""
        b, a = self.butter_bandpass(lowcut, highcut, order=order)
        y = lfilter(b, a, data)
        return y
        
    def resample(self, signal: np.ndarray, original_sampling_rate: int, target_sampling_rate: int) -> np.ndarray:
        """Resample o sinal ECG para uma nova taxa de amostragem."""
        if original_sampling_rate == target_sampling_rate:
            return signal
            
        num_samples = int(signal.shape[-1] * target_sampling_rate / original_sampling_rate)
        
        # Resample cada derivação independentemente
        resampled_signal = np.zeros((signal.shape[0], num_samples))
        for i in range(signal.shape[0]):
            resampled_signal[i, :] = resample(signal[i, :], num_samples)
            
        logger.debug(f"Sinal reamostrado de {original_sampling_rate}Hz para {target_sampling_rate}Hz. "
                     f"De {signal.shape[-1]} para {resampled_signal.shape[-1]} amostras.")
        return resampled_signal
