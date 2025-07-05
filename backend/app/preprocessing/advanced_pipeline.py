# backend/app/preprocessing/advanced_pipeline.py
import numpy as np
import logging
from scipy.signal import butter, filtfilt, medfilt

from app.services.interfaces import PreprocessingPipelineInterface

logger = logging.getLogger(__name__)

class AdvancedPipeline(PreprocessingPipelineInterface):
    """
    Pipeline de pré-processamento avançado, otimizado para o modelo PTB-XL.
    Inclui filtragem de ruído, remoção de desvio da linha de base e normalização.
    """
    def __init__(self, target_fs: int = 500, target_len: int = 1000):
        self.target_fs = target_fs
        self.target_len = target_len # O modelo espera 1000 amostras (2 segundos a 500Hz)
        logger.info("AdvancedPipeline inicializado.")

    def _bandpass_filter(self, data, lowcut=0.5, highcut=49.0):
        """Aplica um filtro passa-faixa para remover ruído de baixa e alta frequência."""
        nyquist = 0.5 * self.target_fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(2, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)

    def _baseline_wander_removal(self, data):
        """Remove o desvio da linha de base usando um filtro de mediana."""
        # Filtro de mediana com uma janela grande para capturar a linha de base
        win_size = int(self.target_fs * 0.6) # Janela de 600ms
        if win_size % 2 == 0:
            win_size += 1
        baseline = medfilt(data, kernel_size=(win_size, 1))
        return data - baseline

    def _normalize(self, data):
        """Normaliza o sinal para ter média zero e desvio padrão um."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # Evitar divisão por zero para derivações com sinal plano
        std[std == 0] = 1
        return (data - mean) / std

    def _resample(self, data: np.ndarray, original_fs: int) -> np.ndarray:
        """Redimensiona o sinal para a frequência de amostragem alvo (target_fs)."""
        if original_fs == self.target_fs:
            return data
        
        num_samples_original = data.shape[0]
        duration = num_samples_original / original_fs
        num_samples_target = int(duration * self.target_fs)
        
        # Usa resample do scipy para melhor qualidade
        from scipy.signal import resample
        resampled_data = resample(data, num_samples_target, axis=0)
        
        logger.info(f"Sinal reamostrado de {original_fs}Hz para {self.target_fs}Hz. Shape: {resampled_data.shape}")
        return resampled_data

    def _pad_or_truncate(self, data: np.ndarray) -> np.ndarray:
        """Garante que o sinal tenha o comprimento alvo (target_len)."""
        current_len = data.shape[0]
        if current_len == self.target_len:
            return data
        
        if current_len > self.target_len:
            # Trunca o sinal, pegando o centro
            start = (current_len - self.target_len) // 2
            return data[start : start + self.target_len, :]
        else:
            # Preenche com zeros nas bordas
            pad_needed = self.target_len - current_len
            pad_before = pad_needed // 2
            pad_after = pad_needed - pad_before
            return np.pad(data, ((pad_before, pad_after), (0, 0)), 'constant', constant_values=0)

    async def process(self, signal_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Executa o pipeline completo de pré-processamento.
        """
        logger.info(f"Iniciando pré-processamento. Shape inicial: {signal_data.shape}, Rate: {sampling_rate}Hz")
        
        # 1. Reamostragem para a frequência do modelo
        signal = self._resample(signal_data, sampling_rate)

        # 2. Remoção de desvio da linha de base
        signal = self._baseline_wander_removal(signal)

        # 3. Filtragem de ruído
        signal = self._bandpass_filter(signal)
        
        # 4. Normalização
        signal = self._normalize(signal)
        
        # 5. Padding ou truncagem para o tamanho esperado pelo modelo
        signal = self._pad_or_truncate(signal)
        
        # 6. Adicionar dimensão de batch
        signal_batch = np.expand_dims(signal, axis=0)
        
        logger.info(f"Pré-processamento concluído. Shape final para o modelo: {signal_batch.shape}")
        return signal_batch.astype(np.float32)

