# backend/app/utils/ecg_processor.py
import numpy as np
import logging
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

class ECGProcessor:
    """
    Classe de utilitários para processar sinais de ECG e extrair
    características clínicas básicas.
    """
    def calculate_heart_rate(self, signal: np.ndarray, fs: int) -> float:
        """
        Calcula a frequência cardíaca média a partir dos picos R em uma derivação.
        Usa a derivação II por padrão, que geralmente tem a melhor visibilidade dos picos R.
        """
        try:
            # Usa a derivação II (índice 1) se disponível, senão a primeira
            lead_index = 1 if signal.shape[1] > 1 else 0
            lead_signal = signal[:, lead_index]

            # Encontra picos R. Ajusta a altura e distância para robustez.
            min_height = np.std(lead_signal) * 1.5 # Heurística para altura mínima do pico
            min_distance = fs * 0.4 # Distância mínima de 400ms entre batimentos (max 150 bpm)
            
            peaks, _ = find_peaks(lead_signal, height=min_height, distance=min_distance)

            if len(peaks) < 2:
                logger.warning("Não foi possível calcular a frequência cardíaca, poucos picos R detectados.")
                return 0.0

            # Calcula os intervalos RR em segundos e depois a frequência cardíaca
            rr_intervals = np.diff(peaks) / fs
            heart_rate_bpm = 60 / np.mean(rr_intervals)

            return round(heart_rate_bpm, 2)
        except Exception as e:
            logger.error(f"Erro ao calcular a frequência cardíaca: {e}", exc_info=True)
            return 0.0
            
    def calculate_qrs_duration(self, signal: np.ndarray, fs: int) -> float:
        """
        Estima a duração média do complexo QRS.
        Esta é uma simplificação e pode não ser clinicamente perfeita.
        """
        try:
            lead_index = 1 if signal.shape[1] > 1 else 0
            lead_signal = signal[:, lead_index]

            # Encontra picos R
            peaks, _ = find_peaks(lead_signal, height=np.std(lead_signal), distance=fs*0.4)

            if len(peaks) == 0:
                return 0.0

            qrs_durations = []
            for peak in peaks:
                # Heurística: procura o início (Q) e o fim (S) do complexo QRS
                # em uma janela ao redor do pico R.
                window_size = int(fs * 0.1) # Janela de 100ms
                start_window = max(0, peak - window_size)
                end_window = min(len(lead_signal), peak + window_size)
                
                window = lead_signal[start_window:end_window]
                
                # Q é o mínimo antes do pico, S é o mínimo depois do pico
                try:
                    q_point_relative = np.argmin(lead_signal[start_window:peak])
                    s_point_relative = np.argmin(lead_signal[peak:end_window])
                    
                    q_point = start_window + q_point_relative
                    s_point = peak + s_point_relative

                    duration_samples = s_point - q_point
                    duration_ms = (duration_samples / fs) * 1000
                    
                    # Filtra durações irrealistas
                    if 50 < duration_ms < 150:
                        qrs_durations.append(duration_ms)
                except ValueError:
                    continue # Janela vazia

            if not qrs_durations:
                return 0.0

            return np.mean(qrs_durations)
        except Exception as e:
            logger.error(f"Erro ao calcular a duração do QRS: {e}", exc_info=True)
            return 0.0

