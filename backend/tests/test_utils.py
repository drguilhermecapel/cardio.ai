"""
Utilidades para testes do CardioAI Pro
"""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import numpy as np
from typing import Tuple, List
import scipy.signal as signal


class ECGTestGenerator:
    """Gerador de sinais ECG para testes."""
    
    @staticmethod
    def generate_clean_ecg(duration: int, fs: int, heart_rate: int = 72) -> Tuple[np.ndarray, List[int]]:
        """Gera ECG limpo com R-peaks conhecidos.
        
        Args:
            duration: Duração em segundos
            fs: Taxa de amostragem
            heart_rate: Frequência cardíaca em bpm
            
        Returns:
            Tupla (sinal_ecg, posições_r_peaks)
        """
        samples = duration * fs
        t = np.linspace(0, duration, samples)
        
        # Gerar ECG sintético
        ecg = np.zeros(samples)
        r_peaks = []
        
        # Intervalo entre batimentos
        beat_interval = 60.0 / heart_rate  # em segundos
        beat_samples = int(beat_interval * fs)
        
        # Gerar batimentos
        for i in range(0, samples, beat_samples):
            if i + 100 < samples:
                # Onda P
                p_wave = 0.1 * signal.windows.gaussian(int(0.08 * fs), std=fs*0.01)
                ecg[i:i+len(p_wave)] += p_wave
                
                # Complexo QRS
                qrs_start = i + int(0.12 * fs)
                qrs_width = max(int(0.08 * fs), 1)  # Garantir pelo menos 1 sample
                if qrs_start + qrs_width < samples:
                    qrs = 1.0 * signal.windows.gaussian(qrs_width, std=qrs_width/6)
                    ecg[qrs_start:qrs_start+len(qrs)] += qrs
                    r_peaks.append(qrs_start + qrs_width//2)
                
                # Onda T
                t_start = qrs_start + int(0.15 * fs)
                if t_start + int(0.2 * fs) < samples:
                    t_wave = 0.2 * signal.windows.gaussian(int(0.2 * fs), std=fs*0.03)
                    ecg[t_start:t_start+len(t_wave)] += t_wave
        
        return ecg, r_peaks
    
    @staticmethod
    def add_gaussian_noise(ecg: np.ndarray, snr_db: float) -> np.ndarray:
        """Adiciona ruído gaussiano ao ECG."""
        signal_power = np.mean(ecg ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(ecg))
        return ecg + noise
    
    @staticmethod
    def add_baseline_wander(ecg: np.ndarray, amplitude: float = 0.5) -> np.ndarray:
        """Adiciona deriva de linha de base."""
        t = np.arange(len(ecg))
        wander = amplitude * np.sin(2 * np.pi * 0.1 * t / len(ecg))
        return ecg + wander
    
    @staticmethod
    def add_powerline_interference(ecg: np.ndarray, amplitude: float = 0.1) -> np.ndarray:
        """Adiciona interferência de linha de energia (50/60 Hz)."""
        t = np.arange(len(ecg))
        fs = 360  # Taxa de amostragem padrão
        interference = amplitude * np.sin(2 * np.pi * 50 * t / fs)
        return ecg + interference
    
    @staticmethod
    def generate_pathological_ecg(pathology: str, duration: int = 10, fs: int = 360) -> np.ndarray:
        """Gera ECG com patologia específica."""
        if pathology == "afib":
            # Fibrilação atrial - ritmo irregular
            ecg = np.random.randn(duration * fs) * 0.1
            # Adicionar complexos QRS irregulares
            intervals = np.random.exponential(0.8, size=int(duration * 100/60))
            current_pos = 0
            for interval in intervals:
                pos = current_pos + int(interval * fs)
                if pos < len(ecg) - 50:
                    ecg[pos:pos+20] += np.random.randn() * 0.8
                current_pos = pos
            return ecg
        
        elif pathology == "vt":
            # Taquicardia ventricular
            ecg, _ = ECGTestGenerator.generate_clean_ecg(duration, fs, heart_rate=180)
            # Alargar QRS
            return signal.savgol_filter(ecg, 51, 3)
        
        else:
            # Retornar ECG normal como fallback
            ecg, _ = ECGTestGenerator.generate_clean_ecg(duration, fs)
            return ecg
