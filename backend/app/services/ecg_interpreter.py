"""
Interpretador Principal de ECG com Inteligência Artificial
Sistema completo para análise e interpretação de eletrocardiogramas
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ECGInterpreter:
    """Interpretador principal de ECG com IA."""
    
    def __init__(self):
        self.version = "1.0.0"
        self.model_loaded = False
        self.analysis_count = 0
        
    def load_model(self) -> bool:
        """Carrega o modelo de IA para interpretação."""
        try:
            # Simulação de carregamento do modelo
            logger.info("Carregando modelo de IA para interpretação de ECG...")
            self.model_loaded = True
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def preprocess_ecg_signal(self, ecg_data: np.ndarray, sampling_rate: int = 500) -> np.ndarray:
        """Pré-processa o sinal de ECG."""
        try:
            # Filtro passa-banda (0.5-40 Hz)
            from scipy import signal
            
            # Normalização
            ecg_normalized = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)
            
            # Filtro passa-alta para remover baseline drift
            sos_high = signal.butter(4, 0.5, btype='high', fs=sampling_rate, output='sos')
            ecg_filtered = signal.sosfilt(sos_high, ecg_normalized)
            
            # Filtro passa-baixa para remover ruído de alta frequência
            sos_low = signal.butter(4, 40, btype='low', fs=sampling_rate, output='sos')
            ecg_clean = signal.sosfilt(sos_low, ecg_filtered)
            
            return ecg_clean
            
        except Exception as e:
            logger.error(f"Erro no pré-processamento: {e}")
            return ecg_data
    
    def detect_r_peaks(self, ecg_signal: np.ndarray, sampling_rate: int = 500) -> List[int]:
        """Detecta picos R no sinal de ECG."""
        try:
            from scipy import signal
            
            # Encontrar picos com altura mínima e distância
            height = np.std(ecg_signal) * 0.5
            distance = int(0.6 * sampling_rate)  # Mínimo 0.6s entre picos
            
            peaks, _ = signal.find_peaks(ecg_signal, height=height, distance=distance)
            return peaks.tolist()
            
        except Exception as e:
            logger.error(f"Erro na detecção de picos R: {e}")
            return []
    
    def calculate_heart_rate(self, r_peaks: List[int], sampling_rate: int = 500) -> float:
        """Calcula a frequência cardíaca."""
        if len(r_peaks) < 2:
            return 0.0
        
        # Calcular intervalos RR
        rr_intervals = np.diff(r_peaks) / sampling_rate
        
        # Frequência cardíaca média
        heart_rate = 60.0 / np.mean(rr_intervals)
        
        return round(heart_rate, 1)
    
    def analyze_rhythm(self, r_peaks: List[int], sampling_rate: int = 500) -> Dict[str, Any]:
        """Analisa o ritmo cardíaco."""
        if len(r_peaks) < 3:
            return {"rhythm": "Insuficiente", "regularity": "Indeterminado"}
        
        # Calcular intervalos RR
        rr_intervals = np.diff(r_peaks) / sampling_rate
        
        # Variabilidade dos intervalos RR
        rr_variability = np.std(rr_intervals) / np.mean(rr_intervals)
        
        # Classificação do ritmo
        if rr_variability < 0.1:
            regularity = "Regular"
        elif rr_variability < 0.2:
            regularity = "Irregularmente regular"
        else:
            regularity = "Irregularmente irregular"
        
        # Frequência cardíaca
        heart_rate = 60.0 / np.mean(rr_intervals)
        
        # Classificação básica do ritmo
        if 60 <= heart_rate <= 100 and regularity == "Regular":
            rhythm = "Ritmo sinusal normal"
        elif heart_rate < 60:
            rhythm = "Bradicardia"
        elif heart_rate > 100:
            rhythm = "Taquicardia"
        else:
            rhythm = "Arritmia"
        
        return {
            "rhythm": rhythm,
            "regularity": regularity,
            "heart_rate": round(heart_rate, 1),
            "rr_variability": round(rr_variability, 3)
        }
    
    def detect_abnormalities(self, ecg_signal: np.ndarray, r_peaks: List[int]) -> List[Dict[str, Any]]:
        """Detecta anormalidades no ECG."""
        abnormalities = []
        
        try:
            # Análise de amplitude dos picos R
            if len(r_peaks) > 0:
                r_amplitudes = [ecg_signal[peak] for peak in r_peaks]
                mean_amplitude = np.mean(r_amplitudes)
                
                # Verificar variações significativas na amplitude
                for i, amplitude in enumerate(r_amplitudes):
                    if abs(amplitude - mean_amplitude) > 2 * np.std(r_amplitudes):
                        abnormalities.append({
                            "type": "Variação de amplitude",
                            "location": r_peaks[i],
                            "severity": "Moderada",
                            "description": "Variação significativa na amplitude do complexo QRS"
                        })
            
            # Análise de intervalos RR
            if len(r_peaks) > 2:
                rr_intervals = np.diff(r_peaks)
                mean_rr = np.mean(rr_intervals)
                
                for i, interval in enumerate(rr_intervals):
                    if abs(interval - mean_rr) > 2 * np.std(rr_intervals):
                        abnormalities.append({
                            "type": "Arritmia",
                            "location": r_peaks[i],
                            "severity": "Leve",
                            "description": "Intervalo RR anormal detectado"
                        })
            
        except Exception as e:
            logger.error(f"Erro na detecção de anormalidades: {e}")
        
        return abnormalities
    
    def generate_interpretation(self, analysis_results: Dict[str, Any]) -> str:
        """Gera interpretação textual do ECG."""
        interpretation = []
        
        # Ritmo e frequência
        rhythm_info = analysis_results.get("rhythm_analysis", {})
        heart_rate = rhythm_info.get("heart_rate", 0)
        rhythm = rhythm_info.get("rhythm", "Indeterminado")
        
        interpretation.append(f"Frequência cardíaca: {heart_rate} bpm")
        interpretation.append(f"Ritmo: {rhythm}")
        
        # Anormalidades
        abnormalities = analysis_results.get("abnormalities", [])
        if abnormalities:
            interpretation.append(f"Anormalidades detectadas: {len(abnormalities)}")
            for abnormality in abnormalities[:3]:  # Máximo 3 principais
                interpretation.append(f"- {abnormality['type']}: {abnormality['description']}")
        else:
            interpretation.append("Nenhuma anormalidade significativa detectada")
        
        # Conclusão
        if heart_rate < 60:
            interpretation.append("CONCLUSÃO: Bradicardia - Recomenda-se avaliação médica")
        elif heart_rate > 100:
            interpretation.append("CONCLUSÃO: Taquicardia - Recomenda-se avaliação médica")
        elif abnormalities:
            interpretation.append("CONCLUSÃO: ECG com alterações - Recomenda-se avaliação médica")
        else:
            interpretation.append("CONCLUSÃO: ECG dentro dos parâmetros normais")
        
        return "\n".join(interpretation)
    
    def analyze_ecg(self, ecg_data: np.ndarray, sampling_rate: int = 500, 
                   patient_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Análise completa do ECG."""
        
        if not self.model_loaded:
            self.load_model()
        
        self.analysis_count += 1
        analysis_id = f"ECG_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.analysis_count}"
        
        try:
            # Pré-processamento
            ecg_clean = self.preprocess_ecg_signal(ecg_data, sampling_rate)
            
            # Detecção de picos R
            r_peaks = self.detect_r_peaks(ecg_clean, sampling_rate)
            
            # Análise do ritmo
            rhythm_analysis = self.analyze_rhythm(r_peaks, sampling_rate)
            
            # Detecção de anormalidades
            abnormalities = self.detect_abnormalities(ecg_clean, r_peaks)
            
            # Resultados da análise
            results = {
                "analysis_id": analysis_id,
                "timestamp": datetime.now().isoformat(),
                "patient_info": patient_info or {},
                "signal_quality": "Boa" if len(r_peaks) > 5 else "Ruim",
                "rhythm_analysis": rhythm_analysis,
                "r_peaks_count": len(r_peaks),
                "abnormalities": abnormalities,
                "confidence_score": 0.85 if len(r_peaks) > 5 else 0.60
            }
            
            # Gerar interpretação
            results["interpretation"] = self.generate_interpretation(results)
            
            logger.info(f"Análise ECG concluída: {analysis_id}")
            return results
            
        except Exception as e:
            logger.error(f"Erro na análise do ECG: {e}")
            return {
                "analysis_id": analysis_id,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "Erro na análise"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status do interpretador."""
        return {
            "version": self.version,
            "model_loaded": self.model_loaded,
            "analyses_performed": self.analysis_count,
            "status": "Ativo" if self.model_loaded else "Inativo"
        }


# Instância global do interpretador
ecg_interpreter = ECGInterpreter()


def create_sample_ecg_data(duration: int = 10, sampling_rate: int = 500) -> np.ndarray:
    """Cria dados de ECG simulados para teste."""
    t = np.linspace(0, duration, duration * sampling_rate)
    
    # Simular ECG com componentes básicos
    heart_rate = 75  # bpm
    frequency = heart_rate / 60  # Hz
    
    # Onda P, QRS, T simplificada
    ecg_signal = np.zeros_like(t)
    
    for beat in range(int(duration * frequency)):
        beat_time = beat / frequency
        beat_samples = int(beat_time * sampling_rate)
        
        if beat_samples < len(ecg_signal) - 100:
            # Complexo QRS simplificado
            qrs_width = int(0.08 * sampling_rate)  # 80ms
            qrs_start = beat_samples
            qrs_end = min(qrs_start + qrs_width, len(ecg_signal))
            
            # Pico R
            r_peak_pos = qrs_start + qrs_width // 2
            if r_peak_pos < len(ecg_signal):
                ecg_signal[r_peak_pos] = 1.0
                
                # Ondas Q e S
                if qrs_start < len(ecg_signal):
                    ecg_signal[qrs_start] = -0.2
                if qrs_end - 1 < len(ecg_signal):
                    ecg_signal[qrs_end - 1] = -0.3
    
    # Adicionar ruído realista
    noise = np.random.normal(0, 0.05, len(ecg_signal))
    ecg_signal += noise
    
    return ecg_signal

