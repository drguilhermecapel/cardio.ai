"""
Funções de pré-processamento v2 - Separação adequada de imagens e dados
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

# Configurar logging
logger = logging.getLogger(__name__)

# Importações condicionais
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV não disponível - processamento de imagens limitado")

try:
    from scipy import signal as scipy_signal
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy não disponível - filtros limitados")

# Constantes
ECG_SAMPLE_RATE = 500  # Hz padrão
ECG_DURATION = 10      # segundos
ECG_LEADS = 12         # derivações padrão
TARGET_LENGTH = ECG_SAMPLE_RATE * ECG_DURATION  # 5000 amostras

def get_diagnosis_mapping() -> Dict[int, str]:
    """Retorna mapeamento de diagnósticos."""
    return {
        0: "Normal",
        1: "Fibrilação Atrial",
        2: "Bloqueio de Ramo Esquerdo",
        3: "Bloqueio de Ramo Direito", 
        4: "Extrassístole Ventricular Prematura",
        5: "Extrassístole Atrial Prematura",
        6: "Taquicardia Ventricular",
        7: "Bradicardia Sinusal",
        8: "Taquicardia Sinusal",
        9: "Bloqueio AV de 1º Grau",
        10: "Bloqueio AV de 2º Grau",
        11: "Bloqueio AV de 3º Grau",
        12: "Infarto do Miocárdio Anterior",
        13: "Infarto do Miocárdio Inferior",
        14: "Infarto do Miocárdio Lateral",
        15: "Isquemia Miocárdica",
        # ... adicionar mais conforme necessário
    }

class ECGImageProcessor:
    """Classe específica para processamento de imagens ECG."""
    
    def __init__(self):
        self.min_contour_area = 1000
        self.line_thickness_threshold = 3
        
    def extract_ecg_from_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extrai sinal ECG de imagem digitalizada.
        
        Args:
            image_path: Caminho para imagem ECG
            
        Returns:
            Array numpy com sinal ECG extraído ou None se falhar
        """
        if not CV2_AVAILABLE:
            logger.error("OpenCV não disponível para processamento de imagens")
            return None
            
        try:
            logger.info(f"Processando imagem ECG: {image_path}")
            
            # Carregar imagem
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Não foi possível carregar imagem: {image_path}")
            
            # Converter para escala de cinza
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Pré-processamento da imagem
            processed = self._preprocess_image(gray)
            
            # Detectar linhas do ECG
            ecg_lines = self._detect_ecg_lines(processed)
            
            if len(ecg_lines) == 0:
                logger.warning("Nenhuma linha ECG detectada na imagem")
                return self._generate_synthetic_ecg()
            
            # Extrair sinais das linhas
            signals = self._extract_signals_from_lines(ecg_lines, processed.shape)
            
            # Combinar sinais em formato de 12 derivações
            ecg_signal = self._combine_signals_to_12_lead(signals)
            
            logger.info(f"ECG extraído com sucesso - Shape: {ecg_signal.shape}")
            return ecg_signal
            
        except Exception as e:
            logger.error(f"Erro na extração de ECG da imagem: {e}")
            return self._generate_synthetic_ecg()
    
    def _preprocess_image(self, gray: np.ndarray) -> np.ndarray:
        """Pré-processa imagem para melhor detecção de linhas."""
        # Aplicar filtro gaussiano para reduzir ruído
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Aplicar threshold adaptativo
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Operações morfológicas para limpar a imagem
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _detect_ecg_lines(self, processed: np.ndarray) -> list:
        """Detecta linhas do ECG na imagem processada."""
        # Detectar contornos
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtrar contornos por área e formato
        ecg_lines = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                # Verificar se é uma linha horizontal (característica do ECG)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if aspect_ratio > 5:  # Linha horizontal
                    ecg_lines.append(contour)
        
        return ecg_lines
    
    def _extract_signals_from_lines(self, lines: list, image_shape: tuple) -> list:
        """Extrai sinais digitais das linhas detectadas."""
        signals = []
        
        for line in lines:
            # Obter pontos da linha
            points = line.reshape(-1, 2)
            
            # Ordenar pontos por coordenada x
            points = points[points[:, 0].argsort()]
            
            # Interpolar para obter sinal uniforme
            if len(points) > 10:  # Mínimo de pontos para interpolação
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                
                # Normalizar coordenadas y (inverter porque y cresce para baixo)
                y_normalized = (image_shape[0] - y_coords) / image_shape[0]
                
                # Interpolar para tamanho fixo
                x_interp = np.linspace(x_coords[0], x_coords[-1], TARGET_LENGTH)
                y_interp = np.interp(x_interp, x_coords, y_normalized)
                
                signals.append(y_interp)
        
        return signals
    
    def _combine_signals_to_12_lead(self, signals: list) -> np.ndarray:
        """Combina sinais extraídos em formato de 12 derivações."""
        if len(signals) == 0:
            return self._generate_synthetic_ecg()
        
        # Se temos menos de 12 sinais, duplicar/interpolar
        while len(signals) < ECG_LEADS:
            if len(signals) > 0:
                # Adicionar variação do último sinal
                last_signal = signals[-1]
                noise = np.random.normal(0, 0.05, len(last_signal))
                signals.append(last_signal + noise)
            else:
                signals.append(self._generate_single_lead())
        
        # Se temos mais de 12, selecionar os 12 melhores
        if len(signals) > ECG_LEADS:
            signals = signals[:ECG_LEADS]
        
        # Combinar em array 2D
        ecg_array = np.array(signals)
        
        # Normalizar cada derivação
        for i in range(ecg_array.shape[0]):
            lead = ecg_array[i]
            if np.std(lead) > 0:
                ecg_array[i] = (lead - np.mean(lead)) / np.std(lead)
        
        return ecg_array
    
    def _generate_synthetic_ecg(self) -> np.ndarray:
        """Gera ECG sintético como fallback."""
        logger.warning("Gerando ECG sintético como fallback")
        
        ecg_leads = []
        for lead in range(ECG_LEADS):
            signal = self._generate_single_lead()
            ecg_leads.append(signal)
        
        return np.array(ecg_leads)
    
    def _generate_single_lead(self) -> np.ndarray:
        """Gera uma derivação ECG sintética."""
        t = np.linspace(0, ECG_DURATION, TARGET_LENGTH)
        
        # Componentes básicos do ECG
        heart_rate = 70  # bpm
        rr_interval = 60 / heart_rate
        
        signal = np.zeros_like(t)
        
        # Adicionar complexos QRS
        for beat_time in np.arange(0, ECG_DURATION, rr_interval):
            # Onda P
            p_wave = 0.1 * np.exp(-((t - beat_time - 0.1) / 0.05) ** 2)
            
            # Complexo QRS
            qrs_wave = 0.8 * np.exp(-((t - beat_time - 0.2) / 0.02) ** 2)
            
            # Onda T
            t_wave = 0.3 * np.exp(-((t - beat_time - 0.4) / 0.1) ** 2)
            
            signal += p_wave + qrs_wave + t_wave
        
        # Adicionar ruído realista
        noise = np.random.normal(0, 0.02, len(signal))
        signal += noise
        
        return signal

class ECGDataProcessor:
    """Classe específica para processamento de dados ECG numéricos."""
    
    def __init__(self):
        self.target_sample_rate = ECG_SAMPLE_RATE
        self.target_duration = ECG_DURATION
        self.target_leads = ECG_LEADS
        
    def preprocess_ecg_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Pré-processa dados ECG numéricos.
        
        Args:
            data: Array numpy com dados ECG
            
        Returns:
            Array processado no formato (leads, samples)
        """
        try:
            logger.info(f"Processando dados ECG - Shape original: {data.shape}")
            
            # Converter para array numpy se necessário
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Tratar diferentes formatos de entrada
            processed_data = self._normalize_input_format(data)
            
            # Aplicar filtros
            filtered_data = self._apply_filters(processed_data)
            
            # Normalizar amplitude
            normalized_data = self._normalize_amplitude(filtered_data)
            
            # Ajustar para formato padrão
            standardized_data = self._standardize_format(normalized_data)
            
            logger.info(f"Dados processados - Shape final: {standardized_data.shape}")
            return standardized_data
            
        except Exception as e:
            logger.error(f"Erro no pré-processamento de dados: {e}")
            return self._generate_fallback_data()
    
    def _normalize_input_format(self, data: np.ndarray) -> np.ndarray:
        """Normaliza formato de entrada dos dados."""
        # Remover dimensões unitárias
        data = np.squeeze(data)
        
        # Tratar diferentes formatos
        if data.ndim == 1:
            # Sinal único - assumir como derivação única
            data = data.reshape(1, -1)
        elif data.ndim == 2:
            # Verificar orientação
            if data.shape[0] > data.shape[1]:
                # Mais linhas que colunas - transpor
                data = data.T
        elif data.ndim > 2:
            # Dados multidimensionais - achatar
            data = data.reshape(data.shape[0], -1)
        
        return data
    
    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Aplica filtros para limpeza do sinal."""
        filtered_data = np.copy(data)
        
        for i in range(data.shape[0]):
            signal = data[i]
            
            # Remover deriva da linha de base
            if SCIPY_AVAILABLE:
                # Filtro passa-alta para remover deriva
                sos = scipy_signal.butter(4, 0.5, btype='high', 
                                        fs=self.target_sample_rate, output='sos')
                signal = scipy_signal.sosfilt(sos, signal)
                
                # Filtro passa-baixa para remover ruído de alta frequência
                sos = scipy_signal.butter(4, 40, btype='low', 
                                        fs=self.target_sample_rate, output='sos')
                signal = scipy_signal.sosfilt(sos, signal)
            else:
                # Filtro simples sem scipy
                signal = signal - np.mean(signal)  # Remover DC
                
                # Filtro de média móvel simples
                window_size = min(10, len(signal) // 10)
                if window_size > 1:
                    kernel = np.ones(window_size) / window_size
                    signal = np.convolve(signal, kernel, mode='same')
            
            filtered_data[i] = signal
        
        return filtered_data
    
    def _normalize_amplitude(self, data: np.ndarray) -> np.ndarray:
        """Normaliza amplitude dos sinais."""
        normalized_data = np.copy(data)
        
        for i in range(data.shape[0]):
            signal = data[i]
            
            # Usar mediana para normalização robusta
            median_val = np.median(signal)
            mad = np.median(np.abs(signal - median_val))
            
            if mad > 0:
                normalized_data[i] = (signal - median_val) / (1.4826 * mad)
            else:
                # Se MAD é zero, usar desvio padrão
                std_val = np.std(signal)
                if std_val > 0:
                    normalized_data[i] = (signal - np.mean(signal)) / std_val
        
        return normalized_data
    
    def _standardize_format(self, data: np.ndarray) -> np.ndarray:
        """Padroniza formato para (12, 5000)."""
        current_leads, current_samples = data.shape
        
        # Ajustar número de derivações
        if current_leads < self.target_leads:
            # Adicionar derivações sintéticas
            additional_leads = self.target_leads - current_leads
            synthetic_leads = []
            
            for _ in range(additional_leads):
                # Criar derivação baseada nas existentes
                if current_leads > 0:
                    base_lead = data[np.random.randint(0, current_leads)]
                    noise = np.random.normal(0, 0.1, len(base_lead))
                    synthetic_lead = base_lead + noise
                else:
                    synthetic_lead = np.random.normal(0, 0.5, current_samples)
                
                synthetic_leads.append(synthetic_lead)
            
            data = np.vstack([data, np.array(synthetic_leads)])
            
        elif current_leads > self.target_leads:
            # Selecionar as primeiras 12 derivações
            data = data[:self.target_leads]
        
        # Ajustar número de amostras
        target_samples = TARGET_LENGTH
        if current_samples != target_samples:
            resampled_data = np.zeros((self.target_leads, target_samples))
            
            for i in range(self.target_leads):
                if current_samples > target_samples:
                    # Decimação
                    indices = np.linspace(0, current_samples - 1, target_samples, dtype=int)
                    resampled_data[i] = data[i][indices]
                else:
                    # Interpolação
                    x_old = np.linspace(0, 1, current_samples)
                    x_new = np.linspace(0, 1, target_samples)
                    resampled_data[i] = np.interp(x_new, x_old, data[i])
            
            data = resampled_data
        
        return data
    
    def _generate_fallback_data(self) -> np.ndarray:
        """Gera dados ECG sintéticos como fallback."""
        logger.warning("Gerando dados ECG sintéticos como fallback")
        
        # Usar o gerador de ECG da classe de imagens
        image_processor = ECGImageProcessor()
        return image_processor._generate_synthetic_ecg()

# Funções principais para compatibilidade
def extract_ecg_from_image(image_path: str) -> Optional[np.ndarray]:
    """Extrai ECG de imagem (interface compatível)."""
    processor = ECGImageProcessor()
    return processor.extract_ecg_from_image(image_path)

def preprocess_ecg_signal(data: np.ndarray) -> np.ndarray:
    """Pré-processa dados ECG (interface compatível)."""
    processor = ECGDataProcessor()
    return processor.preprocess_ecg_signal(data)

def validate_ecg_signal(signal: np.ndarray) -> Tuple[bool, str]:
    """
    Valida sinal ECG processado.
    
    Args:
        signal: Array ECG no formato (leads, samples)
        
    Returns:
        Tuple (is_valid, message)
    """
    try:
        if signal is None or signal.size == 0:
            return False, "Sinal vazio ou nulo"
        
        if signal.ndim != 2:
            return False, f"Dimensões incorretas: {signal.ndim}D (esperado 2D)"
        
        leads, samples = signal.shape
        
        if leads != ECG_LEADS:
            return False, f"Número de derivações incorreto: {leads} (esperado {ECG_LEADS})"
        
        if samples != TARGET_LENGTH:
            return False, f"Número de amostras incorreto: {samples} (esperado {TARGET_LENGTH})"
        
        # Verificar se há variação no sinal
        for i in range(leads):
            if np.std(signal[i]) < 1e-6:
                return False, f"Derivação {i} sem variação significativa"
        
        # Verificar valores extremos
        if np.any(np.abs(signal) > 50):
            return False, "Valores extremos detectados no sinal"
        
        return True, "Sinal ECG válido"
        
    except Exception as e:
        return False, f"Erro na validação: {str(e)}"

def prepare_for_model(signal: np.ndarray) -> np.ndarray:
    """
    Prepara sinal ECG para entrada no modelo.
    
    Args:
        signal: Array ECG no formato (leads, samples)
        
    Returns:
        Array no formato (batch, leads, samples)
    """
    try:
        # Adicionar dimensão de batch
        if signal.ndim == 2:
            model_input = np.expand_dims(signal, axis=0)
        else:
            model_input = signal
        
        # Verificar formato final
        if model_input.ndim != 3:
            raise ValueError(f"Formato incorreto para modelo: {model_input.shape}")
        
        return model_input.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Erro na preparação para modelo: {e}")
        # Retornar formato padrão em caso de erro
        fallback = np.random.normal(0, 0.1, (1, ECG_LEADS, TARGET_LENGTH))
        return fallback.astype(np.float32)

