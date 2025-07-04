"""
Funções de pré-processamento melhoradas para ECG
Inclui correções para extração de imagens e processamento robusto
"""

import numpy as np
import scipy.signal as sg
from scipy import ndimage
import cv2
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

def preprocess_ecg_signal(sig: np.ndarray, fs_in: int = 500, fs_target: int = 100) -> np.ndarray:
    """
    Pré-processa sinal de ECG para uso com modelo PTB-XL.
    
    Args:
        sig: Sinal ECG com shape (amostras, derivações) ou (derivações, amostras)
        fs_in: Frequência de amostragem de entrada (Hz)
        fs_target: Frequência de amostragem alvo (Hz)
        
    Returns:
        Sinal pré-processado com shape (12, 1000) para o modelo
    """
    try:
        # Converter para numpy array se necessário
        if not isinstance(sig, np.ndarray):
            sig = np.array(sig)
        
        # Verificar se o sinal não está vazio
        if sig.size == 0:
            raise ValueError("Sinal ECG vazio")
        
        # Verificar e corrigir orientação
        if len(sig.shape) == 1:
            # Sinal unidimensional - assumir uma derivação
            sig = sig.reshape(1, -1)
        elif sig.shape[0] > sig.shape[1]:
            # Assumir que a dimensão maior é o tempo
            sig = sig.T
        
        # Garantir 12 derivações
        if sig.shape[0] < 12:
            # Preencher com zeros se necessário
            pad_leads = np.zeros((12 - sig.shape[0], sig.shape[1]))
            sig = np.vstack([sig, pad_leads])
        elif sig.shape[0] > 12:
            # Usar apenas as primeiras 12 derivações
            sig = sig[:12, :]
        
        # Reamostrar para frequência alvo se necessário
        if fs_in != fs_target and fs_in > 0:
            new_length = int(sig.shape[1] * fs_target / fs_in)
            if new_length > 0:
                resampled = np.zeros((sig.shape[0], new_length))
                for i in range(sig.shape[0]):
                    if np.any(np.isfinite(sig[i])):
                        resampled[i] = sg.resample(sig[i], new_length)
                    else:
                        resampled[i] = np.zeros(new_length)
                sig = resampled
        
        # Garantir comprimento de 1000 pontos (10s a 100Hz)
        target_length = 1000
        if sig.shape[1] < target_length:
            # Preencher com zeros
            pad_length = target_length - sig.shape[1]
            sig = np.pad(sig, ((0, 0), (0, pad_length)), 'constant')
        elif sig.shape[1] > target_length:
            # Cortar para target_length pontos (pegar do meio)
            start_idx = (sig.shape[1] - target_length) // 2
            sig = sig[:, start_idx:start_idx + target_length]
        
        # Filtrar ruído e remover linha de base
        for i in range(sig.shape[0]):
            if np.any(np.isfinite(sig[i])) and np.std(sig[i]) > 0:
                # Filtro passa-alta para remover deriva da linha de base
                sos_high = sg.butter(4, 0.5, btype='high', fs=fs_target, output='sos')
                sig[i] = sg.sosfilt(sos_high, sig[i])
                
                # Filtro passa-baixa para remover ruído de alta frequência
                sos_low = sg.butter(4, 40, btype='low', fs=fs_target, output='sos')
                sig[i] = sg.sosfilt(sos_low, sig[i])
                
                # Remover artefatos usando filtro mediano
                try:
                    kernel_size = min(21, len(sig[i]) // 10)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    baseline = sg.medfilt(sig[i], kernel_size=kernel_size)
                    sig[i] = sig[i] - baseline
                except:
                    # Se falhar, usar remoção simples da média
                    sig[i] = sig[i] - np.mean(sig[i])
            else:
                # Se a derivação está vazia ou inválida, preencher com zeros
                sig[i] = np.zeros(sig.shape[1])
        
        # Normalização robusta
        for i in range(sig.shape[0]):
            if np.std(sig[i]) > 0:
                # Normalização Z-score robusta usando mediana
                median = np.median(sig[i])
                mad = np.median(np.abs(sig[i] - median))
                if mad > 0:
                    sig[i] = (sig[i] - median) / (1.4826 * mad)
                else:
                    sig[i] = sig[i] - median
            
        # Clipar valores extremos
        sig = np.clip(sig, -10, 10)
        
        return sig.astype('float32')
        
    except Exception as e:
        logger.error(f"Erro no pré-processamento do sinal ECG: {e}")
        # Retornar sinal zerado em caso de erro
        return np.zeros((12, 1000), dtype='float32')


def extract_ecg_from_image(image_path: str, 
                          leads_expected: int = 12,
                          duration_seconds: float = 10.0,
                          sampling_rate: int = 100) -> np.ndarray:
    """
    Extrai sinal ECG de uma imagem digitalizada.
    
    Args:
        image_path: Caminho para a imagem do ECG
        leads_expected: Número de derivações esperadas
        duration_seconds: Duração do ECG em segundos
        sampling_rate: Taxa de amostragem desejada
        
    Returns:
        Sinal ECG extraído com shape (leads, samples)
    """
    try:
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Pré-processamento da imagem
        # Aplicar filtro gaussiano para reduzir ruído
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Binarização adaptativa para destacar as linhas do ECG
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Detectar linhas horizontais (derivações do ECG)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Encontrar contornos das derivações
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Ordenar contornos por posição vertical
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        
        # Extrair sinais de cada derivação
        signals = []
        target_samples = int(duration_seconds * sampling_rate)
        
        for i, contour in enumerate(contours[:leads_expected]):
            # Obter região da derivação
            x, y, w, h = cv2.boundingRect(contour)
            
            # Expandir região para capturar toda a derivação
            y_start = max(0, y - h//2)
            y_end = min(gray.shape[0], y + h + h//2)
            roi = binary[y_start:y_end, x:x+w]
            
            # Extrair perfil vertical médio
            if roi.size > 0:
                profile = np.mean(roi, axis=0)
                
                # Interpolar para obter número correto de amostras
                if len(profile) > 1:
                    x_old = np.linspace(0, 1, len(profile))
                    x_new = np.linspace(0, 1, target_samples)
                    signal = np.interp(x_new, x_old, profile)
                    
                    # Inverter se necessário (picos para cima)
                    if np.mean(signal) > 127:
                        signal = 255 - signal
                    
                    # Normalizar para range [-1, 1]
                    signal = (signal - 127.5) / 127.5
                    signals.append(signal)
        
        # Preencher com zeros se não temos derivações suficientes
        while len(signals) < leads_expected:
            signals.append(np.zeros(target_samples))
        
        # Converter para array numpy
        ecg_signal = np.array(signals[:leads_expected])
        
        return ecg_signal.astype('float32')
        
    except Exception as e:
        logger.error(f"Erro na extração de ECG da imagem: {e}")
        # Retornar sinal zerado em caso de erro
        target_samples = int(duration_seconds * sampling_rate)
        return np.zeros((leads_expected, target_samples), dtype='float32')


def validate_ecg_signal(signal: np.ndarray) -> Tuple[bool, str]:
    """
    Valida se o sinal ECG está em formato correto para o modelo.
    
    Args:
        signal: Sinal ECG para validar
        
    Returns:
        Tuple (is_valid, error_message)
    """
    try:
        if not isinstance(signal, np.ndarray):
            return False, "Sinal deve ser um numpy array"
        
        if signal.ndim != 2:
            return False, f"Sinal deve ter 2 dimensões, encontrado {signal.ndim}"
        
        if signal.shape[0] != 12:
            return False, f"Sinal deve ter 12 derivações, encontrado {signal.shape[0]}"
        
        if signal.shape[1] != 1000:
            return False, f"Sinal deve ter 1000 amostras, encontrado {signal.shape[1]}"
        
        if not np.isfinite(signal).all():
            return False, "Sinal contém valores não finitos (NaN ou Inf)"
        
        if np.all(signal == 0):
            return False, "Sinal está completamente zerado"
        
        # Verificar se há variação suficiente
        for i in range(signal.shape[0]):
            if np.std(signal[i]) < 1e-6:
                logger.warning(f"Derivação {i} tem variação muito baixa")
        
        return True, "Sinal válido"
        
    except Exception as e:
        return False, f"Erro na validação: {str(e)}"


def prepare_for_model(signal: np.ndarray) -> np.ndarray:
    """
    Prepara o sinal ECG para entrada no modelo.
    
    Args:
        signal: Sinal ECG com shape (12, 1000)
        
    Returns:
        Sinal preparado com shape (1, 12, 1000) para batch
    """
    try:
        # Validar sinal
        is_valid, error_msg = validate_ecg_signal(signal)
        if not is_valid:
            logger.error(f"Sinal inválido: {error_msg}")
            signal = np.zeros((12, 1000), dtype='float32')
        
        # Adicionar dimensão de batch
        return np.expand_dims(signal, axis=0)
        
    except Exception as e:
        logger.error(f"Erro na preparação para o modelo: {e}")
        return np.zeros((1, 12, 1000), dtype='float32')


def get_ptbxl_leads_order():
    """Retorna a ordem das derivações no PTB-XL."""
    return ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def get_diagnosis_mapping():
    """Retorna mapeamento de classes para diagnósticos."""
    return {
        0: "Normal",
        1: "Fibrilação Atrial",
        2: "Bradicardia Sinusal",
        3: "Taquicardia Sinusal", 
        4: "Arritmia Ventricular",
        5: "Bloqueio AV",
        6: "Isquemia",
        7: "Infarto do Miocárdio",
        8: "Hipertrofia Ventricular Esquerda",
        9: "Anormalidade Inespecífica"
    }


# Função de compatibilidade com código existente
def preprocess(sig, fs_in=500, fs_target=100):
    """Função de compatibilidade com código existente."""
    return preprocess_ecg_signal(sig, fs_in, fs_target)

