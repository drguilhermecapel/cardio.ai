
import numpy as np
import scipy.signal as sg

def preprocess_ecg(sig, fs_in=500, fs_target=100):
    """
    Pré-processa sinal de ECG para uso com modelo PTB-XL.
    
    Args:
        sig: Sinal ECG com shape (amostras, derivações) ou (derivações, amostras)
        fs_in: Frequência de amostragem de entrada (Hz)
        fs_target: Frequência de amostragem alvo (Hz)
        
    Returns:
        Sinal pré-processado com shape (1000, 12)
    """
    # Verificar e corrigir orientação
    if sig.shape[0] > sig.shape[1]:
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
    
    # Reamostrar para 100 Hz (padrão PTB-XL)
    if fs_in != fs_target:
        new_length = int(sig.shape[1] * fs_target / fs_in)
        resampled = np.zeros((sig.shape[0], new_length))
        for i in range(sig.shape[0]):
            resampled[i] = sg.resample(sig[i], new_length)
        sig = resampled
    
    # Garantir comprimento de 1000 pontos (10s a 100Hz)
    if sig.shape[1] < 1000:
        # Preencher com zeros
        pad_length = 1000 - sig.shape[1]
        sig = np.pad(sig, ((0, 0), (0, pad_length)), 'constant')
    elif sig.shape[1] > 1000:
        # Cortar para 1000 pontos
        sig = sig[:, :1000]
    
    # Remover linha de base
    for i in range(sig.shape[0]):
        try:
            sig[i] = sig[i] - sg.medfilt(sig[i], kernel_size=201)
        except:
            # Se falhar, usar filtro mais simples
            sig[i] = sig[i] - np.mean(sig[i])
    
    # Normalizar para mV
    sig = sig / 1000.0  # Assumindo entrada em µV
    
    # Manter formato (derivações, amostras) - formato PTB-XL
    # Não transpor aqui, pois o modelo espera (batch, derivações, amostras)
    
    return sig.astype('float32')

def get_ptbxl_leads_order():
    """Retorna a ordem das derivações no PTB-XL."""
    return ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
