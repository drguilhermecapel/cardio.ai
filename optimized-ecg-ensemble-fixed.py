#!/usr/bin/env python3
"""
Sistema Ensemble ECG Otimizado com Análise Paramétrica Completa
Versão corrigida e otimizada para máxima performance diagnóstica
Target: AUC > 99%
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, NamedTuple
from collections import defaultdict, OrderedDict, Counter
from datetime import datetime
import warnings
from tqdm import tqdm
from dataclasses import dataclass, field
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn

from scipy import signal as scipy_signal
from scipy.signal import find_peaks, butter, filtfilt, hilbert, welch
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, skew
import pywt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    classification_report, multilabel_confusion_matrix, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight

# Configurações otimizadas
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Otimizações CUDA
cudnn.benchmark = True
cudnn.deterministic = False

# Seed para reprodutibilidade
def set_all_seeds(seed=42):
    """Define todas as seeds para reprodutibilidade"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(42)

# ==================== CONFIGURAÇÃO MÉDICA ====================

@dataclass
class MedicalECGConfig:
    """Configuração otimizada para análise ECG médica"""
    # Dados
    sampling_rate: int = 100
    signal_length: int = 1000
    num_leads: int = 12
    use_multilabel: bool = True
    
    # Modelo
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 16
    dropout: float = 0.1
    activation: str = 'gelu'
    
    # Treinamento
    batch_size: int = 64
    num_epochs: int = 300
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Augmentação
    augment_prob: float = 0.8
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Loss
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    
    # Validação
    early_stopping_patience: int = 30
    eval_every_n_epochs: int = 1

# ==================== PATOLOGIAS ECG COMPLETAS ====================

COMPLETE_ECG_PATHOLOGIES = {
    # Arritmias
    'NORM': {'name': 'Normal ECG', 'severity': 0, 'category': 'rhythm'},
    'SR': {'name': 'Sinus rhythm', 'severity': 0, 'category': 'rhythm'},
    'AFIB': {'name': 'Atrial fibrillation', 'severity': 3, 'category': 'rhythm'},
    'AFLT': {'name': 'Atrial flutter', 'severity': 3, 'category': 'rhythm'},
    'STACH': {'name': 'Sinus tachycardia', 'severity': 1, 'category': 'rhythm'},
    'SBRAD': {'name': 'Sinus bradycardia', 'severity': 1, 'category': 'rhythm'},
    'PAC': {'name': 'Premature atrial contraction', 'severity': 1, 'category': 'rhythm'},
    'PVC': {'name': 'Premature ventricular contraction', 'severity': 2, 'category': 'rhythm'},
    'VT': {'name': 'Ventricular tachycardia', 'severity': 5, 'category': 'rhythm'},
    'VF': {'name': 'Ventricular fibrillation', 'severity': 5, 'category': 'rhythm'},
    
    # Bloqueios
    'AVB1': {'name': 'First degree AV block', 'severity': 1, 'category': 'conduction'},
    'AVB2': {'name': 'Second degree AV block', 'severity': 3, 'category': 'conduction'},
    'AVB3': {'name': 'Third degree AV block', 'severity': 5, 'category': 'conduction'},
    'RBBB': {'name': 'Right bundle branch block', 'severity': 2, 'category': 'conduction'},
    'LBBB': {'name': 'Left bundle branch block', 'severity': 2, 'category': 'conduction'},
    
    # Isquemia
    'MI': {'name': 'Myocardial infarction', 'severity': 4, 'category': 'ischemia'},
    'STEMI': {'name': 'ST elevation MI', 'severity': 5, 'category': 'ischemia'},
    'STTC': {'name': 'ST-T changes', 'severity': 2, 'category': 'ischemia'},
    
    # Hipertrofia
    'LVH': {'name': 'Left ventricular hypertrophy', 'severity': 2, 'category': 'hypertrophy'},
    'RVH': {'name': 'Right ventricular hypertrophy', 'severity': 2, 'category': 'hypertrophy'},
}

# ==================== PARÂMETROS ECG CLÍNICOS ====================

@dataclass
class ECGParameters:
    """Parâmetros clínicos extraídos do ECG"""
    # Frequência Cardíaca
    heart_rate: float = 0.0
    heart_rate_variability: float = 0.0
    rhythm_regularity: float = 0.0
    
    # Amplitudes (em mV)
    p_amplitude: Dict[str, float] = field(default_factory=dict)
    qrs_amplitude: Dict[str, float] = field(default_factory=dict)
    t_amplitude: Dict[str, float] = field(default_factory=dict)
    st_level: Dict[str, float] = field(default_factory=dict)
    
    # Durações (em ms)
    p_duration: float = 0.0
    pr_interval: float = 0.0
    qrs_duration: float = 0.0
    qt_interval: float = 0.0
    qtc_interval: float = 0.0
    
    # Morfologia
    p_morphology: Dict[str, str] = field(default_factory=dict)
    qrs_morphology: Dict[str, str] = field(default_factory=dict)
    t_morphology: Dict[str, str] = field(default_factory=dict)
    
    # Eixos
    p_axis: float = 0.0
    qrs_axis: float = 0.0
    t_axis: float = 0.0
    
    # Características especiais
    q_waves: Dict[str, bool] = field(default_factory=dict)
    delta_waves: bool = False
    epsilon_waves: bool = False
    j_point_elevation: Dict[str, float] = field(default_factory=dict)
    
    # Variabilidade e complexidade
    qrs_variability: float = 0.0
    rr_entropy: float = 0.0
    
    # Índices derivados
    sokolow_lyon_index: float = 0.0
    cornell_index: float = 0.0
    romhilt_estes_score: int = 0

# ==================== DETECTOR DE ONDAS AVANÇADO ====================

class AdvancedWaveDetector:
    """Detector avançado de ondas P, QRS e T com análise morfológica"""
    
    def __init__(self, sampling_rate=100):
        self.fs = sampling_rate
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    def detect_r_peaks(self, ecg_signal):
        """Detecção robusta de picos R usando múltiplos métodos"""
        if len(ecg_signal) < self.fs:
            return np.array([])
            
        # Método 1: Pan-Tompkins modificado
        r_peaks_pt = self._pan_tompkins_detector(ecg_signal)
        
        # Método 2: Baseado em wavelets
        r_peaks_wt = self._wavelet_detector(ecg_signal)
        
        # Método 3: Baseado em energia
        r_peaks_energy = self._energy_detector(ecg_signal)
        
        # Consenso entre métodos
        r_peaks = self._consensus_peaks(r_peaks_pt, r_peaks_wt, r_peaks_energy)
        
        # Refinar com busca local
        r_peaks = self._refine_r_peaks(ecg_signal, r_peaks)
        
        return r_peaks
    
    def _pan_tompkins_detector(self, signal):
        """Implementação otimizada do Pan-Tompkins"""
        try:
            # Filtro passa-banda
            nyquist = self.fs / 2
            low = 5 / nyquist
            high = min(15 / nyquist, 0.99)
            
            if low >= high:
                return np.array([])
                
            b, a = butter(2, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)
            
            # Derivada
            diff = np.diff(filtered)
            
            # Elevar ao quadrado
            squared = diff ** 2
            
            # Integração com janela móvel
            window = int(0.15 * self.fs)
            integrated = np.convolve(squared, np.ones(window)/window, mode='same')
            
            # Encontrar picos
            peaks, properties = find_peaks(
                integrated,
                distance=int(0.2 * self.fs),
                height=np.percentile(integrated, 75)
            )
            
            return peaks
        except:
            return np.array([])
    
    def _wavelet_detector(self, signal):
        """Detecção usando transformada wavelet"""
        try:
            # Decomposição wavelet
            max_level = min(4, pywt.dwt_max_level(len(signal), 'db4'))
            coeffs = pywt.wavedec(signal, 'db4', level=max_level)
            
            # Reconstruir com escalas relevantes
            if len(coeffs) > 3:
                d3 = coeffs[-3]
                d4 = coeffs[-4] if len(coeffs) > 4 else coeffs[-3]
                
                # Reconstruir sinal das escalas
                reconstructed = np.zeros_like(signal)
                for i, c in enumerate(coeffs):
                    if i >= len(coeffs) - 4:
                        rec = pywt.waverec([np.zeros_like(c) if j != i else c 
                                           for j in range(len(coeffs))], 'db4')
                        if len(rec) > len(signal):
                            rec = rec[:len(signal)]
                        elif len(rec) < len(signal):
                            rec = np.pad(rec, (0, len(signal) - len(rec)))
                        reconstructed += rec
                
                # Detectar picos
                peaks, _ = find_peaks(
                    np.abs(reconstructed),
                    distance=int(0.2 * self.fs),
                    height=np.percentile(np.abs(reconstructed), 80)
                )
                
                return peaks
            else:
                return np.array([])
        except:
            return np.array([])
    
    def _energy_detector(self, signal):
        """Detecção baseada em energia do sinal"""
        try:
            # Transformada de Hilbert para envelope
            analytic_signal = hilbert(signal)
            envelope = np.abs(analytic_signal)
            
            # Suavizar envelope
            window = int(0.1 * self.fs)
            envelope_smooth = np.convolve(envelope, np.ones(window)/window, mode='same')
            
            # Detectar picos
            peaks, _ = find_peaks(
                envelope_smooth,
                distance=int(0.2 * self.fs),
                height=np.mean(envelope_smooth) + np.std(envelope_smooth)
            )
            
            return peaks
        except:
            return np.array([])
    
    def _consensus_peaks(self, peaks1, peaks2, peaks3):
        """Encontra consenso entre diferentes detectores"""
        all_peaks = []
        tolerance = int(0.05 * self.fs)
        
        # Para cada pico no detector 1
        for p1 in peaks1:
            count = 1
            # Verificar se existe em outros detectores
            if any(abs(p2 - p1) < tolerance for p2 in peaks2):
                count += 1
            if any(abs(p3 - p1) < tolerance for p3 in peaks3):
                count += 1
            
            # Se pelo menos 2 detectores concordam
            if count >= 2:
                all_peaks.append(p1)
        
        return np.array(sorted(set(all_peaks)))
    
    def _refine_r_peaks(self, signal, r_peaks):
        """Refina posição dos picos R"""
        refined_peaks = []
        window = int(0.05 * self.fs)
        
        for peak in r_peaks:
            start = max(0, peak - window)
            end = min(len(signal), peak + window)
            
            if start < end:
                # Encontrar máximo local
                local_segment = signal[start:end]
                if len(local_segment) > 0:
                    local_max = start + np.argmax(np.abs(local_segment))
                    refined_peaks.append(local_max)
        
        return np.array(refined_peaks)

# ==================== EXTRATOR DE PARÂMETROS CLÍNICOS ====================

class ClinicalParameterExtractor:
    """Extrai todos os parâmetros clínicos relevantes do ECG"""
    
    def __init__(self, sampling_rate=100):
        self.fs = sampling_rate
        self.wave_detector = AdvancedWaveDetector(sampling_rate)
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    def extract_all_parameters(self, ecg_12lead):
        """
        Extrai todos os parâmetros clínicos de um ECG 12 derivações
        
        Args:
            ecg_12lead: Array (12, n_samples) com ECG 12 derivações
            
        Returns:
            ECGParameters: Objeto com todos os parâmetros extraídos
        """
        try:
            # Usar derivação II para detecção principal
            lead_ii = ecg_12lead[1] if ecg_12lead.shape[0] > 1 else ecg_12lead[0]
            
            # Detectar ondas
            r_peaks = self.wave_detector.detect_r_peaks(lead_ii)
            
            if len(r_peaks) < 2:
                return ECGParameters()
            
            # Criar parâmetros com valores default
            params = ECGParameters(
                heart_rate=self._calculate_heart_rate(r_peaks),
                heart_rate_variability=self._calculate_hrv(r_peaks),
                rhythm_regularity=self._calculate_rhythm_regularity(r_peaks),
                qrs_duration=self._calculate_qrs_duration(ecg_12lead, r_peaks),
                pr_interval=200.0,  # Default
                qt_interval=400.0,  # Default
                qtc_interval=420.0,  # Default
                qrs_axis=self._calculate_axis(ecg_12lead, r_peaks, 'QRS'),
                sokolow_lyon_index=self._calculate_sokolow_lyon(ecg_12lead, r_peaks),
                cornell_index=self._calculate_cornell_index(ecg_12lead, r_peaks)
            )
            
            return params
            
        except Exception as e:
            logger.warning(f"Erro ao extrair parâmetros: {e}")
            return ECGParameters()
    
    def _calculate_heart_rate(self, r_peaks):
        """Calcula frequência cardíaca média"""
        if len(r_peaks) < 2:
            return 75.0  # Default
        
        rr_intervals = np.diff(r_peaks) / self.fs
        heart_rate = 60 / np.mean(rr_intervals)
        
        # Limitar a valores fisiológicos
        return np.clip(heart_rate, 30, 200)
    
    def _calculate_hrv(self, r_peaks):
        """Calcula variabilidade da frequência cardíaca (SDNN)"""
        if len(r_peaks) < 3:
            return 50.0  # Default
        
        rr_intervals = np.diff(r_peaks) / self.fs * 1000
        return np.std(rr_intervals)
    
    def _calculate_rhythm_regularity(self, r_peaks):
        """Calcula regularidade do ritmo"""
        if len(r_peaks) < 3:
            return 0.9
        
        rr_intervals = np.diff(r_peaks)
        cv = np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-8)
        
        return 1 / (1 + cv)
    
    def _calculate_qrs_duration(self, ecg_12lead, r_peaks):
        """Calcula duração do complexo QRS"""
        if len(r_peaks) == 0:
            return 90.0  # Default
        
        durations = []
        
        for r_peak in r_peaks[:min(5, len(r_peaks))]:
            # Usar derivação com maior amplitude
            lead_idx = 1  # Lead II
            if ecg_12lead.shape[0] > lead_idx:
                signal = ecg_12lead[lead_idx]
                
                # Janela ao redor do pico R
                window = int(0.1 * self.fs)
                start = max(0, r_peak - window)
                end = min(len(signal), r_peak + window)
                
                if end > start:
                    segment = signal[start:end]
                    
                    # Estimar duração baseada em energia
                    energy = segment ** 2
                    threshold = 0.1 * np.max(energy)
                    
                    above_threshold = energy > threshold
                    if np.any(above_threshold):
                        indices = np.where(above_threshold)[0]
                        duration = (indices[-1] - indices[0]) / self.fs * 1000
                        
                        if 40 < duration < 200:
                            durations.append(duration)
        
        return np.mean(durations) if durations else 90.0
    
    def _calculate_axis(self, ecg_12lead, r_peaks, wave_type):
        """Calcula eixo elétrico"""
        if len(r_peaks) == 0 or ecg_12lead.shape[0] < 6:
            return 0.0
        
        try:
            # Usar derivações I e aVF
            lead_i = ecg_12lead[0]
            lead_avf = ecg_12lead[5]
            
            # Calcular amplitude média nas ondas R
            amp_i = []
            amp_avf = []
            
            for peak in r_peaks[:min(5, len(r_peaks))]:
                window = int(0.05 * self.fs)
                start = max(0, peak - window)
                end = min(len(lead_i), peak + window)
                
                if end > start:
                    amp_i.append(np.max(lead_i[start:end]) - np.min(lead_i[start:end]))
                    amp_avf.append(np.max(lead_avf[start:end]) - np.min(lead_avf[start:end]))
            
            if amp_i and amp_avf:
                mean_i = np.mean(amp_i)
                mean_avf = np.mean(amp_avf)
                
                # Calcular ângulo
                axis_degrees = np.degrees(np.arctan2(mean_avf, mean_i))
                return axis_degrees
            
        except:
            pass
        
        return 0.0
    
    def _calculate_sokolow_lyon(self, ecg_12lead, r_peaks):
        """Calcula índice de Sokolow-Lyon para HVE"""
        if len(r_peaks) == 0 or ecg_12lead.shape[0] < 12:
            return 20.0  # Default
        
        try:
            # S em V1 + R em V5 ou V6
            amplitudes = []
            
            for r_peak in r_peaks[:min(5, len(r_peaks))]:
                window = int(0.05 * self.fs)
                
                # S em V1 (derivação 6)
                v1_seg = ecg_12lead[6][max(0, r_peak-window):min(len(ecg_12lead[6]), r_peak+window)]
                s_v1 = abs(np.min(v1_seg)) if len(v1_seg) > 0 else 0
                
                # R em V5 (derivação 10)
                v5_seg = ecg_12lead[10][max(0, r_peak-window):min(len(ecg_12lead[10]), r_peak+window)]
                r_v5 = np.max(v5_seg) if len(v5_seg) > 0 else 0
                
                # R em V6 (derivação 11)
                v6_seg = ecg_12lead[11][max(0, r_peak-window):min(len(ecg_12lead[11]), r_peak+window)]
                r_v6 = np.max(v6_seg) if len(v6_seg) > 0 else 0
                
                amplitudes.append(s_v1 + max(r_v5, r_v6))
            
            return np.mean(amplitudes) * 1000 if amplitudes else 20.0  # Converter para mm
            
        except:
            return 20.0
    
    def _calculate_cornell_index(self, ecg_12lead, r_peaks):
        """Calcula índice de Cornell para HVE"""
        if len(r_peaks) == 0 or ecg_12lead.shape[0] < 12:
            return 15.0  # Default
        
        try:
            amplitudes = []
            
            for r_peak in r_peaks[:min(5, len(r_peaks))]:
                window = int(0.05 * self.fs)
                
                # R em aVL (derivação 4)
                avl_seg = ecg_12lead[4][max(0, r_peak-window):min(len(ecg_12lead[4]), r_peak+window)]
                r_avl = np.max(avl_seg) if len(avl_seg) > 0 else 0
                
                # S em V3 (derivação 8)
                v3_seg = ecg_12lead[8][max(0, r_peak-window):min(len(ecg_12lead[8]), r_peak+window)]
                s_v3 = abs(np.min(v3_seg)) if len(v3_seg) > 0 else 0
                
                amplitudes.append(r_avl + s_v3)
            
            return np.mean(amplitudes) * 1000 if amplitudes else 15.0  # Converter para mm
            
        except:
            return 15.0

# ==================== DATASET OTIMIZADO ====================

class OptimizedECGDataset(Dataset):
    """Dataset otimizado com cache e augmentação avançada"""
    
    def __init__(self, X, Y, config, parameter_cache=None, is_training=True):
        self.X = X
        self.Y = Y
        self.config = config
        self.is_training = is_training
        
        # Cache de parâmetros
        self.parameter_cache = parameter_cache or {}
        
        # Extrator de parâmetros
        self.param_extractor = ClinicalParameterExtractor(config.sampling_rate)
        
        # Normalização robusta
        self.normalizer = RobustScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.normalizer.fit(X_reshaped)
        
        logger.info(f"Dataset criado: {len(X)} amostras, training={is_training}")
    
    def _params_to_vector(self, params):
        """Converte ECGParameters para vetor de features"""
        features = [
            params.heart_rate / 100.0,  # Normalizar
            params.heart_rate_variability / 100.0,
            params.rhythm_regularity,
            params.qrs_duration / 100.0,
            params.pr_interval / 200.0,
            params.qt_interval / 400.0,
            params.qtc_interval / 400.0,
            params.qrs_axis / 180.0,
            params.sokolow_lyon_index / 50.0,
            params.cornell_index / 30.0,
        ]
        
        return np.array(features, dtype=np.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # ECG signal
        ecg = self.X[idx].copy()
        
        # Normalização
        ecg_reshaped = ecg.reshape(-1, ecg.shape[-1])
        ecg_normalized = self.normalizer.transform(ecg_reshaped)
        ecg = ecg_normalized.reshape(ecg.shape)
        
        # Parâmetros clínicos
        if idx in self.parameter_cache:
            params = self.parameter_cache[idx]
        else:
            ecg_params = self.param_extractor.extract_all_parameters(self.X[idx])
            params = self._params_to_vector(ecg_params)
            self.parameter_cache[idx] = params
        
        # Augmentação durante treino
        if self.is_training and np.random.rand() < self.config.augment_prob:
            ecg = self._apply_augmentation(ecg)
        
        # Converter para tensores
        ecg_tensor = torch.FloatTensor(ecg)
        params_tensor = torch.FloatTensor(params)
        
        if self.config.use_multilabel:
            label = torch.FloatTensor(self.Y[idx])
        else:
            label = torch.LongTensor([self.Y[idx]])
        
        return ecg_tensor, params_tensor, label
    
    def _apply_augmentation(self, ecg):
        """Aplica augmentação avançada"""
        # 1. Time warping
        if np.random.rand() < 0.5:
            ecg = self._time_warp(ecg)
        
        # 2. Amplitude scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            ecg = ecg * scale
        
        # 3. Baseline wander
        if np.random.rand() < 0.3:
            ecg = self._add_baseline_wander(ecg)
        
        # 4. Gaussian noise
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.01, ecg.shape)
            ecg = ecg + noise
        
        # 5. Lead dropout
        if np.random.rand() < 0.2:
            num_drops = np.random.randint(1, 4)
            leads_to_drop = np.random.choice(12, num_drops, replace=False)
            ecg[leads_to_drop] = 0
        
        return ecg
    
    def _time_warp(self, ecg):
        """Aplica time warping"""
        warp_factor = np.random.uniform(0.9, 1.1)
        old_length = ecg.shape[1]
        new_length = int(old_length * warp_factor)
        
        warped = np.zeros((ecg.shape[0], old_length))
        for i in range(ecg.shape[0]):
            stretched = np.interp(
                np.linspace(0, new_length-1, old_length),
                np.arange(new_length),
                np.interp(np.linspace(0, old_length-1, new_length), 
                         np.arange(old_length), ecg[i])
            )
            warped[i] = stretched
        
        return warped
    
    def _add_baseline_wander(self, ecg):
        """Adiciona baseline wander realista"""
        t = np.linspace(0, 10, ecg.shape[1])
        
        for i in range(ecg.shape[0]):
            # Frequências respiratórias típicas (0.15-0.3 Hz)
            freq = np.random.uniform(0.15, 0.3)
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.05, 0.15)
            
            baseline = amplitude * np.sin(2 * np.pi * freq * t + phase)
            
            # Adicionar componente de baixa frequência
            freq2 = np.random.uniform(0.05, 0.1)
            baseline += 0.5 * amplitude * np.sin(2 * np.pi * freq2 * t)
            
            ecg[i] += baseline
        
        return ecg

# ==================== BLOCOS DE CONSTRUÇÃO OTIMIZADOS ====================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block para atenção por canal"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class ResidualBlock(nn.Module):
    """Bloco residual otimizado com SE"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.se = SEBlock(out_channels) if use_se else None
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.activation = nn.GELU()
    
    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.se is not None:
            out = self.se(out)
        
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class MultiScaleBlock(nn.Module):
    """Bloco multi-escala para capturar padrões em diferentes resoluções"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 4 == 0
        c = out_channels // 4
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm1d(c),
            nn.GELU()
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels, c, 5, padding=2, bias=False),
            nn.BatchNorm1d(c),
            nn.GELU()
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels, c, 7, padding=3, bias=False),
            nn.BatchNorm1d(c),
            nn.GELU()
        )
        
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, c, 1, bias=False),
            nn.BatchNorm1d(c),
            nn.GELU()
        )
    
    def forward(self, x):
        return torch.cat([
            self.conv3(x),
            self.conv5(x),
            self.conv7(x),
            self.pool_conv(x)
        ], dim=1)

# ==================== MODELOS ESPECIALIZADOS OTIMIZADOS ====================

class UltraOptimizedRhythmAnalyzer(nn.Module):
    """Analisador de ritmo ultra-otimizado"""
    
    def __init__(self, num_classes, config):
        super().__init__()
        self.config = config
        
        # Processamento inicial multi-escala
        self.input_conv = MultiScaleBlock(12, 128)
        
        # Blocos residuais profundos
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128, 128) for _ in range(6)
        ])
        
        # Atenção temporal bidirecional
        self.temporal_attention = nn.MultiheadAttention(
            128, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # GRU bidirecional para padrões sequenciais
        self.gru = nn.GRU(128, 256, num_layers=3, 
                          batch_first=True, bidirectional=True, dropout=0.1)
        
        # Fusão de parâmetros
        self.param_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.GELU()
        )
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim // 2, num_classes)
        )
    
    def forward(self, ecg, params):
        # Processamento multi-escala
        x = self.input_conv(ecg)
        
        # Blocos residuais
        for block in self.res_blocks:
            x = block(x)
        
        # Preparar para atenção
        x = x.transpose(1, 2)
        
        # Auto-atenção temporal
        x_att, _ = self.temporal_attention(x, x, x)
        x = x + x_att
        
        # GRU bidirecional
        x_gru, _ = self.gru(x)
        
        # Pooling adaptativo
        x_mean = x_gru.mean(dim=1)
        x_max = x_gru.max(dim=1)[0]
        x_features = torch.cat([x_mean, x_max], dim=1)[:, :512]
        
        # Processar parâmetros
        param_features = self.param_encoder(params)
        
        # Fusão e classificação
        combined = torch.cat([x_features, param_features], dim=1)
        output = self.classifier(combined)
        
        return output

class UltraOptimizedMorphologyAnalyzer(nn.Module):
    """Analisador morfológico ultra-otimizado"""
    
    def __init__(self, num_classes, config):
        super().__init__()
        self.config = config
        
        # Convolução inicial com diferentes tamanhos de kernel
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(12, 64, kernel_size=k, padding=k//2, bias=False),
                nn.BatchNorm1d(64),
                nn.GELU()
            ) for k in [3, 5, 7, 9, 11]
        ])
        
        # Fusão das branches
        self.fusion = nn.Conv1d(320, 256, 1)
        
        # Blocos residuais com SE
        self.res_blocks = nn.ModuleList([
            ResidualBlock(256, 256, use_se=True) for _ in range(8)
        ])
        
        # Atenção por derivação
        self.lead_attention = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.GELU(),
            nn.Conv1d(128, 12, 1),
            nn.Sigmoid()
        )
        
        # Pooling hierárquico
        self.hierarchical_pool = nn.ModuleList([
            nn.AdaptiveAvgPool1d(size) for size in [32, 16, 8, 4, 1]
        ])
        
        # Processador de parâmetros
        self.param_processor = nn.Sequential(
            nn.Linear(10, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU()
        )
        
        # Classificador final
        total_features = 256 * 5 + 256  # Hierarchical pooling + params
        self.classifier = nn.Sequential(
            nn.Linear(total_features, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, num_classes)
        )
    
    def forward(self, ecg, params):
        # Multi-branch convolution
        branch_outputs = [branch(ecg) for branch in self.conv_branches]
        x = torch.cat(branch_outputs, dim=1)
        x = self.fusion(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Lead attention
        att_weights = self.lead_attention(x)
        x = x * att_weights
        
        # Hierarchical pooling
        pooled_features = []
        for pool in self.hierarchical_pool:
            pooled = pool(x)
            pooled_features.append(pooled.flatten(1))
        
        x_features = torch.cat(pooled_features, dim=1)
        
        # Process parameters
        param_features = self.param_processor(params)
        
        # Combine and classify
        combined = torch.cat([x_features, param_features], dim=1)
        output = self.classifier(combined)
        
        return output

class UltraOptimizedTransformerIntegrator(nn.Module):
    """Integrador Transformer ultra-otimizado"""
    
    def __init__(self, num_classes, config):
        super().__init__()
        self.config = config
        
        # Projeção inicial com positional encoding aprendível
        self.input_projection = nn.Conv1d(12, config.hidden_dim, 1)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, config.hidden_dim) * 0.02)
        
        # Transformer encoder layers otimizados
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # Cross-attention com parâmetros
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        
        # Projeção de parâmetros
        self.param_projection = nn.Sequential(
            nn.Linear(10, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )
        
        # Pooling com atenção
        self.attention_pool = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, num_classes)
        )
    
    def forward(self, ecg, params):
        # Projeção e positional encoding
        x = self.input_projection(ecg)
        x = x.transpose(1, 2)
        
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Cross-attention com parâmetros
        param_features = self.param_projection(params).unsqueeze(1)
        x_cross, _ = self.cross_attention(x, param_features, param_features)
        x = x + x_cross
        
        # Attention pooling
        att_weights = self.attention_pool(x)
        x_pooled = (x * att_weights).sum(dim=1)
        
        # Combinar com features globais
        x_global = x.mean(dim=1)
        combined = torch.cat([x_pooled, x_global], dim=1)
        
        # Classificação
        output = self.classifier(combined)
        
        return output

# ==================== ENSEMBLE MÉDICO ULTRA-OTIMIZADO ====================

class UltraOptimizedMedicalEnsemble(nn.Module):
    """Ensemble médico ultra-otimizado para máxima performance"""
    
    def __init__(self, num_classes, config):
        super().__init__()
        self.num_classes = num_classes
        self.config = config
        
        # Modelos especializados
        self.rhythm_analyzer = UltraOptimizedRhythmAnalyzer(num_classes, config)
        self.morphology_analyzer = UltraOptimizedMorphologyAnalyzer(num_classes, config)
        self.transformer_integrator = UltraOptimizedTransformerIntegrator(num_classes, config)
        
        # Analisador de parâmetros direto
        self.parameter_analyzer = nn.Sequential(
            nn.Linear(10, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Meta-learner com gating
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * 4, config.hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim // 2, num_classes)
        )
        
        # Mecanismo de gating adaptativo
        self.gating_network = nn.Sequential(
            nn.Linear(10 + num_classes * 4, 256),
            nn.GELU(),
            nn.Linear(256, 4),
            nn.Softmax(dim=1)
        )
        
        # Temperature scaling para calibração
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, ecg, params, return_all_outputs=False):
        # Obter saídas de todos os modelos
        rhythm_out = self.rhythm_analyzer(ecg, params)
        morph_out = self.morphology_analyzer(ecg, params)
        trans_out = self.transformer_integrator(ecg, params)
        param_out = self.parameter_analyzer(params)
        
        # Stack outputs para meta-learning
        all_outputs = torch.stack([rhythm_out, morph_out, trans_out, param_out], dim=1)
        
        # Calcular pesos adaptativos
        concat_features = torch.cat([
            params,
            rhythm_out, morph_out, trans_out, param_out
        ], dim=1)
        
        weights = self.gating_network(concat_features)
        
        # Aplicar pesos
        weighted_outputs = all_outputs * weights.unsqueeze(-1)
        weighted_sum = weighted_outputs.sum(dim=1)
        
        # Meta-learner
        meta_input = torch.cat([rhythm_out, morph_out, trans_out, param_out], dim=1)
        meta_out = self.meta_learner(meta_input)
        
        # Combinação final com temperature scaling
        final_output = (0.6 * meta_out + 0.4 * weighted_sum) / self.temperature
        
        if return_all_outputs:
            return final_output, {
                'rhythm': rhythm_out,
                'morphology': morph_out,
                'transformer': trans_out,
                'parameters': param_out,
                'weights': weights,
                'meta': meta_out
            }
        
        return final_output

# ==================== LOSS FUNCTIONS OTIMIZADAS ====================

class OptimizedMedicalLoss(nn.Module):
    """Loss function otimizada para máxima performance em ECG"""
    
    def __init__(self, num_classes, class_weights=None, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        
        # Pesos baseados em severidade clínica
        if class_weights is None:
            # Criar pesos baseados em severidade
            severity_weights = []
            for pathology in COMPLETE_ECG_PATHOLOGIES.values():
                weight = 1.0 + pathology['severity'] * 0.5
                severity_weights.append(weight)
            
            # Preencher com peso padrão se necessário
            while len(severity_weights) < num_classes:
                severity_weights.append(1.0)
            
            self.class_weights = torch.tensor(severity_weights[:num_classes]).to(device)
        else:
            self.class_weights = class_weights.to(device)
        
        # Componentes da loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        
        # Label smoothing
        self.label_smoothing = 0.1
    
    def forward(self, predictions, targets, params=None):
        # Apply label smoothing
        smoothed_targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2
        
        # BCE base loss
        bce_loss = self.bce_loss(predictions, smoothed_targets)
        
        # Focal loss modification
        p_t = torch.where(targets == 1, torch.sigmoid(predictions), 1 - torch.sigmoid(predictions))
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        # Apply class weights and focal weight
        weighted_loss = bce_loss * self.class_weights * focal_weight
        
        # Additional penalty for critical conditions
        critical_mask = (targets[:, :10].sum(dim=1) > 0).float()  # Primeiras 10 são críticas
        critical_weight = 1.0 + critical_mask * 0.5
        
        # Final loss
        loss = (weighted_loss * critical_weight.unsqueeze(1)).mean()
        
        # Regularização baseada em parâmetros clínicos (se disponível)
        if params is not None:
            param_reg = self.parameter_regularization(predictions, targets, params)
            loss = loss + 0.1 * param_reg
        
        return loss
    
    def parameter_regularization(self, predictions, targets, params):
        """Regularização baseada em consistência com parâmetros"""
        reg_loss = 0.0
        
        # Heart rate consistency
        hr = params[:, 0] * 100  # Desnormalizar
        
        # Se HR > 150, deveria ter maior probabilidade de taquicardia
        if self.num_classes > 4:  # Assumindo que índice 4 é taquicardia
            tach_prob = torch.sigmoid(predictions[:, 4])
            hr_mask = (hr > 150).float()
            reg_loss += F.mse_loss(tach_prob * hr_mask, hr_mask)
        
        # Se HR < 50, deveria ter maior probabilidade de bradicardia
        if self.num_classes > 5:  # Assumindo que índice 5 é bradicardia
            brad_prob = torch.sigmoid(predictions[:, 5])
            hr_mask = (hr < 50).float()
            reg_loss += F.mse_loss(brad_prob * hr_mask, hr_mask)
        
        return reg_loss

# ==================== TRAINER ULTRA-OTIMIZADO ====================

class UltraOptimizedTrainer:
    """Trainer otimizado para máxima performance"""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Otimizador com diferentes learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': model.rhythm_analyzer.parameters(), 'lr': config.learning_rate},
            {'params': model.morphology_analyzer.parameters(), 'lr': config.learning_rate},
            {'params': model.transformer_integrator.parameters(), 'lr': config.learning_rate},
            {'params': model.parameter_analyzer.parameters(), 'lr': config.learning_rate * 2},
            {'params': model.meta_learner.parameters(), 'lr': config.learning_rate * 0.5},
            {'params': model.gating_network.parameters(), 'lr': config.learning_rate * 0.5}
        ], weight_decay=config.weight_decay)
        
        # Scheduler com warmup
        total_steps = config.num_epochs * 100  # Estimativa
        self.scheduler = self.get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = OptimizedMedicalLoss(model.num_classes, device=device)
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Metrics tracking
        self.best_auc = 0.0
        self.metrics_history = defaultdict(list)
    
    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """Cria scheduler com warmup cosine"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (ecg, params, labels) in enumerate(progress_bar):
            ecg = ecg.to(self.device)
            params = params.to(self.device)
            labels = labels.to(self.device)
            
            # Mixed precision training
            with autocast():
                outputs = self.model(ecg, params)
                loss = self.criterion(outputs, labels, params)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.append(torch.sigmoid(outputs).detach().cpu())
            all_targets.append(labels.cpu())
            
            # Update progress
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_outputs = defaultdict(list)
        
        with torch.no_grad():
            for ecg, params, labels in tqdm(val_loader, desc="Validation"):
                ecg = ecg.to(self.device)
                params = params.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass with all outputs
                with autocast():
                    outputs, individual_outputs = self.model(ecg, params, return_all_outputs=True)
                    loss = self.criterion(outputs, labels, params)
                
                total_loss += loss.item()
                all_predictions.append(torch.sigmoid(outputs).cpu())
                all_targets.append(labels.cpu())
                
                # Store individual model outputs
                for key, value in individual_outputs.items():
                    if key != 'weights':
                        all_outputs[key].append(torch.sigmoid(value).cpu())
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = total_loss / len(val_loader)
        
        # Calculate individual model metrics
        for key in ['rhythm', 'morphology', 'transformer', 'parameters']:
            if key in all_outputs:
                model_preds = torch.cat(all_outputs[key])
                model_auc = self.calculate_auc(model_preds, all_targets)
                metrics[f'{key}_auc'] = model_auc
        
        return metrics
    
    def calculate_metrics(self, predictions, targets):
        """Calcula métricas abrangentes"""
        metrics = {}
        
        # AUC (principal métrica)
        try:
            # AUC médio por classe
            auc_scores = []
            for i in range(predictions.shape[1]):
                if targets[:, i].sum() > 0:  # Só calcular se houver positivos
                    auc = roc_auc_score(targets[:, i], predictions[:, i])
                    auc_scores.append(auc)
            
            metrics['auc'] = np.mean(auc_scores) if auc_scores else 0.0
            
            # AUC macro (considerando todas as classes)
            metrics['auc_macro'] = roc_auc_score(targets, predictions, average='macro')
            
            # AUC weighted
            metrics['auc_weighted'] = roc_auc_score(targets, predictions, average='weighted')
            
        except Exception as e:
            logger.warning(f"Erro ao calcular AUC: {e}")
            metrics['auc'] = 0.0
            metrics['auc_macro'] = 0.0
            metrics['auc_weighted'] = 0.0
        
        # Average Precision
        try:
            metrics['avg_precision'] = average_precision_score(targets, predictions, average='macro')
        except:
            metrics['avg_precision'] = 0.0
        
        # F1 Score com threshold otimizado
        thresholds = self.optimize_thresholds(predictions, targets)
        binary_preds = predictions > thresholds.unsqueeze(0)
        
        metrics['f1_macro'] = f1_score(targets.numpy(), binary_preds.numpy(), average='macro')
        metrics['f1_weighted'] = f1_score(targets.numpy(), binary_preds.numpy(), average='weighted')
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(targets.numpy().flatten(), binary_preds.numpy().flatten())
        
        # Métricas para condições críticas
        critical_indices = [i for i, (code, info) in enumerate(COMPLETE_ECG_PATHOLOGIES.items()) 
                          if info['severity'] >= 4 and i < predictions.shape[1]]
        
        if critical_indices:
            critical_preds = predictions[:, critical_indices]
            critical_targets = targets[:, critical_indices]
            
            # Sensibilidade média para condições críticas
            sensitivities = []
            for i in range(critical_preds.shape[1]):
                if critical_targets[:, i].sum() > 0:
                    tp = ((critical_preds[:, i] > 0.5) & (critical_targets[:, i] == 1)).sum()
                    fn = ((critical_preds[:, i] <= 0.5) & (critical_targets[:, i] == 1)).sum()
                    sens = tp.float() / (tp + fn) if (tp + fn) > 0 else 0
                    sensitivities.append(sens.item())
            
            metrics['critical_sensitivity'] = np.mean(sensitivities) if sensitivities else 0.0
        
        return metrics
    
    def calculate_auc(self, predictions, targets):
        """Calcula AUC robusto"""
        try:
            auc_scores = []
            for i in range(predictions.shape[1]):
                if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                    auc = roc_auc_score(targets[:, i], predictions[:, i])
                    auc_scores.append(auc)
            return np.mean(auc_scores) if auc_scores else 0.0
        except:
            return 0.0
    
    def optimize_thresholds(self, predictions, targets):
        """Otimiza thresholds por classe para máximo F1"""
        thresholds = []
        
        for i in range(predictions.shape[1]):
            if targets[:, i].sum() > 0:
                # Para condições críticas, priorizar sensibilidade
                if i < len(COMPLETE_ECG_PATHOLOGIES) and list(COMPLETE_ECG_PATHOLOGIES.values())[i]['severity'] >= 4:
                    # Threshold mais baixo para aumentar sensibilidade
                    threshold = 0.3
                else:
                    # Otimizar F1 score
                    best_threshold = 0.5
                    best_f1 = 0
                    
                    for t in np.linspace(0.1, 0.9, 20):
                        binary = (predictions[:, i] > t).float()
                        f1 = f1_score(targets[:, i], binary)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = t
                    
                    threshold = best_threshold
            else:
                threshold = 0.5
            
            thresholds.append(threshold)
        
        return torch.tensor(thresholds)
    
    def train(self, train_loader, val_loader, num_epochs):
        """Loop de treinamento principal"""
        logger.info("Iniciando treinamento otimizado...")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"AUC: {train_metrics['auc']:.4f}, "
                       f"F1: {train_metrics['f1_macro']:.4f}")
            
            # Validation
            if (epoch + 1) % self.config.eval_every_n_epochs == 0:
                val_metrics = self.validate(val_loader)
                logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                           f"AUC: {val_metrics['auc']:.4f}, "
                           f"AUC Macro: {val_metrics['auc_macro']:.4f}, "
                           f"Critical Sensitivity: {val_metrics.get('critical_sensitivity', 0):.4f}")
                
                # Log individual model performance
                for model in ['rhythm', 'morphology', 'transformer', 'parameters']:
                    if f'{model}_auc' in val_metrics:
                        logger.info(f"  {model.capitalize()} AUC: {val_metrics[f'{model}_auc']:.4f}")
                
                # Save best model
                if val_metrics['auc'] > self.best_auc:
                    self.best_auc = val_metrics['auc']
                    self.save_checkpoint(epoch, val_metrics)
                    logger.info(f"Novo melhor modelo! AUC: {self.best_auc:.4f}")
                
                # Early stopping check
                if len(self.metrics_history['val_auc']) > self.config.early_stopping_patience:
                    recent_aucs = self.metrics_history['val_auc'][-self.config.early_stopping_patience:]
                    if max(recent_aucs) <= self.best_auc - 0.001:
                        logger.info("Early stopping triggered!")
                        break
                
                self.metrics_history['val_auc'].append(val_metrics['auc'])
        
        return self.best_auc
    
    def save_checkpoint(self, epoch, metrics):
        """Salva checkpoint do modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_auc': self.best_auc,
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, 'ultra_optimized_ecg_best.pth')
        
        # Salvar relatório detalhado
        report = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v 
                       for k, v in metrics.items()},
            'best_auc': float(self.best_auc),
            'model_config': {
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'architecture': 'UltraOptimizedMedicalEnsemble'
            }
        }
        
        with open('training_report_ultra_optimized.json', 'w') as f:
            json.dump(report, f, indent=2)

# ==================== FUNÇÃO PRINCIPAL ====================

def main():
    """Função principal otimizada"""
    parser = argparse.ArgumentParser(description="Sistema ECG Ultra-Otimizado")
    parser.add_argument('--data-path', type=str, 
                       default=r"C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro2\ptbxl_processing\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\processed_npy\ptbxl_100hz")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("SISTEMA ECG ULTRA-OTIMIZADO - TARGET AUC > 99%")
    logger.info("="*80)
    
    # Configuração
    config = MedicalECGConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Carregar dados
    logger.info(f"Carregando dados de: {args.data_path}")
    data_path = Path(args.data_path)
    
    try:
        X = np.load(data_path / 'X.npy')
        Y_multi = np.load(data_path / 'Y_multilabel.npy')
        
        logger.info(f"Dados carregados: X={X.shape}, Y={Y_multi.shape}")
        
        # Ajustar configuração baseada nos dados
        config.signal_length = X.shape[2]
        config.num_leads = X.shape[1]
        num_classes = Y_multi.shape[1]
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        logger.info("Tentando formato alternativo...")
        
        # Tentar outros formatos
        try:
            # Formato consolidado
            X = np.load(data_path / 'X.npy')
            Y = np.load(data_path / 'Y.npy')
            
            # Criar Y_multi dummy se necessário
            if Y.ndim == 1:
                num_classes = len(np.unique(Y))
                Y_multi = np.zeros((len(Y), num_classes))
                for i, label in enumerate(Y):
                    Y_multi[i, label] = 1
            else:
                Y_multi = Y
            
            logger.info(f"Dados carregados (formato alternativo): X={X.shape}, Y_multi={Y_multi.shape}")
            
        except Exception as e2:
            logger.error(f"Falha ao carregar dados: {e2}")
            return
    
    # Verificar e ajustar formato dos dados
    if X.ndim == 2:
        # Assumir que está no formato (samples, features)
        # Reshape para (samples, leads, length)
        n_samples = X.shape[0]
        n_leads = 12
        signal_length = X.shape[1] // n_leads
        X = X.reshape(n_samples, n_leads, signal_length)
        logger.info(f"Dados reformatados para: {X.shape}")
    
    # Cache de parâmetros
    param_cache_file = data_path / 'parameter_cache_optimized.pkl'
    parameter_cache = {}
    
    if param_cache_file.exists():
        logger.info("Carregando cache de parâmetros...")
        try:
            with open(param_cache_file, 'rb') as f:
                parameter_cache = pickle.load(f)
            logger.info(f"Cache carregado: {len(parameter_cache)} entradas")
        except:
            logger.warning("Falha ao carregar cache, será recriado")
    
    # Dividir dados com estratificação
    # Para multilabel, usar a classe mais prevalente para estratificação
    stratify_labels = Y_multi.argmax(axis=1)
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y_multi, test_size=0.2, random_state=42, stratify=stratify_labels
    )
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Criar datasets
    train_dataset = OptimizedECGDataset(
        X_train, Y_train, config,
        parameter_cache={i: parameter_cache.get(i) for i in range(len(X_train)) if i in parameter_cache},
        is_training=True
    )
    
    val_dataset = OptimizedECGDataset(
        X_val, Y_val, config,
        parameter_cache={i: parameter_cache.get(i + len(X_train)) for i in range(len(X_val)) 
                        if (i + len(X_train)) in parameter_cache},
        is_training=False
    )
    
    # Salvar cache atualizado
    if len(train_dataset.parameter_cache) > len(parameter_cache):
        logger.info("Salvando cache de parâmetros atualizado...")
        all_cache = {}
        all_cache.update(train_dataset.parameter_cache)
        all_cache.update({i + len(X_train): v for i, v in val_dataset.parameter_cache.items()})
        
        with open(param_cache_file, 'wb') as f:
            pickle.dump(all_cache, f)
    
    # DataLoaders com sampler balanceado
    # Calcular pesos para balanceamento
    class_counts = Y_train.sum(axis=0)
    class_weights = 1.0 / (class_counts + 1)
    sample_weights = (Y_train * class_weights).sum(axis=1)
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Criar modelo
    logger.info("Criando modelo ultra-otimizado...")
    model = UltraOptimizedMedicalEnsemble(num_classes, config)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total de parâmetros: {total_params:,}")
    logger.info(f"Parâmetros treináveis: {trainable_params:,}")
    
    # Criar trainer
    trainer = UltraOptimizedTrainer(model, config, device)
    
    # Ajustar steps do scheduler
    trainer.scheduler = trainer.get_cosine_schedule_with_warmup(
        trainer.optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs
    )
    
    # Treinar
    logger.info("\nIniciando treinamento...")
    logger.info(f"Configuração:")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Epochs: {config.num_epochs}")
    logger.info(f"  - Warmup steps: {config.warmup_steps}")
    
    best_auc = trainer.train(train_loader, val_loader, config.num_epochs)
    
    logger.info("\n" + "="*80)
    logger.info("TREINAMENTO CONCLUÍDO!")
    logger.info("="*80)
    logger.info(f"Melhor AUC alcançado: {best_auc:.4f}")
    logger.info("Modelo salvo: ultra_optimized_ecg_best.pth")
    logger.info("Relatório: training_report_ultra_optimized.json")
    
    # Teste final
    logger.info("\nCarregando melhor modelo para teste final...")
    checkpoint = torch.load('ultra_optimized_ecg_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics = trainer.validate(val_loader)
    logger.info("\nMétricas finais:")
    for key, value in final_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")

if __name__ == "__main__":
    main()
