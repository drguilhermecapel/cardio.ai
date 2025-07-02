#!/usr/bin/env python3
"""
Sistema de Treinamento ECG com Deep Learning
"""

# Importações do sistema Python
import os
import sys
import time
import json
import yaml
import logging
import warnings
import argparse
import datetime
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import defaultdict, Counter, OrderedDict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from functools import partial, wraps # Adicionado wraps
import pickle
import random
import shutil
import math
import copy
from enum import Enum # Adicionado Enum
import threading # Adicionado threading

# Importações numéricas e científicas
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy import integrate
from scipy import stats
from scipy.stats import zscore
from scipy.interpolate import interp1d
import pywt

# Importações de Machine Learning
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score, # Adicionado
    precision_recall_fscore_support # Adicionado
)
from sklearn.utils.class_weight import compute_class_weight

# Importações do PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, 
    ReduceLROnPlateau, 
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Importações de utilidades e visualização
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# Configurações globais
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configurar matplotlib para não mostrar plots
plt.switch_backend("Agg")

# Seed para reprodutibilidade
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Importações opcionais com verificação
try:
    import wfdb # Adicionado import
    WFDB_AVAILABLE = True
except ImportError:
    print("AVISO: wfdb-python não instalado. Algumas funcionalidades estarão limitadas.")
    print("Instale com: pip install wfdb")
    WFDB_AVAILABLE = False

try:
    import pywt # Adicionado import
    PYWT_AVAILABLE = True
except ImportError:
    print("AVISO: PyWavelets não instalado. Denoising wavelet limitado.")
    print("Instale com: pip install PyWavelets")
    PYWT_AVAILABLE = False

# Configurar caminhos com validação e segurança
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

# Validação de segurança de caminhos - Compatível com Python 3.7+
def validate_path(path: Path, base_dir: Path) -> Path:
    """Valida e sanitiza caminhos para prevenir path traversal"""
    resolved_path = path.resolve()
    resolved_base = base_dir.resolve()
    
    # Compatibilidade com Python < 3.9
    try:
        # Python 3.9+
        resolved_path.relative_to(resolved_base)
    except ValueError:
        raise ValueError(f"Caminho inválido: {path}")
    except AttributeError:
        # Python < 3.9 - verificação manual
        if not str(resolved_path).startswith(str(resolved_base)):
            raise ValueError(f"Caminho inválido: {path}")
    
    return resolved_path

BACKEND_DIR = SCRIPT_DIR  # Usa o diretório atual do projeto
PROJECT_ROOT = BACKEND_DIR.parent if (BACKEND_DIR.parent / 'train_ecg.py').exists() else BACKEND_DIR

sys.path.insert(0, str(BACKEND_DIR))

# Suprimir warnings não críticos
warnings.filterwarnings('ignore', category=UserWarning)

# ==================== CONSTANTES E CONFIGURAÇÕES ====================

# Constantes médicas com documentação científica
NOISE_MEDIAN_FACTOR = 0.6745  # Fator para MAD (Median Absolute Deviation)
MIN_HEART_RATE = 20  # bpm - limite fisiológico inferior
MAX_HEART_RATE = 300  # bpm - limite fisiológico superior
MIN_QT_INTERVAL = 200  # ms
MAX_QT_INTERVAL = 600  # ms
MIN_PR_INTERVAL = 50  # ms
MAX_PR_INTERVAL = 400  # ms

# Requisitos clínicos baseados em diretrizes
CLINICAL_REQUIREMENTS = {
    'target_auc': 0.90,
    'min_sensitivity': 0.85,
    'min_specificity': 0.85,
    'max_inference_time': 100,  # ms
    'calibration_factor': 10.0,  # mm/mV padrão
}

# Critérios diagnósticos atualizados conforme diretrizes 2023
DIAGNOSTIC_CRITERIA = {
    'lvh': {
        'sokolow_lyon': 35,  # mm (3.5 mV)
        'cornell_male': 28,  # mm (2.8 mV)
        'cornell_female': 20,  # mm (2.0 mV)
        'romhilt_estes': 5,  # pontos
    },
    'rvh': {
        'r_v1': 7,  # mm
        'r_v1_s_v5v6': 11,  # mm
        'r_v1_s_v5v6_ratio': 1.0,
    },
    'lae': {
        'p_duration': 120,  # ms
        'p_terminal_force': -0.04,  # mm·s
        'p_amplitude': 2.5,  # mm
    },
    'rae': {
        'p_amplitude_lead2': 2.5,  # mm
        'p_amplitude_v1': 1.5,  # mm
    },
    'mi': {
        'q_duration': 40,  # ms
        'q_r_ratio': 0.25,
        'st_elevation_j': 0.1,  # mV limb leads
        'st_elevation_j_precordial': 0.2,  # mV precordial
    }
}

# Thread-safe cache decorator
def thread_safe_cache(maxsize=128):
    """Decorator para cache thread-safe com LRU"""
    def decorator(func):
        cache = OrderedDict()
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            
            with lock:
                if key in cache:
                    # Move to end (LRU)
                    cache.move_to_end(key)
                    return cache[key]
            
            result = func(*args, **kwargs)
            
            with lock:
                cache[key] = result
                if len(cache) > maxsize:
                    cache.popitem(last=False)
            
            return result
        
        wrapper.cache_clear = lambda: cache.clear()
        return wrapper
    
    return decorator

# Enumeração de categorias de patologias
class SCPCategory(Enum):
    RHYTHM = "rhythm"
    CONDUCTION = "conduction"
    ISCHEMIA = "ischemia"
    HYPERTROPHY = "hypertrophy"
    AXIS = "axis"
    REPOLARIZATION = "repolarization"
    OTHER = "other"

# Dicionário completo de condições SCP-ECG com critérios diagnósticos
SCP_ECG_CONDITIONS = {
    'NORM': {'name': 'Normal ECG', 'category': SCPCategory.RHYTHM, 'severity': 0},
    'SR': {'name': 'Sinus rhythm', 'category': SCPCategory.RHYTHM, 'severity': 0},
    'AFIB': {'name': 'Atrial fibrillation', 'category': SCPCategory.RHYTHM, 'severity': 3},
    'AFLT': {'name': 'Atrial flutter', 'category': SCPCategory.RHYTHM, 'severity': 3},
    'STACH': {'name': 'Sinus tachycardia', 'category': SCPCategory.RHYTHM, 'severity': 1},
    'SBRAD': {'name': 'Sinus bradycardia', 'category': SCPCategory.RHYTHM, 'severity': 1},
    'PAC': {'name': 'Premature atrial contraction', 'category': SCPCategory.RHYTHM, 'severity': 1},
    'PVC': {'name': 'Premature ventricular contraction', 'category': SCPCategory.RHYTHM, 'severity': 2},
    'VT': {'name': 'Ventricular tachycardia', 'category': SCPCategory.RHYTHM, 'severity': 4},
    'SVTACH': {'name': 'Supraventricular tachycardia', 'category': SCPCategory.RHYTHM, 'severity': 2},
    'PACED': {'name': 'Paced rhythm', 'category': SCPCategory.RHYTHM, 'severity': 1},
    
    'AVB1': {'name': 'First degree AV block', 'category': SCPCategory.CONDUCTION, 'severity': 1},
    'AVB2': {'name': 'Second degree AV block', 'category': SCPCategory.CONDUCTION, 'severity': 2},
    'AVB3': {'name': 'Third degree AV block', 'category': SCPCategory.CONDUCTION, 'severity': 4},
    'RBBB': {'name': 'Right bundle branch block', 'category': SCPCategory.CONDUCTION, 'severity': 2},
    'LBBB': {'name': 'Left bundle branch block', 'category': SCPCategory.CONDUCTION, 'severity': 2},
    'LAFB': {'name': 'Left anterior fascicular block', 'category': SCPCategory.CONDUCTION, 'severity': 1},
    'LPFB': {'name': 'Left posterior fascicular block', 'category': SCPCategory.CONDUCTION, 'severity': 1},
    'WPW': {'name': 'Wolff-Parkinson-White syndrome', 'category': SCPCategory.CONDUCTION, 'severity': 3},
    
    'MI': {'name': 'Myocardial infarction', 'category': SCPCategory.ISCHEMIA, 'severity': 4},
    'AMI': {'name': 'Acute myocardial infarction', 'category': SCPCategory.ISCHEMIA, 'severity': 5},
    'STTC': {'name': 'ST-T changes', 'category': SCPCategory.ISCHEMIA, 'severity': 2},
    'STE': {'name': 'ST elevation', 'category': SCPCategory.ISCHEMIA, 'severity': 4},
    'STD': {'name': 'ST depression', 'category': SCPCategory.ISCHEMIA, 'severity': 3},
    
    'LVH': {'name': 'Left ventricular hypertrophy', 'category': SCPCategory.HYPERTROPHY, 'severity': 2},
    'RVH': {'name': 'Right ventricular hypertrophy', 'category': SCPCategory.HYPERTROPHY, 'severity': 2},
    'LAE': {'name': 'Left atrial enlargement', 'category': SCPCategory.HYPERTROPHY, 'severity': 1},
    'RAE': {'name': 'Right atrial enlargement', 'category': SCPCategory.HYPERTROPHY, 'severity': 1},
    
    'LAD': {'name': 'Left axis deviation', 'category': SCPCategory.AXIS, 'severity': 1},
    'RAD': {'name': 'Right axis deviation', 'category': SCPCategory.AXIS, 'severity': 1},
    'ERAD': {'name': 'Extreme right axis deviation', 'category': SCPCategory.AXIS, 'severity': 2},
    
    'TWI': {'name': 'T wave inversion', 'category': SCPCategory.REPOLARIZATION, 'severity': 2},
    'TWF': {'name': 'T wave flattening', 'category': SCPCategory.REPOLARIZATION, 'severity': 1},
    'LQT': {'name': 'Long QT syndrome', 'category': SCPCategory.REPOLARIZATION, 'severity': 4},
    'SQT': {'name': 'Short QT syndrome', 'category': SCPCategory.REPOLARIZATION, 'severity': 3},
    'BRUG': {'name': 'Brugada syndrome', 'category': SCPCategory.REPOLARIZATION, 'severity': 4},
    'ERS': {'name': 'Early repolarization syndrome', 'category': SCPCategory.REPOLARIZATION, 'severity': 2},
    
    'LOWV': {'name': 'Low voltage', 'category': SCPCategory.OTHER, 'severity': 1},
    'PRWP': {'name': 'Poor R wave progression', 'category': SCPCategory.OTHER, 'severity': 1},
    'DIG': {'name': 'Digitalis effect', 'category': SCPCategory.OTHER, 'severity': 1},
    'ELEV': {'name': 'Electrode misplacement', 'category': SCPCategory.OTHER, 'severity': 0},
}

# ==================== CONFIGURAÇÃO OTIMIZADA ====================

@dataclass
class EnhancedECGAnalysisConfig:
    """Configuração otimizada e validada para análise de ECG"""
    # Parâmetros do sinal
    sampling_rate: int = 500
    signal_length: int = 5000
    num_leads: int = 12
    
    # Filtros adaptativos
    use_adaptive_filtering: bool = True
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0
    notch_freq: float = 60.0
    notch_quality: float = 30.0
    
    # Wavelet denoising
    use_wavelet_denoising: bool = True
    wavelet_name: str = 'db4'
    wavelet_level: int = 5
    wavelet_threshold_method: str = 'soft'
    
    # Augmentação otimizada
    augmentation_prob: float = 0.7
    amplitude_scaling: Tuple[float, float] = (0.8, 1.2)
    noise_types: List[str] = field(default_factory=lambda: ['gaussian', 'baseline', 'powerline', 'muscle'])
    time_warping: bool = True
    lead_dropout: bool = True
    max_lead_dropout: int = 3
    
    # Treinamento otimizado
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    early_stopping_patience: int = 15
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: Optional[float] = None
    use_focal_loss: bool = True
    label_smoothing: float = 0.1
    test_size: float = 0.2
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    
    # Performance
    num_workers: int = 4
    use_cache: bool = True
    cache_size: int = 1000
    use_parallel: bool = True
    max_parallel_workers: int = 4
    
    # Thresholds médicos adaptativos
    use_adaptive_thresholds: bool = True
    st_elevation_threshold_limb: float = 0.1  # mV
    st_elevation_threshold_precordial: float = 0.2  # mV
    
    # Validação médica
    validate_physiological_ranges: bool = True
    min_valid_amplitude: float = 0.05  # mV
    max_valid_amplitude: float = 5.0  # mV
    
    # EMA e TTA
    use_ema: bool = True
    ema_decay: float = 0.999
    use_tta: bool = True
    tta_augmentations: int = 5
    
    # Calibração
    calibration_factor: float = 10.0  # mm/mV padrão
    
    def __post_init__(self):
        """Validação pós-inicialização"""
        assert self.sampling_rate > 0, "Taxa de amostragem deve ser positiva"
        assert self.signal_length > 0, "Comprimento do sinal deve ser positivo"
        assert 0 < self.augmentation_prob <= 1, "Probabilidade de augmentação deve estar entre 0 e 1"
        assert self.batch_size > 0, "Tamanho do batch deve ser positivo"
        assert self.num_epochs > 0, "Número de épocas deve ser positivo"

# ==================== LOGGER CLÍNICO ====================

class ClinicalLogger:
    """Logger especializado para eventos clínicos e auditoria médica"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or PROJECT_ROOT / 'logs' / 'clinical'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Arquivo de log específico para eventos clínicos
        self.clinical_log_file = self.log_dir / f'clinical_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        self.events = []
        self._lock = threading.Lock()
        
        # Contadores de performance
        self.pathology_counts = defaultdict(int)
        self.pathology_tp = defaultdict(int)
        self.pathology_fp = defaultdict(int)
        self.pathology_fn = defaultdict(int)
        self.pathology_tn = defaultdict(int)
    
    def log_clinical_event(self, event_type: str, data: Dict[str, Any]):
        """Registra evento clínico com timestamp e dados"""
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        
        with self._lock:
            self.events.append(event)
            
            # Salvar incrementalmente
            with open(self.clinical_log_file, 'a') as f:
                json.dump(event, f)
                f.write('\n')
    
    def log_pathology_performance(self, pathology_code: str, metrics: Dict[str, float]):
        """Registra performance por patologia"""
        self.log_clinical_event('pathology_performance', {
            'pathology_code': pathology_code,
            'pathology_name': SCP_ECG_CONDITIONS.get(pathology_code, {}).get('name', 'Unknown'),
            'metrics': metrics
        })
        
        # Atualizar contadores
        with self._lock:
            if 'tp' in metrics:
                self.pathology_tp[pathology_code] += metrics['tp']
            if 'fp' in metrics:
                self.pathology_fp[pathology_code] += metrics['fp']
            if 'fn' in metrics:
                self.pathology_fn[pathology_code] += metrics['fn']
            if 'tn' in metrics:
                self.pathology_tn[pathology_code] += metrics['tn'] # Corrigido

    def get_overall_performance(self) -> Dict[str, Any]:
        """Calcula e retorna métricas de performance gerais"""
        total_tp = sum(self.pathology_tp.values())
        total_fp = sum(self.pathology_fp.values())
        total_fn = sum(self.pathology_fn.values())
        total_tn = sum(self.pathology_tn.values())

        # Evitar divisão por zero
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0

        return {
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'total_true_negatives': total_tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        }

    def save_log(self):
        """Salva todos os eventos registrados em um arquivo JSON"""
        with self._lock:
            with open(self.clinical_log_file, 'w') as f:
                json.dump(self.events, f, indent=4)

    def load_log(self):
        """Carrega eventos de um arquivo JSON"""
        if self.clinical_log_file.exists():
            with self._lock:
                with open(self.clinical_log_file, 'r') as f:
                    self.events = [json.loads(line) for line in f]

# ==================== UTILITÁRIOS DE PRÉ-PROCESSAMENTO ====================

class Preprocessor:
    """Classe para pré-processamento de sinais ECG, incluindo filtragem e normalização"""
    def __init__(self, config: EnhancedECGAnalysisConfig):
        self.config = config

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Aplica filtro passa-banda ao sinal"""
        nyquist = 0.5 * self.config.sampling_rate
        low = self.config.bandpass_low / nyquist
        high = self.config.bandpass_high / nyquist
        b, a = scipy_signal.butter(3, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, signal)

    def notch_filter(self, signal: np.ndarray) -> np.ndarray:
        """Aplica filtro notch para remover ruído de linha de energia"""
        nyquist = 0.5 * self.config.sampling_rate
        freq = self.config.notch_freq / nyquist
        q = self.config.notch_quality
        b, a = scipy_signal.iirnotch(freq, q)
        return scipy_signal.filtfilt(b, a, signal)

    def wavelet_denoise(self, signal: np.ndarray) -> np.ndarray:
        """Aplica denoising wavelet ao sinal"""
        if not PYWT_AVAILABLE:
            logger.warning("PyWavelets não disponível para denoising wavelet.")
            return signal
        
        coeffs = pywt.wavedec(signal, self.config.wavelet_name, level=self.config.wavelet_level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        if self.config.wavelet_threshold_method == 'soft':
            coeffs[1:] = (pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:])
        elif self.config.wavelet_threshold_method == 'hard':
            coeffs[1:] = (pywt.threshold(c, threshold, mode='hard') for c in coeffs[1:])
        
        return pywt.waverec(coeffs, self.config.wavelet_name)

    def normalize_signal(self, signal: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """Normaliza o sinal usando o método especificado"""
        if method == 'zscore':
            return zscore(signal)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            return scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        elif method == 'robust':
            scaler = RobustScaler()
            return scaler.fit_transform(signal.reshape(-1, 1)).flatten()
        else:
            raise ValueError(f"Método de normalização desconhecido: {method}")

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Processa o sinal ECG aplicando filtros e normalização"""
        processed_signal = signal.copy()
        if self.config.use_adaptive_filtering:
            processed_signal = self.bandpass_filter(processed_signal)
            processed_signal = self.notch_filter(processed_signal)
        if self.config.use_wavelet_denoising:
            processed_signal = self.wavelet_denoise(processed_signal)
        
        # Normalização é geralmente a última etapa para manter a escala dos filtros
        processed_signal = self.normalize_signal(processed_signal)
        return processed_signal

# ==================== AUMENTAÇÃO DE DADOS ====================

class Augmenter:
    """Classe para aumento de dados de ECG, aplicando diversas transformações"""
    def __init__(self, config: EnhancedECGAnalysisConfig):
        self.config = config

    def apply_augmentation(self, signal: np.ndarray) -> np.ndarray:
        """Aplica augmentação ao sinal com uma dada probabilidade"""
        if random.random() < self.config.augmentation_prob:
            signal = self._random_amplitude_scaling(signal)
            signal = self._add_random_noise(signal)
            if self.config.time_warping:
                signal = self._time_warping(signal)
            if self.config.lead_dropout:
                signal = self._lead_dropout(signal)
        return signal

    def _random_amplitude_scaling(self, signal: np.ndarray) -> np.ndarray:
        """Ajusta a amplitude do sinal aleatoriamente"""
        scale_factor = random.uniform(self.config.amplitude_scaling[0], self.config.amplitude_scaling[1])
        return signal * scale_factor

    def _add_random_noise(self, signal: np.ndarray) -> np.ndarray:
        """Adiciona ruído aleatório ao sinal"""
        noise_type = random.choice(self.config.noise_types)
        if noise_type == 'gaussian':
            noise = np.random.normal(0, 0.05, signal.shape) # Exemplo: desvio padrão 0.05
        elif noise_type == 'baseline':
            noise = scipy_signal.savgol_filter(np.random.normal(0, 0.02, signal.shape), 51, 3) # Baseline wander
        elif noise_type == 'powerline':
            t = np.linspace(0, len(signal) / self.config.sampling_rate, len(signal), endpoint=False)
            noise = 0.01 * np.sin(2 * np.pi * self.config.notch_freq * t) # Ruído de 60Hz
        elif noise_type == 'muscle':
            noise = np.random.normal(0, 0.03, signal.shape) * np.random.randint(0, 2, signal.shape) # Ruído muscular intermitente
        else:
            noise = np.zeros(signal.shape)
        return signal + noise

    def _time_warping(self, signal: np.ndarray) -> np.ndarray:
        """Aplica distorção temporal ao sinal"""
        # Exemplo simples de time warping usando interpolação
        original_indices = np.arange(len(signal))
        warped_indices = np.sort(original_indices + np.random.normal(0, 0.1 * len(signal), len(signal)))
        warped_indices = np.clip(warped_indices, 0, len(signal) - 1)
        
        interp_func = interp1d(original_indices, signal, kind='linear', bounds_error=False, fill_value="extrapolate")
        return interp_func(warped_indices)

    def _lead_dropout(self, signal: np.ndarray) -> np.ndarray:
        """Aleatoriamente zera alguns canais (leads) do ECG"""
        if signal.ndim == 1: # Se for um único lead, não faz sentido aplicar dropout de leads
            return signal

        num_leads_to_drop = random.randint(1, min(self.config.max_lead_dropout, signal.shape[0]))
        leads_to_drop = random.sample(range(signal.shape[0]), num_leads_to_drop)
        
        augmented_signal = signal.copy()
        augmented_signal[leads_to_drop, :] = 0  # Zera os leads selecionados
        return augmented_signal

# ==================== DATASET E DATALOADER ====================

class ECGDataset(Dataset):
    """Dataset para carregar e pré-processar dados de ECG"""
    def __init__(self, data: pd.DataFrame, labels: pd.Series, config: EnhancedECGAnalysisConfig, is_train: bool = True):
        self.data = data
        self.labels = labels
        self.config = config
        self.is_train = is_train
        self.preprocessor = Preprocessor(config)
        self.augmenter = Augmenter(config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data.iloc[idx].values.astype(np.float32).reshape(self.config.num_leads, -1)
        label = self.labels.iloc[idx]

        # Pré-processamento
        processed_signal = np.array([self.preprocessor.process(s) for s in signal])

        # Augmentação (apenas no treinamento)
        if self.is_train:
            processed_signal = np.array([self.augmenter.apply_augmentation(s) for s in processed_signal])
        
        # Redimensionar para o comprimento do sinal configurado, se necessário
        if processed_signal.shape[1] != self.config.signal_length:
            # Interpolar ou truncar/preencher
            new_signal = np.zeros((self.config.num_leads, self.config.signal_length), dtype=np.float32)
            for i in range(self.config.num_leads):
                interp_func = interp1d(
                    np.linspace(0, 1, processed_signal.shape[1]), 
                    processed_signal[i,:], 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value="extrapolate"
                )
                new_signal[i,:] = interp_func(np.linspace(0, 1, self.config.signal_length))
            processed_signal = new_signal

        return torch.tensor(processed_signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ==================== MODELOS DE DEEP LEARNING ====================

class BasicCNN(nn.Module):
    """CNN básica para classificação de ECG"""
    def __init__(self, num_leads: int, signal_length: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(num_leads, 32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calcular o tamanho da saída após as camadas convolucionais e de pooling
        self.feature_length = self._get_conv_output_length(signal_length)
        self.fc1 = nn.Linear(128 * self.feature_length, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_conv_output_length(self, length):
        length = (length + 2*2 - 5) // 1 + 1 # conv1
        length = (length - 2) // 2 + 1 # pool1
        length = (length + 2*2 - 5) // 1 + 1 # conv2
        length = (length - 2) // 2 + 1 # pool2
        length = (length + 2*2 - 5) // 1 + 1 # conv3
        length = (length - 2) // 2 + 1 # pool3
        return length

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1) # Achatar
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetECG(nn.Module):
    """ResNet adaptada para classificação de ECG"""
    def __init__(self, num_leads: int, signal_length: int, num_classes: int, num_blocks: List[int] = [2, 2, 2, 2]):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNetBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        x = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output))
        return x

class ECGTransformer(nn.Module):
    """Transformer adaptado para classificação de ECG"""
    def __init__(self, num_leads: int, signal_length: int, num_classes: int, embed_dim: int = 128, num_heads: int = 8, ff_dim: int = 512, num_transformer_blocks: int = 4):
        super().__init__()
        self.conv_embed = nn.Conv1d(num_leads, embed_dim, kernel_size=5, stride=1, padding=2)
        self.pos_embedding = nn.Parameter(torch.randn(1, signal_length, embed_dim))
        
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_transformer_blocks)
        ])
        
        self.classifier = nn.Linear(embed_dim * signal_length, num_classes)

    def forward(self, x):
        x = self.conv_embed(x) # (batch_size, embed_dim, signal_length)
        x = x.permute(0, 2, 1) # (batch_size, signal_length, embed_dim)
        x += self.pos_embedding
        
        x = self.transformer_blocks(x)
        
        x = x.view(x.size(0), -1) # Achatar
        x = self.classifier(x)
        return x

# ==================== FUNÇÕES DE PERDA ====================

class FocalLoss(nn.Module):
    """Focal Loss para lidar com desequilíbrio de classes"""
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt)**self.gamma * BCE_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            F_loss = alpha_t * F_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# ==================== TREINAMENTO E AVALIAÇÃO ====================

class Trainer:
    """Classe para gerenciar o treinamento e avaliação do modelo"""
    def __init__(self, model: nn.Module, config: EnhancedECGAnalysisConfig, class_weights: Optional[torch.Tensor] = None, clinical_logger: Optional[ClinicalLogger] = None):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        if self.config.use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights, gamma=self.config.focal_loss_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            
        self.scaler = GradScaler() if self.config.use_mixed_precision else None
        self.clinical_logger = clinical_logger or ClinicalLogger()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)

            for param in self.model.parameters():
                param.grad = None # Zera os gradientes

            with autocast(enabled=self.config.use_mixed_precision):
                output = self.model(data)
                loss = self.criterion(output, target)
                loss = loss / self.config.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Clipagem de gradiente
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
            total_loss += loss.item() * self.config.gradient_accumulation_steps
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0
        with torch.no_grad():
            for data, target in tqdm(dataloader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                with autocast(enabled=self.config.use_mixed_precision):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                total_loss += loss.item()
                all_preds.extend(output.argmax(dim=1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        
        # Calcular métricas
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        # roc_auc = roc_auc_score(all_targets, F.softmax(torch.tensor(output.cpu().numpy()), dim=1), multi_class='ovr') # Precisa de scores, não de preds
        
        # Para ROC AUC, precisamos das probabilidades, não das classes preditas
        # Coletar probabilidades durante a avaliação
        all_probs = []
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                with autocast(enabled=self.config.use_mixed_precision):
                    output = self.model(data)
                all_probs.extend(F.softmax(output, dim=1).cpu().numpy())
        
        all_probs = np.array(all_probs)
        
        # Certificar-se de que all_targets e all_probs têm o mesmo número de amostras
        if len(all_targets) != len(all_probs):
            logger.error("Inconsistência no número de amostras entre targets e probabilidades.")
            # Lidar com o erro, talvez truncar ou levantar uma exceção
            min_len = min(len(all_targets), len(all_probs))
            all_targets = all_targets[:min_len]
            all_probs = all_probs[:min_len]

        roc_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, model_save_path: Path):
        best_val_f1 = -1
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            train_loss = self.train_epoch(train_dataloader)
            val_metrics = self.evaluate(val_dataloader)

            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_score']:.4f}, Val ROC AUC: {val_metrics['roc_auc']:.4f}")

            # Logar performance clínica
            self.clinical_logger.log_clinical_event('epoch_summary', {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_f1_score': val_metrics['f1_score'],
                'val_roc_auc': val_metrics['roc_auc']
            })

            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                torch.save(self.model.state_dict(), model_save_path)
                logger.info(f"Modelo salvo em {model_save_path} com F1-score de {best_val_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"Paciência: {patience_counter}/{self.config.early_stopping_patience}")
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info("Early stopping ativado.")
                    break
        
        self.clinical_logger.save_log()

# ==================== FUNÇÃO PRINCIPAL ====================

def main(config_path: Optional[Path] = None):
    set_seed(42)

    # Carregar configuração
    if config_path:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = EnhancedECGAnalysisConfig(**config_dict)
    else:
        config = EnhancedECGAnalysisConfig()

    logger.info(f"Configuração utilizada: {config}")

    # Simular carregamento de dados (substituir por dados reais)
    # Gerar dados sintéticos para teste
    num_samples = 1000
    num_leads = config.num_leads
    signal_length = config.signal_length
    num_classes = len(SCP_ECG_CONDITIONS) # Número de classes baseado nas condições SCP-ECG

    # Dados de ECG simulados (num_samples, num_leads * signal_length)
    synthetic_data = np.random.rand(num_samples, num_leads * signal_length)
    # Rótulos simulados
    synthetic_labels = np.random.randint(0, num_classes, num_samples)

    df_data = pd.DataFrame(synthetic_data)
    s_labels = pd.Series(synthetic_labels)

    # Dividir dados em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(df_data, s_labels, test_size=config.test_size, random_state=42, stratify=s_labels)

    # Calcular pesos de classe para lidar com desequilíbrio
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Criar Datasets e Dataloaders
    train_dataset = ECGDataset(X_train, y_train, config, is_train=True)
    val_dataset = ECGDataset(X_val, y_val, config, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Inicializar modelo (exemplo: BasicCNN)
    model = BasicCNN(num_leads, signal_length, num_classes) # Ou ResNetECG, ECGTransformer
    # model = ResNetECG(num_leads, signal_length, num_classes)
    # model = ECGTransformer(num_leads, signal_length, num_classes)

    # Inicializar ClinicalLogger
    clinical_logger = ClinicalLogger()

    # Inicializar Trainer
    trainer = Trainer(model, config, class_weights, clinical_logger)

    # Caminho para salvar o modelo
    model_save_path = PROJECT_ROOT / 'best_ecg_model.pth'

    # Treinar o modelo
    trainer.train(train_dataloader, val_dataloader, model_save_path)

    logger.info("Treinamento concluído.")

    # Carregar o melhor modelo e avaliar no conjunto de validação
    model.load_state_dict(torch.load(model_save_path))
    final_metrics = trainer.evaluate(val_dataloader)
    logger.info(f"Métricas finais no conjunto de validação: {final_metrics}")

    # Exemplo de uso do logger clínico para performance por patologia (simulado)
    for i in range(num_classes):
        pathology_code = list(SCP_ECG_CONDITIONS.keys())[i]
        # Simular algumas métricas para cada patologia
        simulated_metrics = {
            'tp': random.randint(10, 50),
            'fp': random.randint(1, 10),
            'fn': random.randint(1, 10),
            'tn': random.randint(100, 200),
        }
        clinical_logger.log_pathology_performance(pathology_code, simulated_metrics)
    
    overall_performance = clinical_logger.get_overall_performance()
    logger.info(f"Performance clínica geral: {overall_performance}")

    clinical_logger.save_log()
    logger.info(f"Log clínico salvo em: {clinical_logger.clinical_log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema de Treinamento ECG com Deep Learning")
    parser.add_argument('--config', type=str, help='Caminho para o arquivo de configuração YAML', default=None)
    args = parser.parse_args()
    
    try:
        main(Path(args.config) if args.config else None)
    except Exception as e:
        logger.error(f"Ocorreu um erro: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


