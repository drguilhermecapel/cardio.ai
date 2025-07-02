#!/usr/bin/env python3
"""
Sistema Ensemble ECG Otimizado - Versão Corrigida para Windows
Corrige problemas de multiprocessing e serialização
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

# Importações das versões anteriores (scipy, sklearn, etc.)
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

# Configurações
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IMPORTANTE: Para Windows
if sys.platform == 'win32':
    # Forçar spawn method
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

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

# ==================== IMPORTS DOS COMPONENTES ANTERIORES ====================
# Importar todas as classes e configurações do arquivo anterior
# Para brevidade, vou incluir apenas as modificações necessárias

# Copie aqui todas as definições de:
# - MedicalECGConfig
# - COMPLETE_ECG_PATHOLOGIES
# - ECGParameters
# - AdvancedWaveDetector
# - ClinicalParameterExtractor
# - Todos os modelos de rede neural

# ==================== DATASET CORRIGIDO PARA WINDOWS ====================

class WindowsOptimizedECGDataset(Dataset):
    """Dataset otimizado para funcionar corretamente no Windows"""
    
    def __init__(self, X, Y, config, is_training=True):
        self.X = X
        self.Y = Y
        self.config = config
        self.is_training = is_training
        
        # Não usar objetos complexos que não são picklable
        self.sampling_rate = config.sampling_rate
        self.augment_prob = config.augment_prob if is_training else 0.0
        
        # Normalização - fazer fit aqui e salvar apenas os parâmetros
        normalizer = RobustScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        normalizer.fit(X_reshaped)
        
        # Salvar apenas os parâmetros, não o objeto
        self.scale_ = normalizer.scale_
        self.center_ = normalizer.center_
        
        logger.info(f"Dataset criado: {len(X)} amostras, training={is_training}")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # ECG signal
        ecg = self.X[idx].copy()
        
        # Normalização manual usando parâmetros salvos
        ecg_reshaped = ecg.reshape(-1, ecg.shape[-1])
        ecg_normalized = (ecg_reshaped - self.center_) / self.scale_
        ecg = ecg_normalized.reshape(ecg.shape)
        
        # Parâmetros clínicos simplificados
        # Não usar objetos complexos, calcular inline
        params = self._extract_simple_params(ecg)
        
        # Augmentação durante treino
        if self.is_training and np.random.rand() < self.augment_prob:
            ecg = self._apply_simple_augmentation(ecg)
        
        # Converter para tensores
        ecg_tensor = torch.FloatTensor(ecg)
        params_tensor = torch.FloatTensor(params)
        
        # Labels
        if self.Y.ndim > 1:  # multilabel
            label = torch.FloatTensor(self.Y[idx])
        else:
            label = torch.LongTensor([self.Y[idx]])
        
        return ecg_tensor, params_tensor, label
    
    def _extract_simple_params(self, ecg):
        """Extração simplificada de parâmetros"""
        # Usar apenas cálculos básicos que são picklable
        params = []
        
        # Heart rate estimation simples
        lead_ii = ecg[1] if ecg.shape[0] > 1 else ecg[0]
        
        # Detecção R simples
        try:
            # Filtro passa-banda básico
            nyquist = self.sampling_rate / 2
            low = 5 / nyquist
            high = min(15 / nyquist, 0.99)
            
            if low < high:
                b, a = butter(2, [low, high], btype='band')
                filtered = filtfilt(b, a, lead_ii)
                
                # Encontrar picos
                peaks, _ = find_peaks(filtered, distance=int(0.2 * self.sampling_rate))
                
                if len(peaks) > 1:
                    rr_intervals = np.diff(peaks) / self.sampling_rate
                    heart_rate = 60 / np.mean(rr_intervals)
                    heart_rate = np.clip(heart_rate, 30, 200)
                    hrv = np.std(rr_intervals * 1000)
                else:
                    heart_rate = 75.0
                    hrv = 50.0
            else:
                heart_rate = 75.0
                hrv = 50.0
        except:
            heart_rate = 75.0
            hrv = 50.0
        
        # Parâmetros básicos normalizados
        params = [
            heart_rate / 100.0,
            hrv / 100.0,
            0.9,  # rhythm_regularity placeholder
            90.0 / 100.0,  # qrs_duration placeholder
            200.0 / 200.0,  # pr_interval placeholder
            400.0 / 400.0,  # qt_interval placeholder
            420.0 / 400.0,  # qtc_interval placeholder
            0.0,  # qrs_axis placeholder
            20.0 / 50.0,  # sokolow_lyon_index placeholder
            15.0 / 30.0,  # cornell_index placeholder
        ]
        
        return np.array(params, dtype=np.float32)
    
    def _apply_simple_augmentation(self, ecg):
        """Augmentação simplificada"""
        # Amplitude scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            ecg = ecg * scale
        
        # Gaussian noise
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.01, ecg.shape)
            ecg = ecg + noise
        
        return ecg

# ==================== FUNÇÃO PARA CRIAR DATALOADER SEGURO ====================

def create_safe_dataloader(dataset, batch_size, shuffle=True, num_workers=0, **kwargs):
    """Cria DataLoader seguro para Windows"""
    
    # No Windows, sempre usar num_workers=0 ou implementar worker_init_fn
    if sys.platform == 'win32' and num_workers > 0:
        logger.warning("Windows detectado: Ajustando num_workers para 0")
        num_workers = 0
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )

# ==================== MAIN CORRIGIDO ====================

def main():
    """Função principal corrigida para Windows"""
    parser = argparse.ArgumentParser(description="Sistema ECG Ultra-Otimizado - Windows Fix")
    parser.add_argument('--data-path', type=str, 
                       default=r"C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro2\ptbxl_processing\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\processed_npy\ptbxl_100hz")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=0)  # Default 0 para Windows
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("SISTEMA ECG ULTRA-OTIMIZADO - VERSÃO WINDOWS")
    logger.info("="*80)
    
    # Verificar se está no Windows
    if sys.platform == 'win32':
        logger.info("Sistema Windows detectado - usando configurações otimizadas")
        if args.num_workers > 0:
            logger.warning(f"num_workers={args.num_workers} pode causar problemas no Windows")
            logger.warning("Considere usar --num-workers 0 para evitar erros")
    
    # Configuração
    from optimized_ecg_ensemble_fixed import MedicalECGConfig
    config = MedicalECGConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Definir seed
    set_all_seeds(42)
    
    # Carregar dados
    logger.info(f"Carregando dados de: {args.data_path}")
    data_path = Path(args.data_path)
    
    try:
        X = np.load(data_path / 'X.npy')
        Y_multi = np.load(data_path / 'Y_multilabel.npy')
        
        logger.info(f"Dados carregados: X={X.shape}, Y={Y_multi.shape}")
        
        # Ajustar configuração
        config.signal_length = X.shape[2]
        config.num_leads = X.shape[1]
        num_classes = Y_multi.shape[1]
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return
    
    # Dividir dados
    stratify_labels = Y_multi.argmax(axis=1)
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y_multi, test_size=0.2, random_state=42, stratify=stratify_labels
    )
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Criar datasets usando versão corrigida
    train_dataset = WindowsOptimizedECGDataset(
        X_train, Y_train, config, is_training=True
    )
    
    val_dataset = WindowsOptimizedECGDataset(
        X_val, Y_val, config, is_training=False
    )
    
    # DataLoaders seguros para Windows
    # Calcular pesos para balanceamento
    class_counts = Y_train.sum(axis=0)
    class_weights = 1.0 / (class_counts + 1)
    sample_weights = (Y_train * class_weights).sum(axis=1)
    
    if args.num_workers == 0:
        # Sem multiprocessing
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=0,  # Força 0 workers
            pin_memory=(device.type == 'cuda'),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,  # Força 0 workers
            pin_memory=(device.type == 'cuda')
        )
    else:
        # Tentar com multiprocessing (pode falhar no Windows)
        logger.warning("Tentando usar multiprocessing - pode falhar no Windows!")
        
        # Usar collate_fn simples
        def simple_collate(batch):
            ecg, params, labels = zip(*batch)
            return torch.stack(ecg), torch.stack(params), torch.stack(labels)
        
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
            collate_fn=simple_collate,
            persistent_workers=True,  # Ajuda no Windows
            pin_memory=(device.type == 'cuda'),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=simple_collate,
            persistent_workers=True,
            pin_memory=(device.type == 'cuda')
        )
    
    # Importar e criar modelo
    try:
        from optimized_ecg_ensemble_fixed import UltraOptimizedMedicalEnsemble
        logger.info("Criando modelo ultra-otimizado...")
        model = UltraOptimizedMedicalEnsemble(num_classes, config)
    except ImportError:
        logger.error("Não foi possível importar o modelo. Certifique-se de que optimized-ecg-ensemble-fixed.py está no mesmo diretório")
        return
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total de parâmetros: {total_params:,}")
    logger.info(f"Parâmetros treináveis: {trainable_params:,}")
    
    # Criar trainer
    try:
        from optimized_ecg_ensemble_fixed import UltraOptimizedTrainer
        trainer = UltraOptimizedTrainer(model, config, device)
    except ImportError:
        logger.error("Não foi possível importar o trainer")
        return
    
    # Treinar
    logger.info("\nIniciando treinamento...")
    logger.info(f"Configuração:")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Epochs: {config.num_epochs}")
    logger.info(f"  - Workers: {args.num_workers}")
    
    try:
        best_auc = trainer.train(train_loader, val_loader, config.num_epochs)
        
        logger.info("\n" + "="*80)
        logger.info("TREINAMENTO CONCLUÍDO!")
        logger.info("="*80)
        logger.info(f"Melhor AUC alcançado: {best_auc:.4f}")
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        logger.error("Tente executar com --num-workers 0")
        raise

if __name__ == "__main__":
    # CRÍTICO para Windows!
    main()
