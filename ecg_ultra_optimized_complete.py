#!/usr/bin/env python3
"""
Módulo de suporte para ECG Ultimate Ensemble
Implementa componentes essenciais para processamento de ECG
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import kurtosis, skew
import pywt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import warnings
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

class ClinicalFeatureExtractor:
    """Extrator de características clínicas do ECG"""
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        
    def extract_features(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Extrai características clínicas de um sinal ECG 12 derivações
        
        Args:
            ecg_signal: Array (12, samples) com ECG 12 derivações
            
        Returns:
            Array com 50 características extraídas
        """
        features = []
        
        for lead in range(12):
            signal = ecg_signal[lead]
            
            # Características temporais
            features.append(np.mean(signal))
            features.append(np.std(signal))
            features.append(kurtosis(signal))
            features.append(skew(signal))
            
            # Detecção de picos R
            peaks, _ = find_peaks(signal, height=0.5*np.max(signal), distance=int(0.6*self.sampling_rate))
            
            if len(peaks) > 1:
                # Heart rate variability
                rr_intervals = np.diff(peaks) / self.sampling_rate
                features.append(np.mean(rr_intervals))
                features.append(np.std(rr_intervals))
            else:
                features.extend([0, 0])
            
            # Características espectrais (simplificadas)
            freqs, psd = scipy_signal.welch(signal, fs=self.sampling_rate, nperseg=min(256, len(signal)))
            
            # Energia em bandas de frequência
            lf_band = (0.04, 0.15)
            hf_band = (0.15, 0.4)
            
            lf_idx = np.where((freqs >= lf_band[0]) & (freqs <= lf_band[1]))[0]
            hf_idx = np.where((freqs >= hf_band[0]) & (freqs <= hf_band[1]))[0]
            
            lf_power = np.trapz(psd[lf_idx], freqs[lf_idx]) if len(lf_idx) > 0 else 0
            hf_power = np.trapz(psd[hf_idx], freqs[hf_idx]) if len(hf_idx) > 0 else 0
            
            features.append(lf_power)
            features.append(hf_power)
        
        # Características inter-derivações
        # Correlação entre derivações adjacentes
        for i in range(11):
            corr = np.corrcoef(ecg_signal[i], ecg_signal[i+1])[0, 1]
            features.append(corr if not np.isnan(corr) else 0)
        
        # Pad para 50 características
        features = np.array(features)
        if len(features) < 50:
            features = np.pad(features, (0, 50 - len(features)), 'constant')
        else:
            features = features[:50]
        
        return features.astype(np.float32)

class AdvancedAugmentation:
    """Augmentação avançada para sinais ECG"""
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        
    def __call__(self, ecg: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica augmentações aleatórias"""
        if np.random.rand() < 0.5:
            ecg = self.add_baseline_wander(ecg)
        if np.random.rand() < 0.3:
            ecg = self.add_noise(ecg)
        if np.random.rand() < 0.3:
            ecg = self.time_warp(ecg)
        if np.random.rand() < 0.2:
            ecg = self.amplitude_scale(ecg)
        
        return ecg, label
    
    def add_baseline_wander(self, ecg: np.ndarray) -> np.ndarray:
        """Adiciona variação de linha de base"""
        wander_freq = np.random.uniform(0.15, 0.3)
        time = np.arange(ecg.shape[1]) / self.sampling_rate
        wander = 0.05 * np.sin(2 * np.pi * wander_freq * time)
        return ecg + wander
    
    def add_noise(self, ecg: np.ndarray) -> np.ndarray:
        """Adiciona ruído gaussiano"""
        noise_level = np.random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_level, ecg.shape)
        return ecg + noise
    
    def time_warp(self, ecg: np.ndarray) -> np.ndarray:
        """Distorção temporal"""
        factor = np.random.uniform(0.9, 1.1)
        new_length = int(ecg.shape[1] * factor)
        warped = np.zeros((ecg.shape[0], new_length))
        
        for i in range(ecg.shape[0]):
            warped[i] = np.interp(
                np.linspace(0, ecg.shape[1]-1, new_length),
                np.arange(ecg.shape[1]),
                ecg[i]
            )
        
        # Redimensionar de volta ao tamanho original
        if new_length != ecg.shape[1]:
            original = np.zeros_like(ecg)
            for i in range(ecg.shape[0]):
                original[i] = np.interp(
                    np.arange(ecg.shape[1]),
                    np.linspace(0, ecg.shape[1]-1, new_length),
                    warped[i]
                )
            return original
        
        return warped
    
    def amplitude_scale(self, ecg: np.ndarray) -> np.ndarray:
        """Escala de amplitude"""
        scale = np.random.uniform(0.8, 1.2)
        return ecg * scale

class UltraOptimizedECGDataset(Dataset):
    """Dataset otimizado para ECG com cache de features"""
    def __init__(self, X, Y, feature_extractor, augmenter=None, 
                 feature_cache=None, is_training=True):
        self.X = X
        self.Y = Y
        self.feature_extractor = feature_extractor
        self.augmenter = augmenter if is_training else None
        self.feature_cache = feature_cache if feature_cache is not None else {}
        self.is_training = is_training
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        ecg = self.X[idx].copy()
        label = self.Y[idx].copy()
        
        # Normalização
        ecg = (ecg - ecg.mean(axis=1, keepdims=True)) / (ecg.std(axis=1, keepdims=True) + 1e-8)
        
        # Augmentação
        if self.augmenter:
            ecg, label = self.augmenter(ecg, label)
        
        # Extrair features (com cache)
        if idx in self.feature_cache:
            features = self.feature_cache[idx]
        else:
            features = self.feature_extractor.extract_features(ecg)
            if not self.is_training:  # Só faz cache em validação
                self.feature_cache[idx] = features
        
        return (
            torch.FloatTensor(ecg),
            torch.FloatTensor(features),
            torch.FloatTensor(label)
        )

class OptimizedMultilabelLoss(nn.Module):
    """Loss otimizada para classificação multilabel com class imbalance"""
    def __init__(self, class_frequencies, device):
        super().__init__()
        # Calcular pesos baseados em frequências
        total_samples = class_frequencies.sum()
        class_weights = total_samples / (len(class_frequencies) * class_frequencies + 1)
        self.class_weights = torch.FloatTensor(class_weights).to(device)
        
    def forward(self, outputs, targets):
        # Binary cross entropy com pesos
        bce = F.binary_cross_entropy_with_logits(
            outputs, targets, reduction='none'
        )
        
        # Aplicar pesos por classe
        weighted_bce = bce * self.class_weights
        
        # Focal loss para exemplos difíceis
        p = torch.sigmoid(outputs)
        ce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_loss = ce_loss * ((1 - p_t) ** 2)
        
        # Combinar losses
        total_loss = 0.7 * weighted_bce.mean() + 0.3 * focal_loss.mean()
        
        return total_loss

def calculate_multilabel_metrics(y_true, y_pred):
    """Calcula métricas para classificação multilabel"""
    metrics = {}
    
    # Converter para numpy se necessário
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # AUC por classe
    auc_per_class = []
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            auc_per_class.append(auc)
        except:
            auc_per_class.append(0.5)
    
    metrics['auc_per_class'] = auc_per_class
    metrics['auc_mean'] = np.mean(auc_per_class)
    
    # AUC macro
    try:
        metrics['auc_macro'] = roc_auc_score(y_true, y_pred, average='macro')
    except:
        metrics['auc_macro'] = 0.5
    
    # F1 score
    y_pred_binary = (y_pred > 0.5).astype(int)
    metrics['f1_micro'] = f1_score(y_true, y_pred_binary, average='micro')
    metrics['f1_macro'] = f1_score(y_true, y_pred_binary, average='macro')
    
    return metrics

def create_weighted_sampler(labels):
    """Cria sampler com pesos para balancear classes"""
    # Calcular pesos por amostra
    class_counts = labels.sum(axis=0)
    class_weights = 1.0 / (class_counts + 1)
    
    # Peso de cada amostra baseado em suas labels
    sample_weights = []
    for label in labels:
        weight = np.sum(label * class_weights)
        sample_weights.append(weight)
    
    sample_weights = torch.FloatTensor(sample_weights)
    
    return WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )

# Componente adicional necessário - HolisticAnalyzer (versão simplificada)
class HolisticAnalyzer(nn.Module):
    """Analisador holístico baseado em Transformer"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        # Encoder para ECG
        self.ecg_encoder = nn.Sequential(
            nn.Conv1d(12, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        
        # Transformer
        self.positional_encoding = nn.Parameter(torch.randn(1, 64, 128))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Feature fusion
        self.feature_proj = nn.Linear(num_features, 128)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, ecg, features):
        # Encode ECG
        x = self.ecg_encoder(ecg)  # (B, 128, 64)
        x = x.transpose(1, 2)  # (B, 64, 128)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Transformer
        x = self.transformer(x)
        
        # Global pooling
        x = x.mean(dim=1)  # (B, 128)
        
        # Add features
        feat = self.feature_proj(features)
        x = x + feat
        
        # Classify
        return self.classifier(x)
