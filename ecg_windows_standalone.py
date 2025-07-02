#!/usr/bin/env python3
"""
Sistema ECG Completo para Windows - Versão Standalone
Não requer importações externas além das bibliotecas padrão
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
import warnings
from tqdm import tqdm
from dataclasses import dataclass, field
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn

from scipy import signal as scipy_signal
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import RobustScaler

# Configurações
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Para Windows
if sys.platform == 'win32':
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except:
        pass

# Otimizações CUDA
cudnn.benchmark = True
cudnn.deterministic = False

# Seed
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(42)

# ==================== CONFIGURAÇÃO ====================

@dataclass
class ECGConfig:
    """Configuração para treinamento ECG"""
    # Dados
    sampling_rate: int = 100
    signal_length: int = 1000
    num_leads: int = 12
    
    # Modelo
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.3
    
    # Treinamento
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Augmentação
    augment_prob: float = 0.5
    
    # Validação
    early_stopping_patience: int = 15

# ==================== DATASET ====================

class ECGDataset(Dataset):
    """Dataset ECG compatível com Windows"""
    
    def __init__(self, X, Y, config, is_training=True):
        self.X = X
        self.Y = Y
        self.config = config
        self.is_training = is_training
        
        # Normalização
        self.normalizer = RobustScaler()
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.normalizer.fit(X_reshaped)
        
        # Salvar parâmetros de normalização
        self.scale_ = self.normalizer.scale_
        self.center_ = self.normalizer.center_
        
        logger.info(f"Dataset criado: {len(X)} amostras")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # ECG
        ecg = self.X[idx].copy()
        
        # Normalizar
        ecg_reshaped = ecg.reshape(-1, ecg.shape[-1])
        ecg_normalized = (ecg_reshaped - self.center_) / self.scale_
        ecg = ecg_normalized.reshape(ecg.shape)
        
        # Augmentação simples
        if self.is_training and np.random.rand() < self.config.augment_prob:
            # Scaling
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                ecg = ecg * scale
            
            # Noise
            if np.random.rand() < 0.3:
                noise = np.random.normal(0, 0.01, ecg.shape)
                ecg = ecg + noise
        
        # Converter para tensores
        ecg_tensor = torch.FloatTensor(ecg)
        
        if self.Y.ndim > 1:  # multilabel
            label = torch.FloatTensor(self.Y[idx])
        else:
            label = torch.LongTensor([self.Y[idx]])
        
        return ecg_tensor, label

# ==================== MODELOS ====================

class ResBlock(nn.Module):
    """Bloco Residual"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ECGModel(nn.Module):
    """Modelo ECG principal"""
    
    def __init__(self, num_classes, config):
        super().__init__()
        self.config = config
        
        # Encoder
        self.conv1 = nn.Conv1d(12, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, config.hidden_dim, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(config.hidden_dim)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(config.hidden_dim) for _ in range(config.num_layers)
        ])
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        self.fc2 = nn.Linear(config.hidden_dim // 2, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Encoder
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ==================== LOSS ====================

class ECGLoss(nn.Module):
    """Loss function para ECG"""
    
    def __init__(self, num_classes, multilabel=True):
        super().__init__()
        self.multilabel = multilabel
        
        if multilabel:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        return self.criterion(predictions, targets)

# ==================== TRAINER ====================

class ECGTrainer:
    """Trainer para modelo ECG"""
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Otimizador
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        
        # Loss
        self.criterion = ECGLoss(model.num_classes)
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Tracking
        self.best_val_metric = 0.0
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_idx, (ecg, labels) in enumerate(tqdm(train_loader, desc="Training")):
            ecg = ecg.to(self.device)
            labels = labels.to(self.device)
            
            # Mixed precision
            with autocast():
                outputs = self.model(ecg)
                loss = self.criterion(outputs, labels)
            
            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Tracking
            total_loss += loss.item()
            
            with torch.no_grad():
                if self.criterion.multilabel:
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())
        
        # Métricas
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        metrics = self.calculate_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for ecg, labels in tqdm(val_loader, desc="Validation"):
                ecg = ecg.to(self.device)
                labels = labels.to(self.device)
                
                with autocast():
                    outputs = self.model(ecg)
                    loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                if self.criterion.multilabel:
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)
                
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())
        
        # Métricas
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        metrics = self.calculate_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def calculate_metrics(self, predictions, targets):
        """Calcula métricas de avaliação"""
        metrics = {}
        
        try:
            if self.criterion.multilabel:
                # Multilabel
                binary_preds = (predictions > 0.5).float()
                
                # AUC por classe
                auc_scores = []
                for i in range(predictions.shape[1]):
                    if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                        auc = roc_auc_score(targets[:, i], predictions[:, i])
                        auc_scores.append(auc)
                
                metrics['auc'] = np.mean(auc_scores) if auc_scores else 0.0
                metrics['accuracy'] = accuracy_score(targets.numpy().flatten(), 
                                                   binary_preds.numpy().flatten())
            else:
                # Multiclass
                preds = predictions.argmax(dim=1)
                metrics['accuracy'] = accuracy_score(targets.numpy(), preds.numpy())
                
                # AUC para multiclass
                if predictions.shape[1] > 2:
                    try:
                        metrics['auc'] = roc_auc_score(targets.numpy(), predictions.numpy(), 
                                                      multi_class='ovr', average='macro')
                    except:
                        metrics['auc'] = 0.0
        except Exception as e:
            logger.warning(f"Erro ao calcular métricas: {e}")
            metrics['auc'] = 0.0
            metrics['accuracy'] = 0.0
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        """Loop de treinamento principal"""
        logger.info("Iniciando treinamento...")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, "
                       f"AUC: {train_metrics.get('auc', 0):.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"AUC: {val_metrics.get('auc', 0):.4f}")
            
            # Scheduler
            self.scheduler.step(val_metrics['loss'])
            
            # Save best
            current_metric = val_metrics.get('auc', val_metrics['accuracy'])
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.save_checkpoint(epoch, val_metrics)
                logger.info(f"Novo melhor modelo! Métrica: {self.best_val_metric:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info("Early stopping ativado!")
                    break
        
        return self.best_val_metric
    
    def save_checkpoint(self, epoch, metrics):
        """Salva checkpoint do modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, 'ecg_model_best.pth')
        
        # Salvar relatório
        report = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v 
                       for k, v in metrics.items()},
            'best_metric': float(self.best_val_metric)
        }
        
        with open('training_report.json', 'w') as f:
            json.dump(report, f, indent=2)

# ==================== MAIN ====================

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="ECG Training - Windows Compatible")
    parser.add_argument('--data-path', type=str, 
                       default=r"C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro2\ptbxl_processing\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\processed_npy\ptbxl_100hz")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("ECG TRAINING SYSTEM - WINDOWS OPTIMIZED")
    logger.info("="*60)
    
    # Configuração
    config = ECGConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Carregar dados
    logger.info(f"\nCarregando dados de: {args.data_path}")
    data_path = Path(args.data_path)
    
    try:
        X = np.load(data_path / 'X.npy')
        
        # Tentar carregar Y_multilabel primeiro
        try:
            Y = np.load(data_path / 'Y_multilabel.npy')
            multilabel = True
            logger.info("Usando labels multilabel")
        except:
            Y = np.load(data_path / 'Y.npy')
            multilabel = False
            logger.info("Usando labels single-label")
            
            # Converter para one-hot se necessário
            if Y.ndim == 1:
                num_classes = len(np.unique(Y))
                Y_onehot = np.zeros((len(Y), num_classes))
                for i, label in enumerate(Y):
                    Y_onehot[i, int(label)] = 1
                Y = Y_onehot
                multilabel = True
        
        logger.info(f"Dados carregados: X={X.shape}, Y={Y.shape}")
        
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return
    
    # Ajustar configuração
    config.signal_length = X.shape[2]
    config.num_leads = X.shape[1]
    num_classes = Y.shape[1] if Y.ndim > 1 else len(np.unique(Y))
    
    # Dividir dados
    if multilabel:
        stratify_labels = Y.argmax(axis=1)
    else:
        stratify_labels = Y
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=stratify_labels
    )
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Criar datasets
    train_dataset = ECGDataset(X_train, Y_train, config, is_training=True)
    val_dataset = ECGDataset(X_val, Y_val, config, is_training=False)
    
    # DataLoaders (sem multiprocessing)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Importante para Windows!
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Importante para Windows!
        pin_memory=(device.type == 'cuda')
    )
    
    # Criar modelo
    logger.info(f"\nCriando modelo com {num_classes} classes...")
    model = ECGModel(num_classes, config)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total de parâmetros: {total_params:,}")
    
    # Ajustar loss para tipo de problema
    model.num_classes = num_classes
    
    # Criar trainer
    trainer = ECGTrainer(model, config, device)
    trainer.criterion.multilabel = multilabel
    
    # Treinar
    logger.info("\nIniciando treinamento...")
    logger.info(f"Configuração:")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Epochs: {config.num_epochs}")
    logger.info(f"  - Multilabel: {multilabel}")
    
    try:
        best_metric = trainer.train(train_loader, val_loader, config.num_epochs)
        
        logger.info("\n" + "="*60)
        logger.info("TREINAMENTO CONCLUÍDO!")
        logger.info(f"Melhor métrica: {best_metric:.4f}")
        logger.info("Modelo salvo: ecg_model_best.pth")
        logger.info("Relatório: training_report.json")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.info("\n\nTreinamento interrompido pelo usuário!")
    except Exception as e:
        logger.error(f"Erro durante treinamento: {e}")
        raise

if __name__ == "__main__":
    main()
