#!/usr/bin/env python3
"""
Sistema de Ensemble para Treinamento de ECG com PTB-XL
Combina múltiplos modelos para melhor performance
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
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar modelos base do train_ecg.py
sys.path.append(str(Path(__file__).parent))

try:
    from train_ecg import (
        EnhancedECGAnalysisConfig,
        ECGDataset,
        BasicCNN,
        ResNetECG,
        ECGTransformer,
        Preprocessor,
        Augmenter,
        ClinicalLogger,
        set_seed
    )
    TRAIN_ECG_AVAILABLE = True
except ImportError:
    logger.warning("train_ecg.py não encontrado. Definindo modelos localmente.")
    TRAIN_ECG_AVAILABLE = False

# ==================== CONFIGURAÇÃO DO ENSEMBLE ====================

class EnsembleConfig:
    """Configuração específica para ensemble"""
    def __init__(self):
        # Modelos a incluir no ensemble
        self.models = ['cnn', 'resnet', 'transformer']
        
        # Método de combinação
        self.ensemble_method = 'weighted_voting'  # 'voting', 'weighted_voting', 'stacking'
        
        # Pesos para weighted voting (serão aprendidos se None)
        self.model_weights = None
        
        # Configurações de treinamento
        self.train_models_separately = True
        self.use_cross_validation = True
        self.n_folds = 5
        
        # Stacking
        self.meta_learner = 'logistic_regression'  # 'logistic_regression', 'neural_network'
        self.use_probabilities = True  # Usar probabilidades ao invés de predições
        
        # Otimização
        self.optimize_weights = True
        self.weight_optimization_metric = 'f1_score'
        
        # Salvamento
        self.save_individual_models = True
        self.checkpoint_frequency = 10

# ==================== MODELOS LOCAIS (caso train_ecg.py não esteja disponível) ====================

if not TRAIN_ECG_AVAILABLE:
    # Definir modelos básicos aqui
    class BasicCNN(nn.Module):
        def __init__(self, num_leads: int, signal_length: int, num_classes: int):
            super().__init__()
            self.conv1 = nn.Conv1d(num_leads, 32, kernel_size=5, padding=2)
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
            self.pool2 = nn.MaxPool1d(2)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.pool3 = nn.MaxPool1d(2)
            
            # Calcular tamanho após convoluções
            conv_output_size = signal_length // 8 * 128
            self.fc1 = nn.Linear(conv_output_size, 256)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, num_classes)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = self.pool3(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

# ==================== MODELOS AVANÇADOS PARA ENSEMBLE ====================

class InceptionBlock(nn.Module):
    """Bloco Inception para captura multi-escala"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Branch 1x1
        self.branch1x1 = nn.Conv1d(in_channels, out_channels//4, kernel_size=1)
        
        # Branch 3x3
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=3, padding=1)
        )
        
        # Branch 5x5
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=5, padding=2)
        )
        
        # Branch pool
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1)
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class InceptionECG(nn.Module):
    """Modelo Inception adaptado para ECG"""
    def __init__(self, num_leads: int, signal_length: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)
        
        self.inception1 = InceptionBlock(64, 128)
        self.inception2 = InceptionBlock(128, 256)
        self.inception3 = InceptionBlock(256, 512)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class AttentionBlock(nn.Module):
    """Bloco de atenção para ECG"""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 8),
            nn.ReLU(),
            nn.Linear(in_channels // 8, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, channels, length)
        avg_pool = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        attention_weights = self.attention(avg_pool).view(x.size(0), x.size(1), 1)
        return x * attention_weights

class AttentionCNN(nn.Module):
    """CNN com mecanismo de atenção"""
    def __init__(self, num_leads: int, signal_length: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.attention1 = AttentionBlock(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.attention2 = AttentionBlock(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.attention3 = AttentionBlock(256)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.attention1(x)
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.attention2(x)
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.attention3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ==================== ENSEMBLE MODEL ====================

class ECGEnsemble(nn.Module):
    """Modelo Ensemble para classificação de ECG"""
    def __init__(self, models: List[nn.Module], num_classes: int, ensemble_config: EnsembleConfig):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.ensemble_config = ensemble_config
        
        # Pesos para weighted voting
        if ensemble_config.model_weights is None:
            self.model_weights = nn.Parameter(torch.ones(len(models)) / len(models))
        else:
            self.model_weights = torch.tensor(ensemble_config.model_weights)
        
        # Meta-learner para stacking
        if ensemble_config.ensemble_method == 'stacking':
            if ensemble_config.meta_learner == 'neural_network':
                input_size = len(models) * num_classes if ensemble_config.use_probabilities else len(models)
                self.meta_model = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, num_classes)
                )
    
    def forward(self, x):
        # Obter predições de todos os modelos
        outputs = []
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Combinar predições baseado no método
        if self.ensemble_config.ensemble_method == 'voting':
            # Votação simples
            predictions = torch.stack([F.softmax(out, dim=1) for out in outputs])
            ensemble_output = predictions.mean(dim=0)
            return torch.log(ensemble_output + 1e-8)  # Log para compatibilidade com CrossEntropy
            
        elif self.ensemble_config.ensemble_method == 'weighted_voting':
            # Votação ponderada
            predictions = torch.stack([F.softmax(out, dim=1) for out in outputs])
            weights = F.softmax(self.model_weights, dim=0)
            weighted_preds = predictions * weights.view(-1, 1, 1)
            ensemble_output = weighted_preds.sum(dim=0)
            return torch.log(ensemble_output + 1e-8)
            
        elif self.ensemble_config.ensemble_method == 'stacking':
            # Stacking
            if self.ensemble_config.use_probabilities:
                # Usar probabilidades como features
                predictions = torch.stack([F.softmax(out, dim=1) for out in outputs])
                meta_input = predictions.transpose(0, 1).contiguous()
                meta_input = meta_input.view(x.size(0), -1)
            else:
                # Usar predições como features
                predictions = torch.stack([out.argmax(dim=1) for out in outputs])
                meta_input = predictions.transpose(0, 1).float()
            
            return self.meta_model(meta_input)
    
    def get_individual_predictions(self, x):
        """Obter predições individuais de cada modelo"""
        predictions = {}
        for i, model in enumerate(self.models):
            output = model(x)
            predictions[f'model_{i}'] = F.softmax(output, dim=1)
        return predictions

# ==================== TRAINER PARA ENSEMBLE ====================

class EnsembleTrainer:
    """Trainer especializado para ensemble"""
    def __init__(self, ensemble: ECGEnsemble, config: EnsembleConfig, device: torch.device):
        self.ensemble = ensemble
        self.config = config
        self.device = device
        self.ensemble.to(device)
        
        # Otimizadores separados para cada modelo
        self.optimizers = []
        for model in self.ensemble.models:
            self.optimizers.append(torch.optim.Adam(model.parameters(), lr=1e-3))
        
        # Otimizador para meta-model (se stacking)
        if config.ensemble_method == 'stacking' and hasattr(self.ensemble, 'meta_model'):
            self.meta_optimizer = torch.optim.Adam(self.ensemble.meta_model.parameters(), lr=1e-3)
        
        # Otimizador para pesos (se weighted voting)
        if config.ensemble_method == 'weighted_voting' and config.optimize_weights:
            self.weight_optimizer = torch.optim.Adam([self.ensemble.model_weights], lr=0.01)
        
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()
        
        # Histórico de treinamento
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
    
    def train_individual_model(self, model_idx: int, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50):
        """Treina um modelo individual do ensemble"""
        model = self.ensemble.models[model_idx]
        optimizer = self.optimizers[model_idx]
        
        logger.info(f"Treinando modelo {model_idx + 1}/{len(self.ensemble.models)}")
        
        best_val_f1 = 0
        patience = 0
        max_patience = 10
        
        for epoch in range(epochs):
            # Treino
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                with autocast():
                    output = model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                train_preds.extend(output.argmax(dim=1).cpu().numpy())
                train_labels.extend(target.cpu().numpy())
            
            # Validação
            val_metrics = self.evaluate_model(model, val_loader)
            
            # Métricas de treino
            train_f1 = f1_score(train_labels, train_preds, average='weighted')
            avg_train_loss = train_loss / len(train_loader)
            
            logger.info(f"Model {model_idx} - Epoch {epoch+1}: "
                       f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
            
            # Early stopping
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                patience = 0
                # Salvar checkpoint
                if self.config.save_individual_models:
                    torch.save(model.state_dict(), f"model_{model_idx}_best.pth")
            else:
                patience += 1
                if patience >= max_patience:
                    logger.info(f"Early stopping para modelo {model_idx}")
                    break
        
        return best_val_f1
    
    def train_ensemble(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 100):
        """Treina o ensemble completo"""
        logger.info("=== Iniciando Treinamento do Ensemble ===")
        
        # Fase 1: Treinar modelos individuais
        if self.config.train_models_separately:
            logger.info("Fase 1: Treinando modelos individuais")
            for i in range(len(self.ensemble.models)):
                self.train_individual_model(i, train_loader, val_loader, epochs=epochs//2)
        
        # Fase 2: Treinar ensemble completo
        logger.info("Fase 2: Otimizando ensemble")
        
        best_val_f1 = 0
        patience = 0
        max_patience = 15
        
        for epoch in range(epochs//2):
            # Treino
            self.ensemble.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Ensemble Epoch {epoch+1}")):
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero gradients
                for opt in self.optimizers:
                    opt.zero_grad()
                if hasattr(self, 'meta_optimizer'):
                    self.meta_optimizer.zero_grad()
                if hasattr(self, 'weight_optimizer'):
                    self.weight_optimizer.zero_grad()
                
                with autocast():
                    output = self.ensemble(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                # Atualizar todos os otimizadores
                for opt in self.optimizers:
                    self.scaler.step(opt)
                if hasattr(self, 'meta_optimizer'):
                    self.scaler.step(self.meta_optimizer)
                if hasattr(self, 'weight_optimizer'):
                    self.scaler.step(self.weight_optimizer)
                
                self.scaler.update()
                
                train_loss += loss.item()
                train_preds.extend(output.argmax(dim=1).cpu().numpy())
                train_labels.extend(target.cpu().numpy())
            
            # Validação
            val_metrics = self.evaluate_ensemble(val_loader)
            
            # Métricas
            train_f1 = f1_score(train_labels, train_preds, average='weighted')
            avg_train_loss = train_loss / len(train_loader)
            
            # Log dos pesos do ensemble
            if self.config.ensemble_method == 'weighted_voting':
                weights = F.softmax(self.ensemble.model_weights, dim=0)
                logger.info(f"Pesos do ensemble: {weights.detach().cpu().numpy()}")
            
            logger.info(f"Ensemble Epoch {epoch+1}: "
                       f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1_score']:.4f}, "
                       f"Val AUC: {val_metrics['roc_auc']:.4f}")
            
            # Salvar histórico
            self.train_history['loss'].append(avg_train_loss)
            self.train_history['f1_score'].append(train_f1)
            self.val_history['loss'].append(val_metrics['loss'])
            self.val_history['f1_score'].append(val_metrics['f1_score'])
            self.val_history['roc_auc'].append(val_metrics['roc_auc'])
            
            # Early stopping
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                patience = 0
                # Salvar melhor modelo
                torch.save(self.ensemble.state_dict(), "ensemble_best.pth")
                logger.info(f"Novo melhor modelo! F1: {best_val_f1:.4f}")
            else:
                patience += 1
                if patience >= max_patience:
                    logger.info("Early stopping ativado")
                    break
            
            # Checkpoint periódico
            if (epoch + 1) % self.config.checkpoint_frequency == 0:
                torch.save({
                    'epoch': epoch,
                    'ensemble_state_dict': self.ensemble.state_dict(),
                    'optimizers_state_dict': [opt.state_dict() for opt in self.optimizers],
                    'train_history': self.train_history,
                    'val_history': self.val_history,
                }, f"checkpoint_epoch_{epoch+1}.pth")
        
        return best_val_f1
    
    def evaluate_model(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Avalia um modelo individual"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                with autocast():
                    output = model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                probs = F.softmax(output, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(output.argmax(dim=1).cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Calcular métricas
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # ROC AUC para multi-classe
        all_probs = np.array(all_probs)
        try:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except:
            roc_auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    
    def evaluate_ensemble(self, dataloader: DataLoader) -> Dict[str, float]:
        """Avalia o ensemble completo"""
        self.ensemble.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        # Coletar predições individuais
        individual_predictions = defaultdict(list)
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                with autocast():
                    output = self.ensemble(data)
                    loss = self.criterion(output, target)
                    
                    # Predições individuais
                    ind_preds = self.ensemble.get_individual_predictions(data)
                    for model_name, preds in ind_preds.items():
                        individual_predictions[model_name].extend(preds.cpu().numpy())
                
                total_loss += loss.item()
                probs = F.softmax(output, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(output.argmax(dim=1).cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Calcular métricas do ensemble
        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_score': f1_score(all_labels, all_preds, average='weighted'),
        }
        
        # ROC AUC
        all_probs = np.array(all_probs)
        try:
            metrics['roc_auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = 0.0
        
        # Métricas individuais dos modelos
        for i, (model_name, preds) in enumerate(individual_predictions.items()):
            preds = np.array(preds)
            model_preds = preds.argmax(axis=1)
            metrics[f'{model_name}_accuracy'] = accuracy_score(all_labels, model_preds)
            metrics[f'{model_name}_f1'] = f1_score(all_labels, model_preds, average='weighted')
        
        return metrics

# ==================== FUNÇÃO PRINCIPAL ====================

def load_ptbxl_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Carrega dados do PTB-XL"""
    data_path = Path(data_path)
    logger.info(f"Carregando dados de: {data_path}")
    
    # Listar arquivos
    npy_files = list(data_path.glob("*.npy"))
    logger.info(f"Arquivos encontrados: {[f.name for f in npy_files]}")
    
    X_data = None
    y_data = None
    
    # Tentar diferentes padrões de nomenclatura
    patterns = [
        ('X.npy', 'Y.npy'),
        ('x.npy', 'y.npy'),
        ('data.npy', 'labels.npy'),
        ('X_train.npy', 'y_train.npy'),
    ]
    
    for x_pattern, y_pattern in patterns:
        x_path = data_path / x_pattern
        y_path = data_path / y_pattern
        
        if x_path.exists() and y_path.exists():
            X_data = np.load(x_path)
            y_data = np.load(y_path)
            logger.info(f"Dados carregados: X={X_data.shape}, Y={y_data.shape}")
            break
    
    if X_data is None:
        raise FileNotFoundError("Não foi possível encontrar os arquivos de dados")
    
    # Metadados
    metadata = {}
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return X_data, y_data, metadata

def create_ensemble_model(num_leads: int, signal_length: int, num_classes: int, 
                         model_types: List[str], device: torch.device) -> ECGEnsemble:
    """Cria o modelo ensemble com os tipos especificados"""
    models = []
    
    for model_type in model_types:
        logger.info(f"Criando modelo: {model_type}")
        
        if model_type == 'cnn':
            model = BasicCNN(num_leads, signal_length, num_classes)
        elif model_type == 'resnet':
            if TRAIN_ECG_AVAILABLE:
                model = ResNetECG(num_leads, signal_length, num_classes)
            else:
                logger.warning("ResNetECG não disponível, usando BasicCNN")
                model = BasicCNN(num_leads, signal_length, num_classes)
        elif model_type == 'transformer':
            if TRAIN_ECG_AVAILABLE:
                model = ECGTransformer(num_leads, signal_length, num_classes)
            else:
                logger.warning("ECGTransformer não disponível, usando BasicCNN")
                model = BasicCNN(num_leads, signal_length, num_classes)
        elif model_type == 'inception':
            model = InceptionECG(num_leads, signal_length, num_classes)
        elif model_type == 'attention':
            model = AttentionCNN(num_leads, signal_length, num_classes)
        else:
            logger.warning(f"Tipo de modelo desconhecido: {model_type}, usando BasicCNN")
            model = BasicCNN(num_leads, signal_length, num_classes)
        
        models.append(model)
    
    # Criar configuração do ensemble
    ensemble_config = EnsembleConfig()
    ensemble_config.models = model_types
    
    # Criar ensemble
    ensemble = ECGEnsemble(models, num_classes, ensemble_config)
    
    return ensemble

def main():
    parser = argparse.ArgumentParser(description="Treinamento de Ensemble para ECG")
    parser.add_argument('--data-path', type=str, required=True, help='Caminho para os dados PTB-XL')
    parser.add_argument('--models', nargs='+', default=['cnn', 'resnet', 'inception'], 
                       help='Modelos a incluir no ensemble')
    parser.add_argument('--ensemble-method', type=str, default='weighted_voting',
                       choices=['voting', 'weighted_voting', 'stacking'],
                       help='Método de combinação do ensemble')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamanho do batch')
    parser.add_argument('--output-dir', type=str, default='./ensemble_output',
                       help='Diretório de saída')
    
    args = parser.parse_args()
    
    # Configuração
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Dispositivo: {device}")
    
    # Criar diretório de saída
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar dados
    try:
        X_data, y_data, metadata = load_ptbxl_data(args.data_path)
        
        # Preparar dados
        if X_data.ndim == 3 and X_data.shape[2] == 12:
            X_data = np.transpose(X_data, (0, 2, 1))
        
        # Converter multi-label para single-label se necessário
        if y_data.ndim > 1 and y_data.shape[1] > 1:
            y_data = y_data.argmax(axis=1)
        
        # Parâmetros
        num_samples = X_data.shape[0]
        num_leads = X_data.shape[1] if X_data.ndim > 2 else 12
        signal_length = X_data.shape[2] if X_data.ndim > 2 else X_data.shape[1] // num_leads
        num_classes = len(np.unique(y_data))
        
        logger.info(f"Dados: {num_samples} amostras, {num_leads} derivações, "
                   f"{signal_length} pontos, {num_classes} classes")
        
        # Criar configuração
        if TRAIN_ECG_AVAILABLE:
            config = EnhancedECGAnalysisConfig(
                sampling_rate=100,
                signal_length=signal_length,
                num_leads=num_leads,
                batch_size=args.batch_size,
                num_epochs=args.epochs
            )
        
        # Preparar dados para PyTorch
        X_flat = X_data.reshape(num_samples, -1)
        df_data = pd.DataFrame(X_flat)
        s_labels = pd.Series(y_data)
        
        # Dividir dados
        X_train, X_val, y_train, y_val = train_test_split(
            df_data, s_labels, test_size=0.2, random_state=42, stratify=s_labels
        )
        
        # Criar datasets
        if TRAIN_ECG_AVAILABLE:
            train_dataset = ECGDataset(X_train, y_train, config, is_train=True)
            val_dataset = ECGDataset(X_val, y_val, config, is_train=False)
        else:
            # Dataset simples
            class SimpleDataset(Dataset):
                def __init__(self, X, y, num_leads, signal_length):
                    self.X = X
                    self.y = y
                    self.num_leads = num_leads
                    self.signal_length = signal_length
                
                def __len__(self):
                    return len(self.X)
                
                def __getitem__(self, idx):
                    x = self.X.iloc[idx].values.reshape(self.num_leads, self.signal_length)
                    y = self.y.iloc[idx]
                    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
            
            train_dataset = SimpleDataset(X_train, y_train, num_leads, signal_length)
            val_dataset = SimpleDataset(X_val, y_val, num_leads, signal_length)
        
        # Criar dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=4, pin_memory=True)
        
        # Criar modelo ensemble
        logger.info(f"Criando ensemble com modelos: {args.models}")
        ensemble = create_ensemble_model(num_leads, signal_length, num_classes, 
                                       args.models, device)
        
        # Configurar ensemble
        ensemble_config = EnsembleConfig()
        ensemble_config.ensemble_method = args.ensemble_method
        ensemble_config.models = args.models
        
        # Criar trainer
        trainer = EnsembleTrainer(ensemble, ensemble_config, device)
        
        # Treinar
        logger.info("Iniciando treinamento do ensemble...")
        best_f1 = trainer.train_ensemble(train_loader, val_loader, epochs=args.epochs)
        
        logger.info(f"Treinamento concluído! Melhor F1-score: {best_f1:.4f}")
        
        # Avaliar modelo final
        logger.info("Avaliação final do ensemble:")
        final_metrics = trainer.evaluate_ensemble(val_loader)
        
        logger.info("Métricas finais:")
        for metric, value in final_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Salvar resultados
        results = {
            'config': {
                'models': args.models,
                'ensemble_method': args.ensemble_method,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
            },
            'metrics': final_metrics,
            'history': {
                'train': dict(trainer.train_history),
                'val': dict(trainer.val_history)
            }
        }
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Resultados salvos em: {output_dir}")
        
    except Exception as e:
        logger.error(f"Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
