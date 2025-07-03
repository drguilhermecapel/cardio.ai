#!/usr/bin/env python3
"""
Sistema ECG Ultimate Ensemble - Estado da Arte 2024
Implementa as técnicas mais avançadas de ensemble para classificação ECG
Target: AUC > 0.97
"""

import os
import sys
import json
import logging
import warnings
import pickle
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional, Any, Union
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal as scipy_signal
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import kurtosis, skew
import pywt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import RobustScaler

# Configurações
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Windows fix
if sys.platform == 'win32':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

# CUDA optimizations
if torch.cuda.is_available():
    cudnn.benchmark = True
    cudnn.deterministic = False

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(42)

# ==================== COMPONENTES AVANÇADOS ====================

class MambaBlock(nn.Module):
    """State Space Model (Mamba) block para eficiência em sequências longas"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # CORREÇÃO APLICADA: projetar para d_inner + d_state + d_state
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + self.d_state + self.d_state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        self.A = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model)
        
    def forward(self, x):
        """SSM forward pass"""
        batch, length, dim = x.shape
        
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(self.d_inner, dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :length]
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        
        # SSM computations
        delta, B, C = self.x_proj(x).split([self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        # Discretization
        A = -torch.exp(self.A.float())
        
        # SSM step
        y = self.selective_scan(x, delta, A, B, C, self.D.float())
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(self, u, delta, A, B, C, D):
        """Efficient selective scan implementation"""
        batch, length, d_in = u.shape
        n = A.shape[1]
        
        # Discretize
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        
        # Scan
        x = torch.zeros(batch, d_in, n, device=u.device, dtype=u.dtype)
        ys = []
        
        for i in range(length):
            x = deltaA[:, i] * x + deltaB[:, i] * u[:, i].unsqueeze(-1)
            y = (x @ C[:, i].unsqueeze(-1)).squeeze(-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        
        y = y + D * u
        
        return y

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network com dilated convolutions"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                self._make_layer(
                    in_channels, out_channels, kernel_size, 
                    dilation=dilation_size, dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        
    def _make_layer(self, in_channels, out_channels, kernel_size, dilation, dropout):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size-1) * dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                     padding=(kernel_size-1) * dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.network(x)

class ECGGraphNetwork(nn.Module):
    """Graph Neural Network para modelar relações entre derivações"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Adjacency matrix for 12-lead ECG (based on anatomical relationships)
        self.register_buffer('adj_matrix', self._create_ecg_adjacency())
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        
    def _create_ecg_adjacency(self):
        """Create adjacency matrix based on ECG lead relationships"""
        # 12 leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        adj = torch.zeros(12, 12)
        
        # Limb leads connections
        limb_connections = [
            (0, 1), (0, 2), (1, 2),  # I, II, III
            (0, 3), (1, 3), (2, 3),  # aVR connections
            (0, 4), (1, 4),          # aVL connections
            (1, 5), (2, 5)           # aVF connections
        ]
        
        # Precordial leads connections (adjacent leads)
        precordial_connections = [
            (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)  # V1-V6
        ]
        
        # Add connections
        for i, j in limb_connections + precordial_connections:
            adj[i, j] = 1
            adj[j, i] = 1
        
        # Self-connections
        adj += torch.eye(12)
        
        return adj
    
    def forward(self, x, edge_index=None):
        """x: (batch, 12, features)"""
        batch_size = x.shape[0]
        
        # Create edge index from adjacency matrix if not provided
        if edge_index is None:
            edge_index = self.adj_matrix.nonzero().t()
        
        # Reshape for GCN
        x = x.view(-1, x.shape[-1])  # (batch*12, features)
        
        # Apply GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        # Reshape back
        x = x.view(batch_size, 12, -1)
        
        # Global pooling
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        
        return torch.cat([x_mean, x_max], dim=1)

# ==================== MODELOS ESPECIALIZADOS DO ENSEMBLE ====================

class WaveNetECG(nn.Module):
    """WaveNet adaptado para ECG com dilated causal convolutions"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        self.input_conv = nn.Conv1d(12, 128, 1)
        
        # Dilated convolution blocks
        self.dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.conv_blocks = nn.ModuleList()
        
        for dilation in self.dilations:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(128, 256, 3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Conv1d(256, 128, 1),
                    nn.BatchNorm1d(128)
                )
            )
        
        # Feature integration
        self.feature_proj = nn.Linear(num_features, 128)
        
        # Output layers
        self.output_conv = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Linear(64 + 128, num_classes)
        
    def forward(self, ecg, features):
        x = self.input_conv(ecg)
        
        # Residual connections through dilated convs
        for conv_block in self.conv_blocks:
            residual = x
            x = conv_block(x)
            x = x + residual
            x = F.relu(x)
        
        # Global features
        x = self.output_conv(x).squeeze(-1)
        
        # Integrate clinical features
        feat = self.feature_proj(features)
        
        combined = torch.cat([x, feat], dim=1)
        return self.classifier(combined)

class MambaECG(nn.Module):
    """State Space Model para ECG - eficiente para sequências longas"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        self.input_proj = nn.Conv1d(12, 256, 1)
        
        # Stack of Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(256, d_state=32, d_conv=4, expand=2)
            for _ in range(6)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(256) for _ in range(6)
        ])
        
        # Feature integration
        self.feature_proj = nn.Linear(num_features, 256)
        
        # Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, ecg, features):
        # Project input
        x = self.input_proj(ecg)
        x = x.transpose(1, 2)  # (B, L, C)
        
        # Apply Mamba blocks
        for i, (mamba, norm) in enumerate(zip(self.mamba_blocks, self.norm_layers)):
            residual = x
            x = mamba(x)
            x = norm(x + residual)
        
        # Pool and classify
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.pool(x).squeeze(-1)
        
        # Add features
        feat = self.feature_proj(features)
        x = x + feat
        
        return self.classifier(x)

class TCNExpert(nn.Module):
    """Temporal Convolutional Network Expert"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        # TCN backbone
        self.tcn = TemporalConvNet(
            num_inputs=12,
            num_channels=[64, 128, 256, 512],
            kernel_size=3,
            dropout=0.2
        )
        
        # Feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, ecg, features):
        # TCN processing
        x = self.tcn(ecg)
        x = self.global_pool(x).squeeze(-1)
        
        # Feature processing
        feat = self.feature_net(features)
        
        # Combine and classify
        combined = torch.cat([x, feat], dim=1)
        return self.classifier(combined)

class GraphECGExpert(nn.Module):
    """Graph Neural Network Expert for inter-lead relationships"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        # Initial feature extraction per lead
        self.lead_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 64, 7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            ) for _ in range(12)
        ])
        
        # Graph network
        self.graph_net = ECGGraphNetwork(128, 256, 128)
        
        # Feature integration
        self.feature_proj = nn.Linear(num_features, 128)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, num_classes)
        )
        
    def forward(self, ecg, features):
        # Extract features per lead
        lead_features = []
        for i in range(12):
            lead_feat = self.lead_encoders[i](ecg[:, i:i+1, :])
            lead_features.append(lead_feat.squeeze(-1))
        
        # Stack features
        x = torch.stack(lead_features, dim=1)  # (batch, 12, 128)
        
        # Apply graph network
        graph_features = self.graph_net(x)
        
        # Add clinical features
        feat = self.feature_proj(features)
        
        # Combine and classify
        combined = torch.cat([graph_features, feat], dim=1)
        return self.classifier(combined)

class DeepResNetECG(nn.Module):
    """Very deep ResNet optimized for ECG"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(12, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 1024, 3, stride=2)
        
        # Feature integration
        self.feature_proj = nn.Linear(num_features, 512)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 512, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, ecg, features):
        x = self.conv1(ecg)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x).squeeze(-1)
        
        # Add features
        feat = self.feature_proj(features)
        combined = torch.cat([x, feat], dim=1)
        
        return self.classifier(combined)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ==================== ENSEMBLE HIERÁRQUICO ====================

class PathologyGroupClassifier(nn.Module):
    """Classificador para grupos de patologias"""
    def __init__(self, input_dim, num_groups=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_groups)
        )
    
    def forward(self, x):
        return self.classifier(x)

class UltimateECGEnsemble(nn.Module):
    """Ultimate Ensemble com 6 modelos especializados + hierarquia"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        # Modelos base do ensemble
        self.wavenet = WaveNetECG(num_classes, num_features)
        self.mamba = MambaECG(num_classes, num_features)
        self.tcn = TCNExpert(num_classes, num_features)
        self.graph = GraphECGExpert(num_classes, num_features)
        self.resnet = DeepResNetECG(num_classes, num_features)
        
        # Modelo transformer do ensemble anterior (mantém o melhor)
        from ecg_ultra_optimized_complete import HolisticAnalyzer
        self.transformer = HolisticAnalyzer(num_classes, num_features)
        
        # Classificador hierárquico de grupos
        self.group_classifier = PathologyGroupClassifier(num_features, num_groups=5)
        
        # Dynamic Model Selection Network
        self.model_selector = nn.Sequential(
            nn.Linear(num_features + 5, 128),  # features + group predictions
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  # 6 models
            nn.Softmax(dim=1)
        )
        
        # Advanced Meta-Learner with Mixture of Experts
        self.expert_gates = nn.ModuleList([
            nn.Linear(num_features + 5, 1) for _ in range(num_classes)
        ])
        
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * 6 + num_features + 5, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Confidence calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, ecg, features, return_all=False):
        # Get all model predictions
        wavenet_out = self.wavenet(ecg, features)
        mamba_out = self.mamba(ecg, features)
        tcn_out = self.tcn(ecg, features)
        graph_out = self.graph(ecg, features)
        resnet_out = self.resnet(ecg, features)
        transformer_out = self.transformer(ecg, features)
        
        # Group classification for hierarchical guidance
        group_logits = self.group_classifier(features)
        group_probs = F.softmax(group_logits, dim=1)
        
        # Dynamic model selection based on features and groups
        selector_input = torch.cat([features, group_probs], dim=1)
        model_weights = self.model_selector(selector_input)
        
        # Stack all outputs
        all_outputs = torch.stack([
            wavenet_out, mamba_out, tcn_out, 
            graph_out, resnet_out, transformer_out
        ], dim=1)  # (batch, 6, num_classes)
        
        # Weighted combination
        weighted_outputs = (all_outputs * model_weights.unsqueeze(-1)).sum(dim=1)
        
        # Mixture of Experts for each class
        expert_weights = []
        for gate in self.expert_gates:
            weight = torch.sigmoid(gate(selector_input))
            expert_weights.append(weight)
        
        expert_weights = torch.cat(expert_weights, dim=1)  # (batch, num_classes)
        
        # Meta-learning with all information
        meta_input = torch.cat([
            all_outputs.view(all_outputs.size(0), -1),  # Flatten all predictions
            features,
            group_probs
        ], dim=1)
        
        meta_output = self.meta_learner(meta_input)
        
        # Final combination with expert weighting
        final_output = (
            expert_weights * weighted_outputs + 
            (1 - expert_weights) * meta_output
        )
        
        # Temperature scaling for calibration
        final_output = final_output / self.temperature
        
        if return_all:
            return final_output, {
                'wavenet': wavenet_out,
                'mamba': mamba_out,
                'tcn': tcn_out,
                'graph': graph_out,
                'resnet': resnet_out,
                'transformer': transformer_out,
                'model_weights': model_weights,
                'group_probs': group_probs,
                'expert_weights': expert_weights
            }
        
        return final_output

# ==================== SNAPSHOT ENSEMBLE ====================

class SnapshotEnsembleTrainer:
    """Trainer com Snapshot Ensemble para múltiplos modelos durante treinamento"""
    def __init__(self, base_model, device, config):
        self.base_model = base_model
        self.device = device
        self.config = config
        
        # Lista para armazenar snapshots
        self.snapshots = []
        self.snapshot_epochs = []
        
        # Optimizer com cosine annealing
        self.optimizer = torch.optim.SGD(
            base_model.parameters(), 
            lr=config['lr_max'], 
            momentum=0.9,
            weight_decay=1e-4
        )
        
        self.criterion = OptimizedMultilabelLoss(
            config['class_frequencies'], 
            device
        )
        
        # Cosine annealing com múltiplos ciclos
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['cycle_length'],
            eta_min=config['lr_min']
        )
        
    def train_with_snapshots(self, train_loader, val_loader, num_cycles=5):
        """Treina com snapshot ensemble"""
        total_epochs = num_cycles * self.config['cycle_length']
        
        for epoch in range(total_epochs):
            # Training
            self.base_model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}"):
                ecg, features, labels = [x.to(self.device) for x in batch]
                
                self.optimizer.zero_grad()
                outputs = self.base_model(ecg, features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Salvar snapshot no final de cada ciclo
            if (epoch + 1) % self.config['cycle_length'] == 0:
                logger.info(f"Salvando snapshot no epoch {epoch+1}")
                snapshot = self.base_model.state_dict()
                self.snapshots.append({k: v.cpu().clone() for k, v in snapshot.items()})
                self.snapshot_epochs.append(epoch + 1)
            
            # Update scheduler
            self.scheduler.step()
            
            # Validation periodicamente
            if (epoch + 1) % 5 == 0:
                val_metrics = self._validate(val_loader)
                logger.info(f"Epoch {epoch+1} - Val AUC: {val_metrics['auc']:.4f}")
        
        return self.snapshots
    
    def _validate(self, val_loader):
        """Validação básica"""
        self.base_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                ecg, features, labels = [x.to(self.device) for x in batch]
                outputs = self.base_model(ecg, features)
                all_preds.append(torch.sigmoid(outputs))
                all_labels.append(labels)
        
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        
        auc = roc_auc_score(all_labels, all_preds, average='macro')
        return {'auc': auc}

# ==================== KNOWLEDGE DISTILLATION ====================

class DistilledECGModel(nn.Module):
    """Modelo destilado mais eficiente"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        # Arquitetura mais leve
        self.ecg_encoder = nn.Sequential(
            nn.Conv1d(12, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, ecg, features):
        ecg_feat = self.ecg_encoder(ecg).squeeze(-1)
        feat = self.feature_encoder(features)
        combined = torch.cat([ecg_feat, feat], dim=1)
        return self.classifier(combined)

def knowledge_distillation(teacher_model, student_model, train_loader, 
                          device, epochs=50, temperature=3.0):
    """Realiza knowledge distillation"""
    teacher_model.eval()
    student_model.to(device)
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        student_model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Distillation Epoch {epoch+1}/{epochs}"):
            ecg, features, labels = [x.to(device) for x in batch]
            
            # Teacher predictions
            with torch.no_grad():
                teacher_outputs = teacher_model(ecg, features)
                teacher_probs = F.sigmoid(teacher_outputs / temperature)
            
            # Student predictions
            student_outputs = student_model(ecg, features)
            student_probs = F.sigmoid(student_outputs / temperature)
            
            # Distillation loss
            distill_loss = F.binary_cross_entropy(student_probs, teacher_probs)
            
            # Standard loss
            standard_loss = F.binary_cross_entropy_with_logits(student_outputs, labels)
            
            # Combined loss
            loss = 0.7 * distill_loss + 0.3 * standard_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        logger.info(f"Distillation Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")
    
    return student_model

# ==================== IMPORTS E HELPERS DO SISTEMA ANTERIOR ====================

from ecg_ultra_optimized_complete import (
    ClinicalFeatureExtractor,
    AdvancedAugmentation,
    UltraOptimizedECGDataset,
    OptimizedMultilabelLoss,
    calculate_multilabel_metrics,
    create_weighted_sampler
)

# ==================== TRAINER ULTIMATE ====================

class UltimateEnsembleTrainer:
    """Trainer para o Ultimate Ensemble"""
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Diferentes learning rates para diferentes partes
        param_groups = [
            {'params': model.wavenet.parameters(), 'lr': config['lr']},
            {'params': model.mamba.parameters(), 'lr': config['lr']},
            {'params': model.tcn.parameters(), 'lr': config['lr']},
            {'params': model.graph.parameters(), 'lr': config['lr']},
            {'params': model.resnet.parameters(), 'lr': config['lr']},
            {'params': model.transformer.parameters(), 'lr': config['lr']},
            {'params': model.group_classifier.parameters(), 'lr': config['lr'] * 0.5},
            {'params': model.model_selector.parameters(), 'lr': config['lr'] * 0.1},
            {'params': model.expert_gates.parameters(), 'lr': config['lr'] * 0.1},
            {'params': model.meta_learner.parameters(), 'lr': config['lr'] * 0.5},
        ]
        
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
        
        # Loss function
        self.criterion = OptimizedMultilabelLoss(
            config['class_frequencies'], 
            device
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[group['lr'] for group in param_groups],
            epochs=config['epochs'],
            steps_per_epoch=config['steps_per_epoch'],
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Mixed precision - desabilitado para CPU
        self.scaler = None  # Forçar None para CPU
        
        # Tracking
        self.best_auc = 0.0
        self.history = defaultdict(list)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (ecg, features, labels) in enumerate(pbar):
            ecg = ecg.to(self.device)
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass sem mixed precision para CPU
            outputs = self.model(ecg, features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            self.scheduler.step()
            
            # Tracking
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).detach())
            all_labels.append(labels)
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        metrics = calculate_multilabel_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        individual_outputs = defaultdict(list)
        
        with torch.no_grad():
            for ecg, features, labels in tqdm(val_loader, desc="Validation"):
                ecg = ecg.to(self.device)
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions with individual model outputs
                outputs, model_outputs = self.model(ecg, features, return_all=True)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                all_preds.append(torch.sigmoid(outputs))
                all_labels.append(labels)
                
                # Store individual model predictions
                for key, value in model_outputs.items():
                    if key.endswith('_out') or key in ['wavenet', 'mamba', 'tcn', 'graph', 'resnet', 'transformer']:
                        individual_outputs[key].append(torch.sigmoid(value))
        
        # Calculate ensemble metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        metrics = calculate_multilabel_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(val_loader)
        
        # Calculate individual model metrics
        for model_name, preds in individual_outputs.items():
            if preds:
                model_preds = torch.cat(preds)
                model_auc = roc_auc_score(
                    all_labels.cpu().numpy(),
                    model_preds.cpu().numpy(),
                    average='macro'
                )
                metrics[f'{model_name}_auc'] = model_auc
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        logger.info("Iniciando treinamento do Ultimate Ensemble...")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"AUC: {train_metrics['auc_mean']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"AUC: {val_metrics['auc_mean']:.4f}, "
                       f"AUC Macro: {val_metrics['auc_macro']:.4f}")
            
            # Log individual model performance
            model_names = ['wavenet', 'mamba', 'tcn', 'graph', 'resnet', 'transformer']
            logger.info("Individual Model AUCs:")
            for name in model_names:
                if f'{name}_auc' in val_metrics:
                    logger.info(f"  {name}: {val_metrics[f'{name}_auc']:.4f}")
            
            # Save best model
            if val_metrics['auc_mean'] > self.best_auc:
                self.best_auc = val_metrics['auc_mean']
                self.save_checkpoint(epoch, val_metrics)
                logger.info(f"Novo melhor modelo! AUC: {self.best_auc:.4f}")
            
            # History tracking
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_auc'].append(train_metrics['auc_mean'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc_mean'])
            
            # Early stopping
            if epoch > 40:
                recent_aucs = self.history['val_auc'][-20:]
                if max(recent_aucs) <= self.best_auc - 0.001:
                    logger.info("Early stopping triggered!")
                    break
        
        return self.best_auc
    
    def save_checkpoint(self, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_auc': self.best_auc,
            'metrics': metrics,
            'history': dict(self.history),
            'config': self.config
        }
        torch.save(checkpoint, 'ultimate_ecg_ensemble.pth')
        
        # Detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in metrics.items()},
            'best_auc': float(self.best_auc),
            'model_info': {
                'architecture': 'UltimateECGEnsemble',
                'num_base_models': 6,
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'includes': [
                    'WaveNet', 'Mamba (SSM)', 'TCN', 'GNN', 
                    'Deep ResNet', 'Transformer', 'Hierarchical Classifier',
                    'Dynamic Model Selection', 'Mixture of Experts'
                ]
            }
        }
        
        with open('ultimate_ensemble_report.json', 'w') as f:
            json.dump(report, f, indent=2)

# ==================== MAIN ====================

def main():
    logger.info("="*80)
    logger.info("ULTIMATE ECG ENSEMBLE - ESTADO DA ARTE 2024")
    logger.info("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Data path
    data_path = Path(r"C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro2\ptbxl_processing\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\processed_npy\ptbxl_100hz")
    
    logger.info(f"\nCarregando dados de: {data_path}")
    
    # Load data
    try:
        X = np.load(data_path / 'X.npy')
        Y_multi = np.load(data_path / 'Y_multilabel.npy')
        logger.info(f"Dados carregados: X={X.shape}, Y={Y_multi.shape}")
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return
    
    # Class frequencies
    class_frequencies = Y_multi.sum(axis=0)
    
    # Train/val split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y_multi, test_size=0.2, random_state=42, 
        stratify=Y_multi.argmax(axis=1)
    )
    
    # Create feature extractor and augmenter
    feature_extractor = ClinicalFeatureExtractor(sampling_rate=100)
    augmenter = AdvancedAugmentation(sampling_rate=100)
    
    # Load or create feature cache
    cache_file = data_path / 'clinical_features_cache_ultimate.pkl'
    if cache_file.exists():
        logger.info("Carregando cache de features...")
        with open(cache_file, 'rb') as f:
            feature_cache = pickle.load(f)
    else:
        feature_cache = {}
    
    # Create datasets
    train_dataset = UltraOptimizedECGDataset(
        X_train, Y_train, feature_extractor, augmenter,
        feature_cache=feature_cache, is_training=True
    )
    
    val_dataset = UltraOptimizedECGDataset(
        X_val, Y_val, feature_extractor, augmenter,
        feature_cache=feature_cache, is_training=False
    )
    
    # Create weighted sampler
    train_sampler = create_weighted_sampler(Y_train)
    
    # DataLoaders - batch size otimizado para CPU
    batch_size = 8  # Reduzido para CPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False  # Desabilitado para CPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False  # Desabilitado para CPU
    )
    
    # Save cache
    if len(train_dataset.feature_cache) > len(feature_cache):
        logger.info("Salvando cache atualizado...")
        all_cache = {**train_dataset.feature_cache, **val_dataset.feature_cache}
        with open(cache_file, 'wb') as f:
            pickle.dump(all_cache, f)
    
    # Create model
    num_classes = Y_multi.shape[1]
    num_features = 50
    
    logger.info(f"\nCriando Ultimate Ensemble com 6 modelos especializados...")
    model = UltimateECGEnsemble(num_classes, num_features)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total de parâmetros: {total_params:,}")
    logger.info("Modelos incluídos:")
    logger.info("  1. WaveNet (Dilated Causal Convolutions)")
    logger.info("  2. Mamba (State Space Model)")
    logger.info("  3. TCN (Temporal Convolutional Network)")
    logger.info("  4. Graph Neural Network")
    logger.info("  5. Deep ResNet")
    logger.info("  6. Transformer")
    logger.info("  + Hierarchical Classifier")
    logger.info("  + Dynamic Model Selection")
    logger.info("  + Mixture of Experts")
    
    # Training config
    config = {
        'lr': 1e-3,
        'epochs': 100,
        'class_frequencies': class_frequencies,
        'steps_per_epoch': len(train_loader)
    }
    
    # Create trainer
    trainer = UltimateEnsembleTrainer(model, device, config)
    
    # Train
    logger.info("\nIniciando treinamento...")
    best_auc = trainer.train(train_loader, val_loader, config['epochs'])
    
    logger.info("\n" + "="*80)
    logger.info("TREINAMENTO CONCLUÍDO!")
    logger.info(f"Melhor AUC: {best_auc:.4f}")
    logger.info("Modelo salvo: ultimate_ecg_ensemble.pth")
    logger.info("="*80)
    
    # Optional: Knowledge Distillation
    logger.info("\nCriando modelo destilado para deployment...")
    student_model = DistilledECGModel(num_classes, num_features)
    
    # Distill knowledge
    distilled_model = knowledge_distillation(
        model, student_model, train_loader, device, epochs=30
    )
    
    torch.save(distilled_model.state_dict(), 'ecg_distilled_model.pth')
    logger.info("Modelo destilado salvo: ecg_distilled_model.pth")
    
    # Calculate distilled model size
    distilled_params = sum(p.numel() for p in distilled_model.parameters())
    logger.info(f"Redução de parâmetros: {total_params:,} → {distilled_params:,} "
                f"({(1 - distilled_params/total_params)*100:.1f}% menor)")

if __name__ == "__main__":
    main()
