#!/usr/bin/env python3
"""
Sistema ECG Ultimate Ensemble - Estado da Arte 2024
Versão HIGH PERFORMANCE para 32GB RAM
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
import psutil
import gc

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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(42)

# ==================== MONITORAMENTO DE SISTEMA ====================

def log_system_info():
    """Log informações do sistema"""
    process = psutil.Process(os.getpid())
    mem_info = psutil.virtual_memory()
    
    logger.info(f"Sistema: {sys.platform}")
    logger.info(f"CPU: {psutil.cpu_count()} cores")
    logger.info(f"RAM Total: {mem_info.total / (1024**3):.1f} GB")
    logger.info(f"RAM Disponível: {mem_info.available / (1024**3):.1f} GB")
    logger.info(f"GPU: {'Disponível' if torch.cuda.is_available() else 'Não disponível'}")
    if torch.cuda.is_available():
        logger.info(f"GPU Nome: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memória: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

# ==================== COMPONENTES AVANÇADOS COMPLETOS ====================

class MambaBlock(nn.Module):
    """State Space Model (Mamba) block - versão completa para alta performance"""
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
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
        
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + self.d_state + self.d_state)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        self.A = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model)
        
        # Inicialização melhorada
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(self, x):
        """SSM forward pass with optimized memory usage"""
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
        
        # SSM step otimizado para 32GB RAM
        y = self.selective_scan_fast(x, delta, A, B, C, self.D.float())
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)
        
        return output
    
    def selective_scan_fast(self, u, delta, A, B, C, D):
        """Versão otimizada do selective scan para alta performance"""
        batch, length, d_in = u.shape
        n = A.shape[1]
        
        # Processar em chunks maiores (aproveitando RAM disponível)
        chunk_size = min(200, length)
        
        # Pre-alocar output
        y = torch.zeros_like(u)
        
        # Estado inicial
        x = torch.zeros(batch, d_in, n, device=u.device, dtype=u.dtype)
        
        for i in range(0, length, chunk_size):
            end_idx = min(i + chunk_size, length)
            
            # Processar chunk com operações vetorizadas
            u_chunk = u[:, i:end_idx]
            delta_chunk = delta[:, i:end_idx]
            B_chunk = B[:, i:end_idx]
            C_chunk = C[:, i:end_idx]
            
            # Computação eficiente
            deltaA_chunk = torch.exp(torch.einsum('blh,hc->blhc', delta_chunk, A))
            deltaB_chunk = torch.einsum('blh,blc->blhc', delta_chunk, B_chunk)
            
            # Scan otimizado
            for j in range(end_idx - i):
                x = deltaA_chunk[:, j] * x + deltaB_chunk[:, j] * u_chunk[:, j].unsqueeze(-1)
                y[:, i + j] = torch.einsum('bhc,bc->bh', x, C_chunk[:, j])
        
        # Add skip connection
        y = y + D * u
        
        return y

class TemporalConvNet(nn.Module):
    """TCN aprimorado com Squeeze-and-Excitation"""
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
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
            nn.Dropout(dropout),
            # Squeeze-and-Excitation
            SqueezeExcitation(out_channels)
        )
    
    def forward(self, x):
        return self.network(x)

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block para TCN"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ECGGraphNetwork(nn.Module):
    """Graph Neural Network aprimorada com attention"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Adjacency matrix for 12-lead ECG
        self.register_buffer('adj_matrix', self._create_ecg_adjacency())
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        self.dropout = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def _create_ecg_adjacency(self):
        """Create adjacency matrix based on ECG lead relationships"""
        adj = torch.zeros(12, 12)
        
        # Anatomical connections
        connections = [
            # Limb leads
            (0, 1), (0, 2), (1, 2),  # I, II, III
            (0, 3), (1, 3), (2, 3),  # aVR
            (0, 4), (1, 4),          # aVL
            (1, 5), (2, 5),          # aVF
            # Precordial leads
            (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),  # V1-V6
            # Cross connections
            (0, 6), (1, 9), (2, 11)  # Limb to precordial
        ]
        
        for i, j in connections:
            adj[i, j] = 1
            adj[j, i] = 1
        
        adj += torch.eye(12)
        return adj
    
    def forward(self, x, edge_index=None):
        batch_size = x.shape[0]
        
        if edge_index is None:
            edge_index = self.adj_matrix.nonzero().t()
        
        # GCN processing
        x_flat = x.view(-1, x.shape[-1])
        
        x_flat = F.relu(self.conv1(x_flat, edge_index))
        x_flat = self.dropout(x_flat)
        x_flat = F.relu(self.conv2(x_flat, edge_index))
        x_flat = self.dropout(x_flat)
        
        # Reshape for attention
        x_conv = x_flat.view(batch_size, 12, -1)
        
        # Self-attention
        x_att, _ = self.attention(x_conv, x_conv, x_conv)
        x_conv = self.norm1(x_conv + x_att)
        
        # Final GCN
        x_flat = x_conv.view(-1, x_conv.shape[-1])
        x_flat = self.conv3(x_flat, edge_index)
        x_out = x_flat.view(batch_size, 12, -1)
        
        # Global pooling with attention weights
        attention_weights = F.softmax(x_out.mean(dim=-1), dim=1)
        x_weighted = (x_out * attention_weights.unsqueeze(-1)).sum(dim=1)
        x_max = x_out.max(dim=1)[0]
        
        return torch.cat([x_weighted, x_max], dim=1)

# ==================== MODELOS ESPECIALIZADOS (VERSÃO COMPLETA) ====================

class WaveNetECG(nn.Module):
    """WaveNet completo com skip connections globais"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        self.input_conv = nn.Conv1d(12, 256, 1)
        
        # Full dilated convolution stack
        self.dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16]
        self.conv_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for dilation in self.dilations:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(256, 512, 3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Conv1d(512, 256, 1),
                    nn.BatchNorm1d(256)
                )
            )
            self.skip_convs.append(nn.Conv1d(256, 256, 1))
        
        # Feature integration
        self.feature_proj = nn.Linear(num_features, 256)
        
        # Output layers
        self.output_conv = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Linear(128 + 256, num_classes)
        
    def forward(self, ecg, features):
        x = self.input_conv(ecg)
        
        # WaveNet with skip connections
        skip_sum = 0
        for i, (conv_block, skip_conv) in enumerate(zip(self.conv_blocks, self.skip_convs)):
            residual = x
            x = conv_block(x)
            skip = skip_conv(x)
            x = x + residual
            x = F.relu(x)
            skip_sum = skip_sum + skip
        
        # Global features from skip connections
        x = self.output_conv(skip_sum).squeeze(-1)
        
        # Integrate clinical features
        feat = self.feature_proj(features)
        
        combined = torch.cat([x, feat], dim=1)
        return self.classifier(combined)

class MambaECG(nn.Module):
    """State Space Model completo para ECG"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        self.input_proj = nn.Conv1d(12, 384, 1)
        
        # Full Mamba stack
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(384, d_state=64, d_conv=4, expand=2)
            for _ in range(8)  # 8 blocos para máxima capacidade
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(384) for _ in range(8)
        ])
        
        # Feature integration with cross-attention
        self.feature_proj = nn.Linear(num_features, 384)
        self.cross_attention = nn.MultiheadAttention(384, num_heads=8, batch_first=True)
        
        # Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pre_classifier = nn.Sequential(
            nn.Linear(384, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, ecg, features):
        # Project input
        x = self.input_proj(ecg)
        x = x.transpose(1, 2)  # (B, L, C)
        
        # Apply Mamba blocks with residual connections
        for i, (mamba, norm) in enumerate(zip(self.mamba_blocks, self.norm_layers)):
            residual = x
            x = mamba(x)
            x = norm(x + residual)
            
            # Progressive feature fusion every 2 blocks
            if i % 2 == 1:
                feat_expanded = self.feature_proj(features).unsqueeze(1)
                x_att, _ = self.cross_attention(x, feat_expanded, feat_expanded)
                x = x + 0.1 * x_att
        
        # Pool and classify
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.pool(x).squeeze(-1)
        
        # Deep classification head
        x = self.pre_classifier(x)
        return self.classifier(x)

class TCNExpert(nn.Module):
    """TCN Expert com multi-scale processing"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        # Multi-scale TCN branches
        self.tcn_small = TemporalConvNet(
            num_inputs=12,
            num_channels=[64, 128, 256],
            kernel_size=3,
            dropout=0.2
        )
        
        self.tcn_large = TemporalConvNet(
            num_inputs=12,
            num_channels=[128, 256, 512, 1024],
            kernel_size=5,
            dropout=0.2
        )
        
        # Feature processing with self-attention
        self.feature_net = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(256 + 1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, ecg, features):
        # Multi-scale TCN processing
        x_small = self.tcn_small(ecg)
        x_large = self.tcn_large(ecg)
        
        # Global pooling
        x_small = self.global_pool(x_small).squeeze(-1)
        x_large = self.global_pool(x_large).squeeze(-1)
        
        # Fusion
        x_tcn = torch.cat([x_small, x_large], dim=1)
        x_tcn = self.fusion(x_tcn)
        
        # Feature processing
        feat = self.feature_net(features)
        
        # Combine and classify
        combined = torch.cat([x_tcn, feat], dim=1)
        return self.classifier(combined)

class GraphECGExpert(nn.Module):
    """Graph Neural Network Expert com hierarchical processing"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        # Hierarchical lead encoders
        self.lead_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 128, 7, padding=3),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, 5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(128, 256, 5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, 3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            ) for _ in range(12)
        ])
        
        # Enhanced Graph network
        self.graph_net = ECGGraphNetwork(256, 512, 256)
        
        # Feature integration with gating
        self.feature_proj = nn.Linear(num_features, 256)
        self.feature_gate = nn.Sequential(
            nn.Linear(num_features + 512, 512),
            nn.Sigmoid()
        )
        
        # Deep classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 256, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, ecg, features):
        # Extract features per lead
        lead_features = []
        for i in range(12):
            lead_feat = self.lead_encoders[i](ecg[:, i:i+1, :])
            lead_features.append(lead_feat.squeeze(-1))
        
        # Stack features
        x = torch.stack(lead_features, dim=1)  # (batch, 12, 256)
        
        # Apply graph network
        graph_features = self.graph_net(x)
        
        # Gated feature fusion
        feat = self.feature_proj(features)
        gate_input = torch.cat([graph_features, features], dim=1)
        gate = self.feature_gate(gate_input)
        graph_features = graph_features * gate
        
        # Combine and classify
        combined = torch.cat([graph_features, feat], dim=1)
        return self.classifier(combined)

class DeepResNetECG(nn.Module):
    """Very deep ResNet with dense connections"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(12, 128, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        
        # ResNet blocks with increasing depth
        self.layer1 = self._make_layer(128, 256, 4)
        self.layer2 = self._make_layer(256, 512, 6, stride=2)
        self.layer3 = self._make_layer(512, 1024, 8, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 4, stride=2)
        
        # Feature integration
        self.feature_proj = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
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
    """Enhanced Residual Block with SE"""
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
        
        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(out_channels)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ==================== ENSEMBLE HIERÁRQUICO AVANÇADO ====================

class PathologyGroupClassifier(nn.Module):
    """Classificador hierárquico de patologias"""
    def __init__(self, input_dim, num_groups=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_groups)
        )
    
    def forward(self, x):
        return self.classifier(x)

class UltimateECGEnsemble(nn.Module):
    """Ultimate Ensemble para alta performance"""
    def __init__(self, num_classes, num_features=50):
        super().__init__()
        
        # Modelos base do ensemble
        self.wavenet = WaveNetECG(num_classes, num_features)
        self.mamba = MambaECG(num_classes, num_features)
        self.tcn = TCNExpert(num_classes, num_features)
        self.graph = GraphECGExpert(num_classes, num_features)
        self.resnet = DeepResNetECG(num_classes, num_features)
        
        # Transformer do ensemble anterior
        from ecg_ultra_optimized_complete import HolisticAnalyzer
        self.transformer = HolisticAnalyzer(num_classes, num_features)
        
        # Classificador hierárquico
        self.group_classifier = PathologyGroupClassifier(num_features, num_groups=5)
        
        # Dynamic Model Selection com attention
        self.model_selector = nn.Sequential(
            nn.Linear(num_features + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6),
            nn.Softmax(dim=1)
        )
        
        # Model confidence estimator
        self.confidence_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_classes, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(6)
        ])
        
        # Advanced Meta-Learner
        self.expert_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features + 5, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(num_classes)
        ])
        
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * 6 + num_features + 5 + 6, 1024),  # +6 for confidences
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(num_classes * 6, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, ecg, features, return_all=False):
        # Get all model predictions
        wavenet_out = self.wavenet(ecg, features)
        mamba_out = self.mamba(ecg, features)
        tcn_out = self.tcn(ecg, features)
        graph_out = self.graph(ecg, features)
        resnet_out = self.resnet(ecg, features)
        transformer_out = self.transformer(ecg, features)
        
        # Stack all outputs
        all_outputs = torch.stack([
            wavenet_out, mamba_out, tcn_out, 
            graph_out, resnet_out, transformer_out
        ], dim=1)  # (batch, 6, num_classes)
        
        # Estimate model confidences
        confidences = []
        for i, (output, estimator) in enumerate(zip(
            [wavenet_out, mamba_out, tcn_out, graph_out, resnet_out, transformer_out],
            self.confidence_estimator
        )):
            conf = estimator(output)
            confidences.append(conf)
        confidences = torch.cat(confidences, dim=1)  # (batch, 6)
        
        # Group classification
        group_logits = self.group_classifier(features)
        group_probs = F.softmax(group_logits, dim=1)
        
        # Dynamic model selection with confidence weighting
        selector_input = torch.cat([features, group_probs], dim=1)
        model_weights = self.model_selector(selector_input)
        model_weights = model_weights * confidences  # Weight by confidence
        model_weights = model_weights / (model_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted combination
        weighted_outputs = (all_outputs * model_weights.unsqueeze(-1)).sum(dim=1)
        
        # Mixture of Experts
        expert_weights = []
        for gate in self.expert_gates:
            weight = torch.sigmoid(gate(selector_input))
            expert_weights.append(weight)
        expert_weights = torch.cat(expert_weights, dim=1)
        
        # Meta-learning with all information
        meta_input = torch.cat([
            all_outputs.view(all_outputs.size(0), -1),
            features,
            group_probs,
            confidences
        ], dim=1)
        
        meta_output = self.meta_learner(meta_input)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(all_outputs.view(all_outputs.size(0), -1))
        uncertainty = torch.sigmoid(uncertainty)
        
        # Final combination with expert weighting and uncertainty
        final_output = (
            expert_weights * weighted_outputs + 
            (1 - expert_weights) * meta_output
        ) * (1 - uncertainty * 0.1)  # Slight penalty for uncertainty
        
        # Temperature scaling
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
                'confidences': confidences,
                'group_probs': group_probs,
                'expert_weights': expert_weights,
                'uncertainty': uncertainty
            }
        
        return final_output

# ==================== IMPORTS E HELPERS ====================

from ecg_ultra_optimized_complete import (
    ClinicalFeatureExtractor,
    AdvancedAugmentation,
    UltraOptimizedECGDataset,
    OptimizedMultilabelLoss,
    calculate_multilabel_metrics,
    create_weighted_sampler
)

# ==================== TRAINER DE ALTA PERFORMANCE ====================

class UltimateEnsembleTrainer:
    """Trainer otimizado para alta performance com 32GB RAM"""
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Gradient accumulation para batch size efetivo maior
        self.accumulation_steps = config.get('accumulation_steps', 4)
        
        # Diferentes learning rates
        param_groups = [
            {'params': model.wavenet.parameters(), 'lr': config['lr']},
            {'params': model.mamba.parameters(), 'lr': config['lr']},
            {'params': model.tcn.parameters(), 'lr': config['lr']},
            {'params': model.graph.parameters(), 'lr': config['lr']},
            {'params': model.resnet.parameters(), 'lr': config['lr']},
            {'params': model.transformer.parameters(), 'lr': config['lr']},
            {'params': model.group_classifier.parameters(), 'lr': config['lr'] * 0.5},
            {'params': model.model_selector.parameters(), 'lr': config['lr'] * 0.1},
            {'params': model.confidence_estimator.parameters(), 'lr': config['lr'] * 0.5},
            {'params': model.expert_gates.parameters(), 'lr': config['lr'] * 0.1},
            {'params': model.meta_learner.parameters(), 'lr': config['lr'] * 0.5},
            {'params': model.uncertainty_head.parameters(), 'lr': config['lr'] * 0.5},
        ]
        
        # Optimizer com lookahead
        base_optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
        self.optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
        
        # Loss function
        self.criterion = OptimizedMultilabelLoss(
            config['class_frequencies'], 
            device
        )
        
        # Scheduler - Cosine with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision se GPU disponível
        self.scaler = GradScaler() if device.type == 'cuda' else None
        
        # Tracking
        self.best_auc = 0.0
        self.history = defaultdict(list)
        
        # EMA para model averaging
        self.ema = ExponentialMovingAverage(model.parameters(), decay=0.995)
    
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
            
            # Mixed precision training
            if self.scaler:
                with autocast():
                    outputs = self.model(ecg, features)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.ema.update()
            else:
                outputs = self.model(ecg, features)
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.ema.update()
            
            # Tracking
            total_loss += loss.item() * self.accumulation_steps
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                all_preds.append(torch.sigmoid(outputs).detach())
                all_labels.append(labels)
            
            pbar.set_postfix({
                'loss': loss.item() * self.accumulation_steps,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Scheduler step
            self.scheduler.step()
        
        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        metrics = calculate_multilabel_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, val_loader):
        self.model.eval()
        self.ema.store()
        self.ema.copy_to()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_uncertainties = []
        individual_outputs = defaultdict(list)
        
        with torch.no_grad():
            for ecg, features, labels in tqdm(val_loader, desc="Validation"):
                ecg = ecg.to(self.device)
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions
                outputs, model_outputs = self.model(ecg, features, return_all=True)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                all_preds.append(torch.sigmoid(outputs))
                all_labels.append(labels)
                all_uncertainties.append(model_outputs['uncertainty'])
                
                # Store individual model predictions
                for key in ['wavenet', 'mamba', 'tcn', 'graph', 'resnet', 'transformer']:
                    individual_outputs[key].append(torch.sigmoid(model_outputs[key]))
        
        self.ema.restore()
        
        # Calculate metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_uncertainties = torch.cat(all_uncertainties)
        
        metrics = calculate_multilabel_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(val_loader)
        metrics['uncertainty_mean'] = all_uncertainties.mean().item()
        
        # Individual model metrics
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
        logger.info("="*80)
        logger.info("Iniciando treinamento Ultimate Ensemble - High Performance")
        logger.info(f"Configuração:")
        logger.info(f"  - Batch size: {self.config.get('batch_size', 16)}")
        logger.info(f"  - Gradient accumulation: {self.accumulation_steps}")
        logger.info(f"  - Effective batch size: {self.config.get('batch_size', 16) * self.accumulation_steps}")
        logger.info(f"  - Learning rate: {self.config['lr']}")
        logger.info(f"  - Device: {self.device}")
        logger.info("="*80)
        
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
                       f"Uncertainty: {val_metrics['uncertainty_mean']:.4f}")
            
            # Log individual model performance
            logger.info("Performance individual dos modelos:")
            for name in ['wavenet', 'mamba', 'tcn', 'graph', 'resnet', 'transformer']:
                if f'{name}_auc' in val_metrics:
                    logger.info(f"  {name}: {val_metrics[f'{name}_auc']:.4f}")
            
            # Save best model
            if val_metrics['auc_mean'] > self.best_auc:
                self.best_auc = val_metrics['auc_mean']
                self.save_checkpoint(epoch, val_metrics)
                logger.info(f"🏆 Novo melhor modelo! AUC: {self.best_auc:.4f}")
            
            # History
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_auc'].append(train_metrics['auc_mean'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc_mean'])
            
            # Early stopping com patience maior
            if epoch > 50:
                recent_aucs = self.history['val_auc'][-30:]
                if max(recent_aucs) <= self.best_auc - 0.001:
                    logger.info("Early stopping triggered!")
                    break
        
        return self.best_auc
    
    def save_checkpoint(self, epoch, metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'best_auc': self.best_auc,
            'metrics': metrics,
            'history': dict(self.history),
            'config': self.config
        }
        
        torch.save(checkpoint, 'ultimate_ecg_ensemble_best.pth')
        
        # Report detalhado
        report = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in metrics.items()},
            'best_auc': float(self.best_auc),
            'model_info': {
                'architecture': 'UltimateECGEnsemble High Performance',
                'num_base_models': 6,
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'features': [
                    'Full MambaBlock with d_state=64',
                    'Deep architectures preserved',
                    'Gradient accumulation',
                    'Model confidence estimation',
                    'Uncertainty quantification',
                    'EMA model averaging',
                    'Lookahead optimizer'
                ]
            }
        }
        
        with open('ultimate_ensemble_report.json', 'w') as f:
            json.dump(report, f, indent=2)

# ==================== COMPONENTES AUXILIARES ====================

class Lookahead(torch.optim.Optimizer):
    """Lookahead optimizer wrapper"""
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = base_optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = base_optimizer.state
        
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            fast.data.copy_(slow + self.alpha * (fast.data - slow))
            slow.copy_(fast.data)
    
    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

class ExponentialMovingAverage:
    """EMA for model parameters"""
    def __init__(self, parameters, decay=0.995):
        self.parameters = list(parameters)
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in self.parameters]
        self.backup_params = []
    
    def update(self):
        for s_param, param in zip(self.shadow_params, self.parameters):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def copy_to(self):
        for s_param, param in zip(self.shadow_params, self.parameters):
            param.data.copy_(s_param.data)
    
    def store(self):
        self.backup_params = [p.clone() for p in self.parameters]
    
    def restore(self):
        for b_param, param in zip(self.backup_params, self.parameters):
            param.data.copy_(b_param.data)
    
    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow_params": self.shadow_params,
        }

# ==================== MAIN ====================

def main():
    logger.info("="*80)
    logger.info("ULTIMATE ECG ENSEMBLE - HIGH PERFORMANCE VERSION")
    logger.info("Otimizado para 32GB RAM")
    logger.info("="*80)
    
    # Log system info
    log_system_info()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\nDevice selecionado: {device}")
    
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
    
    # Feature extractor and augmenter
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
    
    # DataLoaders - batch size otimizado para 32GB RAM
    batch_size = 32 if device.type == 'cuda' else 16
    num_workers = min(8, os.cpu_count() - 1)  # Usar múltiplos workers
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=True if num_workers > 0 else False
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
    
    logger.info(f"\nCriando Ultimate Ensemble - High Performance...")
    model = UltimateECGEnsemble(num_classes, num_features)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total de parâmetros: {total_params:,}")
    logger.info("\nCaracterísticas do modelo:")
    logger.info("  ✓ 6 modelos especializados (WaveNet, Mamba, TCN, GNN, ResNet, Transformer)")
    logger.info("  ✓ MambaBlock completo com d_state=64")
    logger.info("  ✓ Arquiteturas profundas preservadas")
    logger.info("  ✓ Model confidence estimation")
    logger.info("  ✓ Uncertainty quantification")
    logger.info("  ✓ Hierarchical pathology classification")
    logger.info("  ✓ Dynamic model selection with attention")
    
    # Training config
    config = {
        'lr': 1e-3,
        'epochs': 100,
        'batch_size': batch_size,
        'accumulation_steps': 4,  # Effective batch size = 64 ou 128
        'class_frequencies': class_frequencies,
        'steps_per_epoch': len(train_loader)
    }
    
    # Create trainer
    trainer = UltimateEnsembleTrainer(model, device, config)
    
    # Train
    logger.info("\nIniciando treinamento de alta performance...")
    logger.info(f"Effective batch size: {batch_size * config['accumulation_steps']}")
    
    best_auc = trainer.train(train_loader, val_loader, config['epochs'])
    
    logger.info("\n" + "="*80)
    logger.info("🎯 TREINAMENTO CONCLUÍDO!")
    logger.info(f"🏆 Melhor AUC: {best_auc:.4f}")
    logger.info("💾 Modelo salvo: ultimate_ecg_ensemble_best.pth")
    logger.info("📊 Relatório salvo: ultimate_ensemble_report.json")
    logger.info("="*80)
    
    # Garbage collection
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    main()
