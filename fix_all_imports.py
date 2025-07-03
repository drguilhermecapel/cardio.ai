#!/usr/bin/env python3
"""
Script para adicionar TODAS as importações necessárias ao train_ecg.py
"""

import sys
from pathlib import Path

# TODAS as importações necessárias em ordem correta
ALL_IMPORTS = '''#!/usr/bin/env python3
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
from functools import partial
import pickle
import random
import shutil
import math
import copy

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
    average_precision_score
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
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar matplotlib para não mostrar plots
plt.switch_backend('Agg')

# Seed para reprodutibilidade
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

'''

def fix_imports_in_file(file_path='train_ecg.py'):
    """Corrige todas as importações no arquivo"""
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ Erro: Arquivo '{file_path}' não encontrado!")
        return False
    
    try:
        # Fazer backup
        backup_path = file_path.with_suffix('.py.backup')
        
        # Ler o conteúdo atual
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Salvar backup
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"✅ Backup criado: {backup_path}")
        
        # Encontrar onde começam as importações ou código real
        lines = original_content.split('\n')
        
        # Remover shebang e docstrings do início se existirem
        start_idx = 0
        in_docstring = False
        docstring_chars = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Pular linhas vazias no início
            if not stripped and not in_docstring:
                continue
            
            # Detectar docstrings
            if not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = True
                    docstring_chars = stripped[:3]
                    if stripped.endswith(docstring_chars) and len(stripped) > 6:
                        in_docstring = False
                    continue
            else:
                if stripped.endswith(docstring_chars):
                    in_docstring = False
                    start_idx = i + 1
                    break
                continue
            
            # Se não é docstring nem linha vazia, encontramos o início do código
            if not stripped.startswith('#!') and not in_docstring:
                start_idx = i
                break
        
        # Encontrar onde terminam as importações existentes
        import_end_idx = start_idx
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if line and not (line.startswith('import ') or 
                           line.startswith('from ') or 
                           line.startswith('#') or
                           line == ''):
                import_end_idx = i
                break
        
        # Remover importações antigas e adicionar as novas
        new_lines = []
        
        # Adicionar shebang e docstrings originais se existirem
        for i in range(start_idx):
            new_lines.append(lines[i])
        
        # Adicionar todas as importações
        new_lines.extend(ALL_IMPORTS.split('\n')[start_idx:])  # Pular shebang duplicado
        
        # Adicionar o resto do código (pulando importações antigas)
        for i in range(import_end_idx, len(lines)):
            line = lines[i]
            # Pular linhas que tentam importar coisas que já importamos
            if (line.strip().startswith('import ') or 
                line.strip().startswith('from ') or
                line.strip() == 'warnings.filterwarnings'):
                continue
            new_lines.append(line)
        
        # Escrever o arquivo corrigido
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        
        print(f"✅ Arquivo '{file_path}' atualizado com sucesso!")
        print("✅ Todas as importações foram adicionadas!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao processar arquivo: {e}")
        traceback.print_exc()
        
        # Tentar restaurar o backup
        if backup_path.exists():
            try:
                shutil.copy2(backup_path, file_path)
                print("✅ Arquivo original restaurado do backup")
            except:
                print("❌ Erro ao restaurar backup!")
        
        return False

def test_imports():
    """Testa se as importações funcionam"""
    print("\n🔍 Testando importações...")
    
    test_imports = [
        "from tqdm import tqdm",
        "from scipy import integrate",
        "import warnings",
        "import pywt",
        "from einops import rearrange"
    ]
    
    all_ok = True
    for imp in test_imports:
        try:
            exec(imp)
            print(f"  ✅ {imp}")
        except ImportError as e:
            print(f"  ❌ {imp} - Erro: {e}")
            all_ok = False
    
    return all_ok

def main():
    """Função principal"""
    print("="*60)
    print("🔧 CORREÇÃO COMPLETA DE IMPORTAÇÕES - train_ecg.py")
    print("="*60)
    
    # Verificar se o arquivo existe
    file_path = "train_ecg.py"
    if not Path(file_path).exists():
        file_path = input("Digite o caminho completo para train_ecg.py: ").strip()
    
    # Testar importações primeiro
    if test_imports():
        print("\n✅ Todas as dependências estão instaladas!")
    else:
        print("\n⚠️  Algumas dependências estão faltando!")
        return
    
    # Perguntar se deseja continuar
    print(f"\n📄 Arquivo a ser corrigido: {file_path}")
    response = input("\nDeseja continuar? (s/n): ").strip().lower()
    
    if response == 's':
        success = fix_imports_in_file(file_path)
        
        if success:
            print("\n" + "="*60)
            print("✅ SUCESSO! Agora você pode executar:")
            print(f"   python {file_path} --mode train --data-path D:\\ptb-xl\\npy_lr --model-type ensemble --config-file config_ptbxl.json --num-epochs 100")
            print("="*60)
    else:
        print("\n❌ Operação cancelada!")

if __name__ == "__main__":
    main()