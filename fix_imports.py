#!/usr/bin/env python3
"""
Script para adicionar automaticamente as importações faltantes ao train_ecg.py
"""

import re
from pathlib import Path

# Importações essenciais que devem estar presentes
ESSENTIAL_IMPORTS = """
# Importações essenciais
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

# Bibliotecas de processamento
from scipy import signal as scipy_signal
from scipy import integrate
from scipy.stats import zscore
import pywt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Utilitários
from tqdm import tqdm
import logging
import json
import yaml
import time
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd

# Configurações
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
"""

def check_existing_imports(file_content):
    """Verifica quais importações já existem no arquivo"""
    import_pattern = r'^(import|from)\s+[\w\.]+'
    existing_imports = set()
    
    for line in file_content.split('\n'):
        if re.match(import_pattern, line.strip()):
            existing_imports.add(line.strip())
    
    return existing_imports

def add_missing_imports(file_path):
    """Adiciona importações faltantes ao arquivo"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Erro: Arquivo {file_path} não encontrado!")
        return False
    
    # Fazer backup
    backup_path = file_path.with_suffix('.py.backup')
    
    try:
        # Ler conteúdo original
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Salvar backup
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Backup criado: {backup_path}")
        
        # Verificar importações existentes
        existing_imports = check_existing_imports(content)
        
        # Encontrar onde inserir as importações
        lines = content.split('\n')
        insert_index = 0
        
        # Procurar após shebang e docstrings
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                if 'import' in line or 'from' in line:
                    insert_index = i
                    break
        
        # Preparar novas importações
        new_imports = []
        for import_line in ESSENTIAL_IMPORTS.strip().split('\n'):
            if import_line.strip() and not any(import_line.strip() in existing for existing in existing_imports):
                new_imports.append(import_line)
        
        if new_imports:
            # Inserir importações
            import_block = '\n'.join(new_imports) + '\n\n'
            lines.insert(insert_index, import_block)
            
            # Escrever arquivo atualizado
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            print(f"✓ {len(new_imports)} importações adicionadas ao arquivo")
            return True
        else:
            print("✓ Todas as importações essenciais já estão presentes")
            return True
            
    except Exception as e:
        print(f"✗ Erro ao processar arquivo: {e}")
        
        # Restaurar backup em caso de erro
        if backup_path.exists():
            import shutil
            shutil.copy2(backup_path, file_path)
            print("✓ Arquivo original restaurado do backup")
        
        return False

def create_requirements_file():
    """Cria arquivo requirements.txt com todas as dependências"""
    requirements = """# Dependências do Sistema ECG
numpy>=1.21.0
scipy>=1.7.0
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0
tqdm>=4.60.0
PyWavelets>=1.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
pandas>=1.3.0
tensorboard>=2.6.0
einops>=0.4.0
pyyaml>=5.4.0
Pillow>=8.0.0
h5py>=3.0.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✓ Arquivo requirements.txt criado")

def main():
    """Função principal"""
    print("="*50)
    print("Correção de Importações - train_ecg.py")
    print("="*50 + "\n")
    
    # Perguntar pelo caminho do arquivo
    file_path = input("Digite o caminho para train_ecg.py (ou pressione Enter para usar o padrão): ").strip()
    if not file_path:
        file_path = "train_ecg.py"
    
    # Adicionar importações
    success = add_missing_imports(file_path)
    
    if success:
        print("\n✓ Importações corrigidas com sucesso!")
        
        # Criar requirements.txt
        response = input("\nDeseja criar um arquivo requirements.txt? (s/n): ")
        if response.lower() == 's':
            create_requirements_file()
    
    print("\n" + "="*50)
    print("Processo concluído!")
    print("="*50)

if __name__ == "__main__":
    main()