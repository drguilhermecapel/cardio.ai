#!/usr/bin/env python3
"""
Script de configuração do ambiente de treinamento para cardio.ai.pro
Versão sem emojis - Compatível com Windows cp1252
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil
import json

def check_python_version():
    """Verifica se a versão do Python é adequada"""
    print("[INFO] Verificando versao do Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"[ERRO] Python {version.major}.{version.minor} detectado. Python 3.8+ necessario.")
        sys.exit(1)
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro} detectado")

def create_directories():
    """Cria estrutura de diretórios necessária"""
    print("\n[INFO] Criando estrutura de diretorios...")
    
    base_dir = Path(__file__).parent.parent
    directories = [
        "data/raw",
        "data/processed", 
        "data/features",
        "models/checkpoints",
        "models/trained",
        "logs/training",
        "logs/evaluation",
        "outputs/predictions",
        "outputs/reports",
        "configs/model",
        "configs/training"
    ]
    
    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"   [OK] {dir_path}")

def create_venv():
    """Cria ambiente virtual Python"""
    print("\n[INFO] Configurando ambiente virtual...")
    
    venv_path = Path(__file__).parent.parent / "venv"
    
    if venv_path.exists():
        print("   [INFO] Ambiente virtual ja existe")
        return venv_path
    
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print("   [OK] Ambiente virtual criado")
        return venv_path
    except subprocess.CalledProcessError as e:
        print(f"   [ERRO] Erro ao criar ambiente virtual: {e}")
        sys.exit(1)

def get_pip_command(venv_path):
    """Retorna o comando pip apropriado para o SO"""
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "pip.exe")
    else:
        return str(venv_path / "bin" / "pip")

def get_python_command(venv_path):
    """Retorna o comando python apropriado para o SO"""
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "python.exe")
    else:
        return str(venv_path / "bin" / "python")

def install_requirements(venv_path):
    """Instala dependências do projeto"""
    print("\n[INFO] Instalando dependencias...")
    
    pip_cmd = get_pip_command(venv_path)
    python_cmd = get_python_command(venv_path)
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("   [AVISO] requirements.txt nao encontrado. Criando arquivo base...")
        create_basic_requirements(requirements_file)
    
    # Atualiza pip (método compatível com Windows)
    try:
        if platform.system() == "Windows":
            subprocess.run([python_cmd, "-m", "pip", "install", "--upgrade", "pip"], 
                         capture_output=True, text=True)
        else:
            subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        print("   [OK] pip atualizado (ou ja esta atualizado)")
    except subprocess.CalledProcessError:
        print("   [AVISO] Nao foi possivel atualizar pip, continuando com versao atual...")
    
    # Instala requirements
    try:
        print("   [INFO] Instalando pacotes (isso pode levar alguns minutos)...")
        
        # Instala pacotes essenciais primeiro
        essential_packages = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.4.0"
        ]
        
        for package in essential_packages:
            try:
                subprocess.run([pip_cmd, "install", package], 
                             capture_output=True, text=True, check=True)
                print(f"   [OK] {package.split('>=')[0]} instalado")
            except subprocess.CalledProcessError:
                print(f"   [AVISO] Erro ao instalar {package}: continuando...")
        
        # Instala o resto do requirements
        result = subprocess.run([pip_cmd, "install", "-r", str(requirements_file)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   [OK] Todas as dependencias instaladas com sucesso!")
        else:
            print("   [AVISO] Algumas dependencias podem nao ter sido instaladas.")
            print("   [DICA] Voce pode tentar instalar manualmente depois.")
            
    except Exception as e:
        print(f"   [ERRO] Erro inesperado: {e}")
        print("   [DICA] Tente instalar as dependencias manualmente apos ativar o ambiente.")

def create_basic_requirements(requirements_file):
    """Cria arquivo requirements.txt básico para projetos de ML cardíaco"""
    basic_requirements = """# Core ML/DL
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.10.0
tensorflow>=2.7.0
xgboost>=1.5.0

# Data processing
scipy>=1.7.0
wfdb>=3.4.0  # Para dados ECG
pyedflib>=0.1.0  # Para arquivos EDF
biosppy>=0.7.0  # Processamento de sinais biomedicos
heartpy>=1.2.0  # Analise de frequencia cardiaca

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# ML utilities
optuna>=2.10.0  # Otimizacao de hiperparametros
mlflow>=1.20.0  # Tracking de experimentos
wandb>=0.12.0  # Tracking alternativo

# Medical imaging (se aplicavel)
# opencv-python>=4.5.0
# pydicom>=2.2.0
# SimpleITK>=2.1.0

# Utils
tqdm>=4.62.0
pyyaml>=5.4.0
jupyter>=1.0.0
pytest>=6.2.0
black>=21.0.0
flake8>=3.9.0
"""
    
    with open(requirements_file, 'w', encoding='utf-8') as f:
        f.write(basic_requirements)
    print(f"   [OK] {requirements_file} criado")

def create_config_templates():
    """Cria templates de configuração"""
    print("\n[INFO] Criando arquivos de configuracao...")
    
    base_dir = Path(__file__).parent.parent
    
    # Config de treinamento
    training_config = {
        "model": {
            "type": "cnn_lstm",
            "input_shape": [1000, 12],  # 1000 samples, 12 leads ECG
            "num_classes": 5,
            "architecture": {
                "cnn_filters": [32, 64, 128],
                "lstm_units": [128, 64],
                "dropout_rate": 0.3
            }
        },
        "training": {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy", "auc"],
            "early_stopping": {
                "patience": 10,
                "monitor": "val_loss",
                "mode": "min"
            }
        },
        "data": {
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "sampling_rate": 500,
            "preprocessing": {
                "normalize": True,
                "remove_baseline": True,
                "filter_noise": True
            }
        }
    }
    
    config_file = base_dir / "configs" / "training" / "default_config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(training_config, f, indent=4)
    print(f"   [OK] {config_file.name} criado")
    
    # Config de modelo
    model_config = {
        "ecg_classifier": {
            "model_type": "hybrid_cnn_lstm",
            "input_specs": {
                "signal_length": 1000,
                "num_leads": 12,
                "sampling_rate": 500
            },
            "classes": [
                "Normal",
                "Atrial Fibrillation",
                "Myocardial Infarction",
                "Heart Failure",
                "Other Arrhythmia"
            ],
            "preprocessing": {
                "bandpass_filter": [0.5, 40],
                "notch_filter": 60,
                "normalize": "z-score"
            }
        }
    }
    
    model_config_file = base_dir / "configs" / "model" / "ecg_classifier.json"
    with open(model_config_file, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=4)
    print(f"   [OK] {model_config_file.name} criado")

def create_example_scripts():
    """Cria scripts de exemplo para treinamento"""
    print("\n[INFO] Criando scripts de exemplo...")
    
    train_script = '''#!/usr/bin/env python3
"""
Script de exemplo para treinamento de modelo de classificacao ECG
"""

import json
import argparse
from pathlib import Path
import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import mlflow

def load_config(config_path):
    """Carrega configuracao de treinamento"""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_model(config):
    """Cria modelo baseado na configuracao"""
    # TODO: Implementar arquitetura do modelo aqui
    print("[INFO] Modelo sera criado com base em:", config)
    pass

def train_epoch(model, dataloader, criterion, optimizer):
    """Treina uma epoca"""
    # TODO: Implementar loop de treinamento
    total_loss = 0
    return total_loss

def main():
    parser = argparse.ArgumentParser(description='Treinar modelo cardiaco')
    parser.add_argument('--config', type=str, default='configs/training/default_config.json')
    parser.add_argument('--data', type=str, default='data/processed')
    parser.add_argument('--experiment', type=str, default='ecg_classification')
    args = parser.parse_args()
    
    # Carregar configuracao
    print("[INFO] Carregando configuracao de:", args.config)
    config = load_config(args.config)
    
    # Configurar MLflow (quando disponivel)
    # mlflow.set_experiment(args.experiment)
    
    # with mlflow.start_run():
    #     mlflow.log_params(config['training'])
    
    # Criar modelo
    model = create_model(config['model'])
    
    # Treinar modelo
    print("[INFO] Iniciando treinamento...")
    print("      Dados em:", args.data)
    print("      Experimento:", args.experiment)
    # TODO: Implementar treinamento completo
    
    print("[OK] Treinamento concluido!")

if __name__ == "__main__":
    main()
'''
    
    train_file = Path(__file__).parent.parent / "scripts" / "train_model.py"
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(train_script)
    print(f"   [OK] train_model.py criado")

def create_activation_script():
    """Cria scripts de ativação do ambiente"""
    print("\n[INFO] Criando scripts de ativacao...")
    
    base_dir = Path(__file__).parent.parent
    
    if platform.system() == "Windows":
        # Script PowerShell
        ps_script = f'''# Ativa ambiente virtual no Windows PowerShell
$venvPath = "{base_dir}\\venv\\Scripts\\Activate.ps1"
if (Test-Path $venvPath) {{
    & $venvPath
    Write-Host "[OK] Ambiente virtual ativado!" -ForegroundColor Green
    Write-Host "[INFO] Diretorio do projeto: {base_dir}" -ForegroundColor Cyan
}} else {{
    Write-Host "[ERRO] Ambiente virtual nao encontrado!" -ForegroundColor Red
    Write-Host "Execute primeiro: python scripts/setup_training_clean.py" -ForegroundColor Yellow
}}
'''
        activate_file = base_dir / "activate_env.ps1"
        with open(activate_file, 'w', encoding='utf-8') as f:
            f.write(ps_script)
        print(f"   [OK] {activate_file.name} criado")
        
        # Script CMD
        cmd_script = f'''@echo off
REM Ativa ambiente virtual no Windows CMD
set VENV_PATH={base_dir}\\venv\\Scripts\\activate.bat
if exist "%VENV_PATH%" (
    call "%VENV_PATH%"
    echo [OK] Ambiente virtual ativado!
    echo [INFO] Diretorio do projeto: {base_dir}
) else (
    echo [ERRO] Ambiente virtual nao encontrado!
    echo Execute primeiro: python scripts/setup_training_clean.py
)
'''
        cmd_file = base_dir / "activate_env.bat"
        with open(cmd_file, 'w', encoding='utf-8') as f:
            f.write(cmd_script)
        print(f"   [OK] {cmd_file.name} criado")

def print_next_steps():
    """Imprime próximos passos"""
    print("\n" + "="*50)
    print("[OK] CONFIGURACAO CONCLUIDA!")
    print("="*50)
    
    print("\n[INFO] Proximos passos:")
    
    if platform.system() == "Windows":
        print("\n1. Ativar ambiente virtual:")
        print("   PowerShell: .\\activate_env.ps1")
        print("   CMD: activate_env.bat")
    else:
        print("\n1. Ativar ambiente virtual:")
        print("   source venv/bin/activate")
    
    print("\n2. Adicionar seus dados em: data/raw/")
    print("\n3. Ajustar configuracoes em: configs/training/default_config.json")
    print("\n4. Executar treinamento:")
    print("   python scripts/train_model.py --config configs/training/default_config.json")
    
    print("\n[DICAS]:")
    print("   - Use MLflow UI para acompanhar experimentos: mlflow ui")
    print("   - Logs de treinamento em: logs/training/")
    print("   - Modelos salvos em: models/trained/")
    print("\n[AVISO]: Algumas bibliotecas pesadas podem demorar para instalar")
    print("         Instale uma por vez se houver problemas:")
    print("         pip install torch")
    print("         pip install tensorflow")
    print("         pip install wfdb biosppy heartpy")

from backend.training.config.dataset_configs import get_dataset_config

def main():
    """Função principal"""
    print("[CARDIO.AI.PRO] Setup de Ambiente de Treinamento")
    print("="*50)
    
    # Verificações
    check_python_version()
    

    # Verificações
    check_python_version()

    # Setup
    create_directories()
    venv_path = create_venv()
    install_requirements(venv_path)
    create_config_templates()
    setup_ptb_xl_training()
    create_example_scripts()
    create_activation_script()

    
    # Finalização
    print_next_steps()

def setup_ptb_xl_training():
    """Configura e inicia o treinamento usando o dataset PTB-XL"""
    print("[INFO] Iniciando configuração do treinamento para o dataset PTB-XL...")
    from backend.training.data_loader import PTBXLLoader
    from backend.training.models import ECGModel
    from backend.training.train import train

    # Carregar configuração do dataset PTB-XL
    config = get_dataset_config('ptbxl')

    # Carregar o dataset PTB-XL
    loader = PTBXLLoader(config)
    train_data, val_data, test_data = loader.load_data()

    # Definir o modelo
    model = ECGModel(input_shape=config.input_shape, num_classes=len(config.classes))

    # Configurar e executar o treinamento
    train(model, train_data, val_data, config)

    print("[INFO] Treinamento para o dataset PTB-XL concluído com sucesso.")

if __name__ == "__main__":
    main()
