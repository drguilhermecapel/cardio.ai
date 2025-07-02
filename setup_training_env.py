# backend/training/scripts/setup_training_env.py
"""
Script para configurar o ambiente de treinamento
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Instala dependências necessárias"""
    print("Instalando dependências do sistema de treinamento...")
    
    requirements_path = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_path.exists():
        print(f"Erro: {requirements_path} não encontrado!")
        return False
        
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
        ])
        print("✓ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Erro ao instalar dependências: {e}")
        return False


def create_directories():
    """Cria estrutura de diretórios necessária"""
    print("\nCriando estrutura de diretórios...")
    
    base_path = Path(__file__).parent.parent
    
    directories = [
        base_path / "data",
        base_path / "data" / "ptbxl",
        base_path / "data" / "mitbih",
        base_path / "data" / "cpsc2018",
        base_path / "checkpoints",
        base_path / "logs",
        base_path / "exported_models",
    ]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path}")
        
    return True


def verify_installation():
    """Verifica se a instalação está correta"""
    print("\nVerificando instalação...")
    
    # Verificar imports críticos
    critical_imports = [
        ("torch", "PyTorch"),
        ("wfdb", "WFDB"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
    ]
    
    all_ok = True
    for module_name, display_name in critical_imports:
        try:
            __import__(module_name)
            print(f"✓ {display_name} instalado corretamente")
        except ImportError:
            print(f"✗ {display_name} NÃO está instalado")
            all_ok = False
            
    # Verificar GPU se disponível
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU disponível: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ GPU não disponível, treinamento será feito na CPU")
    except:
        pass
        
    return all_ok


def main():
    print("=" * 60)
    print("SETUP DO AMBIENTE DE TREINAMENTO - CARDIOAI PRO")
    print("=" * 60)
    
    # Instalar requirements
    if not install_requirements():
        print("\n❌ Falha na instalação das dependências!")
        print("Por favor, instale manualmente:")
        print("pip install -r backend/training/requirements.txt")
        return
        
    # Criar diretórios
    create_directories()
    
    # Verificar instalação
    if verify_installation():
        print("\n✅ Ambiente configurado com sucesso!")
        print("\nPróximos passos:")
        print("1. Baixe os datasets: python backend/training/scripts/download_datasets.py")
        print("2. Inicie o treinamento: python backend/training/main.py --dataset ptbxl --model cnn_lstm")
    else:
        print("\n⚠️ Algumas dependências estão faltando!")
        print("Verifique a instalação e tente novamente.")


if __name__ == "__main__":
    main()

