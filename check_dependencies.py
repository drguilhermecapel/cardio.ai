#!/usr/bin/env python3
"""
Script para verificar todas as dependências necessárias para o sistema ECG
"""

import sys
import subprocess
from importlib import import_module

# Lista de dependências necessárias
REQUIRED_PACKAGES = {
    'numpy': '1.21.0',
    'scipy': '1.7.0',
    'torch': '2.0.0',
    'tqdm': '4.60.0',
    'PyWavelets': '1.1.0',
    'scikit-learn': '0.24.0',
    'matplotlib': '3.3.0',
    'pandas': '1.3.0',
    'tensorboard': '2.6.0',
    'einops': '0.4.0',
    'torchvision': '0.15.0'
}

# Mapeamento de nomes de importação para nomes de pacote
IMPORT_NAMES = {
    'PyWavelets': 'pywt',
    'scikit-learn': 'sklearn',
    'Pillow': 'PIL'
}

def check_package(package_name, min_version=None):
    """Verifica se um pacote está instalado e sua versão"""
    import_name = IMPORT_NAMES.get(package_name, package_name)
    
    try:
        module = import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        print(f"✓ {package_name}: {version}", end="")
        
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f" (⚠️  versão mínima recomendada: {min_version})")
            else:
                print(" ✓")
        else:
            print()
            
        return True
    except ImportError:
        print(f"✗ {package_name}: NÃO INSTALADO")
        return False

def install_missing_packages(missing_packages):
    """Instala pacotes faltantes"""
    if not missing_packages:
        return
    
    print("\n" + "="*50)
    print("Instalando pacotes faltantes...")
    print("="*50 + "\n")
    
    for package in missing_packages:
        print(f"Instalando {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} instalado com sucesso!\n")
        except subprocess.CalledProcessError:
            print(f"✗ Erro ao instalar {package}\n")

def check_cuda():
    """Verifica disponibilidade de CUDA"""
    print("\n" + "="*50)
    print("Verificando CUDA/GPU:")
    print("="*50)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA disponível: {'✓ Sim' if cuda_available else '✗ Não'}")
        
        if cuda_available:
            print(f"Número de GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Versão CUDA: {torch.version.cuda}")
            print(f"Versão cuDNN: {torch.backends.cudnn.version()}")
    except Exception as e:
        print(f"Erro ao verificar CUDA: {e}")

def test_imports():
    """Testa imports específicos do projeto"""
    print("\n" + "="*50)
    print("Testando imports específicos:")
    print("="*50)
    
    test_cases = [
        ("from tqdm import tqdm", "tqdm"),
        ("from scipy import integrate", "scipy.integrate"),
        ("import scipy.signal", "scipy.signal"),
        ("import pywt", "PyWavelets"),
        ("from torch.utils.data import DataLoader", "torch.utils.data"),
        ("from sklearn.preprocessing import StandardScaler", "scikit-learn"),
        ("from einops import rearrange", "einops")
    ]
    
    for import_statement, package_name in test_cases:
        try:
            exec(import_statement)
            print(f"✓ {import_statement}")
        except ImportError as e:
            print(f"✗ {import_statement} - Erro: {e}")

def main():
    """Função principal"""
    print("="*50)
    print("Verificação de Dependências - Sistema ECG")
    print("="*50 + "\n")
    
    # Verificar pacotes
    missing_packages = []
    for package, min_version in REQUIRED_PACKAGES.items():
        if not check_package(package, min_version):
            missing_packages.append(package)
    
    # Instalar pacotes faltantes
    if missing_packages:
        response = input(f"\nDeseja instalar os {len(missing_packages)} pacotes faltantes? (s/n): ")
        if response.lower() == 's':
            install_missing_packages(missing_packages)
    
    # Verificar CUDA
    check_cuda()
    
    # Testar imports específicos
    test_imports()
    
    # Resumo final
    print("\n" + "="*50)
    if not missing_packages:
        print("✓ Todas as dependências estão instaladas!")
    else:
        print(f"⚠️  {len(missing_packages)} pacotes precisam ser instalados.")
    print("="*50)

if __name__ == "__main__":
    main()