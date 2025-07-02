#!/usr/bin/env python3
"""
Script de Verificação e Instalação de TODAS as Bibliotecas
CardioAI Pro v2.0.0
"""

import subprocess
import sys
import importlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lista de TODAS as bibliotecas necessárias
REQUIRED_LIBRARIES = {
    # Bibliotecas principais
    'fastapi': 'fastapi>=0.104.1',
    'uvicorn': 'uvicorn[standard]>=0.24.0',
    'pydantic': 'pydantic>=2.5.0',
    'sqlalchemy': 'sqlalchemy>=2.0.23',
    'numpy': 'numpy>=1.24.0',
    'pandas': 'pandas>=2.1.0',
    'scipy': 'scipy>=1.11.0',
    
    # Machine Learning
    'sklearn': 'scikit-learn>=1.3.0',
    'torch': 'torch>=2.1.0',
    'tensorflow': 'tensorflow>=2.15.0',
    
    # Processamento de ECG
    'wfdb': 'wfdb>=4.1.0',
    'pyedflib': 'pyedflib>=0.1.36',
    'pywt': 'pywt>=1.4.1',
    'neurokit2': 'neurokit2>=0.2.7',
    
    # Visualização
    'matplotlib': 'matplotlib>=3.8.0',
    'plotly': 'plotly>=5.17.0',
    'seaborn': 'seaborn>=0.13.0',
    
    # Utilitários
    'requests': 'requests>=2.31.0',
    'python-dotenv': 'python-dotenv>=1.0.0',
    'tqdm': 'tqdm>=4.66.0',
    
    # Desenvolvimento
    'pytest': 'pytest>=7.4.0',
    'black': 'black>=23.11.0',
}

# Bibliotecas opcionais (não críticas)
OPTIONAL_LIBRARIES = {
    'onnxruntime': 'onnxruntime>=1.16.0',
    'transformers': 'transformers>=4.35.0',
    'shap': 'shap>=0.43.0',
    'lime': 'lime>=0.2.0.1',
    'opencv-python': 'opencv-python>=4.8.0',
    'jupyter': 'jupyter>=1.0.0',
    'redis': 'redis>=5.0.1',
    'celery': 'celery>=5.3.0',
}

def check_library(lib_name):
    """Verificar se uma biblioteca está instalada"""
    try:
        importlib.import_module(lib_name)
        return True
    except ImportError:
        return False

def get_library_version(lib_name):
    """Obter versão de uma biblioteca"""
    try:
        lib = importlib.import_module(lib_name)
        return getattr(lib, '__version__', 'unknown')
    except:
        return 'unknown'

def install_library(package_spec):
    """Instalar uma biblioteca"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_spec])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Função principal"""
    logger.info("🔍 CardioAI Pro v2.0.0 - Verificação de Bibliotecas")
    logger.info("=" * 60)
    
    installed_count = 0
    missing_count = 0
    failed_installs = []
    
    # Verificar bibliotecas obrigatórias
    logger.info("📚 VERIFICANDO BIBLIOTECAS OBRIGATÓRIAS:")
    for lib_name, package_spec in REQUIRED_LIBRARIES.items():
        if check_library(lib_name):
            version = get_library_version(lib_name)
            logger.info(f"✅ {lib_name}: {version}")
            installed_count += 1
        else:
            logger.warning(f"❌ {lib_name}: NÃO INSTALADA")
            logger.info(f"🔧 Instalando {package_spec}...")
            
            if install_library(package_spec):
                logger.info(f"✅ {lib_name}: INSTALADA COM SUCESSO")
                installed_count += 1
            else:
                logger.error(f"❌ {lib_name}: FALHA NA INSTALAÇÃO")
                failed_installs.append(lib_name)
                missing_count += 1
    
    # Verificar bibliotecas opcionais
    logger.info("\n📦 VERIFICANDO BIBLIOTECAS OPCIONAIS:")
    optional_installed = 0
    for lib_name, package_spec in OPTIONAL_LIBRARIES.items():
        if check_library(lib_name):
            version = get_library_version(lib_name)
            logger.info(f"✅ {lib_name}: {version}")
            optional_installed += 1
        else:
            logger.info(f"⚠️ {lib_name}: Opcional - não instalada")
    
    # Relatório final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RELATÓRIO FINAL DE BIBLIOTECAS")
    logger.info("=" * 60)
    
    total_required = len(REQUIRED_LIBRARIES)
    success_rate = (installed_count / total_required) * 100
    
    logger.info(f"📈 BIBLIOTECAS OBRIGATÓRIAS:")
    logger.info(f"   • Total necessárias: {total_required}")
    logger.info(f"   • Instaladas: {installed_count}")
    logger.info(f"   • Faltando: {missing_count}")
    logger.info(f"   • Taxa de sucesso: {success_rate:.1f}%")
    
    logger.info(f"\n📦 BIBLIOTECAS OPCIONAIS:")
    logger.info(f"   • Instaladas: {optional_installed}/{len(OPTIONAL_LIBRARIES)}")
    
    if failed_installs:
        logger.info(f"\n❌ FALHAS NA INSTALAÇÃO:")
        for lib in failed_installs:
            logger.info(f"   • {lib}")
    
    # Status final
    if missing_count == 0:
        logger.info(f"\n🎉 STATUS: TODAS AS BIBLIOTECAS OBRIGATÓRIAS ESTÃO INSTALADAS!")
        return 0
    elif success_rate >= 80:
        logger.info(f"\n✅ STATUS: MAIORIA DAS BIBLIOTECAS INSTALADAS ({success_rate:.1f}%)")
        return 1
    else:
        logger.info(f"\n⚠️ STATUS: MUITAS BIBLIOTECAS FALTANDO ({success_rate:.1f}%)")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

