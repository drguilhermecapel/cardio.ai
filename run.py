#!/usr/bin/env python3
"""
Script de inicialização do CardioAI Pro
"""

import os
import sys
import argparse
import uvicorn
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cardioai")


def setup_environment():
    """Configura o ambiente para execução."""
    # Criar diretórios necessários
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Verificar dependências
    try:
        import numpy
        import pandas
        import fastapi
        logger.info("Dependências básicas verificadas com sucesso")
    except ImportError as e:
        logger.error(f"Dependência faltando: {e}")
        logger.info("Instalando dependências básicas...")
        os.system("pip install numpy pandas fastapi uvicorn")
    
    # Verificar dependências opcionais
    try:
        import tensorflow
        logger.info("TensorFlow disponível")
    except ImportError:
        logger.warning("TensorFlow não disponível - funcionalidades limitadas")
    
    try:
        import torch
        logger.info("PyTorch disponível")
    except ImportError:
        logger.warning("PyTorch não disponível - funcionalidades limitadas")
    
    try:
        import matplotlib
        logger.info("Matplotlib disponível - visualizações habilitadas")
    except ImportError:
        logger.warning("Matplotlib não disponível - visualizações desabilitadas")
    
    try:
        import wfdb
        logger.info("WFDB disponível - suporte a formatos PhysioNet")
    except ImportError:
        logger.warning("WFDB não disponível - suporte limitado a formatos")
    
    try:
        import pyedflib
        logger.info("PyEDFLib disponível - suporte a formato EDF")
    except ImportError:
        logger.warning("PyEDFLib não disponível - sem suporte a EDF")


def run_server(host="0.0.0.0", port=8000, reload=False, workers=1):
    """Executa o servidor FastAPI."""
    logger.info(f"Iniciando CardioAI Pro em {host}:{port}")
    
    # Verificar se o módulo principal existe
    app_path = Path("backend/app/main.py")
    if not app_path.exists():
        logger.error(f"Arquivo principal não encontrado: {app_path}")
        sys.exit(1)
    
    # Iniciar servidor
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    uvicorn.run(
        "backend.app.main_direct:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info"
    )


def run_tests():
    """Executa os testes unitários."""
    import unittest
    
    logger.info("Executando testes unitários...")
    
    # Verificar se o diretório de testes existe
    tests_path = Path("backend/tests")
    if not tests_path.exists():
        logger.error(f"Diretório de testes não encontrado: {tests_path}")
        sys.exit(1)
    
    # Descobrir e executar testes
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("backend/tests")
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Verificar resultado
    if result.wasSuccessful():
        logger.info("Todos os testes passaram com sucesso!")
        return 0
    else:
        logger.error(f"Falha nos testes: {len(result.failures)} falhas, {len(result.errors)} erros")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioAI Pro - Sistema de Análise de ECG com IA")
    
    subparsers = parser.add_subparsers(dest="command", help="Comando a executar")
    
    # Comando run
    run_parser = subparsers.add_parser("run", help="Executar servidor")
    run_parser.add_argument("--host", default="0.0.0.0", help="Host para bind")
    run_parser.add_argument("--port", type=int, default=8000, help="Porta para bind")
    run_parser.add_argument("--reload", action="store_true", help="Habilitar reload automático")
    run_parser.add_argument("--workers", type=int, default=1, help="Número de workers")
    
    # Comando test
    test_parser = subparsers.add_parser("test", help="Executar testes")
    
    # Comando setup
    setup_parser = subparsers.add_parser("setup", help="Configurar ambiente")
    
    args = parser.parse_args()
    
    if args.command == "run":
        setup_environment()
        run_server(args.host, args.port, args.reload, args.workers)
    elif args.command == "test":
        setup_environment()
        sys.exit(run_tests())
    elif args.command == "setup":
        setup_environment()
        logger.info("Ambiente configurado com sucesso")
    else:
        parser.print_help()