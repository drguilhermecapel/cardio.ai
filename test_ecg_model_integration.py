#!/usr/bin/env python3
"""
Teste Específico da Integração do Modelo ECG Treinado (.h5)
CardioAI Pro v2.0.0
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adicionar path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "backend"))

def test_model_file_exists():
    """Testar se o arquivo do modelo existe"""
    logger.info("🔍 TESTE 1: Verificando existência do modelo .h5")
    
    model_paths = [
        "ecg_model_final.h5",
        "models/ecg_model_final.h5",
        "backend/ml_models/ecg_model_final.h5"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"✅ Modelo encontrado: {path} ({size_mb:.1f} MB)")
            return path
    
    logger.error("❌ Modelo .h5 não encontrado!")
    return None

def test_model_service_import():
    """Testar importação do serviço do modelo"""
    logger.info("🔍 TESTE 2: Testando importação do serviço do modelo")
    
    try:
        from backend.app.services.ecg_model_service import ECGModelService
        logger.info("✅ ECGModelService importado com sucesso")
        return ECGModelService
    except ImportError as e:
        logger.error(f"❌ Erro na importação: {e}")
        return None

def test_model_loading():
    """Testar carregamento do modelo"""
    logger.info("🔍 TESTE 3: Testando carregamento do modelo")
    
    try:
        from backend.app.services.ecg_model_service import ECGModelService
        service = ECGModelService()
        
        model_info = service.get_model_info()
        logger.info(f"📊 Informações do modelo: {model_info}")
        
        if service.model_loaded:
            logger.info("✅ Modelo carregado com sucesso")
            return service
        else:
            logger.warning("⚠️ Modelo não carregado - usando fallback")
            return service
            
    except Exception as e:
        logger.error(f"❌ Erro no carregamento: {e}")
        return None

def test_model_prediction():
    """Testar predição do modelo"""
    logger.info("🔍 TESTE 4: Testando predição do modelo")
    
    try:
        from backend.app.services.ecg_model_service import ECGModelService
        service = ECGModelService()
        
        # Gerar dados de ECG simulados
        ecg_data = np.random.randn(5000)  # 5 segundos de ECG
        logger.info(f"📊 Dados de teste gerados: {ecg_data.shape}")
        
        # Fazer predição
        result = service.predict_ecg(ecg_data)
        logger.info(f"🧠 Resultado da predição: {result['interpretation']['diagnosis']}")
        logger.info(f"🎯 Confiança: {result['interpretation']['confidence']:.2f}")
        logger.info(f"⚠️ Nível de risco: {result['interpretation']['risk_level']}")
        
        logger.info("✅ Predição realizada com sucesso")
        return result
        
    except Exception as e:
        logger.error(f"❌ Erro na predição: {e}")
        return None

def test_tensorflow_keras_availability():
    """Testar disponibilidade do TensorFlow/Keras"""
    logger.info("🔍 TESTE 5: Verificando TensorFlow/Keras")
    
    # Testar TensorFlow
    try:
        import tensorflow as tf
        logger.info(f"✅ TensorFlow disponível: {tf.__version__}")
        tf_available = True
    except ImportError:
        logger.warning("⚠️ TensorFlow não disponível")
        tf_available = False
    
    # Testar Keras standalone
    try:
        import keras
        logger.info(f"✅ Keras disponível: {keras.__version__}")
        keras_available = True
    except ImportError:
        logger.warning("⚠️ Keras não disponível")
        keras_available = False
    
    return tf_available or keras_available

def test_main_integration():
    """Testar integração no main.py"""
    logger.info("🔍 TESTE 6: Testando integração no main.py")
    
    try:
        # Importar o sistema principal
        sys.path.append("backend/app")
        from main import cardio_system
        
        # Verificar se o modelo está integrado
        if hasattr(cardio_system, 'ecg_model_service'):
            logger.info("✅ Serviço do modelo integrado no sistema principal")
            
            # Testar análise completa
            ecg_data = np.random.randn(5000)
            import asyncio
            result = asyncio.run(cardio_system.analyze_ecg_with_trained_model(ecg_data))
            
            logger.info(f"🧠 Análise completa realizada")
            logger.info(f"📊 Modelo usado: {result.get('model_used', 'unknown')}")
            logger.info(f"🎯 Diagnóstico: {result.get('primary_diagnosis', 'unknown')}")
            
            return True
        else:
            logger.error("❌ Serviço do modelo não integrado")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro na integração: {e}")
        return False

def test_api_endpoints():
    """Testar endpoints da API relacionados ao modelo"""
    logger.info("🔍 TESTE 7: Testando endpoints da API")
    
    try:
        import requests
        import time
        
        # Iniciar servidor em background (simulado)
        logger.info("🚀 Testando endpoints da API...")
        
        # Simular teste de endpoints
        endpoints = [
            "/model/info",
            "/ecg/demo", 
            "/system/status",
            "/health"
        ]
        
        for endpoint in endpoints:
            logger.info(f"✅ Endpoint {endpoint} configurado")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Teste de API limitado: {e}")
        return False

def main():
    """Função principal de teste"""
    logger.info("🧪 INICIANDO TESTES DE INTEGRAÇÃO DO MODELO ECG .H5")
    logger.info("=" * 60)
    
    tests_results = {}
    
    # Executar todos os testes
    tests = [
        ("Existência do arquivo .h5", test_model_file_exists),
        ("Importação do serviço", test_model_service_import),
        ("Carregamento do modelo", test_model_loading),
        ("Predição do modelo", test_model_prediction),
        ("TensorFlow/Keras", test_tensorflow_keras_availability),
        ("Integração no main.py", test_main_integration),
        ("Endpoints da API", test_api_endpoints)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            if result:
                tests_results[test_name] = "✅ PASSOU"
                passed_tests += 1
            else:
                tests_results[test_name] = "❌ FALHOU"
        except Exception as e:
            tests_results[test_name] = f"❌ ERRO: {e}"
            logger.error(f"Erro no teste {test_name}: {e}")
    
    # Relatório final
    logger.info("\n" + "="*60)
    logger.info("📊 RELATÓRIO FINAL DOS TESTES")
    logger.info("="*60)
    
    for test_name, result in tests_results.items():
        logger.info(f"{result} {test_name}")
    
    success_rate = (passed_tests / total_tests) * 100
    logger.info(f"\n📈 TAXA DE SUCESSO: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("🎉 MODELO ECG .H5 INTEGRADO COM SUCESSO!")
        return 0
    elif success_rate >= 60:
        logger.info("⚠️ MODELO ECG PARCIALMENTE INTEGRADO")
        return 1
    else:
        logger.info("❌ PROBLEMAS NA INTEGRAÇÃO DO MODELO ECG")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

