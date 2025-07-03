#!/usr/bin/env python3
"""
Script de teste para validar o sistema CardioAI integrado
"""

import sys
import os
import numpy as np
import json
import logging
from pathlib import Path

# Adicionar o diretório backend ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Testa se todas as importações funcionam."""
    logger.info("Testando importações...")
    
    try:
        from app.services.model_service import ModelService, ECGClassificationModel
        logger.info("✓ ModelService importado com sucesso")
    except ImportError as e:
        logger.error(f"✗ Erro ao importar ModelService: {e}")
        return False
    
    try:
        from app.services.explainability_service import ExplainabilityService
        logger.info("✓ ExplainabilityService importado com sucesso")
    except ImportError as e:
        logger.error(f"✗ Erro ao importar ExplainabilityService: {e}")
        return False
    
    try:
        from app.schemas.fhir import FHIRObservation, FHIRDiagnosticReport
        logger.info("✓ Schemas FHIR importados com sucesso")
    except ImportError as e:
        logger.error(f"✗ Erro ao importar schemas FHIR: {e}")
        return False
    
    return True


def test_model_service():
    """Testa o serviço de modelos."""
    logger.info("Testando ModelService...")
    
    try:
        from app.services.model_service import ModelService
        
        # Criar instância
        model_service = ModelService()
        
        # Testar listagem de modelos
        models = model_service.list_models()
        logger.info(f"✓ Modelos carregados: {len(models)}")
        
        # Criar dados de teste
        test_ecg = np.random.randn(1000)
        
        # Se não há modelos carregados, criar um modelo de teste
        if len(models) == 0:
            logger.info("Criando modelo de teste...")
            
            # Criar modelo PyTorch simples
            import torch
            import torch.nn as nn
            
            class SimpleECGModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(1000, 5)
                
                def forward(self, x):
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    return self.fc(x)
            
            # Criar e salvar modelo
            model = SimpleECGModel()
            torch.save(model.state_dict(), "test_model.pth")
            
            # Carregar no serviço
            success = model_service.load_pytorch_model("test_model.pth", SimpleECGModel, "test_model")
            if success:
                logger.info("✓ Modelo de teste carregado")
                
                # Testar predição
                result = model_service.predict_ecg("test_model", test_ecg)
                if "error" not in result:
                    logger.info("✓ Predição realizada com sucesso")
                    logger.info(f"  Confiança: {result['confidence']:.3f}")
                else:
                    logger.error(f"✗ Erro na predição: {result['error']}")
                    return False
            else:
                logger.error("✗ Falha ao carregar modelo de teste")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste do ModelService: {e}")
        return False


def test_explainability_service():
    """Testa o serviço de explicabilidade."""
    logger.info("Testando ExplainabilityService...")
    
    try:
        from app.services.explainability_service import ExplainabilityService
        
        # Criar instância
        explainability_service = ExplainabilityService()
        
        # Criar dados de teste
        test_ecg = np.random.randn(1000)
        
        # Criar modelo mock simples
        class MockModel:
            def predict(self, x):
                return np.random.rand(x.shape[0], 5)
        
        mock_model = MockModel()
        
        # Testar análise de importância
        importance_result = explainability_service.generate_feature_importance(mock_model, test_ecg)
        
        if "error" not in importance_result:
            logger.info("✓ Análise de importância realizada com sucesso")
            logger.info(f"  Janelas analisadas: {len(importance_result['importance_scores'])}")
        else:
            logger.error(f"✗ Erro na análise de importância: {importance_result['error']}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste do ExplainabilityService: {e}")
        return False


def test_fhir_schemas():
    """Testa os schemas FHIR."""
    logger.info("Testando schemas FHIR...")
    
    try:
        from app.schemas.fhir import create_ecg_observation, create_ecg_diagnostic_report
        from datetime import datetime
        
        # Dados de teste
        patient_id = "test-patient-123"
        ecg_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        sampling_rate = 500
        analysis_results = {"confidence": 0.95, "predicted_class": 1}
        
        # Testar criação de observação FHIR
        observation = create_ecg_observation(patient_id, ecg_data, sampling_rate, analysis_results)
        
        if observation.resourceType == "Observation":
            logger.info("✓ Observação FHIR criada com sucesso")
            logger.info(f"  ID: {observation.id}")
            logger.info(f"  Status: {observation.status}")
        else:
            logger.error("✗ Erro na criação da observação FHIR")
            return False
        
        # Testar criação de relatório diagnóstico
        report = create_ecg_diagnostic_report(patient_id, ["obs-1"], "ECG normal")
        
        if report.resourceType == "DiagnosticReport":
            logger.info("✓ Relatório diagnóstico FHIR criado com sucesso")
            logger.info(f"  ID: {report.id}")
            logger.info(f"  Status: {report.status}")
        else:
            logger.error("✗ Erro na criação do relatório FHIR")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste dos schemas FHIR: {e}")
        return False


def test_api_structure():
    """Testa a estrutura da API."""
    logger.info("Testando estrutura da API...")
    
    try:
        from app.api.v1.ecg_endpoints import router
        from app.main import app
        
        # Verificar se o router foi criado
        if router:
            logger.info("✓ Router ECG criado com sucesso")
            
            # Verificar rotas
            routes = [route.path for route in router.routes]
            expected_routes = ["/analyze", "/upload-file", "/models"]
            
            for expected_route in expected_routes:
                if any(expected_route in route for route in routes):
                    logger.info(f"✓ Rota {expected_route} encontrada")
                else:
                    logger.warning(f"⚠ Rota {expected_route} não encontrada")
        
        # Verificar aplicação principal
        if app:
            logger.info("✓ Aplicação FastAPI criada com sucesso")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste da estrutura da API: {e}")
        return False


def test_data_processing():
    """Testa processamento de dados ECG."""
    logger.info("Testando processamento de dados ECG...")
    
    try:
        # Simular dados de ECG
        sampling_rate = 500  # Hz
        duration = 10  # segundos
        samples = sampling_rate * duration
        
        # Gerar sinal ECG sintético
        t = np.linspace(0, duration, samples)
        heart_rate = 70  # BPM
        ecg_signal = np.sin(2 * np.pi * heart_rate / 60 * t) + 0.1 * np.random.randn(samples)
        
        logger.info(f"✓ Sinal ECG sintético gerado: {len(ecg_signal)} amostras")
        
        # Testar normalização
        normalized = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        logger.info(f"✓ Normalização realizada: média={np.mean(normalized):.3f}, std={np.std(normalized):.3f}")
        
        # Testar segmentação
        window_size = 1000  # 2 segundos a 500 Hz
        segments = []
        for i in range(0, len(ecg_signal) - window_size, window_size // 2):
            segment = ecg_signal[i:i + window_size]
            segments.append(segment)
        
        logger.info(f"✓ Segmentação realizada: {len(segments)} segmentos")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste de processamento: {e}")
        return False


def run_all_tests():
    """Executa todos os testes."""
    logger.info("=== Iniciando testes do sistema CardioAI ===")
    
    tests = [
        ("Importações", test_imports),
        ("Serviço de Modelos", test_model_service),
        ("Serviço de Explicabilidade", test_explainability_service),
        ("Schemas FHIR", test_fhir_schemas),
        ("Estrutura da API", test_api_structure),
        ("Processamento de Dados", test_data_processing)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Executando teste: {test_name} ---")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                logger.info(f"✓ {test_name}: PASSOU")
            else:
                logger.error(f"✗ {test_name}: FALHOU")
        except Exception as e:
            logger.error(f"✗ {test_name}: ERRO - {e}")
            results[test_name] = False
    
    # Resumo
    logger.info("\n=== Resumo dos Testes ===")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSOU" if result else "✗ FALHOU"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nResultado final: {passed}/{total} testes passaram")
    
    if passed == total:
        logger.info("🎉 Todos os testes passaram! Sistema pronto para deploy.")
        return True
    else:
        logger.warning(f"⚠ {total - passed} teste(s) falharam. Revisar antes do deploy.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

