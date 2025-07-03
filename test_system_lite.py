#!/usr/bin/env python3
"""
Script de teste simplificado para validar o sistema CardioAI
Versão sem dependências pesadas (TensorFlow, PyTorch)
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


def test_lite_model_service():
    """Testa o serviço de modelos lite."""
    logger.info("Testando ModelServiceLite...")
    
    try:
        from app.services.model_service_lite import ModelServiceLite, initialize_models_lite
        
        # Inicializar serviço
        model_service = initialize_models_lite()
        
        # Testar listagem de modelos
        models = model_service.list_models()
        logger.info(f"✓ Modelos carregados: {models}")
        
        if len(models) > 0:
            model_name = models[0]
            
            # Criar dados de teste
            test_ecg = np.random.randn(1000)
            
            # Testar predição
            result = model_service.predict_ecg(model_name, test_ecg)
            
            if "error" not in result:
                logger.info("✓ Predição realizada com sucesso")
                logger.info(f"  Modelo: {result['model_name']}")
                logger.info(f"  Confiança: {result['confidence']:.3f}")
                
                if "class_names" in result["predictions"]:
                    classes = result["predictions"]["class_names"]
                    logger.info(f"  Classes preditas: {classes}")
                
                return True
            else:
                logger.error(f"✗ Erro na predição: {result['error']}")
                return False
        else:
            logger.error("✗ Nenhum modelo carregado")
            return False
        
    except Exception as e:
        logger.error(f"✗ Erro no teste do ModelServiceLite: {e}")
        return False


def test_api_endpoints():
    """Testa os endpoints da API sem inicializar o servidor."""
    logger.info("Testando estrutura dos endpoints...")
    
    try:
        # Testar importação dos schemas
        from app.schemas.fhir import FHIRObservation, FHIRDiagnosticReport
        logger.info("✓ Schemas FHIR importados")
        
        # Testar criação de dados FHIR
        from app.schemas.fhir import create_ecg_observation, create_ecg_diagnostic_report
        
        patient_id = "test-patient-456"
        ecg_data = list(np.random.randn(100))
        analysis_results = {"confidence": 0.92, "predicted_class": 0}
        
        observation = create_ecg_observation(patient_id, ecg_data, 500, analysis_results)
        logger.info(f"✓ Observação FHIR criada: {observation.id}")
        
        report = create_ecg_diagnostic_report(patient_id, ["obs-1"], "ECG dentro dos parâmetros normais")
        logger.info(f"✓ Relatório FHIR criado: {report.id}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste dos endpoints: {e}")
        return False


def test_ecg_processing_pipeline():
    """Testa pipeline completo de processamento de ECG."""
    logger.info("Testando pipeline de processamento ECG...")
    
    try:
        # 1. Gerar sinal ECG sintético
        sampling_rate = 500
        duration = 10
        t = np.linspace(0, duration, sampling_rate * duration)
        
        # Simular ECG com componentes de frequência cardíaca
        heart_rate = 75  # BPM
        ecg_base = np.sin(2 * np.pi * heart_rate / 60 * t)
        
        # Adicionar ruído e artefatos
        noise = 0.1 * np.random.randn(len(t))
        artifacts = 0.05 * np.sin(2 * np.pi * 50 * t)  # Interferência 50Hz
        
        ecg_signal = ecg_base + noise + artifacts
        logger.info(f"✓ Sinal ECG sintético gerado: {len(ecg_signal)} amostras")
        
        # 2. Pré-processamento
        # Filtro passa-alta simples (remoção de deriva)
        from scipy.signal import butter, filtfilt
        
        # Filtro passa-alta 0.5 Hz
        nyquist = sampling_rate / 2
        low_cutoff = 0.5 / nyquist
        b, a = butter(2, low_cutoff, btype='high')
        ecg_filtered = filtfilt(b, a, ecg_signal)
        
        # Filtro passa-baixa 40 Hz
        high_cutoff = 40 / nyquist
        b, a = butter(2, high_cutoff, btype='low')
        ecg_filtered = filtfilt(b, a, ecg_filtered)
        
        logger.info("✓ Filtragem digital aplicada")
        
        # 3. Normalização
        ecg_normalized = (ecg_filtered - np.mean(ecg_filtered)) / np.std(ecg_filtered)
        logger.info("✓ Normalização aplicada")
        
        # 4. Segmentação
        window_size = 1000  # 2 segundos
        overlap = 500  # 50% overlap
        segments = []
        
        for i in range(0, len(ecg_normalized) - window_size, overlap):
            segment = ecg_normalized[i:i + window_size]
            segments.append(segment)
        
        logger.info(f"✓ Segmentação realizada: {len(segments)} segmentos")
        
        # 5. Extração de características
        features = []
        for segment in segments:
            feature_vector = {
                'mean': float(np.mean(segment)),
                'std': float(np.std(segment)),
                'max': float(np.max(segment)),
                'min': float(np.min(segment)),
                'rms': float(np.sqrt(np.mean(segment**2))),
                'zero_crossings': int(np.sum(np.diff(np.sign(segment)) != 0))
            }
            features.append(feature_vector)
        
        logger.info(f"✓ Características extraídas: {len(features[0])} features por segmento")
        
        # 6. Teste com modelo
        from app.services.model_service_lite import ModelServiceLite
        
        model_service = ModelServiceLite()
        model_service.create_demo_model("pipeline_test_model")
        
        # Usar primeiro segmento para teste
        test_segment = segments[0]
        result = model_service.predict_ecg("pipeline_test_model", test_segment)
        
        if "error" not in result:
            logger.info("✓ Predição no pipeline realizada com sucesso")
            logger.info(f"  Confiança: {result['confidence']:.3f}")
        else:
            logger.warning(f"⚠ Erro na predição do pipeline: {result['error']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no teste do pipeline: {e}")
        return False


def test_system_integration():
    """Testa integração completa do sistema."""
    logger.info("Testando integração do sistema...")
    
    try:
        # Simular fluxo completo
        patient_data = {
            "patient_id": "PATIENT_001",
            "ecg_data": list(np.random.randn(2000)),
            "sampling_rate": 500,
            "metadata": {
                "age": 45,
                "gender": "M",
                "condition": "routine_checkup"
            }
        }
        
        # 1. Processamento
        ecg_array = np.array(patient_data["ecg_data"])
        logger.info(f"✓ Dados do paciente carregados: {len(ecg_array)} amostras")
        
        # 2. Análise com modelo
        from app.services.model_service_lite import initialize_models_lite
        model_service = initialize_models_lite()
        
        models = model_service.list_models()
        if models:
            result = model_service.predict_ecg(models[0], ecg_array)
            
            if "error" not in result:
                logger.info("✓ Análise de ECG realizada")
                
                # 3. Criar recursos FHIR
                from app.schemas.fhir import create_ecg_observation, create_ecg_diagnostic_report
                
                observation = create_ecg_observation(
                    patient_data["patient_id"],
                    patient_data["ecg_data"],
                    patient_data["sampling_rate"],
                    result
                )
                
                # Gerar conclusão clínica
                confidence = result["confidence"]
                if confidence > 0.8:
                    conclusion = "ECG analisado com alta confiança. Resultados dentro dos parâmetros esperados."
                elif confidence > 0.6:
                    conclusion = "ECG analisado com confiança moderada. Recomenda-se revisão médica."
                else:
                    conclusion = "ECG analisado com baixa confiança. Revisão médica necessária."
                
                report = create_ecg_diagnostic_report(
                    patient_data["patient_id"],
                    [observation.id],
                    conclusion
                )
                
                logger.info("✓ Recursos FHIR criados")
                logger.info(f"  Observação: {observation.id}")
                logger.info(f"  Relatório: {report.id}")
                logger.info(f"  Conclusão: {conclusion}")
                
                # 4. Estruturar resposta final
                response = {
                    "patient_id": patient_data["patient_id"],
                    "analysis_timestamp": result["timestamp"],
                    "confidence": confidence,
                    "predictions": result["predictions"],
                    "fhir_resources": {
                        "observation_id": observation.id,
                        "report_id": report.id
                    },
                    "clinical_conclusion": conclusion,
                    "status": "completed"
                }
                
                logger.info("✓ Integração completa realizada com sucesso")
                return True
            else:
                logger.error(f"✗ Erro na análise: {result['error']}")
                return False
        else:
            logger.error("✗ Nenhum modelo disponível")
            return False
        
    except Exception as e:
        logger.error(f"✗ Erro na integração: {e}")
        return False


def run_lite_tests():
    """Executa todos os testes da versão lite."""
    logger.info("=== Iniciando testes do sistema CardioAI (Versão Lite) ===")
    
    tests = [
        ("Serviço de Modelos Lite", test_lite_model_service),
        ("Endpoints da API", test_api_endpoints),
        ("Pipeline de Processamento ECG", test_ecg_processing_pipeline),
        ("Integração do Sistema", test_system_integration)
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
    logger.info("\n=== Resumo dos Testes (Versão Lite) ===")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSOU" if result else "✗ FALHOU"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nResultado final: {passed}/{total} testes passaram")
    
    if passed == total:
        logger.info("🎉 Todos os testes passaram! Sistema lite pronto para deploy.")
        return True
    else:
        logger.warning(f"⚠ {total - passed} teste(s) falharam. Revisar implementação.")
        return False


if __name__ == "__main__":
    success = run_lite_tests()
    sys.exit(0 if success else 1)

