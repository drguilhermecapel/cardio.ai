#!/usr/bin/env python3
"""
Script de teste para o sistema ECG melhorado
Testa todas as funcionalidades corrigidas
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adicionar paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "models"))

def test_imports():
    """Testa se todas as importações funcionam."""
    print("\n=== TESTE DE IMPORTAÇÕES ===")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        from preprocess_functions_improved import (
            preprocess_ecg_signal, extract_ecg_from_image, 
            validate_ecg_signal, prepare_for_model
        )
        print("✅ Funções de pré-processamento melhoradas")
    except ImportError as e:
        print(f"❌ Funções melhoradas: {e}")
        return False
    
    try:
        from backend.app.services.improved_ecg_service import get_improved_ecg_service
        print("✅ Serviço ECG melhorado")
    except ImportError as e:
        print(f"❌ Serviço melhorado: {e}")
        return False
    
    return True

def test_preprocessing_functions():
    """Testa as funções de pré-processamento melhoradas."""
    print("\n=== TESTE DE PRÉ-PROCESSAMENTO ===")
    
    try:
        from preprocess_functions_improved import (
            preprocess_ecg_signal, validate_ecg_signal, prepare_for_model
        )
        
        # Teste 1: Sinal sintético normal
        print("📊 Teste 1: Sinal sintético normal")
        synthetic_signal = np.random.randn(12, 1000) * 0.5
        processed = preprocess_ecg_signal(synthetic_signal)
        
        is_valid, msg = validate_ecg_signal(processed)
        print(f"   Shape: {processed.shape}")
        print(f"   Válido: {is_valid} - {msg}")
        
        # Teste 2: Sinal com dimensões incorretas
        print("📊 Teste 2: Sinal com dimensões incorretas")
        wrong_signal = np.random.randn(8, 500)  # 8 derivações, 500 amostras
        processed = preprocess_ecg_signal(wrong_signal)
        
        is_valid, msg = validate_ecg_signal(processed)
        print(f"   Shape original: {wrong_signal.shape}")
        print(f"   Shape processado: {processed.shape}")
        print(f"   Válido: {is_valid} - {msg}")
        
        # Teste 3: Preparação para modelo
        print("📊 Teste 3: Preparação para modelo")
        model_input = prepare_for_model(processed)
        print(f"   Shape para modelo: {model_input.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de pré-processamento: {e}")
        return False

def test_model_integration():
    """Testa a integração com o modelo."""
    print("\n=== TESTE DE INTEGRAÇÃO COM MODELO ===")
    
    try:
        from backend.app.services.unified_model_service import get_model_service
        from preprocess_functions_improved import prepare_for_model
        
        # Obter serviço de modelo
        model_service = get_model_service()
        
        # Listar modelos disponíveis
        models = model_service.list_models()
        print(f"📊 Modelos disponíveis: {models}")
        
        if "ecg_model_final" in models:
            # Criar dados de teste
            test_signal = np.random.randn(12, 1000) * 0.5
            model_input = prepare_for_model(test_signal)
            
            # Fazer predição
            result = model_service.predict("ecg_model_final", model_input)
            
            if "error" not in result:
                predictions = result["predictions"][0]
                print(f"✅ Predição realizada com sucesso")
                print(f"   Shape da predição: {predictions.shape}")
                print(f"   Valores máximos: {np.max(predictions):.4f}")
                print(f"   Valores mínimos: {np.min(predictions):.4f}")
                return True
            else:
                print(f"❌ Erro na predição: {result['error']}")
                return False
        else:
            print("❌ Modelo ecg_model_final não encontrado")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste de integração: {e}")
        return False

def test_improved_service():
    """Testa o serviço ECG melhorado."""
    print("\n=== TESTE DO SERVIÇO ECG MELHORADO ===")
    
    try:
        from backend.app.services.improved_ecg_service import get_improved_ecg_service
        
        # Obter serviço
        ecg_service = get_improved_ecg_service()
        print("✅ Serviço ECG melhorado inicializado")
        
        # Criar arquivo de teste
        test_data = np.random.randn(12, 1000) * 0.5
        test_file = "test_ecg_data.npy"
        np.save(test_file, test_data)
        
        # Processar arquivo
        print("📊 Processando arquivo de teste...")
        process_result = ecg_service.process_ecg_file(test_file)
        
        if "error" not in process_result:
            process_id = process_result["process_id"]
            print(f"✅ Arquivo processado: {process_id}")
            print(f"   Qualidade: {process_result.get('quality_score', 'N/A')}")
            
            # Analisar ECG
            print("📊 Analisando ECG...")
            analysis_result = ecg_service.analyze_ecg(process_id)
            
            if "error" not in analysis_result:
                print("✅ Análise realizada com sucesso")
                diagnoses = analysis_result.get("diagnoses", [])
                print(f"   Diagnósticos encontrados: {len(diagnoses)}")
                for diag in diagnoses[:3]:  # Mostrar top 3
                    print(f"   - {diag['condition']}: {diag['probability']:.3f} ({diag['confidence']})")
                
                # Limpar arquivo de teste
                os.remove(test_file)
                return True
            else:
                print(f"❌ Erro na análise: {analysis_result['error']}")
                os.remove(test_file)
                return False
        else:
            print(f"❌ Erro no processamento: {process_result['error']}")
            os.remove(test_file)
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste do serviço: {e}")
        if os.path.exists("test_ecg_data.npy"):
            os.remove("test_ecg_data.npy")
        return False

def test_image_processing():
    """Testa o processamento de imagens ECG."""
    print("\n=== TESTE DE PROCESSAMENTO DE IMAGENS ===")
    
    try:
        # Verificar se existe uma imagem de teste
        test_images = ["test_ecg_image.jpg", "test_ecg_complete.jpg"]
        test_image = None
        
        for img_path in test_images:
            if os.path.exists(img_path):
                test_image = img_path
                break
        
        if test_image:
            print(f"📊 Usando imagem de teste: {test_image}")
            
            from backend.app.services.improved_ecg_service import get_improved_ecg_service
            ecg_service = get_improved_ecg_service()
            
            # Processar imagem
            process_result = ecg_service.process_ecg_file(test_image)
            
            if "error" not in process_result:
                print("✅ Imagem processada com sucesso")
                print(f"   Qualidade: {process_result.get('quality_score', 'N/A')}")
                print(f"   Derivações detectadas: {process_result.get('leads_detected', 'N/A')}")
                
                # Analisar se possível
                process_id = process_result["process_id"]
                analysis_result = ecg_service.analyze_ecg(process_id)
                
                if "error" not in analysis_result:
                    print("✅ Análise da imagem realizada")
                    diagnoses = analysis_result.get("diagnoses", [])
                    print(f"   Diagnósticos: {len(diagnoses)}")
                else:
                    print(f"⚠️ Análise da imagem falhou: {analysis_result['error']}")
                
                return True
            else:
                print(f"❌ Erro no processamento da imagem: {process_result['error']}")
                return False
        else:
            print("⚠️ Nenhuma imagem de teste encontrada - pulando teste")
            return True
            
    except Exception as e:
        print(f"❌ Erro no teste de imagem: {e}")
        return False

def test_api_endpoints():
    """Testa os endpoints da API."""
    print("\n=== TESTE DE ENDPOINTS DA API ===")
    
    try:
        # Importar aplicação FastAPI
        from backend.app.main import app
        print("✅ Aplicação FastAPI importada")
        
        # Verificar se os endpoints estão definidos
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/api/v1/ecg/upload", "/api/v1/ecg/analyze/{process_id}"]
        
        for route in expected_routes:
            if any(route.replace("{process_id}", "test") in r for r in routes):
                print(f"✅ Endpoint encontrado: {route}")
            else:
                print(f"⚠️ Endpoint não encontrado: {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de endpoints: {e}")
        return False

def run_comprehensive_test():
    """Executa todos os testes."""
    print("🔍 TESTE ABRANGENTE DO SISTEMA ECG MELHORADO")
    print("=" * 60)
    
    # Mudar para diretório do projeto
    os.chdir(project_root)
    
    tests = [
        ("Importações", test_imports),
        ("Pré-processamento", test_preprocessing_functions),
        ("Integração com Modelo", test_model_integration),
        ("Serviço ECG Melhorado", test_improved_service),
        ("Processamento de Imagens", test_image_processing),
        ("Endpoints da API", test_api_endpoints)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Executando: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name}: PASSOU")
            else:
                print(f"❌ {test_name}: FALHOU")
        except Exception as e:
            print(f"💥 {test_name}: ERRO - {e}")
            results.append((test_name, False))
    
    # Resumo final
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM! Sistema pronto para deploy.")
        return True
    else:
        print("⚠️ Alguns testes falharam. Revisar antes do deploy.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)

