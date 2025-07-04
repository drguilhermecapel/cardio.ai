#!/usr/bin/env python3
"""
Script de teste completo para o sistema CardioAI Pro v3.0
Testa todas as funcionalidades incluindo correção de bias
"""

import sys
import os
import numpy as np
from pathlib import Path

# Adicionar paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "backend" / "app" / "services"))

def test_bias_corrected_service():
    """Testa o serviço PTB-XL com correção de bias."""
    print("🧪 Testando serviço PTB-XL com correção de bias...")
    
    try:
        from ptbxl_model_service_bias_corrected import PTBXLModelServiceBiasCorrected
        
        # Inicializar serviço
        service = PTBXLModelServiceBiasCorrected()
        
        # Verificar informações do modelo
        model_info = service.get_model_info()
        print(f"   📊 Tipo de modelo: {model_info['model_type']}")
        print(f"   🔧 Correção de bias: {model_info['bias_correction_applied']}")
        print(f"   📈 Classes totais: {model_info['total_classes']}")
        
        # Testar predição
        test_data = np.random.normal(0, 1, (1, 12, 1000)).astype(np.float32)
        result = service.predict(test_data)
        
        if "error" in result:
            print(f"   ❌ Erro na predição: {result['error']}")
            return False
        
        print(f"   ✅ Predição realizada com sucesso")
        print(f"   📊 Modelo usado: {result['model_used']}")
        print(f"   🔧 Bias corrigido: {result['bias_correction_applied']}")
        print(f"   📋 Diagnósticos: {len(result['diagnoses'])}")
        
        # Mostrar alguns diagnósticos
        for i, diag in enumerate(result['diagnoses'][:3]):
            print(f"      {i+1}. {diag['condition']}: {diag['probability']:.3f} ({diag['confidence']})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no teste: {e}")
        return False

def test_preprocessing_functions():
    """Testa funções de pré-processamento."""
    print("\n🧪 Testando funções de pré-processamento...")
    
    try:
        # Importar da aplicação principal
        sys.path.insert(0, str(current_dir / "backend" / "app"))
        from main_complete_final import preprocess_ecg_for_ptbxl, generate_synthetic_ecg_ptbxl
        
        # Teste 1: ECG sintético
        synthetic_ecg = generate_synthetic_ecg_ptbxl()
        print(f"   ✅ ECG sintético gerado: {synthetic_ecg.shape}")
        
        # Teste 2: Pré-processamento de dados 1D
        data_1d = np.random.normal(0, 1, 5000)
        processed_1d = preprocess_ecg_for_ptbxl(data_1d)
        print(f"   ✅ Dados 1D processados: {data_1d.shape} -> {processed_1d.shape}")
        
        # Teste 3: Pré-processamento de dados 2D (12, 5000)
        data_2d = np.random.normal(0, 1, (12, 5000))
        processed_2d = preprocess_ecg_for_ptbxl(data_2d)
        print(f"   ✅ Dados 2D processados: {data_2d.shape} -> {processed_2d.shape}")
        
        # Teste 4: Pré-processamento de dados 2D (5000, 12)
        data_2d_t = np.random.normal(0, 1, (5000, 12))
        processed_2d_t = preprocess_ecg_for_ptbxl(data_2d_t)
        print(f"   ✅ Dados 2D transpostos processados: {data_2d_t.shape} -> {processed_2d_t.shape}")
        
        # Verificar se todos resultaram em (12, 1000)
        expected_shape = (12, 1000)
        tests = [
            ("Sintético", synthetic_ecg),
            ("1D", processed_1d),
            ("2D", processed_2d),
            ("2D transposto", processed_2d_t)
        ]
        
        all_correct = True
        for name, data in tests:
            if data.shape != expected_shape:
                print(f"   ❌ {name}: formato incorreto {data.shape} != {expected_shape}")
                all_correct = False
        
        if all_correct:
            print(f"   ✅ Todos os formatos corretos: {expected_shape}")
        
        return all_correct
        
    except Exception as e:
        print(f"   ❌ Erro no teste de pré-processamento: {e}")
        return False

def test_image_processing():
    """Testa processamento de imagens."""
    print("\n🧪 Testando processamento de imagens...")
    
    try:
        # Importar da aplicação principal
        from main_complete_final import extract_ecg_from_image_ptbxl
        
        # Criar imagem de teste simples
        import cv2
        test_image = np.ones((400, 800, 3), dtype=np.uint8) * 255
        
        # Desenhar algumas linhas simulando ECG
        for i in range(12):
            y = 50 + i * 25
            cv2.line(test_image, (50, y), (750, y + 10), (0, 0, 0), 2)
        
        # Salvar imagem temporária
        test_image_path = "test_ecg_image.png"
        cv2.imwrite(test_image_path, test_image)
        
        # Processar imagem
        ecg_from_image = extract_ecg_from_image_ptbxl(test_image_path)
        
        # Limpar arquivo temporário
        os.remove(test_image_path)
        
        print(f"   ✅ ECG extraído de imagem: {ecg_from_image.shape}")
        
        # Verificar formato
        if ecg_from_image.shape == (12, 1000):
            print(f"   ✅ Formato correto: {ecg_from_image.shape}")
            return True
        else:
            print(f"   ❌ Formato incorreto: {ecg_from_image.shape} != (12, 1000)")
            return False
        
    except Exception as e:
        print(f"   ❌ Erro no teste de imagem: {e}")
        return False

def test_complete_application():
    """Testa a aplicação completa."""
    print("\n🧪 Testando aplicação completa...")
    
    try:
        # Importar aplicação
        from main_complete_final import app
        
        print(f"   ✅ Aplicação FastAPI carregada")
        print(f"   📋 Título: {app.title}")
        print(f"   🔢 Versão: {app.version}")
        
        # Verificar rotas principais
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/api/v1/ecg/upload", "/api/v1/models"]
        
        missing_routes = []
        for route in expected_routes:
            if route not in routes:
                missing_routes.append(route)
        
        if missing_routes:
            print(f"   ❌ Rotas faltando: {missing_routes}")
            return False
        else:
            print(f"   ✅ Todas as rotas principais encontradas")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no teste da aplicação: {e}")
        return False

def test_model_dimensions():
    """Testa se o modelo aceita as dimensões corretas."""
    print("\n🧪 Testando dimensões do modelo...")
    
    try:
        import tensorflow as tf
        
        # Carregar modelo
        model_path = "models/ecg_model_final.h5"
        if not os.path.exists(model_path):
            model_path = "ecg_model_final.h5"
        
        if not os.path.exists(model_path):
            print(f"   ⚠️ Modelo não encontrado - pulando teste")
            return True
        
        model = tf.keras.models.load_model(model_path)
        print(f"   ✅ Modelo carregado: {model.input_shape}")
        
        # Testar dimensão correta
        test_input = np.random.normal(0, 1, (1, 12, 1000)).astype(np.float32)
        prediction = model.predict(test_input, verbose=0)
        
        print(f"   ✅ Predição realizada: {test_input.shape} -> {prediction.shape}")
        print(f"   📊 Classes de saída: {prediction.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no teste de dimensões: {e}")
        return False

def main():
    """Função principal de teste."""
    print("=" * 70)
    print("🔬 TESTE COMPLETO DO SISTEMA CARDIOAI PRO v3.0")
    print("=" * 70)
    
    tests = [
        ("Serviço PTB-XL com Correção de Bias", test_bias_corrected_service),
        ("Funções de Pré-processamento", test_preprocessing_functions),
        ("Processamento de Imagens", test_image_processing),
        ("Aplicação Completa", test_complete_application),
        ("Dimensões do Modelo", test_model_dimensions)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")
            results.append((test_name, False))
    
    # Resumo final
    print(f"\n{'='*70}")
    print("📋 RESUMO DOS TESTES")
    print(f"{'='*70}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Resultado Final: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM! Sistema pronto para uso.")
        return True
    else:
        print("⚠️ Alguns testes falharam. Verifique os problemas acima.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

