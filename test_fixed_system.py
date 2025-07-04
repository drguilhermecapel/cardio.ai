#!/usr/bin/env python3
"""
Script de teste para sistema CardioAI corrigido
"""

import sys
import os
import numpy as np
from pathlib import Path

# Adicionar paths
sys.path.insert(0, str(Path(__file__).parent / "models"))
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_imports():
    """Testa importações das funções corrigidas."""
    print("🧪 Testando importações...")
    
    try:
        from preprocess_functions_v2 import (
            ECGImageProcessor,
            ECGDataProcessor,
            validate_ecg_signal,
            prepare_for_model,
            get_diagnosis_mapping
        )
        print("✅ Funções v2 importadas com sucesso")
        return True
    except ImportError as e:
        print(f"❌ Erro na importação: {e}")
        return False

def test_data_processing():
    """Testa processamento de dados ECG."""
    print("\n🧪 Testando processamento de dados...")
    
    try:
        from preprocess_functions_v2 import ECGDataProcessor, validate_ecg_signal
        
        processor = ECGDataProcessor()
        
        # Teste 1: Dados 1D
        data_1d = np.random.normal(0, 1, 1000)
        result_1d = processor.preprocess_ecg_signal(data_1d)
        is_valid, msg = validate_ecg_signal(result_1d)
        print(f"📊 Dados 1D: Shape {result_1d.shape}, Válido: {is_valid}")
        
        # Teste 2: Dados 2D
        data_2d = np.random.normal(0, 1, (8, 1200))
        result_2d = processor.preprocess_ecg_signal(data_2d)
        is_valid, msg = validate_ecg_signal(result_2d)
        print(f"📊 Dados 2D: Shape {result_2d.shape}, Válido: {is_valid}")
        
        # Teste 3: Dados com formato incorreto
        data_3d = np.random.normal(0, 1, (5, 10, 100))
        result_3d = processor.preprocess_ecg_signal(data_3d)
        is_valid, msg = validate_ecg_signal(result_3d)
        print(f"📊 Dados 3D: Shape {result_3d.shape}, Válido: {is_valid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no processamento de dados: {e}")
        return False

def test_image_processing():
    """Testa processamento de imagens ECG."""
    print("\n🧪 Testando processamento de imagens...")
    
    try:
        from preprocess_functions_v2 import ECGImageProcessor, validate_ecg_signal
        
        processor = ECGImageProcessor()
        
        # Criar imagem de teste sintética
        test_image_path = "test_ecg_synthetic.jpg"
        create_synthetic_ecg_image(test_image_path)
        
        # Processar imagem
        result = processor.extract_ecg_from_image(test_image_path)
        
        if result is not None:
            is_valid, msg = validate_ecg_signal(result)
            print(f"📊 Imagem processada: Shape {result.shape}, Válido: {is_valid}")
            print(f"   Mensagem: {msg}")
        else:
            print("📊 Imagem retornou None - usando fallback")
        
        # Limpar arquivo de teste
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no processamento de imagem: {e}")
        return False

def test_model_preparation():
    """Testa preparação para modelo."""
    print("\n🧪 Testando preparação para modelo...")
    
    try:
        from preprocess_functions_v2 import ECGDataProcessor, prepare_for_model
        
        processor = ECGDataProcessor()
        
        # Criar dados de teste
        test_data = np.random.normal(0, 1, (12, 5000))
        
        # Preparar para modelo
        model_input = prepare_for_model(test_data)
        
        print(f"📊 Input do modelo: Shape {model_input.shape}, Tipo: {model_input.dtype}")
        
        # Verificar formato
        if model_input.shape == (1, 12, 5000) and model_input.dtype == np.float32:
            print("✅ Formato correto para modelo")
            return True
        else:
            print("❌ Formato incorreto para modelo")
            return False
        
    except Exception as e:
        print(f"❌ Erro na preparação para modelo: {e}")
        return False

def test_json_serialization():
    """Testa serialização JSON."""
    print("\n🧪 Testando serialização JSON...")
    
    try:
        from fixed_app import convert_numpy_types
        
        # Criar dados com tipos numpy
        test_data = {
            "float32": np.float32(3.14),
            "int64": np.int64(42),
            "array": np.array([1, 2, 3]),
            "nested": {
                "float": np.float64(2.71),
                "list": [np.int32(1), np.float32(2.5)]
            }
        }
        
        # Converter tipos
        converted = convert_numpy_types(test_data)
        
        # Tentar serializar para JSON
        import json
        json_str = json.dumps(converted)
        
        print("✅ Serialização JSON bem-sucedida")
        print(f"   Dados convertidos: {type(converted['float32'])}, {type(converted['int64'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na serialização JSON: {e}")
        return False

def test_fixed_app():
    """Testa aplicação corrigida."""
    print("\n🧪 Testando aplicação corrigida...")
    
    try:
        from fixed_app import process_data_file, process_image_file
        
        # Criar arquivo de dados de teste
        test_data_file = "test_data.csv"
        test_data = np.random.normal(0, 1, (12, 1000))
        np.savetxt(test_data_file, test_data, delimiter=',')
        
        # Testar processamento de dados
        result_data = process_data_file(test_data_file, '.csv')
        print(f"📊 Processamento de dados: {result_data['processing_type']}")
        print(f"   Shape: {result_data['signal_shape']}, Válido: {result_data['is_valid']}")
        
        # Limpar arquivo
        os.remove(test_data_file)
        
        # Criar imagem de teste
        test_image_file = "test_image.jpg"
        create_synthetic_ecg_image(test_image_file)
        
        # Testar processamento de imagem
        result_image = process_image_file(test_image_file)
        print(f"📊 Processamento de imagem: {result_image['processing_type']}")
        print(f"   Shape: {result_image['signal_shape']}, Válido: {result_image['is_valid']}")
        
        # Limpar arquivo
        if os.path.exists(test_image_file):
            os.remove(test_image_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na aplicação corrigida: {e}")
        return False

def create_synthetic_ecg_image(filename):
    """Cria imagem ECG sintética para teste."""
    try:
        import cv2
        
        # Criar imagem em branco
        img = np.ones((400, 800, 3), dtype=np.uint8) * 255
        
        # Desenhar linhas de ECG sintéticas
        for i in range(3):
            y_base = 100 + i * 100
            for x in range(0, 800, 2):
                # Simular onda ECG
                y = y_base + int(30 * np.sin(x * 0.02) + 10 * np.sin(x * 0.1))
                cv2.circle(img, (x, y), 1, (0, 0, 0), -1)
        
        # Salvar imagem
        cv2.imwrite(filename, img)
        
    except ImportError:
        # Se OpenCV não estiver disponível, criar arquivo vazio
        with open(filename, 'wb') as f:
            f.write(b'fake_image_data')

def main():
    """Executa todos os testes."""
    print("=" * 60)
    print("🔬 TESTE DO SISTEMA CARDIOAI CORRIGIDO")
    print("=" * 60)
    
    tests = [
        ("Importações", test_imports),
        ("Processamento de Dados", test_data_processing),
        ("Processamento de Imagens", test_image_processing),
        ("Preparação para Modelo", test_model_preparation),
        ("Serialização JSON", test_json_serialization),
        ("Aplicação Corrigida", test_fixed_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Executando: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSOU" if result else "❌ FALHOU"
            print(f"{status}: {test_name}")
        except Exception as e:
            print(f"❌ ERRO: {test_name} - {e}")
            results.append((test_name, False))
    
    # Resumo
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM! Sistema corrigido funcionando.")
    else:
        print("⚠️ Alguns testes falharam. Verificar correções necessárias.")

if __name__ == "__main__":
    main()

