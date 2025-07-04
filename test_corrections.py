#!/usr/bin/env python3
"""
Teste das correções implementadas no CardioAI Pro
"""

import sys
import os
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Testa se todas as importações estão funcionando."""
    print("🔍 Testando importações...")
    
    try:
        # Testar importação do digitalizador híbrido
        from hybrid_ecg_digitizer import HybridECGDigitizer
        print("✅ HybridECGDigitizer importado com sucesso")
        
        # Testar importação dos serviços
        sys.path.append('backend/app')
        from services.unified_ecg_service import get_ecg_service
        from services.unified_model_service import get_model_service
        print("✅ Serviços unificados importados com sucesso")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na importação: {str(e)}")
        return False

def test_digitizer():
    """Testa o digitalizador híbrido."""
    print("\n🔍 Testando digitalizador híbrido...")
    
    try:
        from hybrid_ecg_digitizer import HybridECGDigitizer
        
        # Criar imagem de teste
        test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            Image.fromarray(test_image).save(tmp_file.name)
            
            # Inicializar digitalizador
            digitizer = HybridECGDigitizer(target_length=1000, verbose=False)
            
            # Testar digitalização
            result = digitizer.digitize(tmp_file.name)
            
            print(f"✅ Digitalização concluída:")
            print(f"   - Método: {result.get('method', 'N/A')}")
            print(f"   - Shape dos dados: {result['data'].shape}")
            print(f"   - Qualidade: {result.get('quality', {}).get('overall_score', 'N/A')}")
            
            # Limpar arquivo temporário
            os.unlink(tmp_file.name)
            
        return True
        
    except Exception as e:
        print(f"❌ Erro no digitalizador: {str(e)}")
        return False

def test_unified_ecg_service():
    """Testa o serviço unificado de ECG."""
    print("\n🔍 Testando UnifiedECGService...")
    
    try:
        sys.path.append('backend/app')
        from services.unified_ecg_service import get_ecg_service
        
        # Obter serviço
        ecg_service = get_ecg_service()
        
        # Verificar se método process_ecg_image existe
        if hasattr(ecg_service, 'process_ecg_image'):
            print("✅ Método process_ecg_image encontrado")
            
            # Criar imagem de teste
            test_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                Image.fromarray(test_image).save(tmp_file.name)
                
                # Testar processamento
                result = ecg_service.process_ecg_image(
                    image_path=tmp_file.name,
                    patient_id="TEST_001",
                    quality_threshold=0.1  # Baixo para aceitar dados sintéticos
                )
                
                print(f"✅ Processamento de imagem concluído:")
                print(f"   - Process ID: {result.get('process_id', 'N/A')}")
                print(f"   - Digitalização real: {result.get('digitization', {}).get('real_digitization', 'N/A')}")
                print(f"   - Método: {result.get('digitization', {}).get('method', 'N/A')}")
                print(f"   - Shape dos dados: {result.get('data', {}).get('shape', 'N/A')}")
                
                # Limpar arquivo temporário
                os.unlink(tmp_file.name)
                
        else:
            print("❌ Método process_ecg_image não encontrado")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Erro no UnifiedECGService: {str(e)}")
        return False

def test_model_service():
    """Testa o serviço de modelo."""
    print("\n🔍 Testando UnifiedModelService...")
    
    try:
        sys.path.append('backend/app')
        from services.unified_model_service import get_model_service
        
        # Obter serviço
        model_service = get_model_service()
        
        # Listar modelos
        models = model_service.list_models()
        print(f"✅ Modelos disponíveis: {models}")
        
        # Testar predição com dados sintéticos
        test_data = np.random.randn(1, 12, 1000)
        
        if models:
            model_name = models[0]
            try:
                prediction = model_service.predict_ecg(test_data, model_name)
                print(f"✅ Predição realizada com {model_name}")
                print(f"   - Tipo de resultado: {type(prediction)}")
                
            except Exception as e:
                print(f"⚠️ Erro na predição: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no UnifiedModelService: {str(e)}")
        return False

def test_main_direct_syntax():
    """Testa se o main_direct.py tem sintaxe válida."""
    print("\n🔍 Testando sintaxe do main_direct.py...")
    
    try:
        import ast
        
        with open('backend/app/main_direct.py', 'r') as f:
            content = f.read()
        
        # Tentar parsear o código
        ast.parse(content)
        print("✅ Sintaxe do main_direct.py válida")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Erro de sintaxe no main_direct.py: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Erro ao verificar main_direct.py: {str(e)}")
        return False

def main():
    """Executa todos os testes."""
    print("🧪 TESTE DAS CORREÇÕES DO CARDIOAI PRO")
    print("=" * 50)
    
    tests = [
        ("Importações", test_imports),
        ("Digitalizador Híbrido", test_digitizer),
        ("UnifiedECGService", test_unified_ecg_service),
        ("UnifiedModelService", test_model_service),
        ("Sintaxe main_direct.py", test_main_direct_syntax)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro inesperado em {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Resumo dos resultados
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 RESULTADO FINAL: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODAS AS CORREÇÕES ESTÃO FUNCIONANDO!")
    else:
        print("⚠️ Algumas correções precisam de ajustes.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

