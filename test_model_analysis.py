#!/usr/bin/env python3
"""
Script para análise e teste do modelo ECG
Identifica problemas no carregamento e funcionamento do modelo
"""

import os
import sys
import numpy as np
import json
from pathlib import Path

# Adicionar o diretório do projeto ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

def test_tensorflow_import():
    """Testa se o TensorFlow pode ser importado."""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow importado com sucesso - Versão: {tf.__version__}")
        return True, tf
    except ImportError as e:
        print(f"❌ Erro ao importar TensorFlow: {e}")
        return False, None

def test_model_loading():
    """Testa o carregamento do modelo ECG."""
    print("\n=== TESTE DE CARREGAMENTO DO MODELO ===")
    
    # Verificar se TensorFlow está disponível
    tf_available, tf = test_tensorflow_import()
    if not tf_available:
        return False, None
    
    # Caminhos possíveis para o modelo
    model_paths = [
        "models/ecg_model_final.h5",
        "ecg_model_final.h5",
        "backend/models/ecg_model_final.h5"
    ]
    
    model = None
    model_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"📁 Modelo encontrado em: {path}")
            try:
                model = tf.keras.models.load_model(path)
                model_path = path
                print(f"✅ Modelo carregado com sucesso!")
                break
            except Exception as e:
                print(f"❌ Erro ao carregar modelo de {path}: {e}")
                continue
    
    if model is None:
        print("❌ Nenhum modelo pôde ser carregado")
        return False, None
    
    return True, (model, model_path)

def analyze_model_architecture(model):
    """Analisa a arquitetura do modelo."""
    print("\n=== ANÁLISE DA ARQUITETURA DO MODELO ===")
    
    try:
        # Informações básicas
        print(f"📊 Input shape: {model.input_shape}")
        print(f"📊 Output shape: {model.output_shape}")
        print(f"📊 Número de parâmetros: {model.count_params():,}")
        
        # Resumo da arquitetura
        print("\n📋 Resumo da arquitetura:")
        model.summary()
        
        # Verificar se o modelo foi compilado
        if hasattr(model, 'optimizer') and model.optimizer is not None:
            print(f"✅ Modelo compilado com otimizador: {model.optimizer.__class__.__name__}")
        else:
            print("⚠️ Modelo não compilado")
        
        return True
    except Exception as e:
        print(f"❌ Erro na análise da arquitetura: {e}")
        return False

def test_model_prediction(model):
    """Testa uma predição simples com o modelo."""
    print("\n=== TESTE DE PREDIÇÃO ===")
    
    try:
        # Obter shape de entrada
        input_shape = model.input_shape
        print(f"📊 Shape de entrada esperado: {input_shape}")
        
        # Criar dados de teste sintéticos
        if len(input_shape) == 3:  # (batch, time, features)
            test_data = np.random.randn(1, input_shape[1], input_shape[2])
        elif len(input_shape) == 2:  # (batch, features)
            test_data = np.random.randn(1, input_shape[1])
        else:
            print(f"❌ Shape de entrada não suportado: {input_shape}")
            return False
        
        print(f"📊 Dados de teste criados com shape: {test_data.shape}")
        
        # Fazer predição
        prediction = model.predict(test_data, verbose=0)
        print(f"✅ Predição realizada com sucesso!")
        print(f"📊 Shape da predição: {prediction.shape}")
        print(f"📊 Valores da predição: {prediction[0][:10]}...")  # Primeiros 10 valores
        
        return True, prediction
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        return False, None

def analyze_model_metadata():
    """Analisa metadados do modelo."""
    print("\n=== ANÁLISE DE METADADOS ===")
    
    metadata_paths = [
        "models/model_info.json",
        "model_info.json",
        "models/model_architecture.json"
    ]
    
    for path in metadata_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"📄 Metadados encontrados em: {path}")
                print(json.dumps(metadata, indent=2, ensure_ascii=False))
                return True, metadata
            except Exception as e:
                print(f"❌ Erro ao ler metadados de {path}: {e}")
    
    print("⚠️ Nenhum arquivo de metadados encontrado")
    return False, None

def check_preprocessing_functions():
    """Verifica funções de pré-processamento."""
    print("\n=== VERIFICAÇÃO DE PRÉ-PROCESSAMENTO ===")
    
    preprocess_paths = [
        "models/preprocess_functions.py",
        "backend/app/utils/ecg_processor.py",
        "ecg_processor.py"
    ]
    
    for path in preprocess_paths:
        if os.path.exists(path):
            print(f"📄 Arquivo de pré-processamento encontrado: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                print(f"📊 Tamanho do arquivo: {len(content)} caracteres")
                
                # Verificar funções importantes
                important_functions = [
                    'preprocess', 'normalize', 'filter', 'resample',
                    'extract_features', 'load_ecg', 'process_ecg'
                ]
                
                found_functions = []
                for func in important_functions:
                    if f"def {func}" in content:
                        found_functions.append(func)
                
                print(f"🔍 Funções encontradas: {found_functions}")
                return True, path
            except Exception as e:
                print(f"❌ Erro ao ler {path}: {e}")
    
    print("⚠️ Nenhum arquivo de pré-processamento encontrado")
    return False, None

def identify_issues():
    """Identifica problemas comuns no modelo e pipeline."""
    print("\n=== IDENTIFICAÇÃO DE PROBLEMAS ===")
    
    issues = []
    
    # Verificar dependências
    try:
        import tensorflow as tf
    except ImportError:
        issues.append("TensorFlow não instalado")
    
    try:
        import numpy as np
    except ImportError:
        issues.append("NumPy não instalado")
    
    try:
        import pandas as pd
    except ImportError:
        issues.append("Pandas não instalado")
    
    # Verificar arquivos essenciais
    essential_files = [
        "models/ecg_model_final.h5",
        "models/model_info.json",
        "backend/app/services/unified_model_service.py",
        "backend/app/services/unified_ecg_service.py"
    ]
    
    for file_path in essential_files:
        if not os.path.exists(file_path):
            issues.append(f"Arquivo essencial não encontrado: {file_path}")
    
    if issues:
        print("❌ Problemas identificados:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Nenhum problema crítico identificado")
    
    return issues

def main():
    """Função principal de análise."""
    print("🔍 ANÁLISE COMPLETA DO MODELO ECG")
    print("=" * 50)
    
    # Mudar para o diretório do projeto
    os.chdir(project_root)
    
    # 1. Identificar problemas gerais
    issues = identify_issues()
    
    # 2. Testar carregamento do modelo
    model_loaded, model_data = test_model_loading()
    
    if model_loaded:
        model, model_path = model_data
        
        # 3. Analisar arquitetura
        analyze_model_architecture(model)
        
        # 4. Testar predição
        test_model_prediction(model)
    
    # 5. Analisar metadados
    analyze_model_metadata()
    
    # 6. Verificar pré-processamento
    check_preprocessing_functions()
    
    print("\n" + "=" * 50)
    print("🏁 ANÁLISE CONCLUÍDA")
    
    if issues:
        print(f"⚠️ {len(issues)} problema(s) identificado(s)")
        return False
    else:
        print("✅ Sistema aparenta estar funcionando corretamente")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

