#!/usr/bin/env python3
"""
Script de teste para o Interpretador de ECG
"""

import sys
import os
import numpy as np
import json
from datetime import datetime

# Adicionar o diretório backend ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_ecg_interpreter():
    """Testa o interpretador de ECG."""
    print("=== Teste do Interpretador de ECG CardioAI Pro ===\n")
    
    try:
        # Importar o interpretador
        from app.services.ecg_interpreter import ecg_interpreter, create_sample_ecg_data
        
        print("✓ Interpretador importado com sucesso")
        
        # Verificar status inicial
        status = ecg_interpreter.get_status()
        print(f"✓ Status inicial: {status}")
        
        # Carregar modelo
        model_loaded = ecg_interpreter.load_model()
        print(f"✓ Modelo carregado: {model_loaded}")
        
        # Criar dados de ECG simulados
        print("\n--- Criando dados de ECG simulados ---")
        ecg_data = create_sample_ecg_data(duration=10, sampling_rate=500)
        print(f"✓ Dados criados: {len(ecg_data)} amostras")
        print(f"✓ Duração: 10 segundos")
        print(f"✓ Taxa de amostragem: 500 Hz")
        
        # Informações do paciente de teste
        patient_info = {
            "patient_id": "TEST_001",
            "patient_name": "Paciente Teste",
            "patient_age": 35
        }
        
        # Realizar análise
        print("\n--- Realizando análise de ECG ---")
        results = ecg_interpreter.analyze_ecg(
            ecg_data=ecg_data,
            sampling_rate=500,
            patient_info=patient_info
        )
        
        # Verificar se houve erro
        if "error" in results:
            print(f"✗ Erro na análise: {results['error']}")
            return False
        
        # Exibir resultados
        print("✓ Análise concluída com sucesso!")
        print(f"\n--- Resultados da Análise ---")
        print(f"ID da Análise: {results['analysis_id']}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Qualidade do Sinal: {results['signal_quality']}")
        print(f"Picos R Detectados: {results['r_peaks_count']}")
        print(f"Score de Confiança: {results['confidence_score']}")
        
        # Análise do ritmo
        rhythm = results['rhythm_analysis']
        print(f"\n--- Análise do Ritmo ---")
        print(f"Ritmo: {rhythm['rhythm']}")
        print(f"Regularidade: {rhythm['regularity']}")
        print(f"Frequência Cardíaca: {rhythm['heart_rate']} bpm")
        print(f"Variabilidade RR: {rhythm['rr_variability']}")
        
        # Anormalidades
        abnormalities = results['abnormalities']
        print(f"\n--- Anormalidades ---")
        if abnormalities:
            print(f"Encontradas {len(abnormalities)} anormalidades:")
            for i, abnormality in enumerate(abnormalities, 1):
                print(f"  {i}. {abnormality['type']}: {abnormality['description']}")
        else:
            print("Nenhuma anormalidade detectada")
        
        # Interpretação
        print(f"\n--- Interpretação ---")
        print(results['interpretation'])
        
        # Teste de múltiplas análises
        print(f"\n--- Teste de Múltiplas Análises ---")
        for i in range(3):
            test_data = create_sample_ecg_data(duration=5, sampling_rate=500)
            test_results = ecg_interpreter.analyze_ecg(test_data, 500)
            print(f"✓ Análise {i+1}: {test_results['analysis_id']}")
        
        # Status final
        final_status = ecg_interpreter.get_status()
        print(f"\n--- Status Final ---")
        print(f"Análises realizadas: {final_status['analyses_performed']}")
        print(f"Status: {final_status['status']}")
        
        print(f"\n=== Teste Concluído com Sucesso! ===")
        return True
        
    except Exception as e:
        print(f"✗ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_imports():
    """Testa as importações da API."""
    print("\n=== Teste das Importações da API ===\n")
    
    try:
        from app.api.ecg_api import router
        print("✓ Router da API importado com sucesso")
        
        from app.main import app
        print("✓ Aplicação FastAPI importada com sucesso")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro nas importações da API: {e}")
        return False


if __name__ == "__main__":
    print("Iniciando testes do CardioAI Pro...\n")
    
    # Teste do interpretador
    interpreter_ok = test_ecg_interpreter()
    
    # Teste das importações da API
    api_ok = test_api_imports()
    
    # Resultado final
    print(f"\n{'='*50}")
    print(f"RESULTADO DOS TESTES:")
    print(f"Interpretador de ECG: {'✓ PASSOU' if interpreter_ok else '✗ FALHOU'}")
    print(f"Importações da API: {'✓ PASSOU' if api_ok else '✗ FALHOU'}")
    
    if interpreter_ok and api_ok:
        print(f"\n🎉 TODOS OS TESTES PASSARAM! 🎉")
        print(f"O CardioAI Pro está pronto para uso!")
    else:
        print(f"\n❌ ALGUNS TESTES FALHARAM")
        print(f"Verifique os erros acima")
    
    print(f"{'='*50}")

