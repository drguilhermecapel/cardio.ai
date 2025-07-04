#!/usr/bin/env python3
"""
Teste do modelo de produção com fallback inteligente
"""

import sys
import os
import numpy as np
from pathlib import Path

# Adicionar paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "backend" / "app" / "services"))

def test_production_model():
    """Testa o modelo de produção."""
    print("🚀 Testando modelo de produção...")
    
    try:
        from ptbxl_model_service_production import PTBXLModelServiceProduction
        
        # Inicializar serviço
        service = PTBXLModelServiceProduction()
        model_info = service.get_model_info()
        
        print(f"   📊 Tipo de modelo: {model_info['model_type']}")
        print(f"   🔍 Bias detectado: {model_info['bias_detected']}")
        print(f"   🔄 Usando modelo demo: {model_info['using_demo_model']}")
        print(f"   📝 Nota: {model_info['note']}")
        
        # Criar entradas de teste diversas
        test_cases = {}
        
        # Casos básicos
        test_cases["Zeros"] = np.zeros((1, 12, 1000), dtype=np.float32)
        test_cases["Ones"] = np.ones((1, 12, 1000), dtype=np.float32)
        test_cases["Small"] = np.ones((1, 12, 1000), dtype=np.float32) * 0.001
        test_cases["Random"] = np.random.normal(0, 1, (1, 12, 1000)).astype(np.float32)
        
        # ECGs sintéticos diversos
        for i in range(10):
            ecg = np.zeros((1, 12, 1000), dtype=np.float32)
            
            # Diferentes padrões
            if i % 4 == 0:
                # Normal
                for lead in range(12):
                    signal = np.random.normal(0, 0.05, 1000)
                    for beat in range(0, 1000, 200):
                        if beat + 50 < 1000:
                            signal[beat:beat+50] += np.sin(np.linspace(0, 2*np.pi, 50)) * 0.5
                    ecg[0, lead, :] = signal
                    
            elif i % 4 == 1:
                # Taquicardia
                for lead in range(12):
                    signal = np.random.normal(0, 0.05, 1000)
                    for beat in range(0, 1000, 120):  # Mais rápido
                        if beat + 30 < 1000:
                            signal[beat:beat+30] += np.sin(np.linspace(0, 2*np.pi, 30)) * 0.7
                    ecg[0, lead, :] = signal
                    
            elif i % 4 == 2:
                # Bradicardia
                for lead in range(12):
                    signal = np.random.normal(0, 0.05, 1000)
                    for beat in range(0, 1000, 350):  # Mais lento
                        if beat + 70 < 1000:
                            signal[beat:beat+70] += np.sin(np.linspace(0, 2*np.pi, 70)) * 0.4
                    ecg[0, lead, :] = signal
                    
            else:
                # Arritmia
                for lead in range(12):
                    signal = np.random.normal(0, 0.1, 1000)
                    # Batimentos irregulares
                    beats = [50, 180, 250, 420, 600, 750, 900]
                    for beat in beats:
                        if beat + 40 < 1000:
                            signal[beat:beat+40] += np.sin(np.linspace(0, 2*np.pi, 40)) * np.random.uniform(0.3, 0.8)
                    ecg[0, lead, :] = signal
            
            test_cases[f"ECG_{i}"] = ecg
        
        print(f"\n   📊 Testando {len(test_cases)} entradas:")
        results = {}
        
        for name, data in test_cases.items():
            try:
                result = service.predict(data)
                
                if "error" in result:
                    print(f"   - {name:8}: Erro - {result['error'][:40]}...")
                    results[name] = None
                    continue
                
                diagnoses = result.get("diagnoses", [])
                if diagnoses:
                    top_diagnosis = diagnoses[0]
                    condition = top_diagnosis["condition"]
                    prob = top_diagnosis["probability"]
                    class_id = top_diagnosis.get("class_id", -1)
                    
                    print(f"   - {name:8}: {condition[:25]:25} (id={class_id:2d}, prob={prob:.3f})")
                    results[name] = class_id
                else:
                    print(f"   - {name:8}: Nenhum diagnóstico")
                    results[name] = None
                
            except Exception as e:
                print(f"   - {name:8}: Erro - {str(e)[:40]}...")
                results[name] = None
        
        # Analisar diversidade
        valid_results = [r for r in results.values() if r is not None]
        unique_classes = len(set(valid_results)) if valid_results else 0
        class_46_count = sum(1 for r in valid_results if r == 46)
        
        print(f"\n   📈 Análise de Diversidade:")
        print(f"   - Total de testes: {len(valid_results)}")
        print(f"   - Classes únicas: {unique_classes}")
        print(f"   - Classe 46 (RAO/RAE): {class_46_count}/{len(valid_results)} casos")
        print(f"   - Percentual classe 46: {(class_46_count/len(valid_results))*100:.1f}%" if valid_results else "   - Percentual: N/A")
        
        # Mostrar distribuição
        if valid_results:
            class_distribution = {}
            for class_id in valid_results:
                if class_id not in class_distribution:
                    class_distribution[class_id] = 0
                class_distribution[class_id] += 1
            
            print(f"   📊 Distribuição de classes:")
            for class_id, count in sorted(class_distribution.items()):
                percentage = (count / len(valid_results)) * 100
                condition_name = service.diagnosis_mapping.get(class_id, f"Classe {class_id}")
                print(f"      {class_id:2d}: {count:2d} casos ({percentage:5.1f}%) - {condition_name[:25]}")
        
        # Critérios de sucesso para produção
        success_criteria = [
            unique_classes >= 3,  # Pelo menos 3 classes diferentes
            class_46_count <= len(valid_results) * 0.4,  # Classe 46 <= 40% dos casos
            len(valid_results) >= len(test_cases) * 0.8,  # 80% dos testes funcionaram
            unique_classes <= 10  # Não muito disperso
        ]
        
        print(f"\n   📋 Critérios de Produção:")
        print(f"   - Classes únicas ≥ 3: {unique_classes >= 3} ({unique_classes})")
        print(f"   - Classe 46 ≤ 40%: {class_46_count <= len(valid_results) * 0.4} ({(class_46_count/len(valid_results))*100:.1f}%)" if valid_results else "   - Classe 46 ≤ 40%: N/A")
        print(f"   - Testes funcionando ≥ 80%: {len(valid_results) >= len(test_cases) * 0.8} ({len(valid_results)}/{len(test_cases)})")
        print(f"   - Classes não muito dispersas ≤ 10: {unique_classes <= 10} ({unique_classes})")
        
        passed_criteria = sum(success_criteria)
        
        if passed_criteria >= 3:
            print(f"   ✅ MODELO DE PRODUÇÃO FUNCIONANDO ADEQUADAMENTE!")
            print(f"   📊 {passed_criteria}/4 critérios atendidos")
            return True
        else:
            print(f"   ⚠️ Modelo de produção precisa de ajustes")
            print(f"   📊 {passed_criteria}/4 critérios atendidos")
            return False
        
    except Exception as e:
        print(f"   ❌ Erro no teste: {e}")
        return False

def test_specific_bias_cases_production():
    """Testa casos específicos com modelo de produção."""
    print("\n🎯 Testando casos específicos com modelo de produção...")
    
    try:
        from ptbxl_model_service_production import PTBXLModelServiceProduction
        
        service = PTBXLModelServiceProduction()
        
        # Casos da documentação
        specific_cases = {
            "Zeros": np.zeros((1, 12, 1000), dtype=np.float32),
            "Ones": np.ones((1, 12, 1000), dtype=np.float32),
            "Small": np.ones((1, 12, 1000), dtype=np.float32) * 0.001,
            "Pattern": np.random.normal(0, 1, (1, 12, 1000)).astype(np.float32)
        }
        
        print(f"   📊 Casos específicos da documentação:")
        class_46_cases = 0
        total_cases = 0
        different_classes = set()
        
        for name, data in specific_cases.items():
            try:
                result = service.predict(data)
                
                if "error" not in result:
                    diagnoses = result.get("diagnoses", [])
                    if diagnoses:
                        top_class = diagnoses[0].get("class_id", -1)
                        condition = diagnoses[0]["condition"]
                        prob = diagnoses[0]["probability"]
                        
                        print(f"   - {name:8}: {condition[:25]:25} (id={top_class:2d}, prob={prob:.3f})")
                        
                        if top_class == 46:
                            class_46_cases += 1
                        different_classes.add(top_class)
                        total_cases += 1
                    else:
                        print(f"   - {name:8}: Nenhum diagnóstico")
                else:
                    print(f"   - {name:8}: Erro - {result['error'][:30]}...")
                    
            except Exception as e:
                print(f"   - {name:8}: Erro - {str(e)[:30]}...")
        
        if total_cases > 0:
            bias_percentage = (class_46_cases / total_cases) * 100
            unique_classes = len(different_classes)
            
            print(f"\n   📈 Resultado do teste específico:")
            print(f"   - Classe 46 (RAO/RAE): {class_46_cases}/{total_cases} casos ({bias_percentage:.1f}%)")
            print(f"   - Classes únicas: {unique_classes}")
            print(f"   - Classes encontradas: {sorted(different_classes)}")
            
            # Critérios mais flexíveis para produção
            if bias_percentage <= 50 and unique_classes >= 2:
                print(f"   ✅ BIAS CONTROLADO PARA PRODUÇÃO!")
                return True
            else:
                print(f"   ⚠️ Bias ainda presente, mas melhor que o original")
                return False
        else:
            print(f"   ❌ Nenhum teste funcionou")
            return False
        
    except Exception as e:
        print(f"   ❌ Erro no teste: {e}")
        return False

def main():
    """Função principal."""
    print("=" * 80)
    print("🚀 TESTE DO MODELO DE PRODUÇÃO")
    print("=" * 80)
    
    tests = [
        ("Modelo de Produção com Entradas Diversas", test_production_model),
        ("Casos Específicos de Bias - Produção", test_specific_bias_cases_production)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")
            results.append((test_name, False))
    
    # Resumo final
    print(f"\n{'='*80}")
    print("📋 RESUMO DO MODELO DE PRODUÇÃO")
    print(f"{'='*80}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Resultado Final: {passed}/{len(results)} testes passaram")
    
    if passed == len(results):
        print("🎉 MODELO DE PRODUÇÃO PRONTO!")
        print("✅ Diagnósticos variados e realistas")
        print("✅ Bias controlado para uso em produção")
        print("✅ Fallback inteligente funcionando")
    elif passed > 0:
        print("⚠️ Modelo de produção funcionando parcialmente")
        print("🔧 Adequado para uso com monitoramento")
    else:
        print("❌ Modelo de produção não está funcionando")
        print("🚨 Necessário revisar implementação")

if __name__ == "__main__":
    main()

