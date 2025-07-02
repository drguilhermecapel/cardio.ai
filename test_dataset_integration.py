#!/usr/bin/env python3
"""
Teste de integração para o sistema de datasets públicos de ECG
"""

import sys
import os

sys.path.append(".")

import numpy as np
import time
from pathlib import Path


def test_dataset_service_integration():
    """Testa a integração do serviço de datasets"""
    print("=" * 60)
    print("TESTANDO INTEGRAÇÃO DO SERVIÇO DE DATASETS")
    print("=" * 60)

    try:
        from app.services.dataset_service import DatasetService

        print("✓ DatasetService importado com sucesso")

        service = DatasetService()
        print("✓ DatasetService inicializado")

        print("\n1️⃣ Validando ambiente...")
        dependencies = service.validate_environment()

        print("Status das dependências:")
        for dep, status in dependencies.items():
            status_icon = "✓" if status else "✗"
            print(
                f"  {status_icon} {dep}: {'Disponível' if status else 'Não disponível'}"
            )

        print("\n2️⃣ Obtendo informações dos datasets...")
        datasets_info = service.get_available_datasets()

        print(f"Datasets disponíveis: {len(datasets_info)}")
        for name, info in datasets_info.items():
            print(
                f"  • {name}: {info['records']} registros, {info['sampling_rate']} Hz"
            )

        print("\n3️⃣ Testando integração com HybridECGAnalysisService...")

        from app.services.hybrid_ecg_service import HybridECGAnalysisService

        hybrid_service = HybridECGAnalysisService()

        if hasattr(hybrid_service, "dataset_service_available"):
            if hybrid_service.dataset_service_available:
                print("✓ Serviço de datasets integrado ao HybridECGAnalysisService")
                print(
                    f"✓ Dataset service: {type(hybrid_service.dataset_service).__name__}"
                )
            else:
                print(
                    "⚠ Serviço de datasets não disponível no HybridECGAnalysisService"
                )
        else:
            print("⚠ Integração com HybridECGAnalysisService não detectada")

        print("\n" + "=" * 60)
        print("RESUMO DO TESTE DE INTEGRAÇÃO")
        print("=" * 60)

        success_criteria = []

        success_criteria.append(True)  # Já passou se chegou até aqui
        print("✓ Importação e inicialização: OK")

        basic_deps = ["pandas", "advanced_preprocessor"]
        basic_deps_ok = all(dependencies.get(dep, False) for dep in basic_deps)
        success_criteria.append(basic_deps_ok)

        if basic_deps_ok:
            print("✓ Dependências básicas: OK")
        else:
            print("✗ Dependências básicas: FALTANDO")

        datasets_ok = len(datasets_info) >= 3  # MIT-BIH, PTB-XL, CPSC2018
        success_criteria.append(datasets_ok)

        if datasets_ok:
            print("✓ Informações dos datasets: OK")
        else:
            print("✗ Informações dos datasets: INCOMPLETAS")

        integration_ok = hasattr(hybrid_service, "dataset_service")
        success_criteria.append(integration_ok)

        if integration_ok:
            print("✓ Integração com HybridECGAnalysisService: OK")
        else:
            print("✗ Integração com HybridECGAnalysisService: FALHOU")

        success_rate = sum(success_criteria) / len(success_criteria) * 100
        print(f"\nTaxa de sucesso: {success_rate:.1f}%")

        if success_rate >= 75:
            print("🎉 TESTE DE INTEGRAÇÃO PASSOU!")
            return True
        else:
            print("❌ TESTE DE INTEGRAÇÃO FALHOU!")
            return False

    except Exception as e:
        print(f"✗ Erro durante o teste: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_dataset_classes():
    """Testa as classes principais do sistema de datasets"""
    print("\n" + "=" * 60)
    print("TESTANDO CLASSES DO SISTEMA DE DATASETS")
    print("=" * 60)

    try:
        from app.datasets import (
            ECGDatasetDownloader,
            ECGDatasetLoader,
            ECGDatasetAnalyzer,
            ECGRecord,
        )

        print("✓ Classes principais importadas com sucesso")

        print("\n1️⃣ Testando ECGRecord...")
        test_signal = np.random.randn(1000, 2)

        record = ECGRecord(
            signal=test_signal,
            sampling_rate=360,
            labels=["normal"],
            patient_id="test_001",
            leads=["I", "II"],
        )

        print(f"✓ ECGRecord criado: {record.patient_id}")
        print(f"  - Signal shape: {record.signal.shape}")
        print(f"  - Sampling rate: {record.sampling_rate} Hz")
        print(f"  - Labels: {record.labels}")

        print("\n2️⃣ Testando ECGDatasetDownloader...")
        downloader = ECGDatasetDownloader(base_dir="test_datasets")

        datasets_info = downloader.get_dataset_info("mit-bih")
        print(f"✓ Informações MIT-BIH obtidas: {datasets_info['name']}")

        print("\n3️⃣ Testando ECGDatasetAnalyzer...")
        analyzer = ECGDatasetAnalyzer()

        test_records = []
        for i in range(3):
            test_record = ECGRecord(
                signal=np.random.randn(3600, 2),  # 10s @ 360Hz
                sampling_rate=360,
                labels=["normal", "test"],
                patient_id=f"test_{i:03d}",
                age=30 + i * 10,
                sex="M" if i % 2 == 0 else "F",
            )
            test_records.append(test_record)

        stats = analyzer.analyze_dataset(test_records, "Test Dataset")
        print(f"✓ Análise concluída: {stats['total_records']} registros")

        print("\n✅ TODAS AS CLASSES TESTADAS COM SUCESSO!")
        return True

    except Exception as e:
        print(f"✗ Erro ao testar classes: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Função principal de teste"""
    print("🧪 INICIANDO TESTES DE INTEGRAÇÃO DE DATASETS")

    test1_passed = test_dataset_service_integration()

    test2_passed = test_dataset_classes()

    print("\n" + "=" * 70)
    print("RESULTADO FINAL DOS TESTES")
    print("=" * 70)

    tests_passed = sum([test1_passed, test2_passed])
    total_tests = 2

    print(f"Testes passaram: {tests_passed}/{total_tests}")

    if tests_passed == total_tests:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("\n✅ Sistema de datasets públicos integrado com sucesso!")
        return True
    else:
        print("❌ ALGUNS TESTES FALHARAM!")
        print("\n⚠️  Verifique as dependências e configurações.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
