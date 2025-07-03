#!/usr/bin/env python3
"""
Script para testar se todos os módulos necessários podem ser importados
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


def test_imports():
    """Testa importação de todos os módulos necessários"""
    print("🔍 Testando importações dos módulos do CardioAI Pro...")

    try:
        from app.datasets.ecg_public_datasets import (
            ECGDatasetDownloader,
            ECGDatasetLoader,
        )

        print("✅ ECG datasets module imported successfully")
    except Exception as e:
        print(f"❌ Error importing ECG datasets: {e}")
        return False

    try:
        from app.ml.hybrid_architecture import ModelConfig, create_hybrid_model

        print("✅ Hybrid architecture module imported successfully")
    except Exception as e:
        print(f"❌ Error importing hybrid architecture: {e}")
        return False

    try:
        from app.ml.training_pipeline import TrainingConfig, ECGTrainingPipeline

        print("✅ Training pipeline module imported successfully")
    except Exception as e:
        print(f"❌ Error importing training pipeline: {e}")
        return False

    try:
        from app.services.advanced_ml_service import AdvancedMLService, AdvancedMLConfig

        print("✅ Advanced ML service module imported successfully")
    except Exception as e:
        print(f"❌ Error importing advanced ML service: {e}")
        return False

    print("🚀 All core modules are importable!")
    return True


if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Todos os módulos foram importados com sucesso!")
        print("🎯 O sistema está pronto para executar os próximos passos.")
    else:
        print("\n❌ Alguns módulos falharam na importação.")
        print("Verifique as dependências e configurações.")
        sys.exit(1)
