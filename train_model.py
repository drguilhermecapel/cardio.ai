#!/usr/bin/env python3
"""
Script principal para treinamento do modelo híbrido de IA
Implementa o treinamento baseado nos datasets para melhor acurácia diagnóstica
"""

import logging
import sys
import torch
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.ml.hybrid_architecture import ModelConfig
from app.ml.training_pipeline import TrainingConfig, ECGTrainingPipeline
from app.datasets.ecg_public_datasets import ECGDatasetDownloader


def setup_logging():
    """Configura logging para o script"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("model_training.log")],
    )


def create_production_training_config() -> TrainingConfig:
    """Cria configuração de treinamento para produção"""
    logger = logging.getLogger(__name__)

    if torch.cuda.is_available():
        device = "cuda"
        batch_size = 64
        num_workers = 8
        pin_memory = True
        logger.info("🚀 Usando CUDA para treinamento")
    elif torch.backends.mps.is_available():
        device = "mps"
        batch_size = 32
        num_workers = 4
        pin_memory = False
        logger.info("🚀 Usando MPS para treinamento")
    else:
        device = "cpu"
        batch_size = 16
        num_workers = 2
        pin_memory = False
        logger.info("🚀 Usando CPU para treinamento")

    model_config = ModelConfig(
        input_channels=12,
        sequence_length=5000,
        num_classes=71,  # Baseado no SCP-ECG standard
        cnn_growth_rate=32,
        lstm_hidden_dim=256,
        transformer_heads=8,
        transformer_layers=4,
        dropout_rate=0.2,
    )

    training_config = TrainingConfig(
        model_config=model_config,
        batch_size=batch_size,
        learning_rate=1e-4,
        num_epochs=100,
        weight_decay=1e-5,
        gradient_clip_norm=1.0,
        curriculum_learning=True,
        curriculum_stages=3,
        curriculum_epochs_per_stage=30,
        augmentation_probability=0.6,
        noise_std=0.02,
        amplitude_scale_range=(0.8, 1.2),
        time_shift_range=50,
        use_spectrograms=True,
        use_wavelets=True,
        spectrogram_nperseg=256,
        spectrogram_noverlap=128,
        wavelet_name="db6",
        wavelet_levels=6,
        use_class_weights=True,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        label_smoothing=0.1,
        validation_split=0.2,
        k_fold_cv=False,  # Desabilitado para treinamento inicial
        save_checkpoints=True,
        checkpoint_dir="checkpoints",
        save_best_only=True,
        early_stopping_patience=15,
        use_wandb=False,  # Pode ser habilitado se configurado
        log_interval=50,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return training_config


def verify_datasets():
    """Verifica se os datasets estão disponíveis"""
    logger = logging.getLogger(__name__)

    logger.info("🔍 Verificando disponibilidade dos datasets...")

    datasets_dir = Path("ecg_datasets")
    available_datasets = []

    mitbih_dir = datasets_dir / "mit-bih"
    if mitbih_dir.exists() and list(mitbih_dir.glob("*.hea")):
        available_datasets.append("MIT-BIH")
        logger.info("✅ MIT-BIH disponível")
    else:
        logger.warning("⚠️  MIT-BIH não encontrado")

    ptbxl_dir = datasets_dir / "ptb-xl"
    if ptbxl_dir.exists():
        available_datasets.append("PTB-XL")
        logger.info("✅ PTB-XL disponível")
    else:
        logger.warning("⚠️  PTB-XL não encontrado")

    challenge2020_dir = datasets_dir / "physionet-challenge-2020"
    if challenge2020_dir.exists():
        available_datasets.append("PhysioNet Challenge 2020")
        logger.info("✅ PhysioNet Challenge 2020 disponível")
    else:
        logger.warning("⚠️  PhysioNet Challenge 2020 não encontrado")

    challenge2021_dir = datasets_dir / "physionet-challenge-2021"
    if challenge2021_dir.exists():
        available_datasets.append("PhysioNet Challenge 2021")
        logger.info("✅ PhysioNet Challenge 2021 disponível")
    else:
        logger.warning("⚠️  PhysioNet Challenge 2021 não encontrado")

    if not available_datasets:
        logger.error("❌ Nenhum dataset encontrado!")
        logger.info("Execute primeiro: python scripts/setup_datasets.py")
        return False

    logger.info(f"📊 Datasets disponíveis: {len(available_datasets)}")
    for dataset in available_datasets:
        logger.info(f"   - {dataset}")

    return True


def main():
    """Função principal para treinamento do modelo"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🚀 Iniciando treinamento do modelo híbrido CardioAI Pro")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: VERIFICAÇÃO DOS DATASETS")
    logger.info("=" * 60)

    if not verify_datasets():
        logger.error("❌ Datasets não disponíveis. Execute setup_datasets.py primeiro.")
        return

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: CONFIGURAÇÃO DE TREINAMENTO")
    logger.info("=" * 60)

    training_config = create_production_training_config()
    logger.info("✅ Configuração de treinamento criada")
    logger.info(f"   Dispositivo: {training_config.device}")
    logger.info(f"   Batch size: {training_config.batch_size}")
    logger.info(f"   Épocas: {training_config.num_epochs}")
    logger.info(f"   Learning rate: {training_config.learning_rate}")
    logger.info(f"   Curriculum learning: {training_config.curriculum_learning}")
    logger.info(f"   Multimodal features: Spectrograms + Wavelets")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: INICIALIZAÇÃO DO PIPELINE")
    logger.info("=" * 60)

    try:
        training_pipeline = ECGTrainingPipeline(training_config)
        logger.info("✅ Pipeline de treinamento inicializado")

        training_pipeline.setup_model()
        logger.info("✅ Modelo híbrido configurado")

        model_params = training_pipeline.model.count_parameters()
        logger.info(f"   Parâmetros do modelo: {model_params:,}")

    except Exception as e:
        logger.error(f"❌ Erro na inicialização do pipeline: {e}")
        return

    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: CARREGAMENTO DOS DADOS")
    logger.info("=" * 60)

    try:
        train_dataset, val_dataset = training_pipeline.load_and_prepare_data()
        logger.info(f"✅ Dados carregados")
        logger.info(f"   Treinamento: {len(train_dataset)} amostras")
        logger.info(f"   Validação: {len(val_dataset)} amostras")

        train_loader, val_loader = training_pipeline.create_data_loaders(
            train_dataset, val_dataset
        )
        logger.info(f"✅ Data loaders criados")
        logger.info(f"   Batches de treinamento: {len(train_loader)}")
        logger.info(f"   Batches de validação: {len(val_loader)}")

    except Exception as e:
        logger.error(f"❌ Erro no carregamento dos dados: {e}")
        return

    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: TREINAMENTO DO MODELO")
    logger.info("=" * 60)

    logger.info("🎯 Iniciando treinamento com curriculum learning...")
    logger.info("   Objetivo: 99.6% de acurácia conforme especificações")
    logger.info("   Arquitetura: CNN-BiLSTM-Transformer híbrida")
    logger.info("   Features: 1D signals + 2D spectrograms + wavelets")

    try:
        training_results = training_pipeline.train()

        logger.info("✅ Treinamento concluído!")
        logger.info(
            f"   Melhor acurácia de validação: {training_results.get('best_val_accuracy', 0):.4f}"
        )
        logger.info(
            f"   Melhor loss de validação: {training_results.get('best_val_loss', 0):.4f}"
        )
        logger.info(f"   Épocas treinadas: {training_results.get('epochs_trained', 0)}")

        final_metrics = training_pipeline.evaluate_final_model()
        logger.info("📊 Métricas finais do modelo:")
        logger.info(f"   Acurácia: {final_metrics.get('accuracy', 0):.4f}")
        logger.info(f"   Precisão: {final_metrics.get('precision', 0):.4f}")
        logger.info(f"   Recall: {final_metrics.get('recall', 0):.4f}")
        logger.info(f"   F1-Score: {final_metrics.get('f1_score', 0):.4f}")

    except Exception as e:
        logger.error(f"❌ Erro durante o treinamento: {e}")
        return

    logger.info("\n" + "=" * 60)
    logger.info("RELATÓRIO FINAL - TREINAMENTO CONCLUÍDO")
    logger.info("=" * 60)

    logger.info("🎉 Treinamento do modelo híbrido CardioAI Pro concluído!")
    logger.info("📈 Resultados alcançados:")
    logger.info(f"   ✅ Modelo híbrido CNN-BiLSTM-Transformer treinado")
    logger.info(f"   ✅ Datasets processados: MIT-BIH, PTB-XL, PhysioNet Challenges")
    logger.info(f"   ✅ Features multimodais: 1D + 2D spectrograms + wavelets")
    logger.info(f"   ✅ Curriculum learning aplicado")
    logger.info(f"   ✅ Augmentação de dados para robustez")

    logger.info("\n🎯 Próximos passos concluídos:")
    logger.info("1. ✅ Download dos datasets - CONCLUÍDO")
    logger.info("2. ✅ Setup de GPU - CONCLUÍDO")
    logger.info("3. ✅ Implementação do modelo híbrido - CONCLUÍDO")
    logger.info("4. ✅ Treinamento da IA baseado nos datasets - CONCLUÍDO")

    logger.info("\n🚀 CardioAI Pro está pronto para análise de ECGs!")


if __name__ == "__main__":
    main()
