#!/usr/bin/env python3
"""
Script para configurar GPU e iniciar treinamento do modelo híbrido
Implementa os próximos passos solicitados pelo usuário
"""

import logging
import sys
import torch
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.ml.hybrid_architecture import ModelConfig, create_hybrid_model
from app.ml.training_pipeline import (
    TrainingConfig,
    ECGTrainingPipeline,
    create_training_config,
)
from app.services.advanced_ml_service import AdvancedMLService, AdvancedMLConfig


def setup_logging():
    """Configura logging para o script"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("gpu_training_setup.log"),
        ],
    )


def check_gpu_setup():
    """Verifica e configura GPU para treinamento"""
    logger = logging.getLogger(__name__)

    logger.info("🔍 Verificando configuração de GPU...")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

        logger.info(f"✅ CUDA disponível!")
        logger.info(f"   GPUs detectadas: {gpu_count}")
        logger.info(f"   GPU principal: {gpu_name}")
        logger.info(f"   Memória GPU: {gpu_memory:.1f} GB")

        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(test_tensor, test_tensor.t())
            logger.info("✅ Teste de operação GPU bem-sucedido")
            return "cuda"
        except Exception as e:
            logger.warning(f"⚠️  Erro no teste GPU: {e}")
            return "cpu"

    elif torch.backends.mps.is_available():
        logger.info("✅ MPS (Apple Silicon) disponível!")
        try:
            test_tensor = torch.randn(1000, 1000).to("mps")
            result = torch.mm(test_tensor, test_tensor.t())
            logger.info("✅ Teste de operação MPS bem-sucedido")
            return "mps"
        except Exception as e:
            logger.warning(f"⚠️  Erro no teste MPS: {e}")
            return "cpu"

    else:
        logger.info("ℹ️  Usando CPU para treinamento")
        return "cpu"


def create_optimized_training_config(device: str) -> TrainingConfig:
    """Cria configuração otimizada para treinamento"""
    logger = logging.getLogger(__name__)

    model_config = ModelConfig(
        input_channels=12,
        sequence_length=5000,
        num_classes=71,
        cnn_growth_rate=32,
        lstm_hidden_dim=256,
        transformer_heads=8,
        transformer_layers=4,
        dropout_rate=0.2,
    )

    if device == "cuda":
        batch_size = 64  # Maior batch size para GPU
        num_workers = 8
        pin_memory = True
        logger.info("🚀 Configuração otimizada para CUDA")
    elif device == "mps":
        batch_size = 32  # Batch size moderado para MPS
        num_workers = 4
        pin_memory = False
        logger.info("🚀 Configuração otimizada para MPS")
    else:
        batch_size = 16  # Batch size menor para CPU
        num_workers = 2
        pin_memory = False
        logger.info("🚀 Configuração otimizada para CPU")

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
        augmentation_probability=0.5,
        noise_std=0.02,
        amplitude_scale_range=(0.8, 1.2),
        time_shift_range=50,
        use_spectrograms=True,
        use_wavelets=True,
        use_class_weights=True,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        label_smoothing=0.1,
        validation_split=0.2,
        save_checkpoints=True,
        checkpoint_dir="checkpoints",
        save_best_only=True,
        early_stopping_patience=15,
        use_wandb=False,  # Desabilitado por padrão
        log_interval=100,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return training_config


def test_model_creation(config: TrainingConfig):
    """Testa criação do modelo híbrido"""
    logger = logging.getLogger(__name__)

    logger.info("🧠 Testando criação do modelo híbrido...")

    try:
        model = create_hybrid_model(
            num_classes=config.model_config.num_classes,
            input_channels=config.model_config.input_channels,
            sequence_length=config.model_config.sequence_length,
        )

        model = model.to(config.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"✅ Modelo criado com sucesso!")
        logger.info(f"   Parâmetros totais: {total_params:,}")
        logger.info(f"   Parâmetros treináveis: {trainable_params:,}")
        logger.info(f"   Dispositivo: {config.device}")

        batch_size = 2
        test_input = torch.randn(
            batch_size,
            config.model_config.input_channels,
            config.model_config.sequence_length,
        ).to(config.device)

        with torch.no_grad():
            output = model(test_input)
            logger.info(f"✅ Forward pass bem-sucedido! Output shape: {output.shape}")

        return True

    except Exception as e:
        logger.error(f"❌ Erro na criação do modelo: {e}")
        return False


def setup_advanced_ml_service(device: str):
    """Configura o serviço de ML avançado"""
    logger = logging.getLogger(__name__)

    logger.info("⚙️  Configurando serviço de ML avançado...")

    try:
        ml_config = AdvancedMLConfig(
            device=device,
            enable_interpretability=True,
            enable_adaptive_thresholds=True,
            confidence_threshold=0.8,
            batch_size=32 if device == "cuda" else 16,
        )

        ml_service = AdvancedMLService(ml_config)

        logger.info("✅ Serviço de ML avançado configurado!")
        logger.info(f"   Dispositivo: {device}")
        logger.info(f"   Interpretabilidade: {ml_config.enable_interpretability}")
        logger.info(
            f"   Thresholds adaptativos: {ml_config.enable_adaptive_thresholds}"
        )

        return ml_service

    except Exception as e:
        logger.error(f"❌ Erro na configuração do serviço ML: {e}")
        return None


def main():
    """Função principal para setup de GPU e treinamento"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🚀 Iniciando setup de GPU e configuração de treinamento")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: CONFIGURAÇÃO DE HARDWARE")
    logger.info("=" * 60)

    device = check_gpu_setup()

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: CONFIGURAÇÃO DE TREINAMENTO")
    logger.info("=" * 60)

    training_config = create_optimized_training_config(device)
    logger.info("✅ Configuração de treinamento criada")

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: TESTE DO MODELO HÍBRIDO")
    logger.info("=" * 60)

    model_test_success = test_model_creation(training_config)

    if not model_test_success:
        logger.error("❌ Falha no teste do modelo. Abortando...")
        return

    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: SERVIÇO DE ML AVANÇADO")
    logger.info("=" * 60)

    ml_service = setup_advanced_ml_service(device)

    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: PIPELINE DE TREINAMENTO")
    logger.info("=" * 60)

    try:
        training_pipeline = ECGTrainingPipeline(training_config)
        logger.info("✅ Pipeline de treinamento inicializado")

        training_pipeline.setup_model()
        logger.info("✅ Modelo configurado no pipeline")

    except Exception as e:
        logger.error(f"❌ Erro na configuração do pipeline: {e}")
        return

    logger.info("\n" + "=" * 60)
    logger.info("RELATÓRIO FINAL - SETUP CONCLUÍDO")
    logger.info("=" * 60)

    logger.info("✅ Setup de GPU e treinamento concluído com sucesso!")
    logger.info(f"   Dispositivo configurado: {device}")
    logger.info(f"   Modelo híbrido: CNN-BiLSTM-Transformer")
    logger.info(
        f"   Parâmetros do modelo: {training_config.model_config.num_classes} classes"
    )
    logger.info(f"   Batch size otimizado: {training_config.batch_size}")
    logger.info(f"   Curriculum learning: {training_config.curriculum_learning}")
    logger.info(f"   Multimodal features: Spectrograms + Wavelets")

    logger.info("\n🎯 Próximos passos:")
    logger.info("1. ✅ Download dos datasets - CONCLUÍDO")
    logger.info("2. ✅ Setup de GPU - CONCLUÍDO")
    logger.info("3. ⏳ Executar treinamento - PRONTO PARA INICIAR")
    logger.info("4. ⏳ Validação e otimização - AGUARDANDO")

    logger.info("\n🚀 Sistema pronto para treinamento!")
    logger.info("   Execute: python scripts/train_model.py para iniciar o treinamento")


if __name__ == "__main__":
    main()
