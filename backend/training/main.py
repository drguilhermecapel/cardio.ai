"""
Script principal para treinamento de modelos de ECG
Renomeie este arquivo de main_.py para main.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
from pathlib import Path
import sys
import os

# Adicionar o diretório backend ao path para permitir importações
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Importações do módulo training
from training.config.training_config import training_config
from training.config.model_configs import get_model_config
from training.config.dataset_configs import get_dataset_config
from training.datasets.dataset_factory import DatasetFactory
from training.models.model_factory import ModelFactory
from training.trainers.classification_trainer import ClassificationTrainer
from training.utils.data_utils import split_dataset
from training.utils.model_utils import get_device, count_parameters

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def verify_dependencies():
    """Verifica se todas as dependências necessárias estão instaladas"""
    missing_deps = []
    
    try:
        import wfdb
    except ImportError:
        missing_deps.append("wfdb")
        
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
        
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy")
        
    if missing_deps:
        logger.error(f"Dependências faltando: {', '.join(missing_deps)}")
        logger.error("Execute: pip install -r backend/training/requirements.txt")
        return False
        
    return True


def setup_directories():
    """Cria diretórios necessários"""
    dirs = [
        training_config.DATA_ROOT,
        training_config.CHECKPOINT_ROOT,
        training_config.LOG_ROOT,
        training_config.EXPORT_ROOT
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Diretório criado/verificado: {dir_path}")


def main(args):
    logger.info("=" * 80)
    logger.info("INICIANDO TREINAMENTO DE MODELO ECG - CARDIOAI PRO")
    logger.info("=" * 80)
    
    # Verificar dependências
    if not verify_dependencies():
        return
        
    # Criar diretórios
    setup_directories()
    
    # 1. Configurações
    device = get_device()
    logger.info(f"Dispositivo de treinamento: {device}")
    
    # Carregar configurações do dataset e modelo
    try:
        dataset_cfg = get_dataset_config(args.dataset)
        model_cfg = get_model_config(args.model)
    except ValueError as e:
        logger.error(f"Erro ao carregar configurações: {e}")
        return
    
    # Atualizar training_config com argumentos da linha de comando
    training_config.BATCH_SIZE = args.batch_size
    training_config.EPOCHS = args.epochs
    training_config.LEARNING_RATE = args.lr
    training_config.MODEL_TYPE = args.model
    training_config.SAMPLING_RATE = dataset_cfg.sampling_rate
    training_config.NUM_LEADS = dataset_cfg.num_leads
    
    logger.info(f"Configurações de Treinamento:")
    logger.info(f"  - Modelo: {args.model}")
    logger.info(f"  - Dataset: {args.dataset}")
    logger.info(f"  - Batch Size: {args.batch_size}")
    logger.info(f"  - Épocas: {args.epochs}")
    logger.info(f"  - Learning Rate: {args.lr}")
    logger.info(f"  - Taxa de Amostragem: {dataset_cfg.sampling_rate} Hz")
    logger.info(f"  - Número de Derivações: {dataset_cfg.num_leads}")
    
    # 2. Carregar Dataset
    dataset_path = training_config.DATA_ROOT / args.dataset
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nCarregando dataset de: {dataset_path}")
    
    try:
        # Criar dataset completo
        full_dataset = DatasetFactory.create_dataset(
            dataset_name=args.dataset,
            data_path=dataset_path,
            target_length=training_config.SIGNAL_LENGTH,
            sampling_rate=training_config.SAMPLING_RATE,
            num_leads=training_config.NUM_LEADS,
            normalize=True,
            augment=True,
            filter_noise=True,
            cache_data=False,
            sample_limit=args.sample_limit  # Para testes rápidos
        )
        
        logger.info(f"Dataset carregado com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro ao carregar dataset: {e}")
        logger.error("Certifique-se de que o dataset foi baixado corretamente.")
        return
    
    # Dividir o dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        train_ratio=0.7,  # 70% treino
        val_ratio=0.15,   # 15% validação
        test_ratio=0.15   # 15% teste
    )
    
    logger.info(f"\nDivisão do dataset:")
    logger.info(f"  - Treino: {len(train_dataset)} amostras")
    logger.info(f"  - Validação: {len(val_dataset)} amostras")
    logger.info(f"  - Teste: {len(test_dataset)} amostras")
    
    # Criar DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.BATCH_SIZE,
        shuffle=True,
        num_workers=training_config.NUM_WORKERS if device.type == 'cuda' else 0,
        pin_memory=training_config.PIN_MEMORY and device.type == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.BATCH_SIZE,
        shuffle=False,
        num_workers=training_config.NUM_WORKERS if device.type == 'cuda' else 0,
        pin_memory=training_config.PIN_MEMORY and device.type == 'cuda'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.BATCH_SIZE,
        shuffle=False,
        num_workers=training_config.NUM_WORKERS if device.type == 'cuda' else 0,
        pin_memory=training_config.PIN_MEMORY and device.type == 'cuda'
    )
    
    # 3. Inicializar Modelo
    logger.info(f"\nInicializando modelo {args.model}...")
    
    # Obter número de classes do dataset
    num_classes = len(dataset_cfg.classes)
    
    try:
        model = ModelFactory.create_model(
            model_name=args.model,
            num_classes=num_classes,
            input_channels=training_config.NUM_LEADS,
            pretrained_path=args.pretrained_path
        )
        model = model.to(device)
        
        param_count = count_parameters(model)
        logger.info(f"Modelo criado com {param_count:,} parâmetros")
        
    except Exception as e:
        logger.error(f"Erro ao criar modelo: {e}")
        return
    
    # 4. Otimizador e Função de Perda
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_config.LEARNING_RATE,
        weight_decay=1e-5
    )
    
    # Scheduler para reduzir LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )    
    # Função de perda com pesos para classes desbalanceadas
    if args.weighted_loss:
        # Calcular pesos das classes baseado na frequência
        class_counts = torch.bincount(torch.tensor(train_dataset.labels))
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        logger.info("Usando perda ponderada para classes desbalanceadas")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 5. Criar Treinador
    logger.info("\nInicializando treinador...")
    
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=training_config,
        scheduler=scheduler
    )
    
    # 6. Treinar
    logger.info("\n" + "=" * 80)
    logger.info("INICIANDO TREINAMENTO")
    logger.info("=" * 80)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\nTreinamento interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro durante treinamento: {e}")
        return
    
    # 7. Avaliar no conjunto de teste
    logger.info("\n" + "=" * 80)
    logger.info("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
    logger.info("=" * 80)
    
    test_metrics = trainer.evaluate(test_loader)
    
    logger.info("\nMétricas de Teste:")
    for metric, value in test_metrics.items():
        logger.info(f"  - {metric}: {value:.4f}")
    
    # 8. Salvar modelo final
    if args.save_model:
        model_path = training_config.EXPORT_ROOT / f"{args.model}_{args.dataset}_final.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'model_name': args.model,
                'num_classes': num_classes,
                'input_channels': training_config.NUM_LEADS,
                'dataset': args.dataset
            },
            'test_metrics': test_metrics
        }, model_path)
        logger.info(f"\nModelo salvo em: {model_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Treinamento de Modelos ECG - CardioAI Pro")
    
    # Argumentos principais
    parser.add_argument("--model", type=str, default="cnn_lstm",
                        choices=["heartbeit", "cnn_lstm", "se_resnet1d", "ecg_transformer"],
                        help="Arquitetura do modelo")
    parser.add_argument("--dataset", type=str, default="ptbxl",
                        choices=["ptbxl", "mitbih", "cpsc2018"],
                        help="Dataset a ser usado")
    
    # Hiperparâmetros
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Tamanho do batch")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Número de épocas")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate inicial")
    
    # Opções adicionais
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Caminho para pesos pré-treinados")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Limite de amostras para teste rápido")
    parser.add_argument("--weighted_loss", action="store_true",
                        help="Usar perda ponderada para classes desbalanceadas")
    parser.add_argument("--save_model", action="store_true", default=True,
                        help="Salvar modelo após treinamento")
    
    args = parser.parse_args()
    
    # Executar treinamento
    main(args)

