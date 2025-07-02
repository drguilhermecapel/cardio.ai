"""
Script principal para treinamento de modelos de ECG
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
from pathlib import Path

from backend.training.config.training_config import training_config
from backend.training.config.model_configs import get_model_config
from backend.training.config.dataset_configs import get_dataset_config
from backend.training.datasets.dataset_factory import DatasetFactory
from backend.training.models.model_factory import ModelFactory
from backend.training.trainers.classification_trainer import ClassificationTrainer
from backend.training.utils.data_utils import split_dataset
from backend.training.utils.model_utils import get_device, count_parameters

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main(args):
    logger.info("Iniciando o script de treinamento...")
    
    # 1. Configurações
    device = get_device()
    
    # Carregar configurações do dataset e modelo
    dataset_cfg = get_dataset_config(args.dataset)
    model_cfg = get_model_config(args.model)
    
    # Atualizar training_config com args
    training_config.BATCH_SIZE = args.batch_size
    training_config.EPOCHS = args.epochs
    training_config.LEARNING_RATE = args.lr
    training_config.MODEL_TYPE = args.model
    training_config.SAMPLING_RATE = dataset_cfg.sampling_rate # Usar sampling rate do dataset
    training_config.NUM_LEADS = dataset_cfg.num_leads # Usar num leads do dataset
    
    logger.info(f"Configurações de Treinamento: {training_config.dict()}")
    logger.info(f"Configurações do Modelo: {model_cfg}")
    logger.info(f"Configurações do Dataset: {dataset_cfg}")
    
    # 2. Carregar Dataset
    # O caminho base para os dados deve ser configurável ou inferido
    # Por enquanto, assumimos que os dados estão em training_config.DATA_ROOT / args.dataset
    dataset_path = training_config.DATA_ROOT / args.dataset
    
    # Crie o diretório de dados se não existir (para datasets que serão baixados)
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Instanciar o dataset completo para então dividir
    full_dataset = DatasetFactory.create_dataset(
        dataset_name=args.dataset,
        data_path=dataset_path,
        target_length=training_config.SIGNAL_LENGTH,
        sampling_rate=training_config.SAMPLING_RATE,
        num_leads=training_config.NUM_LEADS,
        normalize=True,
        augment=True,
        filter_noise=True,
        cache_data=False, # Pode ser True para datasets pequenos
        sample_limit=100 # Limita o número de amostras para testes rápidos
    )
    
    # Dividir o dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        train_ratio=training_config.VAL_SPLIT, # Renomeado para train_ratio
        val_ratio=training_config.TEST_SPLIT,  # Renomeado para val_ratio
        test_ratio=1.0 - training_config.VAL_SPLIT - training_config.TEST_SPLIT # Ajuste para test_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.BATCH_SIZE,
        shuffle=True,
        num_workers=training_config.NUM_WORKERS,
        pin_memory=training_config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.BATCH_SIZE,
        shuffle=False,
        num_workers=training_config.NUM_WORKERS,
        pin_memory=training_config.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.BATCH_SIZE,
        shuffle=False,
        num_workers=training_config.NUM_WORKERS,
        pin_memory=training_config.PIN_MEMORY
    )
    
    # 3. Inicializar Modelo
    model = ModelFactory.create_model(
        model_name=args.model,
        num_classes=len(full_dataset._get_label_map()), # Número de classes do dataset
        input_channels=training_config.NUM_LEADS,
        pretrained_path=args.pretrained_path
    )
    count_parameters(model)
    
    # 4. Otimizador e Função de Perda
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Treinador
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    
    # 6. Treinar
    trainer.train()
    
    # 7. Avaliar no conjunto de teste
    logger.info("Iniciando avaliação no conjunto de teste...")
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Métricas de Teste: {test_metrics}")
    
    logger.info("Treinamento concluído com sucesso!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Treinamento de Modelos ECG")
    parser.add_argument("--model", type=str, default="heartbeit",
                        help="Nome do modelo a ser treinado (ex: heartbeit, cnn_lstm)")
    parser.add_argument("--dataset", type=str, default="ptbxl",
                        help="Nome do dataset a ser usado (ex: ptbxl, mitbih)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Tamanho do batch para treinamento")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Número de épocas para treinamento")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate inicial")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Caminho para pesos pré-treinados (opcional)")
    
    args = parser.parse_args()
    main(args)


