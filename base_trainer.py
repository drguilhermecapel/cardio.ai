
"""
Classe base para treinadores de modelos de deep learning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

from ..config.training_config import training_config

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Classe base abstrata para treinadores de modelos de deep learning."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = torch.device("cpu"),
        **kwargs
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.epochs = training_config.EPOCHS
        self.early_stopping_patience = training_config.EARLY_STOPPING_PATIENCE
        self.gradient_clip_val = training_config.GRADIENT_CLIP_VAL
        self.log_every_n_steps = training_config.LOG_EVERY_N_STEPS
        self.save_top_k = training_config.SAVE_TOP_K
        
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        
        logger.info(f"Trainer inicializado. Dispositivo: {self.device}")
        
    @abstractmethod
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Treina o modelo por uma época."""
        pass
        
    @abstractmethod
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Valida o modelo por uma época."""
        pass
        
    def train(self):
        """Inicia o processo de treinamento."""
        logger.info("Iniciando treinamento...")
        for epoch in range(self.epochs):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            
            logger.info(f'Epoch {epoch+1}/{self.epochs} - ' f'Train Loss: {train_metrics["loss"]:.4f}, ' f'Val Loss: {val_metrics["loss"]:.4f}')
                       
            # Early stopping
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics["loss"], is_best=True)
            else:
                self.patience_counter += 1
                logger.info(f"Paciência: {self.patience_counter}/{self.early_stopping_patience}")
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping ativado.")
                    break
                    
        logger.info("Treinamento concluído.")
        
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Salva o estado do modelo."""
        checkpoint_dir = training_config.CHECKPOINT_ROOT
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = self.model.__class__.__name__.lower()
        
        checkpoint_path = checkpoint_dir / f"{model_name}_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"Checkpoint salvo em {checkpoint_path}")
        
        if is_best:
            best_model_path = checkpoint_dir / f"{model_name}_best.pth"
            torch.save(self.model.state_dict(), best_model_path)
            logger.info(f"Melhor modelo salvo em {best_model_path}")
            
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Avalia o modelo em um conjunto de teste."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                signals = batch["signal"].to(self.device)
                labels = batch["label"].squeeze().to(self.device) # Squeeze para remover dimensão extra
                
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(test_loader)
        logger.info(f"Avaliação - Loss: {avg_loss:.4f}")
        return {"loss": avg_loss}


