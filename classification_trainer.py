
"""
Treinador para tarefas de classificação de ECG
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
import logging

from .base_trainer import BaseTrainer
from ..config.training_config import training_config

logger = logging.getLogger(__name__)


class ClassificationTrainer(BaseTrainer):
    """Treinador específico para tarefas de classificação de ECG."""
    
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
        super().__init__(model, train_loader, val_loader, optimizer, criterion, device, **kwargs)
        
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Treina o modelo por uma época para classificação."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            signals = batch["signal"].to(self.device)
            labels = batch["label"].squeeze().to(self.device) # Squeeze para remover dimensão extra
            
            self.optimizer.zero_grad()
            outputs = self.model(signals)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % self.log_every_n_steps == 0:
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_loader)} | "
                           f"Train Loss: {loss.item():.4f}")
                           
        avg_loss = total_loss / len(self.train_loader)
        return {"loss": avg_loss}
        
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Valida o modelo por uma época para classificação."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                signals = batch["signal"].to(self.device)
                labels = batch["label"].squeeze().to(self.device)
                
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct_predictions / total_samples
        
        logger.info(f"Epoch {epoch+1} | Validation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        return {"loss": avg_loss, "accuracy": accuracy}


