"""
Serviço de Integração de Modelos .h5 para CardioAI
Implementa carregamento, cache e inferência de modelos treinados
"""

import os
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


class ModelService:
    """Serviço para gerenciamento de modelos de ML."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        
    def load_h5_model(self, model_path: str, model_name: str) -> bool:
        """Carrega modelo .h5 do TensorFlow/Keras."""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Modelo não encontrado: {model_path}")
                return False
                
            # Carregar modelo
            model = keras.models.load_model(model_path)
            
            # Armazenar no cache
            self.loaded_models[model_name] = model
            
            # Metadados
            self.model_metadata[model_name] = {
                "type": "keras",
                "path": model_path,
                "loaded_at": datetime.now().isoformat(),
                "input_shape": model.input_shape,
                "output_shape": model.output_shape,
                "parameters": model.count_params()
            }
            
            logger.info(f"Modelo {model_name} carregado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_name}: {str(e)}")
            return False
    
    def load_pytorch_model(self, model_path: str, model_class: nn.Module, model_name: str) -> bool:
        """Carrega modelo PyTorch."""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Modelo não encontrado: {model_path}")
                return False
                
            # Carregar estado do modelo
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Criar instância do modelo
            model = model_class()
            model.load_state_dict(state_dict)
            model.eval()
            
            # Armazenar no cache
            self.loaded_models[model_name] = model
            
            # Metadados
            self.model_metadata[model_name] = {
                "type": "pytorch",
                "path": model_path,
                "loaded_at": datetime.now().isoformat(),
                "parameters": sum(p.numel() for p in model.parameters())
            }
            
            logger.info(f"Modelo PyTorch {model_name} carregado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo PyTorch {model_name}: {str(e)}")
            return False
    
    def predict_ecg(self, model_name: str, ecg_data: np.ndarray) -> Dict[str, Any]:
        """Realiza predição em dados de ECG."""
        try:
            if model_name not in self.loaded_models:
                raise ValueError(f"Modelo {model_name} não está carregado")
                
            model = self.loaded_models[model_name]
            model_type = self.model_metadata[model_name]["type"]
            
            # Preprocessar dados se necessário
            processed_data = self._preprocess_ecg_data(ecg_data)
            
            # Realizar predição baseada no tipo do modelo
            if model_type == "keras":
                predictions = model.predict(processed_data)
            elif model_type == "pytorch":
                with torch.no_grad():
                    tensor_data = torch.FloatTensor(processed_data)
                    predictions = model(tensor_data).numpy()
            else:
                raise ValueError(f"Tipo de modelo não suportado: {model_type}")
            
            # Processar resultados
            results = self._process_predictions(predictions)
            
            return {
                "model_name": model_name,
                "predictions": results,
                "confidence": self._calculate_confidence(predictions),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}")
            return {"error": str(e)}
    
    def _preprocess_ecg_data(self, ecg_data: np.ndarray) -> np.ndarray:
        """Preprocessa dados de ECG para o modelo."""
        # Normalização básica
        if ecg_data.ndim == 1:
            ecg_data = ecg_data.reshape(1, -1)
        
        # Normalização Z-score
        mean = np.mean(ecg_data, axis=1, keepdims=True)
        std = np.std(ecg_data, axis=1, keepdims=True)
        normalized = (ecg_data - mean) / (std + 1e-8)
        
        return normalized
    
    def _process_predictions(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Processa predições do modelo."""
        # Assumindo classificação multi-classe
        if predictions.ndim == 2:
            # Softmax para probabilidades
            probabilities = tf.nn.softmax(predictions).numpy()
            predicted_class = np.argmax(probabilities, axis=1)
            max_probability = np.max(probabilities, axis=1)
            
            return {
                "predicted_classes": predicted_class.tolist(),
                "probabilities": probabilities.tolist(),
                "max_probability": max_probability.tolist()
            }
        else:
            return {"raw_predictions": predictions.tolist()}
    
    def _calculate_confidence(self, predictions: np.ndarray) -> float:
        """Calcula score de confiança da predição."""
        if predictions.ndim == 2:
            # Para classificação, usar entropia
            probabilities = tf.nn.softmax(predictions).numpy()
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
            max_entropy = np.log(probabilities.shape[1])
            confidence = 1 - (entropy / max_entropy)
            return float(np.mean(confidence))
        else:
            # Para regressão, usar variância
            return float(1.0 / (1.0 + np.var(predictions)))
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Retorna informações sobre um modelo."""
        if model_name not in self.model_metadata:
            return {"error": f"Modelo {model_name} não encontrado"}
        
        return self.model_metadata[model_name]
    
    def list_models(self) -> List[str]:
        """Lista todos os modelos carregados."""
        return list(self.loaded_models.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """Remove modelo da memória."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            del self.model_metadata[model_name]
            logger.info(f"Modelo {model_name} removido da memória")
            return True
        return False
    
    def auto_discover_models(self) -> List[str]:
        """Descobre automaticamente modelos na pasta de modelos."""
        discovered = []
        
        for file_path in self.models_dir.glob("*.h5"):
            model_name = file_path.stem
            if self.load_h5_model(str(file_path), model_name):
                discovered.append(model_name)
        
        for file_path in self.models_dir.glob("*.pth"):
            # Para PyTorch, precisaria da classe do modelo
            logger.info(f"Modelo PyTorch encontrado: {file_path}")
        
        return discovered


# Instância global do serviço
model_service = ModelService()


class ECGClassificationModel(nn.Module):
    """Modelo PyTorch exemplo para classificação de ECG."""
    
    def __init__(self, input_size: int = 1000, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Adicionar dimensão de canal
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def initialize_models():
    """Inicializa e carrega modelos disponíveis."""
    logger.info("Inicializando serviço de modelos...")
    
    # Descobrir modelos automaticamente
    discovered = model_service.auto_discover_models()
    
    if discovered:
        logger.info(f"Modelos descobertos: {discovered}")
    else:
        logger.warning("Nenhum modelo .h5 encontrado na pasta models/")
    
    return model_service

