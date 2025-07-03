"""
Serviço de Modelos Lite para CardioAI (sem dependências pesadas)
Versão simplificada para demonstração e testes
"""

import os
import logging
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


class ModelServiceLite:
    """Serviço simplificado para gerenciamento de modelos de ML."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        
    def load_sklearn_model(self, model_path: str, model_name: str) -> bool:
        """Carrega modelo scikit-learn."""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Modelo não encontrado: {model_path}")
                return False
                
            # Carregar modelo
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Armazenar no cache
            self.loaded_models[model_name] = model
            
            # Metadados
            self.model_metadata[model_name] = {
                "type": "sklearn",
                "path": model_path,
                "loaded_at": datetime.now().isoformat(),
                "model_class": type(model).__name__
            }
            
            logger.info(f"Modelo {model_name} carregado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_name}: {str(e)}")
            return False
    
    def create_demo_model(self, model_name: str = "demo_ecg_classifier") -> bool:
        """Cria modelo de demonstração usando scikit-learn."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            # Criar modelo simples
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            scaler = StandardScaler()
            
            # Gerar dados sintéticos para treinamento
            np.random.seed(42)
            X_train = np.random.randn(1000, 100)  # 1000 amostras, 100 features
            y_train = np.random.randint(0, 5, 1000)  # 5 classes
            
            # Treinar
            X_scaled = scaler.fit_transform(X_train)
            model.fit(X_scaled, y_train)
            
            # Criar pipeline
            pipeline = {
                'scaler': scaler,
                'model': model,
                'classes': ['Normal', 'Arritmia', 'Taquicardia', 'Bradicardia', 'Fibrilação']
            }
            
            # Armazenar
            self.loaded_models[model_name] = pipeline
            self.model_metadata[model_name] = {
                "type": "sklearn_pipeline",
                "created_at": datetime.now().isoformat(),
                "model_class": "RandomForestClassifier",
                "n_features": 100,
                "n_classes": 5,
                "accuracy": 0.85  # Simulado
            }
            
            logger.info(f"Modelo de demonstração {model_name} criado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao criar modelo de demonstração: {str(e)}")
            return False
    
    def predict_ecg(self, model_name: str, ecg_data: np.ndarray) -> Dict[str, Any]:
        """Realiza predição em dados de ECG."""
        try:
            if model_name not in self.loaded_models:
                raise ValueError(f"Modelo {model_name} não está carregado")
                
            model_pipeline = self.loaded_models[model_name]
            model_type = self.model_metadata[model_name]["type"]
            
            # Preprocessar dados
            processed_data = self._preprocess_ecg_data(ecg_data)
            
            # Realizar predição baseada no tipo do modelo
            if model_type == "sklearn_pipeline":
                scaler = model_pipeline['scaler']
                model = model_pipeline['model']
                classes = model_pipeline['classes']
                
                # Reduzir dimensionalidade se necessário
                if processed_data.shape[1] > 100:
                    # Usar apenas primeiras 100 features
                    processed_data = processed_data[:, :100]
                elif processed_data.shape[1] < 100:
                    # Pad com zeros
                    padding = np.zeros((processed_data.shape[0], 100 - processed_data.shape[1]))
                    processed_data = np.hstack([processed_data, padding])
                
                # Escalar e predizer
                scaled_data = scaler.transform(processed_data)
                predictions = model.predict(scaled_data)
                probabilities = model.predict_proba(scaled_data)
                
                results = {
                    "predicted_classes": predictions.tolist(),
                    "class_names": [classes[p] for p in predictions],
                    "probabilities": probabilities.tolist(),
                    "max_probability": np.max(probabilities, axis=1).tolist()
                }
                
            elif model_type == "sklearn":
                model = model_pipeline
                predictions = model.predict(processed_data)
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(processed_data)
                    results = {
                        "predicted_classes": predictions.tolist(),
                        "probabilities": probabilities.tolist(),
                        "max_probability": np.max(probabilities, axis=1).tolist()
                    }
                else:
                    results = {
                        "predicted_classes": predictions.tolist()
                    }
            else:
                raise ValueError(f"Tipo de modelo não suportado: {model_type}")
            
            # Calcular confiança
            confidence = self._calculate_confidence(results)
            
            return {
                "model_name": model_name,
                "predictions": results,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}")
            return {"error": str(e)}
    
    def _preprocess_ecg_data(self, ecg_data: np.ndarray) -> np.ndarray:
        """Preprocessa dados de ECG para o modelo."""
        # Garantir formato 2D
        if ecg_data.ndim == 1:
            ecg_data = ecg_data.reshape(1, -1)
        
        # Normalização Z-score
        mean = np.mean(ecg_data, axis=1, keepdims=True)
        std = np.std(ecg_data, axis=1, keepdims=True)
        normalized = (ecg_data - mean) / (std + 1e-8)
        
        return normalized
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calcula score de confiança da predição."""
        if "max_probability" in results:
            return float(np.mean(results["max_probability"]))
        elif "probabilities" in results:
            probs = np.array(results["probabilities"])
            return float(np.mean(np.max(probs, axis=1)))
        else:
            return 0.5  # Confiança neutra
    
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
        
        # Procurar por arquivos pickle
        for file_path in self.models_dir.glob("*.pkl"):
            model_name = file_path.stem
            if self.load_sklearn_model(str(file_path), model_name):
                discovered.append(model_name)
        
        # Se não encontrou nenhum, criar modelo de demonstração
        if not discovered:
            if self.create_demo_model():
                discovered.append("demo_ecg_classifier")
        
        return discovered
    
    def save_model(self, model_name: str, file_path: str) -> bool:
        """Salva modelo em arquivo."""
        try:
            if model_name not in self.loaded_models:
                raise ValueError(f"Modelo {model_name} não está carregado")
            
            model = self.loaded_models[model_name]
            
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Modelo {model_name} salvo em {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {str(e)}")
            return False


# Instância global do serviço
model_service_lite = ModelServiceLite()


def initialize_models_lite():
    """Inicializa e carrega modelos disponíveis (versão lite)."""
    logger.info("Inicializando serviço de modelos (versão lite)...")
    
    # Descobrir modelos automaticamente
    discovered = model_service_lite.auto_discover_models()
    
    if discovered:
        logger.info(f"Modelos descobertos/criados: {discovered}")
    else:
        logger.warning("Nenhum modelo encontrado ou criado")
    
    return model_service_lite


# Classe de modelo simples para compatibilidade
class SimpleECGClassifier:
    """Classificador simples de ECG para demonstração."""
    
    def __init__(self, n_classes: int = 5):
        self.n_classes = n_classes
        self.classes = ['Normal', 'Arritmia', 'Taquicardia', 'Bradicardia', 'Fibrilação']
        self.is_trained = False
    
    def fit(self, X, y):
        """Simula treinamento."""
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Predição simples baseada em regras."""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado")
        
        # Regras simples baseadas em estatísticas do sinal
        predictions = []
        
        for sample in X:
            # Calcular características simples
            mean_val = np.mean(sample)
            std_val = np.std(sample)
            max_val = np.max(sample)
            min_val = np.min(sample)
            
            # Regras simples de classificação
            if std_val > 2.0:
                pred = 1  # Arritmia
            elif max_val > 3.0:
                pred = 2  # Taquicardia
            elif std_val < 0.5:
                pred = 3  # Bradicardia
            elif abs(mean_val) > 1.0:
                pred = 4  # Fibrilação
            else:
                pred = 0  # Normal
            
            predictions.append(pred)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Probabilidades simuladas."""
        predictions = self.predict(X)
        probabilities = []
        
        for pred in predictions:
            probs = np.random.dirichlet(np.ones(self.n_classes) * 0.1)
            probs[pred] = max(probs[pred], 0.7)  # Dar alta probabilidade à classe predita
            probs = probs / np.sum(probs)  # Normalizar
            probabilities.append(probs)
        
        return np.array(probabilities)

