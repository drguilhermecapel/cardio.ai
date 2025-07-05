# backend/app/services/ptbxl_model_service_production.py
import os
import numpy as np
import tensorflow as tf
import logging
import json
from typing import Dict, List

from .interfaces import MLModelServiceInterface
from app.schemas.ecg import ECGDiagnosis, DiagnosisResult

logger = logging.getLogger(__name__)

class PTBXLModelServiceProduction(MLModelServiceInterface):
    """
    Serviço de modelo de ML para o dataset PTB-XL, versão de produção.
    Carrega o modelo treinado e os mapeamentos de classe para realizar
    e formatar as predições.
    """
    def __init__(self,
                 model_path: str = "models/ptbxl_model_prod.hdf5",
                 threshold: float = 0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        self.class_mapping = self._load_class_mapping()
        self._load_model()

    def _load_model(self):
        """Carrega o modelo Keras a partir do caminho especificado."""
        model_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), self.model_path)
        if not os.path.exists(model_file):
            logger.error(f"Arquivo do modelo não encontrado em: {model_file}")
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_file}")
        
        try:
            self.model = tf.keras.models.load_model(model_file)
            logger.info(f"Modelo de produção carregado com sucesso de: {model_file}")
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo Keras: {e}", exc_info=True)
            raise

    def _load_class_mapping(self) -> Dict[str, str]:
        """Carrega o mapeamento de classes do arquivo JSON."""
        mapping_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models/ptbxl_classes.json")
        try:
            with open(mapping_path, 'r') as f:
                mapping = json.load(f)
            logger.info("Mapeamento de classes PTB-XL carregado.")
            return mapping
        except Exception as e:
            logger.error(f"Não foi possível carregar o mapeamento de classes: {e}")
            return {}

    async def predict(self, processed_signal: np.ndarray) -> np.ndarray:
        """
        Realiza a predição no sinal pré-processado.
        O modelo espera uma entrada com shape (batch_size, timesteps, features).
        """
        if self.model is None:
            raise RuntimeError("Modelo não foi carregado. Chame `_load_model()` primeiro.")

        # Garante que o input tenha o formato (1, 1000, 12)
        if processed_signal.ndim != 3 or processed_signal.shape[1:] != (1000, 12):
             logger.error(f"Shape do sinal de entrada ({processed_signal.shape}) é incompatível com o esperado pelo modelo (None, 1000, 12).")
             raise ValueError("Shape do sinal de entrada incompatível com o modelo.")

        try:
            prediction = self.model.predict(processed_signal)
            return prediction
        except Exception as e:
            logger.error(f"Erro durante a predição do modelo: {e}")
            raise

    def format_prediction(self, prediction: np.ndarray) -> ECGDiagnosis:
        """Formata a saída bruta do modelo em um objeto de diagnóstico estruturado."""
        if not self.class_mapping:
            raise RuntimeError("Mapeamento de classes não carregado.")
            
        results: List[DiagnosisResult] = []
        is_abnormal = False
        
        prediction_squeezed = prediction.squeeze()

        for i, class_name in self.class_mapping.items():
            prob = prediction_squeezed[int(i)]
            is_present = bool(prob >= self.threshold)
            
            if is_present and class_name != "Normal ECG":
                is_abnormal = True

            results.append(DiagnosisResult(
                condition=class_name,
                probability=float(prob),
                is_present=is_present
            ))
            
        # Ordena por probabilidade descendente
        results.sort(key=lambda r: r.probability, reverse=True)

        # Determina o status geral
        overall_finding = "Ritmo Sinusal Normal"
        if is_abnormal:
            present_conditions = [r.condition for r in results if r.is_present and r.condition != "Normal ECG"]
            overall_finding = f"Anormalidade Detectada: {', '.join(present_conditions)}"

        return ECGDiagnosis(
            results=results,
            is_abnormal=is_abnormal,
            model_version=f"PTB-XL Production v1.0 - Threshold: {self.threshold}",
            overall_finding=overall_finding,
            heart_rate=0,  # Será preenchido pelo serviço unificado
            qrs_duration="0.0 ms" # Será preenchido pelo serviço unificado
        )

