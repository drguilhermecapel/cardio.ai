import numpy as np
import tensorflow as tf
from typing import List, Dict, Any
import os
from logging import getLogger

from app.services.interfaces import ECGModelService, PreprocessingService, ExplanationService
from app.services.model_loader_robust import ModelLoaderRobust
from app.preprocessing.advanced_pipeline import AdvancedPipeline
from app.utils.clinical_explanations import ClinicalExplanationService
from app.core.exceptions import ModelNotLoadedException, InvalidInputException

logger = getLogger(__name__)

class PTBXLModelServiceBiasCorrected(ECGModelService):
    """
    Uma versão aprimorada do serviço de modelo PTB-XL que implementa uma lógica de correção de viés
    para lidar com diagnósticos de "Ritmo Sinusal" e outras anormalidades de forma mais precisa.
    """
    def __init__(self, model_path: str = "models/ptbxl_model.h5", threshold: float = 0.2):
        self.model_loader = ModelLoaderRobust(model_path)
        self.model, self.classes = self.model_loader.load_model_and_classes()
        self.threshold = threshold
        self.preprocessor: PreprocessingService = AdvancedPipeline()
        self.explanation_service: ExplanationService = ClinicalExplanationService()
        logger.info(f"Serviço de Modelo PTB-XL com Correção de Viés inicializado com limiar de {self.threshold}")

    def predict(self, ecg_data: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None or self.classes is None:
            raise ModelNotLoadedException()
        if not isinstance(ecg_data, np.ndarray) or ecg_data.ndim != 2:
            raise InvalidInputException("Os dados do ECG devem ser um array 2D do NumPy.")

        try:
            processed_data = self.preprocessor.preprocess(ecg_data)
            processed_data = np.expand_dims(processed_data, axis=0)
            
            probabilities = self.model.predict(processed_data)[0]
            
            diagnoses = self._map_probabilities_to_diagnoses_corrected(probabilities)
            
            explained_diagnoses = []
            for diag in diagnoses:
                diag["explanation"] = self.explanation_service.get_explanation(diag["label"])
                explained_diagnoses.append(diag)

            return explained_diagnoses
        except Exception as e:
            logger.error(f"Ocorreu um erro durante a predição: {e}", exc_info=True)
            raise

    def _map_probabilities_to_diagnoses_corrected(self, probabilities: np.ndarray) -> List[Dict[str, Any]]:
        diagnoses = []
        
        if 'NORM' not in self.classes:
            logger.warning("A classe 'NORM' não foi encontrada nas classes do modelo. A lógica de fallback será utilizada.")
            # Lógica de fallback se NORM não existir
            for i, prob in enumerate(probabilities):
                if prob >= self.threshold:
                    condition = self.classes[i]
                    diagnoses.append({"label": condition, "probability": float(prob)})
            if not diagnoses:
                diagnoses.append({"label": "ritmo_sinusal", "probability": 1.0})
            return sorted(diagnoses, key=lambda x: x['probability'], reverse=True)

        norm_prob_index = self.classes.index('NORM')
        norm_prob = probabilities[norm_prob_index]

        # Limiar de alta confiança para considerar o ECG como "Normal"
        # Este valor pode ser ajustado com base na validação clínica
        NORMAL_CONFIDENCE_THRESHOLD = 0.8 

        if norm_prob >= NORMAL_CONFIDENCE_THRESHOLD:
            # Se a probabilidade de ser normal for muito alta, retorne apenas "Ritmo Sinusal"
            # Isso evita que achados de baixa probabilidade poluam o laudo.
            return [{
                "label": "ritmo_sinusal", 
                "probability": float(norm_prob)
            }]

        # Coleta todos os diagnósticos que ultrapassam o limiar
        for i, prob in enumerate(probabilities):
            if prob >= self.threshold:
                condition = self.classes[i]
                # Não adiciona 'NORM' aqui, pois será tratado separadamente
                if condition == 'NORM':
                    continue
                diagnoses.append({
                    "label": condition,
                    "probability": float(prob)
                })

        # Se nenhuma anormalidade foi encontrada e a probabilidade de ser normal é razoável,
        # retorna Ritmo Sinusal.
        if not diagnoses and norm_prob >= self.threshold:
            return [{
                "label": "ritmo_sinusal",
                "probability": float(norm_prob)
            }]
        elif not diagnoses:
            # Caso nenhuma anormalidade e nem o ritmo normal atinjam o limiar,
            # pode indicar um traçado de baixa qualidade ou inconclusivo.
            # Retornar o diagnóstico mais provável, mesmo que abaixo do limiar,
            # ou um status de "inconclusivo".
            most_likely_index = np.argmax(probabilities)
            most_likely_prob = probabilities[most_likely_index]
            most_likely_label = self.classes[most_likely_index]
            
            label_to_return = "ritmo_sinusal" if most_likely_label == 'NORM' else most_likely_label

            logger.warning(f"Nenhum diagnóstico atingiu o limiar. Retornando o mais provável: {label_to_return} com prob {most_likely_prob:.2f}")
            return [{
                "label": label_to_return,
                "probability": float(most_likely_prob)
            }]

        return sorted(diagnoses, key=lambda x: x['probability'], reverse=True)

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo carregado."""
        if self.model:
            return {
                "model_path": self.model_loader.model_path,
                "classes": self.classes,
                "threshold": self.threshold,
                "architecture": "Custom CNN (PTB-XL based)"
            }
        return {"model_path": self.model_loader.model_path, "classes": None, "threshold": self.threshold}

