"""
Serviço ECG Melhorado com correções de pré-processamento
Inclui melhor tratamento de imagens e extração de características
"""

import os
import sys
import logging
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
import uuid
import base64
from io import BytesIO

# Adicionar path para importar funções melhoradas
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../models'))
from preprocess_functions_improved import (
    preprocess_ecg_signal,
    extract_ecg_from_image,
    validate_ecg_signal,
    prepare_for_model,
    get_ptbxl_leads_order,
    get_diagnosis_mapping
)

# Importar serviço de modelo
from .unified_model_service import get_model_service

logger = logging.getLogger(__name__)

class ImprovedECGService:
    """
    Serviço ECG melhorado com correções de pré-processamento e extração de imagens.
    """
    
    def __init__(self):
        """Inicializa o serviço ECG melhorado."""
        self.model_service = get_model_service()
        self.diagnosis_mapping = get_diagnosis_mapping()
        self.leads_order = get_ptbxl_leads_order()
        
        # Configurações de processamento
        self.default_sampling_rate = 100
        self.default_duration = 10.0
        self.target_samples = 1000
        self.target_leads = 12
        
        # Cache de processamentos
        self.processed_ecgs = {}
        
        logger.info("Serviço ECG melhorado inicializado")
    
    def process_ecg_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Processa um arquivo de ECG (dados ou imagem).
        
        Args:
            file_path: Caminho para o arquivo ECG
            
        Returns:
            Resultado do processamento com ID único
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"error": f"Arquivo não encontrado: {file_path}"}
            
            # Gerar ID único para o processamento
            process_id = str(uuid.uuid4())
            
            # Determinar tipo de arquivo
            file_extension = file_path.suffix.lower()
            
            if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                # Processar imagem ECG
                result = self._process_ecg_image(file_path, process_id)
            elif file_extension in ['.csv', '.txt', '.dat', '.npy']:
                # Processar dados ECG
                result = self._process_ecg_data(file_path, process_id)
            else:
                return {"error": f"Formato de arquivo não suportado: {file_extension}"}
            
            # Armazenar resultado no cache
            self.processed_ecgs[process_id] = result
            
            return {
                "process_id": process_id,
                "status": "success",
                "file_type": "image" if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] else "data",
                "signal_shape": result.get("signal_shape"),
                "leads_detected": result.get("leads_detected"),
                "duration_seconds": result.get("duration_seconds"),
                "sampling_rate": result.get("sampling_rate"),
                "quality_score": result.get("quality_score"),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro no processamento do arquivo ECG: {e}")
            return {"error": str(e)}
    
    def _process_ecg_image(self, image_path: Path, process_id: str) -> Dict[str, Any]:
        """Processa uma imagem de ECG."""
        try:
            logger.info(f"Processando imagem ECG: {image_path}")
            
            # Extrair sinal da imagem
            signal = extract_ecg_from_image(
                str(image_path),
                leads_expected=self.target_leads,
                duration_seconds=self.default_duration,
                sampling_rate=self.default_sampling_rate
            )
            
            # Validar sinal extraído
            is_valid, validation_msg = validate_ecg_signal(signal)
            
            # Calcular métricas de qualidade
            quality_score = self._calculate_quality_score(signal)
            
            return {
                "signal": signal,
                "signal_shape": signal.shape,
                "leads_detected": signal.shape[0],
                "duration_seconds": signal.shape[1] / self.default_sampling_rate,
                "sampling_rate": self.default_sampling_rate,
                "is_valid": is_valid,
                "validation_message": validation_msg,
                "quality_score": quality_score,
                "source_type": "image",
                "source_path": str(image_path)
            }
            
        except Exception as e:
            logger.error(f"Erro no processamento da imagem ECG: {e}")
            return {"error": str(e)}
    
    def _process_ecg_data(self, data_path: Path, process_id: str) -> Dict[str, Any]:
        """Processa dados de ECG de arquivo."""
        try:
            logger.info(f"Processando dados ECG: {data_path}")
            
            # Carregar dados baseado na extensão
            file_extension = data_path.suffix.lower()
            
            if file_extension == '.npy':
                raw_data = np.load(data_path)
            elif file_extension in ['.csv', '.txt']:
                raw_data = np.loadtxt(data_path, delimiter=',')
            else:
                return {"error": f"Formato não suportado: {file_extension}"}
            
            # Pré-processar sinal
            signal = preprocess_ecg_signal(
                raw_data,
                fs_in=500,  # Assumir 500 Hz por padrão
                fs_target=self.default_sampling_rate
            )
            
            # Validar sinal processado
            is_valid, validation_msg = validate_ecg_signal(signal)
            
            # Calcular métricas de qualidade
            quality_score = self._calculate_quality_score(signal)
            
            return {
                "signal": signal,
                "signal_shape": signal.shape,
                "leads_detected": signal.shape[0],
                "duration_seconds": signal.shape[1] / self.default_sampling_rate,
                "sampling_rate": self.default_sampling_rate,
                "is_valid": is_valid,
                "validation_message": validation_msg,
                "quality_score": quality_score,
                "source_type": "data",
                "source_path": str(data_path)
            }
            
        except Exception as e:
            logger.error(f"Erro no processamento dos dados ECG: {e}")
            return {"error": str(e)}
    
    def _calculate_quality_score(self, signal: np.ndarray) -> float:
        """Calcula uma pontuação de qualidade para o sinal ECG."""
        try:
            if signal.size == 0:
                return 0.0
            
            quality_factors = []
            
            # Fator 1: Variação do sinal (não deve ser muito baixa ou muito alta)
            for lead in range(signal.shape[0]):
                std_dev = np.std(signal[lead])
                if 0.1 <= std_dev <= 5.0:
                    quality_factors.append(1.0)
                elif std_dev < 0.1:
                    quality_factors.append(0.3)  # Muito baixa variação
                else:
                    quality_factors.append(0.7)  # Muito alta variação
            
            # Fator 2: Presença de valores finitos
            finite_ratio = np.sum(np.isfinite(signal)) / signal.size
            quality_factors.append(finite_ratio)
            
            # Fator 3: Ausência de saturação
            saturation_ratio = np.sum(np.abs(signal) >= 9.5) / signal.size
            quality_factors.append(1.0 - saturation_ratio)
            
            # Calcular pontuação média
            return float(np.mean(quality_factors))
            
        except Exception as e:
            logger.error(f"Erro no cálculo da qualidade: {e}")
            return 0.5  # Pontuação neutra em caso de erro
    
    def analyze_ecg(self, process_id: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analisa um ECG previamente processado usando modelo de IA.
        
        Args:
            process_id: ID do processamento prévio
            model_name: Nome do modelo a usar (opcional)
            
        Returns:
            Resultado da análise com diagnóstico
        """
        try:
            # Verificar se o processamento existe
            if process_id not in self.processed_ecgs:
                return {"error": f"Processamento não encontrado: {process_id}"}
            
            processed_data = self.processed_ecgs[process_id]
            
            if "error" in processed_data:
                return processed_data
            
            # Obter sinal processado
            signal = processed_data["signal"]
            
            # Preparar para o modelo
            model_input = prepare_for_model(signal)
            
            # Usar modelo padrão se não especificado
            if model_name is None:
                model_name = "ecg_model_final"
            
            # Fazer predição
            prediction_result = self.model_service.predict(model_name, model_input)
            
            if "error" in prediction_result:
                return prediction_result
            
            # Interpretar resultados
            predictions = prediction_result["predictions"][0]  # Primeiro item do batch
            
            # Encontrar classes com maior probabilidade
            top_indices = np.argsort(predictions)[-5:][::-1]  # Top 5
            
            diagnoses = []
            for idx in top_indices:
                if idx < len(self.diagnosis_mapping) and predictions[idx] > 0.1:
                    diagnoses.append({
                        "condition": self.diagnosis_mapping.get(idx, f"Classe {idx}"),
                        "probability": float(predictions[idx]),
                        "confidence": "high" if predictions[idx] > 0.7 else "medium" if predictions[idx] > 0.3 else "low"
                    })
            
            # Se nenhum diagnóstico com probabilidade significativa
            if not diagnoses:
                diagnoses.append({
                    "condition": "Normal",
                    "probability": 1.0 - np.max(predictions),
                    "confidence": "medium"
                })
            
            return {
                "process_id": process_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": model_name,
                "signal_quality": processed_data.get("quality_score", 0.5),
                "diagnoses": diagnoses,
                "raw_predictions": predictions.tolist(),
                "leads_analyzed": self.leads_order,
                "metadata": {
                    "signal_shape": processed_data.get("signal_shape"),
                    "sampling_rate": processed_data.get("sampling_rate"),
                    "duration_seconds": processed_data.get("duration_seconds"),
                    "source_type": processed_data.get("source_type")
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na análise do ECG: {e}")
            return {"error": str(e)}
    
    def get_processing_status(self, process_id: str) -> Dict[str, Any]:
        """Obtém status de um processamento."""
        if process_id in self.processed_ecgs:
            data = self.processed_ecgs[process_id]
            return {
                "process_id": process_id,
                "status": "completed",
                "has_error": "error" in data,
                "quality_score": data.get("quality_score"),
                "signal_shape": data.get("signal_shape")
            }
        else:
            return {
                "process_id": process_id,
                "status": "not_found"
            }
    
    def list_processed_ecgs(self) -> List[str]:
        """Lista todos os ECGs processados."""
        return list(self.processed_ecgs.keys())
    
    def clear_cache(self):
        """Limpa o cache de processamentos."""
        self.processed_ecgs.clear()
        logger.info("Cache de processamentos limpo")


# Instância global do serviço
_improved_ecg_service = None

def get_improved_ecg_service() -> ImprovedECGService:
    """Obtém instância singleton do serviço ECG melhorado."""
    global _improved_ecg_service
    if _improved_ecg_service is None:
        _improved_ecg_service = ImprovedECGService()
    return _improved_ecg_service

