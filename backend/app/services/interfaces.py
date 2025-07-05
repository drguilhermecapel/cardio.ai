"""
Interfaces e tipos comuns para evitar imports circulares.
"""

from typing import Protocol, Dict, Any, Optional, List, Tuple
import numpy as np
from abc import ABC, abstractmethod

class IMLService(Protocol):
    """Interface para serviços de ML."""
    
    async def analyze_ecg_advanced(
        self,
        ecg_signal: np.ndarray,
        sampling_rate: float = 500.0,
        patient_context: Optional[Dict[str, Any]] = None,
        return_interpretability: bool = False,
    ) -> Dict[str, Any]:
        """Análise avançada de ECG."""
        ...

class IInterpretabilityService(Protocol):
    """Interface para serviços de interpretabilidade."""
    
    async def explain_prediction(
        self,
        model_output: Dict[str, Any],
        ecg_signal: np.ndarray,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Explica predição do modelo."""
        ...

class IHybridECGService(Protocol):
    """Interface para serviço híbrido de ECG."""
    
    async def analyze_ecg_comprehensive(
        self, 
        file_path: str, 
        patient_id: int, 
        analysis_id: str
    ) -> Dict[str, Any]:
        """Análise abrangente de ECG."""
        ...

# Novas interfaces para a arquitetura unificada

class ECGDigitizerInterface(ABC):
    """Interface para serviços de digitalização de ECG."""
    
    @abstractmethod
    async def digitize(self, image_path: str) -> Tuple[np.ndarray, int]:
        """
        Digitaliza uma imagem de ECG e retorna o sinal e a frequência de amostragem.
        
        Args:
            image_path: Caminho para a imagem do ECG
            
        Returns:
            Tuple contendo (signal_data, sampling_rate)
        """
        pass

class PreprocessingPipelineInterface(ABC):
    """Interface para pipelines de pré-processamento."""
    
    @abstractmethod
    async def process(self, signal_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Processa o sinal de ECG aplicando filtros e normalizações.
        
        Args:
            signal_data: Dados do sinal de ECG
            sampling_rate: Frequência de amostragem
            
        Returns:
            Sinal processado pronto para o modelo
        """
        pass

class MLModelServiceInterface(ABC):
    """Interface para serviços de modelo de ML."""
    
    @abstractmethod
    async def predict(self, processed_signal: np.ndarray) -> np.ndarray:
        """
        Realiza predição no sinal pré-processado.
        
        Args:
            processed_signal: Sinal pré-processado
            
        Returns:
            Predições do modelo
        """
        pass
    
    @abstractmethod
    def format_prediction(self, prediction: np.ndarray):
        """
        Formata a predição bruta em um objeto de diagnóstico estruturado.
        
        Args:
            prediction: Predição bruta do modelo
            
        Returns:
            Objeto de diagnóstico formatado
        """
        pass

class ECGModelService(ABC):
    """Interface base para serviços de modelo de ECG."""
    
    @abstractmethod
    def predict(self, ecg_data: np.ndarray) -> List[Dict[str, Any]]:
        """Realiza predição em dados de ECG."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        pass

class PreprocessingService(ABC):
    """Interface para serviços de pré-processamento."""
    
    @abstractmethod
    def preprocess(self, ecg_data: np.ndarray) -> np.ndarray:
        """Pré-processa dados de ECG."""
        pass

class ExplanationService(ABC):
    """Interface para serviços de explicação."""
    
    @abstractmethod
    def get_explanation(self, diagnosis_code: str) -> str:
        """Retorna explicação para um código de diagnóstico."""
        pass

