"""
Interfaces e tipos comuns para evitar imports circulares.
"""

from typing import Protocol, Dict, Any, Optional
import numpy as np

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
