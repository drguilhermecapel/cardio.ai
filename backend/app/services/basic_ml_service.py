"""Serviço básico de ML para ECG"""
import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BasicMLService:
    def __init__(self):
        self.model_loaded = True
        logger.info("✅ BasicMLService inicializado")
    
    def analyze_ecg(self, data: np.ndarray) -> Dict[str, Any]:
        """Análise básica de ECG"""
        # Simular análise de ML
        heart_rate = 60 + np.random.randint(-20, 40)
        rhythm = "Normal" if 50 <= heart_rate <= 100 else "Anormal"
        
        return {
            "heart_rate": heart_rate,
            "rhythm": rhythm,
            "confidence": 0.85,
            "pathologies": [],
            "analysis_time": "2024-01-01T00:00:00Z"
        }
    
    def health_check(self) -> Dict[str, Any]:
        return {"status": "operational", "model_loaded": self.model_loaded}
