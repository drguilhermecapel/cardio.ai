"""
Schemas Pydantic para validação de dados de ECG.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator

# Schemas existentes mantidos para compatibilidade
class ECGAnalysisBase(BaseModel):
    """Schema base para análise de ECG."""
    patient_id: int = Field(..., description="ID do paciente")
    original_filename: str = Field(..., description="Nome do arquivo original")
    acquisition_date: datetime = Field(..., description="Data de aquisição do ECG")
    sample_rate: int = Field(..., ge=100, le=2000, description="Taxa de amostragem em Hz")
    duration_seconds: float = Field(..., ge=1, description="Duração em segundos")
    leads_count: int = Field(..., ge=1, le=15, description="Número de derivações")
    leads_names: List[str] = Field(..., description="Nomes das derivações")

# Novos schemas para a arquitetura unificada

class ECGSignal(BaseModel):
    """Schema para dados de sinal de ECG."""
    patient_id: Optional[str] = Field(None, description="ID do paciente")
    data: List[List[float]] = Field(..., description="Dados do sinal ECG (derivações x amostras)")
    sampling_rate: int = Field(500, description="Frequência de amostragem em Hz")
    leads: Optional[List[str]] = Field(None, description="Nomes das derivações")
    
    @validator('data')
    def validate_signal_data(cls, v):
        if not v or not v[0]:
            raise ValueError('Dados do sinal não podem estar vazios')
        return v

class DiagnosisResult(BaseModel):
    """Schema para resultado individual de diagnóstico."""
    condition: str = Field(..., description="Nome da condição")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidade da condição")
    is_present: bool = Field(..., description="Se a condição está presente")

class ECGDiagnosis(BaseModel):
    """Schema para diagnóstico completo de ECG."""
    results: List[DiagnosisResult] = Field(..., description="Lista de resultados de diagnóstico")
    is_abnormal: bool = Field(..., description="Se o ECG é anormal")
    model_version: str = Field(..., description="Versão do modelo utilizado")
    overall_finding: str = Field(..., description="Achado geral do diagnóstico")
    heart_rate: float = Field(0.0, description="Frequência cardíaca em BPM")
    qrs_duration: str = Field("0.0 ms", description="Duração do complexo QRS")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp do diagnóstico")
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "condition": "Normal ECG",
                        "probability": 0.85,
                        "is_present": True
                    },
                    {
                        "condition": "Atrial Fibrillation",
                        "probability": 0.12,
                        "is_present": False
                    }
                ],
                "is_abnormal": False,
                "model_version": "PTB-XL Production v1.0",
                "overall_finding": "Ritmo Sinusal Normal",
                "heart_rate": 72.5,
                "qrs_duration": "95.2 ms",
                "timestamp": "2024-01-07T10:30:00"
            }
        }

class ECGAnalysisRequest(BaseModel):
    """Schema para requisição de análise de ECG."""
    signal_data: Optional[List[List[float]]] = Field(None, description="Dados do sinal (para análise direta)")
    image_path: Optional[str] = Field(None, description="Caminho da imagem (para digitalização)")
    patient_id: Optional[str] = Field(None, description="ID do paciente")
    sampling_rate: int = Field(500, description="Frequência de amostragem")
    
    @validator('signal_data', 'image_path')
    def validate_input_source(cls, v, values, field):
        # Pelo menos um dos dois deve estar presente
        if field.name == 'image_path' and not v and not values.get('signal_data'):
            raise ValueError('Deve fornecer signal_data ou image_path')
        return v

class ECGAnalysisResponse(BaseModel):
    """Schema para resposta de análise de ECG."""
    diagnosis: ECGDiagnosis = Field(..., description="Diagnóstico do ECG")
    processing_time_ms: float = Field(..., description="Tempo de processamento em milissegundos")
    success: bool = Field(True, description="Se a análise foi bem-sucedida")
    error_message: Optional[str] = Field(None, description="Mensagem de erro, se houver")
    
    class Config:
        json_schema_extra = {
            "example": {
                "diagnosis": {
                    "results": [
                        {
                            "condition": "Normal ECG",
                            "probability": 0.85,
                            "is_present": True
                        }
                    ],
                    "is_abnormal": False,
                    "model_version": "PTB-XL Production v1.0",
                    "overall_finding": "Ritmo Sinusal Normal",
                    "heart_rate": 72.5,
                    "qrs_duration": "95.2 ms"
                },
                "processing_time_ms": 1250.5,
                "success": True,
                "error_message": None
            }
        }

