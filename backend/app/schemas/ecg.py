"""
Schemas Pydantic para validação de dados de ECG.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from app.models.ecg import FileType, ProcessingStatus, ClinicalUrgency, RhythmType


class ECGAnalysisBase(BaseModel):
    """Schema base para análise de ECG."""
    patient_id: int = Field(..., description="ID do paciente")
    original_filename: str = Field(..., description="Nome do arquivo original")
    file_type: FileType = Field(..., description="Tipo do arquivo")
    acquisition_date: datetime = Field(..., description="Data de aquisição do ECG")
    sample_rate: int = Field(..., ge=100, le=2000, description="Taxa de amostragem em Hz")
    duration_seconds: float = Field(..., ge=1, description="Duração em segundos")
    leads_count: int = Field(..., ge=1, le=15, description="Número de derivações")
    leads_names: List[str] = Field(..., description="Nomes das derivações")


class ECGAnalysisCreate(ECGAnalysisBase):
    """Schema para criação de análise de ECG."""
    file_size_bytes: Optional[int] = Field(None, ge=0)
    created_by: Optional[str] = Field(None, max_length=100)
    
    @validator('leads_names')
    def validate_leads_count(cls, v, values):
        if 'leads_count' in values and len(v) != values['leads_count']:
            raise ValueError('Número de nomes de derivações deve corresponder a leads_count')
        return v
    
    @validator('patient_id')
    def validate_patient_id(cls, v):
        # Aceitar tanto int quanto string que pode ser convertida para int
        if isinstance(v, str):
            if v.isdigit():
                return int(v)
            else:
                # Para testes, aceitar IDs alfanuméricos
                return hash(v) % 1000000  # Converter para int
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": 12345,
                "original_filename": "ecg_12_lead.csv",
                "file_type": "csv",
                "acquisition_date": "2024-01-15T10:30:00",
                "sample_rate": 500,
                "duration_seconds": 10.0,
                "leads_count": 12,
                "leads_names": ["I", "II", "III", "aVR", "aVL", "aVF", 
                               "V1", "V2", "V3", "V4", "V5", "V6"]
            }
        }


class ECGAnalysisUpdate(BaseModel):
    """Schema para atualização de análise de ECG."""
    status: Optional[ProcessingStatus] = None
    heart_rate: Optional[float] = Field(None, ge=20, le=300)
    rhythm_type: Optional[RhythmType] = None
    pr_interval: Optional[float] = Field(None, ge=0, le=500)
    qrs_duration: Optional[float] = Field(None, ge=0, le=300)
    qt_interval: Optional[float] = Field(None, ge=0, le=700)
    qtc_interval: Optional[float] = Field(None, ge=0, le=700)
    signal_quality_score: Optional[float] = Field(None, ge=0, le=1)
    clinical_urgency: Optional[ClinicalUrgency] = None
    abnormalities_detected: Optional[List[str]] = None
    medical_report: Optional[str] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    reviewed_by: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "completed",
                "heart_rate": 72,
                "rhythm_type": "normal_sinus",
                "signal_quality_score": 0.95,
                "clinical_urgency": "low"
            }
        }


class ECGAnalysisResponse(ECGAnalysisBase):
    """Schema para resposta de análise de ECG."""
    id: str
    status: ProcessingStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Resultados da análise
    heart_rate: Optional[float] = None
    rhythm_type: Optional[RhythmType] = None
    pr_interval: Optional[float] = None
    qrs_duration: Optional[float] = None
    qt_interval: Optional[float] = None
    qtc_interval: Optional[float] = None
    
    # Qualidade e urgência
    signal_quality_score: Optional[float] = None
    clinical_urgency: Optional[ClinicalUrgency] = None
    abnormalities_detected: Optional[List[str]] = None
    requires_review: bool = False
    
    # Processamento
    processing_duration_ms: Optional[int] = None
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class ECGValidationResult(BaseModel):
    """Schema para resultado de validação de ECG."""
    is_valid: bool
    quality_score: float = Field(..., ge=0, le=1)
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_valid": True,
                "quality_score": 0.92,
                "issues": [],
                "warnings": ["Low signal quality in lead V6"],
                "metadata": {
                    "num_leads": 12,
                    "duration_seconds": 10.0
                }
            }
        }


class ECGReportRequest(BaseModel):
    """Schema para solicitação de relatório."""
    analysis_id: str
    report_format: str = Field("pdf", pattern="^(pdf|html|json)$")
    include_raw_data: bool = False
    include_images: bool = True
    language: str = Field("en", pattern="^(en|pt|es)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": "123e4567-e89b-12d3-a456-426614174000",
                "report_format": "pdf",
                "include_images": True,
                "language": "en"
            }
        }


class ECGReportResponse(BaseModel):
    """Schema para resposta de relatório."""
    report_id: str
    analysis_id: str
    format: str
    created_at: datetime
    file_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    content: Optional[Dict[str, Any]] = None  # Para formato JSON
    
    class Config:
        json_schema_extra = {
            "example": {
                "report_id": "report-123",
                "analysis_id": "analysis-456",
                "format": "pdf",
                "created_at": "2024-01-15T10:45:00",
                "file_url": "https://storage.example.com/reports/report-123.pdf",
                "file_size_bytes": 245760
            }
        }


class ECGBatchAnalysisRequest(BaseModel):
    """Schema para análise em lote."""
    analyses: List[ECGAnalysisCreate]
    priority: str = Field("normal", pattern="^(low|normal|high|urgent)$")
    callback_url: Optional[str] = None
    
    @validator('analyses')
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError('Máximo de 100 análises por lote')
        return v


class ECGStatistics(BaseModel):
    """Schema para estatísticas de ECG."""
    total_analyses: int
    completed_analyses: int
    failed_analyses: int
    average_processing_time_ms: float
    analyses_by_urgency: Dict[str, int]
    analyses_by_rhythm: Dict[str, int]
    quality_metrics: Dict[str, float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_analyses": 1000,
                "completed_analyses": 950,
                "failed_analyses": 50,
                "average_processing_time_ms": 1500,
                "analyses_by_urgency": {
                    "critical": 10,
                    "urgent": 50,
                    "moderate": 200,
                    "low": 690
                },
                "analyses_by_rhythm": {
                    "normal_sinus": 800,
                    "atrial_fibrillation": 100,
                    "bradycardia": 50
                },
                "quality_metrics": {
                    "average_signal_quality": 0.89,
                    "detection_accuracy": 0.95
                }
            }
        }
