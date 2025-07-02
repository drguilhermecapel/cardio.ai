"""
Schemas para análise de ECG
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from app.core.constants import FileType, AnalysisStatus, ClinicalUrgency, DiagnosisCategory


class ECGAnalysisBase(BaseModel):
    """Schema base para análise de ECG."""
    patient_id: int
    file_url: str
    file_type: FileType
    
    
class ECGAnalysisCreate(ECGAnalysisBase):
    """Schema para criar análise de ECG."""
    analysis_type: Optional[str] = "standard"
    priority: Optional[ClinicalUrgency] = ClinicalUrgency.NORMAL
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Campos adicionais para compatibilidade
    ecg_data: Optional[Dict[str, Any]] = None
    urgency: Optional[ClinicalUrgency] = None
    
    
class ECGAnalysisUpdate(BaseModel):
    """Schema para atualizar análise de ECG."""
    status: Optional[AnalysisStatus] = None
    diagnosis: Optional[str] = None
    findings: Optional[Dict[str, Any]] = None
    risk_score: Optional[float] = Field(None, ge=0, le=1)
    clinical_urgency: Optional[ClinicalUrgency] = None
    notes: Optional[str] = None
    validated: Optional[bool] = None
    validated_by: Optional[int] = None
    validated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)
    
    
class ECGAnalysisResponse(ECGAnalysisBase):
    """Schema de resposta para análise de ECG."""
    id: int
    status: AnalysisStatus
    diagnosis: Optional[str] = None
    findings: Optional[Dict[str, Any]] = None
    risk_score: Optional[float] = None
    clinical_urgency: Optional[ClinicalUrgency] = None
    created_at: datetime
    updated_at: datetime
    validated: Optional[bool] = False
    validated_by: Optional[int] = None
    validated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)
    

class ECGAnalysisList(BaseModel):
    """Lista paginada de análises."""
    items: List[ECGAnalysisResponse]
    total: int
    page: int = 1
    pages: int = 1
    size: int = 20
    

class ECGValidationCreate(BaseModel):
    """Schema para criar validação."""
    analysis_id: int
    notes: Optional[str] = None
    is_correct: bool = True
    corrections: Optional[Dict[str, Any]] = None
    

class ECGValidationResponse(BaseModel):
    """Schema de resposta para validação."""
    id: int
    analysis_id: int
    validator_id: int
    notes: Optional[str] = None
    is_correct: bool
    corrections: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)
