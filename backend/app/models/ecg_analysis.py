"""
Modelo de análise de ECG
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base
from app.core.constants import AnalysisStatus, ClinicalUrgency, DiagnosisCategory, FileType


class ECGAnalysis(Base):
    """Modelo de análise de ECG."""
    __tablename__ = "ecg_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    file_url = Column(String, nullable=False)
    file_type = Column(SQLEnum(FileType), nullable=False)
    
    status = Column(SQLEnum(AnalysisStatus), default=AnalysisStatus.PENDING, nullable=False)
    clinical_urgency = Column(SQLEnum(ClinicalUrgency), default=ClinicalUrgency.NORMAL)
    
    # Resultados da análise
    diagnosis = Column(String, nullable=True)
    diagnosis_category = Column(SQLEnum(DiagnosisCategory), nullable=True)
    findings = Column(JSON, nullable=True)
    risk_score = Column(Float, nullable=True)
    
    # Validação
    validated = Column(Boolean, default=False)
    validated_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    validated_at = Column(DateTime, nullable=True)
    
    # Metadados
    notes = Column(String, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relacionamentos
    patient = relationship("Patient", back_populates="ecg_analyses")
    validator = relationship("User", foreign_keys=[validated_by])
    validations = relationship("ECGValidation", back_populates="analysis")
    
    def __repr__(self):
        return f"<ECGAnalysis(id={self.id}, patient_id={self.patient_id}, status={self.status})>"


# Re-exportar AnalysisStatus para compatibilidade
__all__ = ["ECGAnalysis", "AnalysisStatus"]
