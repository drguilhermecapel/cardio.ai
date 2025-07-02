"""
Modelos de dados para análise de ECG.
Define estruturas de banco de dados e enums para o sistema.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, JSON, Text, 
    Boolean, ForeignKey, Index, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class FileType(str, Enum):
    """Tipos de arquivo suportados para ECG."""
    CSV = "csv"
    XML = "xml"
    JSON = "json"
    EDF = "edf"
    DICOM = "dicom"
    HL7 = "hl7"
    PDF = "pdf"
    MAT = "mat"  # MATLAB files
    TXT = "txt"
    WFDB = "wfdb"  # PhysioNet format


class ProcessingStatus(str, Enum):
    """Status do processamento da análise."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    QUEUED = "queued"
    RETRYING = "retrying"


class ClinicalUrgency(str, Enum):
    """Níveis de urgência clínica."""
    CRITICAL = "critical"
    URGENT = "urgent"
    MODERATE = "moderate"
    LOW = "low"
    ROUTINE = "routine"


class RhythmType(str, Enum):
    """Tipos de ritmo cardíaco."""
    NORMAL_SINUS = "normal_sinus"
    ATRIAL_FIBRILLATION = "atrial_fibrillation"
    ATRIAL_FLUTTER = "atrial_flutter"
    VENTRICULAR_TACHYCARDIA = "ventricular_tachycardia"
    VENTRICULAR_FIBRILLATION = "ventricular_fibrillation"
    BRADYCARDIA = "bradycardia"
    TACHYCARDIA = "tachycardia"
    HEART_BLOCK = "heart_block"
    PACED_RHYTHM = "paced_rhythm"
    IRREGULAR = "irregular"


class ECGAnalysis(Base):
    """Modelo principal para análise de ECG."""
    __tablename__ = "ecg_analyses"
    
    # Identificadores
    id = Column(String(36), primary_key=True)
    patient_id = Column(Integer, nullable=False, index=True)
    
    # Informações do arquivo
    original_filename = Column(String(255), nullable=False)
    file_type = Column(SQLEnum(FileType), nullable=False)
    file_path = Column(String(500))
    file_size_bytes = Column(Integer)
    
    # Metadados da aquisição
    acquisition_date = Column(DateTime, nullable=False)
    sample_rate = Column(Integer, nullable=False)  # Hz
    duration_seconds = Column(Float, nullable=False)
    leads_count = Column(Integer, nullable=False)
    leads_names = Column(JSON, nullable=False)  # Lista de nomes das derivações
    
    # Status e processamento
    status = Column(SQLEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False)
    processing_started_at = Column(DateTime)
    processing_completed_at = Column(DateTime)
    processing_duration_ms = Column(Integer)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    
    # Resultados da análise
    heart_rate = Column(Float)  # BPM
    rhythm_type = Column(SQLEnum(RhythmType))
    pr_interval = Column(Float)  # ms
    qrs_duration = Column(Float)  # ms
    qt_interval = Column(Float)  # ms
    qtc_interval = Column(Float)  # ms (corrigido)
    p_wave_present = Column(Boolean)
    
    # Qualidade do sinal
    signal_quality_score = Column(Float)  # 0-1
    noise_level = Column(Float)
    baseline_wander = Column(Boolean)
    motion_artifacts = Column(Boolean)
    
    # Detecções e anormalidades
    abnormalities_detected = Column(JSON)  # Lista de anormalidades
    clinical_urgency = Column(SQLEnum(ClinicalUrgency))
    requires_review = Column(Boolean, default=False)
    
    # Dados completos
    raw_signal_data = Column(JSON)  # Dados brutos do sinal
    processed_signal_data = Column(JSON)  # Dados processados
    features_extracted = Column(JSON)  # Features para ML
    ml_predictions = Column(JSON)  # Predições do modelo
    
    # Relatório e recomendações
    medical_report = Column(Text)
    recommendations = Column(JSON)
    diagnosis_codes = Column(JSON)  # ICD-10 codes
    
    # Auditoria
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(100))
    reviewed_by = Column(String(100))
    reviewed_at = Column(DateTime)
    
    # Relacionamentos
    files = relationship("ECGFile", back_populates="analysis", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="analysis", cascade="all, delete-orphan")
    
    # Índices para performance
    __table_args__ = (
        Index('idx_patient_date', 'patient_id', 'acquisition_date'),
        Index('idx_status_urgency', 'status', 'clinical_urgency'),
        Index('idx_created_at', 'created_at'),
    )


class ECGFile(Base):
    """Modelo para arquivos de ECG."""
    __tablename__ = "ecg_files"
    
    id = Column(String(36), primary_key=True)
    analysis_id = Column(String(36), ForeignKey("ecg_analyses.id"), nullable=False)
    
    filename = Column(String(255), nullable=False)
    file_type = Column(SQLEnum(FileType), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer)
    checksum = Column(String(64))  # SHA-256
    
    is_original = Column(Boolean, default=True)
    is_compressed = Column(Boolean, default=False)
    compression_ratio = Column(Float)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relacionamento
    analysis = relationship("ECGAnalysis", back_populates="files")


class AuditLog(Base):
    """Modelo para trilha de auditoria."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(String(36), ForeignKey("ecg_analyses.id"), nullable=False)
    
    action = Column(String(100), nullable=False)
    user_id = Column(String(100), nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    
    old_values = Column(JSON)
    new_values = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Relacionamento
    analysis = relationship("ECGAnalysis", back_populates="audit_logs")
    
    # Índice
    __table_args__ = (
        Index('idx_audit_analysis_time', 'analysis_id', 'timestamp'),
    )


class MLModel(Base):
    """Modelo para versionamento de modelos ML."""
    __tablename__ = "ml_models"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    
    model_type = Column(String(50))  # 'classification', 'segmentation', etc.
    architecture = Column(JSON)  # Arquitetura do modelo
    parameters = Column(JSON)  # Hiperparâmetros
    
    accuracy = Column(Float)
    sensitivity = Column(Float)
    specificity = Column(Float)
    auc_score = Column(Float)
    
    training_dataset = Column(String(255))
    training_samples = Column(Integer)
    validation_score = Column(Float)
    
    file_path = Column(String(500))
    file_size_mb = Column(Float)
    
    is_active = Column(Boolean, default=False)
    deployed_at = Column(DateTime)
    
    created_at = Column(DateTime, default=func.now())
    created_by = Column(String(100))
    
    # Índice único
    __table_args__ = (
        Index('idx_model_version', 'name', 'version', unique=True),
    )
