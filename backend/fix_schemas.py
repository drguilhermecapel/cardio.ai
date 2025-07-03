# Corrigir ECGAnalysisCreate schema 
import os 
 
schema_content = '''from datetime import datetime 
from typing import Optional, List, Dict, Any 
from pydantic import BaseModel, Field 
from app.core.constants import FileType, AnalysisStatus, ClinicalUrgency, DiagnosisCategory 
 
class ECGAnalysisCreate(BaseModel): 
    """Schema para criar analise de ECG.""" 
    patient_id: int = Field(..., description="ID do paciente") 
    file_path: str = Field(..., description="Caminho do arquivo") 
    original_filename: str = Field(..., description="Nome original do arquivo") 
    file_type: Optional[FileType] = Field(default=FileType.CSV) 
    acquisition_date: Optional[datetime] = Field(default_factory=datetime.now) 
    sample_rate: Optional[int] = Field(default=500) 
    duration_seconds: Optional[float] = Field(default=10.0) 
    leads_count: Optional[int] = Field(default=12) 
    leads_names: Optional[List[str]] = Field(default_factory=lambda: ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]) 
    device_manufacturer: Optional[str] = Field(default="Unknown") 
    device_model: Optional[str] = Field(default="Unknown") 
    device_serial: Optional[str] = Field(default="Unknown") 
    clinical_notes: Optional[str] = Field(default="") 
ECHO est  desativado.
    class Config: 
        from_attributes = True 
''' 
 
# Adicionar ao arquivo de schemas 
try: 
    with open('app/schemas/ecg_analysis.py', 'r', encoding='utf-8') as f: 
        content = f.read() 
ECHO est  desativado.
    if 'class ECGAnalysisCreate' not in content: 
        content += '\n\n' + schema_content 
    else: 
        # Substituir a classe existente 
        import re 
