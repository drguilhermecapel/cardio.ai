"""
ECG models - importado de constants para compatibilidade
"""

# Re-exportar de constants para manter compatibilidade
from app.core.constants import (
    FileType,
    ClinicalUrgency,
    AnalysisStatus as ProcessingStatus,  # Alias para compatibilidade
    DiagnosisCategory as RhythmType,     # Alias para compatibilidade
)

# Adicionar valores extras se necessário
class ECGLeadType:
    """ECG lead types"""
    I = "I"
    II = "II"
    III = "III"
    AVR = "aVR"
    AVL = "aVL"
    AVF = "aVF"
    V1 = "V1"
    V2 = "V2"
    V3 = "V3"
    V4 = "V4"
    V5 = "V5"
    V6 = "V6"

__all__ = ["FileType", "ProcessingStatus", "ClinicalUrgency", "RhythmType", "ECGLeadType"]
