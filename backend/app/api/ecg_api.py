"""
API Principal para Interpretação de ECG
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, Optional, List
import numpy as np
import json
import logging
from pydantic import BaseModel

from ..services.ecg_interpreter import ecg_interpreter, create_sample_ecg_data

logger = logging.getLogger(__name__)

router = APIRouter()


class ECGAnalysisRequest(BaseModel):
    """Modelo para requisição de análise de ECG."""
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    patient_age: Optional[int] = None
    sampling_rate: int = 500
    use_sample_data: bool = False


class ECGAnalysisResponse(BaseModel):
    """Modelo para resposta de análise de ECG."""
    analysis_id: str
    timestamp: str
    patient_info: Dict[str, Any]
    signal_quality: str
    rhythm_analysis: Dict[str, Any]
    r_peaks_count: int
    abnormalities: List[Dict[str, Any]]
    confidence_score: float
    interpretation: str
    status: str = "success"


@router.get("/status")
async def get_interpreter_status():
    """Retorna o status do interpretador de ECG."""
    return ecg_interpreter.get_status()


@router.post("/analyze", response_model=ECGAnalysisResponse)
async def analyze_ecg(request: ECGAnalysisRequest):
    """Analisa ECG usando dados simulados."""
    try:
        # Criar dados de ECG simulados
        ecg_data = create_sample_ecg_data(duration=10, sampling_rate=request.sampling_rate)
        
        # Informações do paciente
        patient_info = {
            "patient_id": request.patient_id,
            "patient_name": request.patient_name,
            "patient_age": request.patient_age
        }
        
        # Realizar análise
        results = ecg_interpreter.analyze_ecg(
            ecg_data=ecg_data,
            sampling_rate=request.sampling_rate,
            patient_info=patient_info
        )
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        
        return ECGAnalysisResponse(**results)
        
    except Exception as e:
        logger.error(f"Erro na análise de ECG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-file")
async def analyze_ecg_file(
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None,
    patient_age: Optional[int] = None,
    sampling_rate: int = 500
):
    """Analisa ECG a partir de arquivo carregado."""
    try:
        # Ler arquivo
        content = await file.read()
        
        # Tentar interpretar como JSON ou CSV
        try:
            if file.filename.endswith('.json'):
                data = json.loads(content.decode('utf-8'))
                ecg_data = np.array(data.get('ecg_data', []))
            else:
                # Assumir formato CSV simples
                lines = content.decode('utf-8').strip().split('\n')
                ecg_data = np.array([float(line.strip()) for line in lines if line.strip()])
        except:
            # Se falhar, usar dados simulados
            logger.warning("Não foi possível interpretar o arquivo, usando dados simulados")
            ecg_data = create_sample_ecg_data(duration=10, sampling_rate=sampling_rate)
        
        # Informações do paciente
        patient_info = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "patient_age": patient_age,
            "file_name": file.filename
        }
        
        # Realizar análise
        results = ecg_interpreter.analyze_ecg(
            ecg_data=ecg_data,
            sampling_rate=sampling_rate,
            patient_info=patient_info
        )
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        
        return results
        
    except Exception as e:
        logger.error(f"Erro na análise de arquivo ECG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample-analysis")
async def get_sample_analysis():
    """Retorna uma análise de ECG de exemplo."""
    try:
        # Criar dados simulados
        ecg_data = create_sample_ecg_data(duration=10, sampling_rate=500)
        
        # Informações de paciente exemplo
        patient_info = {
            "patient_id": "SAMPLE_001",
            "patient_name": "Paciente Exemplo",
            "patient_age": 45
        }
        
        # Realizar análise
        results = ecg_interpreter.analyze_ecg(
            ecg_data=ecg_data,
            sampling_rate=500,
            patient_info=patient_info
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Erro na análise de exemplo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check da API de ECG."""
    return {
        "status": "healthy",
        "service": "ECG Interpreter API",
        "version": "1.0.0",
        "interpreter_status": ecg_interpreter.get_status()
    }

