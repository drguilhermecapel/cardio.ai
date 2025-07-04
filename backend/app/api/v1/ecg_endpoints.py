"""
Endpoints da API para análise de ECG com suporte FHIR R4
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
import numpy as np
import json
import logging
from datetime import datetime
from pydantic import BaseModel, Field
import base64
import io

from app.services.model_service import model_service
from app.services.explainability_service import explainability_service
from app.repositories.validation_repository import ValidationRepository
from app.schemas.ecg import ECGAnalysisRequest, ECGAnalysisResponse
from app.schemas.fhir import FHIRObservation, FHIRDiagnosticReport

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ecg", tags=["ECG Analysis"])

# Instância do repositório de validação
validation_repo = ValidationRepository()


class ValidationRequest(BaseModel):
    """Request para validação de diagnóstico."""
    ecg_id: str = Field(..., description="ID do ECG analisado")
    ai_diagnosis: str = Field(..., description="Diagnóstico da IA")
    ai_probability: float = Field(..., description="Probabilidade do diagnóstico da IA")
    validator_id: str = Field(..., description="ID do cardiologista validador")
    is_correct: bool = Field(..., description="Se o diagnóstico da IA está correto")
    corrected_diagnosis: Optional[str] = Field(None, description="Diagnóstico correto (se diferente)")
    comments: Optional[str] = Field(None, description="Comentários do validador")
    model_version: Optional[str] = Field(None, description="Versão do modelo usado")
    confidence_score: Optional[float] = Field(None, description="Score de confiança")
    processing_time_ms: Optional[int] = Field(None, description="Tempo de processamento em ms")


class ValidationResponse(BaseModel):
    """Response da validação."""
    success: bool
    message: str
    validation_id: Optional[str] = None


class ECGUploadRequest(BaseModel):
    """Request para upload de ECG."""
    patient_id: str = Field(..., description="ID do paciente")
    ecg_data: List[float] = Field(..., description="Dados do ECG")
    sampling_rate: int = Field(default=500, description="Taxa de amostragem")
    leads: List[str] = Field(default=["I"], description="Derivações do ECG")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Metadados adicionais")


class ECGAnalysisResult(BaseModel):
    """Resultado da análise de ECG."""
    patient_id: str
    analysis_id: str
    predictions: Dict[str, Any]
    confidence: float
    explanations: Optional[Dict[str, Any]] = None
    fhir_observation: Optional[Dict[str, Any]] = None
    timestamp: str


@router.post("/analyze", response_model=ECGAnalysisResult)
async def analyze_ecg(request: ECGUploadRequest):
    """Analisa ECG usando modelos carregados."""
    try:
        # Converter dados para numpy
        ecg_data = np.array(request.ecg_data)
        
        # Validar dados
        if len(ecg_data) == 0:
            raise HTTPException(status_code=400, detail="Dados de ECG vazios")
        
        # Listar modelos disponíveis
        available_models = model_service.list_models()
        if not available_models:
            raise HTTPException(status_code=503, detail="Nenhum modelo disponível")
        
        # Usar primeiro modelo disponível
        model_name = available_models[0]
        
        # Realizar predição
        prediction_result = model_service.predict_ecg(model_name, ecg_data)
        
        if "error" in prediction_result:
            raise HTTPException(status_code=500, detail=prediction_result["error"])
        
        # Gerar explicações
        model = model_service.loaded_models[model_name]
        explanations = explainability_service.generate_comprehensive_report(
            model, ecg_data, model_name
        )
        
        # Gerar ID único para análise
        analysis_id = f"ecg_{request.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Criar observação FHIR
        fhir_observation = create_fhir_observation(
            request.patient_id, 
            prediction_result, 
            request.sampling_rate
        )
        
        return ECGAnalysisResult(
            patient_id=request.patient_id,
            analysis_id=analysis_id,
            predictions=prediction_result["predictions"],
            confidence=prediction_result["confidence"],
            explanations=explanations if "error" not in explanations else None,
            fhir_observation=fhir_observation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erro na análise de ECG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-file")
async def upload_ecg_file(
    patient_id: str,
    file: UploadFile = File(...),
    sampling_rate: int = 500
):
    """Upload de arquivo de ECG."""
    try:
        # Ler conteúdo do arquivo
        content = await file.read()
        
        # Processar baseado no tipo de arquivo
        if file.filename.endswith('.txt') or file.filename.endswith('.csv'):
            # Arquivo de texto
            text_content = content.decode('utf-8')
            ecg_data = [float(x.strip()) for x in text_content.split('\n') if x.strip()]
        elif file.filename.endswith('.npy'):
            # Arquivo NumPy
            ecg_data = np.load(io.BytesIO(content)).tolist()
        else:
            raise HTTPException(status_code=400, detail="Formato de arquivo não suportado")
        
        # Criar request
        request = ECGUploadRequest(
            patient_id=patient_id,
            ecg_data=ecg_data,
            sampling_rate=sampling_rate
        )
        
        # Analisar
        return await analyze_ecg(request)
        
    except Exception as e:
        logger.error(f"Erro no upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_available_models():
    """Lista modelos disponíveis."""
    models = model_service.list_models()
    model_info = {}
    
    for model_name in models:
        model_info[model_name] = model_service.get_model_info(model_name)
    
    return {
        "available_models": models,
        "model_details": model_info,
        "total_models": len(models)
    }


@router.post("/models/{model_name}/load")
async def load_model(model_name: str, model_path: str):
    """Carrega um modelo específico."""
    try:
        success = model_service.load_h5_model(model_path, model_name)
        
        if success:
            return {"message": f"Modelo {model_name} carregado com sucesso"}
        else:
            raise HTTPException(status_code=400, detail="Falha ao carregar modelo")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Remove modelo da memória."""
    success = model_service.unload_model(model_name)
    
    if success:
        return {"message": f"Modelo {model_name} removido da memória"}
    else:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")


@router.post("/explain/{analysis_id}")
async def get_explanation(analysis_id: str, ecg_data: List[float], model_name: str):
    """Gera explicação detalhada para uma análise."""
    try:
        if model_name not in model_service.loaded_models:
            raise HTTPException(status_code=404, detail="Modelo não encontrado")
        
        model = model_service.loaded_models[model_name]
        ecg_array = np.array(ecg_data)
        
        explanation = explainability_service.generate_comprehensive_report(
            model, ecg_array, model_name
        )
        
        return {
            "analysis_id": analysis_id,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro na explicação: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Endpoints FHIR R4
@router.get("/fhir/Observation/{observation_id}")
async def get_fhir_observation(observation_id: str):
    """Retorna observação FHIR R4."""
    # Implementação simplificada - em produção, buscar do banco de dados
    return {
        "resourceType": "Observation",
        "id": observation_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "procedure",
                        "display": "Procedure"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "11524-6",
                    "display": "EKG study"
                }
            ]
        },
        "effectiveDateTime": datetime.now().isoformat(),
        "valueString": "ECG analysis completed"
    }


@router.post("/fhir/DiagnosticReport")
async def create_fhir_diagnostic_report(
    patient_id: str,
    analysis_results: Dict[str, Any]
):
    """Cria relatório diagnóstico FHIR R4."""
    report = {
        "resourceType": "DiagnosticReport",
        "id": f"ecg-report-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "CG",
                        "display": "Cardiology"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "11524-6",
                    "display": "EKG study"
                }
            ]
        },
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "effectiveDateTime": datetime.now().isoformat(),
        "conclusion": generate_clinical_conclusion(analysis_results)
    }
    
    return report


# --- NOVOS ENDPOINTS DE VALIDAÇÃO ---

@router.post("/validate", response_model=ValidationResponse)
async def validate_diagnosis(request: ValidationRequest):
    """
    Endpoint para cardiologistas validarem diagnósticos da IA.
    
    Este endpoint permite que cardiologistas forneçam feedback sobre
    a precisão dos diagnósticos gerados pela IA, criando um loop de
    feedback contínuo para melhoria do modelo.
    """
    try:
        # Preparar dados para salvar
        validation_data = {
            "ecg_id": request.ecg_id,
            "ai_diagnosis": request.ai_diagnosis,
            "ai_probability": request.ai_probability,
            "validator_id": request.validator_id,
            "is_correct": request.is_correct,
            "corrected_diagnosis": request.corrected_diagnosis,
            "comments": request.comments,
            "model_version": request.model_version,
            "confidence_score": request.confidence_score,
            "processing_time_ms": request.processing_time_ms,
            "timestamp": datetime.now().isoformat()
        }
        
        # Salvar validação
        success = validation_repo.save_validation(validation_data)
        
        if success:
            validation_id = f"val_{request.ecg_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Log da validação
            status = "CORRETO" if request.is_correct else "INCORRETO"
            logger.info(f"✅ Validação recebida: ECG {request.ecg_id} - {status} por {request.validator_id}")
            
            return ValidationResponse(
                success=True,
                message="Validação salva com sucesso. Obrigado pelo feedback!",
                validation_id=validation_id
            )
        else:
            raise HTTPException(status_code=500, detail="Erro ao salvar validação")
            
    except Exception as e:
        logger.error(f"❌ Erro no endpoint de validação: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get("/validations/statistics")
async def get_validation_statistics():
    """
    Retorna estatísticas das validações para monitoramento da performance do modelo.
    """
    try:
        stats = validation_repo.get_validation_statistics()
        
        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat(),
            "message": "Estatísticas calculadas com sucesso"
        }
        
    except Exception as e:
        logger.error(f"❌ Erro ao calcular estatísticas: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations")
async def get_validations(
    limit: Optional[int] = 50,
    validator_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Recupera validações com filtros opcionais.
    """
    try:
        validations = validation_repo.get_validations(
            limit=limit,
            validator_id=validator_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "validations": validations,
            "count": len(validations),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erro ao recuperar validações: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validations/feedback-for-retraining")
async def get_feedback_for_retraining(only_incorrect: bool = True):
    """
    Recupera feedback para re-treinamento do modelo.
    
    Args:
        only_incorrect: Se True, retorna apenas validações incorretas
    """
    try:
        feedback_data = validation_repo.get_feedback_for_retraining(only_incorrect)
        
        return {
            "feedback_data": feedback_data,
            "count": len(feedback_data),
            "only_incorrect": only_incorrect,
            "timestamp": datetime.now().isoformat(),
            "message": f"Dados de feedback recuperados: {len(feedback_data)} registros"
        }
        
    except Exception as e:
        logger.error(f"❌ Erro ao recuperar feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validations/export")
async def export_validations(format: str = "csv"):
    """
    Exporta validações em diferentes formatos.
    
    Args:
        format: Formato de exportação ('csv', 'json', 'excel')
    """
    try:
        if format not in ['csv', 'json', 'excel']:
            raise HTTPException(status_code=400, detail="Formato não suportado. Use: csv, json, excel")
        
        export_path = validation_repo.export_validations(format)
        
        return {
            "success": True,
            "export_path": export_path,
            "format": format,
            "timestamp": datetime.now().isoformat(),
            "message": f"Validações exportadas em formato {format}"
        }
        
    except Exception as e:
        logger.error(f"❌ Erro ao exportar validações: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- FIM DOS NOVOS ENDPOINTS DE VALIDAÇÃO ---


def create_fhir_observation(patient_id: str, prediction_result: Dict[str, Any], 
                           sampling_rate: int) -> Dict[str, Any]:
    """Cria observação FHIR R4 para ECG."""
    return {
        "resourceType": "Observation",
        "id": f"ecg-obs-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "procedure",
                        "display": "Procedure"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "11524-6",
                    "display": "EKG study"
                }
            ]
        },
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "effectiveDateTime": datetime.now().isoformat(),
        "valueQuantity": {
            "value": prediction_result.get("confidence", 0.0),
            "unit": "confidence_score",
            "system": "http://unitsofmeasure.org"
        },
        "component": [
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "8867-4",
                            "display": "Heart rate"
                        }
                    ]
                },
                "valueQuantity": {
                    "value": sampling_rate,
                    "unit": "Hz",
                    "system": "http://unitsofmeasure.org"
                }
            }
        ]
    }


def generate_clinical_conclusion(analysis_results: Dict[str, Any]) -> str:
    """Gera conclusão clínica baseada nos resultados."""
    confidence = analysis_results.get("confidence", 0.0)
    predictions = analysis_results.get("predictions", {})
    
    if confidence > 0.9:
        confidence_level = "alta confiança"
    elif confidence > 0.7:
        confidence_level = "confiança moderada"
    else:
        confidence_level = "baixa confiança"
    
    conclusion = f"Análise de ECG realizada com {confidence_level} (score: {confidence:.3f}). "
    
    if "predicted_classes" in predictions:
        classes = predictions["predicted_classes"]
        if classes:
            conclusion += f"Classe predita: {classes[0]}. "
    
    conclusion += "Recomenda-se revisão médica para validação dos resultados."
    
    return conclusion

