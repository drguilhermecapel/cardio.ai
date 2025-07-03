"""
Aplicação CardioAI Pro - Versão Lite
Sistema de análise de ECG sem dependências pesadas
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import logging
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador de ciclo de vida da aplicação."""
    # Startup
    logger.info("Iniciando CardioAI Pro (Versão Lite)...")
    
    try:
        # Inicializar serviços lite
        from app.services.model_service_lite import initialize_models_lite
        model_service = initialize_models_lite()
        logger.info("Serviço de modelos lite inicializado")
        
        # Criar diretórios necessários
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logger.info("CardioAI Pro (Lite) iniciado com sucesso")
        
    except Exception as e:
        logger.error(f"Erro na inicialização: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Encerrando CardioAI Pro (Lite)...")


# Criar aplicação FastAPI
app = FastAPI(
    title="CardioAI Pro (Lite)",
    description="""
    Sistema de Análise de ECG com Inteligência Artificial - Versão Lite
    
    ## Funcionalidades Disponíveis
    
    * **Análise de ECG**: Interpretação automática com modelos simplificados
    * **Upload de Arquivos**: Suporte a CSV, TXT, NPY
    * **FHIR R4**: Compatibilidade com padrões médicos
    * **APIs RESTful**: Endpoints para integração
    * **Documentação**: Swagger UI e ReDoc
    
    ## Versão Lite
    
    Esta versão utiliza modelos simplificados baseados em scikit-learn
    para demonstração e desenvolvimento rápido, sem dependências pesadas
    como TensorFlow ou PyTorch.
    
    ## Endpoints Principais
    
    * `/api/v1/ecg/analyze` - Análise de dados ECG
    * `/api/v1/ecg/upload-file` - Upload de arquivo ECG
    * `/api/v1/ecg/models` - Listar modelos disponíveis
    * `/api/v1/ecg/fhir/observation` - Criar observação FHIR
    """,
    version="2.0.0-lite",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints principais
@app.get("/")
async def root():
    """Endpoint raiz com informações do sistema."""
    return {
        "name": "CardioAI Pro (Lite)",
        "version": "2.0.0-lite",
        "description": "Sistema de Análise de ECG com IA - Versão Lite",
        "status": "running",
        "mode": "lite",
        "features": [
            "Análise de ECG com modelos simplificados",
            "Upload de arquivos (CSV, TXT, NPY)",
            "Compatibilidade FHIR R4",
            "APIs RESTful",
            "Documentação interativa",
            "Processamento em tempo real"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "info": "/info",
            "analyze": "/api/v1/ecg/analyze",
            "upload": "/api/v1/ecg/upload-file",
            "models": "/api/v1/ecg/models",
            "fhir": "/api/v1/ecg/fhir"
        },
        "note": "Versão lite para demonstração - sem dependências pesadas"
    }


@app.get("/health")
async def health_check():
    """Endpoint de health check."""
    try:
        from app.services.model_service_lite import model_service_lite
        
        models = model_service_lite.list_models()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0-lite",
            "services": {
                "model_service": "running",
                "models_loaded": len(models),
                "available_models": models
            },
            "system": {
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                "working_directory": os.getcwd(),
                "memory_usage": "lite"
            }
        }
    except Exception as e:
        logger.error(f"Erro no health check: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/info")
async def system_info():
    """Informações detalhadas do sistema."""
    try:
        from app.services.model_service_lite import model_service_lite
        
        models = model_service_lite.list_models()
        model_info = {}
        
        for model_name in models:
            info = model_service_lite.get_model_info(model_name)
            model_info[model_name] = info
        
        return {
            "system": {
                "name": "CardioAI Pro",
                "version": "2.0.0-lite",
                "mode": "lite",
                "description": "Sistema de análise de ECG com IA",
                "startup_time": datetime.now().isoformat()
            },
            "capabilities": {
                "ecg_analysis": True,
                "file_upload": True,
                "fhir_compatibility": True,
                "real_time_processing": True,
                "batch_processing": True,
                "model_ensemble": False,  # Lite version
                "deep_learning": False,   # Lite version
                "explainability": False   # Lite version
            },
            "models": {
                "total": len(models),
                "available": models,
                "details": model_info
            },
            "supported_formats": [
                "CSV", "TXT", "NPY", "JSON"
            ],
            "api_version": "v1",
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc"
            }
        }
    except Exception as e:
        logger.error(f"Erro ao obter informações: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# API v1 - ECG Endpoints
@app.post("/api/v1/ecg/analyze")
async def analyze_ecg(
    patient_id: str = Form(...),
    ecg_data: str = Form(...),
    sampling_rate: int = Form(500),
    leads: Optional[str] = Form("I")
):
    """Analisa dados de ECG."""
    try:
        from app.services.model_service_lite import model_service_lite
        from app.schemas.fhir import create_ecg_observation
        
        # Parse dos dados ECG
        try:
            if ecg_data.startswith('[') and ecg_data.endswith(']'):
                # JSON array
                ecg_array = np.array(json.loads(ecg_data))
            else:
                # Valores separados por vírgula
                ecg_array = np.array([float(x.strip()) for x in ecg_data.split(',')])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Formato de dados ECG inválido: {str(e)}")
        
        # Obter modelos disponíveis
        models = model_service_lite.list_models()
        if not models:
            raise HTTPException(status_code=503, detail="Nenhum modelo disponível")
        
        # Usar primeiro modelo disponível
        model_name = models[0]
        
        # Realizar análise
        result = model_service_lite.predict_ecg(model_name, ecg_array)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Criar observação FHIR
        observation = create_ecg_observation(
            patient_id, 
            ecg_data, 
            sampling_rate, 
            result
        )
        
        return {
            "patient_id": patient_id,
            "analysis_id": f"analysis_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "model_used": model_name,
            "sampling_rate": sampling_rate,
            "leads": leads.split(',') if leads else ["I"],
            "results": result,
            "fhir_observation": {
                "id": observation.id,
                "status": observation.status.value,
                "resource_type": observation.resourceType
            },
            "recommendations": {
                "confidence_level": "high" if result["confidence"] > 0.8 else "moderate" if result["confidence"] > 0.6 else "low",
                "clinical_review": result["confidence"] < 0.7,
                "follow_up": result["confidence"] < 0.5
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise de ECG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ecg/upload-file")
async def upload_ecg_file(
    patient_id: str = Form(...),
    sampling_rate: int = Form(500),
    file: UploadFile = File(...)
):
    """Upload e análise de arquivo ECG."""
    try:
        from app.services.model_service_lite import model_service_lite
        
        # Verificar tipo de arquivo
        if not file.filename.lower().endswith(('.csv', '.txt', '.npy')):
            raise HTTPException(status_code=400, detail="Formato de arquivo não suportado. Use CSV, TXT ou NPY.")
        
        # Ler conteúdo do arquivo
        content = await file.read()
        
        # Parse baseado na extensão
        if file.filename.lower().endswith('.npy'):
            ecg_array = np.load(io.BytesIO(content))
        elif file.filename.lower().endswith('.csv'):
            # Assumir primeira coluna como dados ECG
            import io
            import pandas as pd
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            ecg_array = df.iloc[:, 0].values
        else:  # .txt
            # Assumir valores separados por linha ou vírgula
            text_content = content.decode('utf-8')
            if ',' in text_content:
                ecg_array = np.array([float(x.strip()) for x in text_content.split(',')])
            else:
                ecg_array = np.array([float(x.strip()) for x in text_content.split('\n') if x.strip()])
        
        # Obter modelos disponíveis
        models = model_service_lite.list_models()
        if not models:
            raise HTTPException(status_code=503, detail="Nenhum modelo disponível")
        
        # Realizar análise
        model_name = models[0]
        result = model_service_lite.predict_ecg(model_name, ecg_array)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "patient_id": patient_id,
            "file_info": {
                "filename": file.filename,
                "size": len(content),
                "samples": len(ecg_array)
            },
            "analysis_id": f"file_analysis_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "model_used": model_name,
            "sampling_rate": sampling_rate,
            "results": result,
            "status": "completed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro no upload de arquivo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ecg/models")
async def list_models():
    """Lista modelos disponíveis."""
    try:
        from app.services.model_service_lite import model_service_lite
        
        models = model_service_lite.list_models()
        model_details = {}
        
        for model_name in models:
            info = model_service_lite.get_model_info(model_name)
            model_details[model_name] = info
        
        return {
            "total_models": len(models),
            "available_models": models,
            "model_details": model_details,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ecg/fhir/observation")
async def create_fhir_observation(
    patient_id: str = Form(...),
    ecg_data: str = Form(...),
    sampling_rate: int = Form(500),
    analysis_results: Optional[str] = Form(None)
):
    """Cria observação FHIR para ECG."""
    try:
        from app.schemas.fhir import create_ecg_observation
        
        # Parse dos resultados de análise se fornecidos
        results = {}
        if analysis_results:
            try:
                results = json.loads(analysis_results)
            except:
                results = {"confidence": 0.5, "predicted_class": 0}
        
        # Criar observação FHIR
        observation = create_ecg_observation(patient_id, ecg_data, sampling_rate, results)
        
        return {
            "fhir_observation": {
                "resourceType": observation.resourceType,
                "id": observation.id,
                "status": observation.status.value,
                "category": [cat.dict() for cat in observation.category],
                "code": observation.code.dict(),
                "subject": observation.subject.dict(),
                "effectiveDateTime": observation.effectiveDateTime,
                "valueQuantity": observation.valueQuantity.dict() if observation.valueQuantity else None
            },
            "created_at": datetime.now().isoformat(),
            "patient_id": patient_id
        }
        
    except Exception as e:
        logger.error(f"Erro ao criar observação FHIR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

