"""
CardioAI Pro - Aplicação Principal Unificada
Sistema completo de análise de ECG com IA
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
import logging
import os
import shutil
from pathlib import Path
from datetime import datetime
import uuid

# Importação do novo roteador para ECG Image Processing
from app.api.v1 import ecg_image_endpoints
# Importação do roteador de diagnósticos do sistema
from app.api.v1 import system_diagnostics

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
    logger.info("Iniciando CardioAI Pro...")
    
    # Inicializar serviços
    try:
        # Importar serviços unificados
        from app.services.unified_model_service import get_model_service
        from app.services.unified_ecg_service import get_ecg_service
        
        # Inicializar serviços
        model_service = get_model_service()
        ecg_service = get_ecg_service()
        
        logger.info("Serviços inicializados com sucesso")
        
        # Criar diretórios necessários
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("uploads", exist_ok=True)
        
        logger.info("CardioAI Pro iniciado com sucesso")
        
    except Exception as e:
        logger.error(f"Erro na inicialização: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Encerrando CardioAI Pro...")


# Criar aplicação FastAPI
app = FastAPI(
    title="CardioAI Pro",
    description="""
    Sistema Avançado de Análise de ECG com Inteligência Artificial
    
    ## Funcionalidades
    
    * **Análise de ECG**: Interpretação automática de eletrocardiogramas
    * **Modelos de IA**: Ensemble de modelos deep learning
    * **Explicabilidade**: Grad-CAM, SHAP e análise de importância
    * **FHIR R4**: Compatibilidade com padrões de interoperabilidade
    * **Incerteza Bayesiana**: Quantificação de confiança nas predições
    * **Auditoria**: Rastreamento completo de operações
    
    ## Arquitetura
    
    O sistema implementa uma arquitetura hierárquica multi-tarefa:
    
    1. **Camada de Aquisição**: Suporte a múltiplos formatos de ECG
    2. **Pré-processamento**: Filtragem digital e normalização
    3. **Modelos de IA**: CNNs, RNNs e Transformers
    4. **Explicabilidade**: Interpretação das decisões
    5. **Validação**: Sistema de incerteza e confiabilidade
    6. **Integração**: APIs FHIR para interoperabilidade
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar o roteador para ECG Image Processing
app.include_router(ecg_image_endpoints.router, prefix="/api/v1/ecg-image", tags=["ECG Image Processing"])

# Registrar o roteador de diagnósticos do sistema
app.include_router(system_diagnostics.router, prefix="/api/v1/system", tags=["System Diagnostics"])

# Servir arquivos estáticos
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Dependências
def get_model_service():
    from app.services.unified_model_service import get_model_service
    return get_model_service()

def get_ecg_service():
    from app.services.unified_ecg_service import get_ecg_service
    return get_ecg_service()


# Endpoints principais
@app.get("/")
async def root():
    """Endpoint raiz com informações do sistema."""
    return {
        "name": "CardioAI Pro",
        "version": "2.0.0",
        "description": "Sistema Avançado de Análise de ECG com IA",
        "status": "running",
        "features": [
            "Análise automática de ECG",
            "Modelos ensemble de deep learning",
            "Explicabilidade com Grad-CAM e SHAP",
            "Compatibilidade FHIR R4",
            "Sistema de incerteza bayesiana",
            "Auditoria completa",
            "APIs RESTful",
            "Interface web responsiva"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "api": "/api/v1",
            "ecg_analysis": "/api/v1/ecg/analyze",
            "models": "/api/v1/models",
            "fhir": "/api/v1/fhir"
        }
    }


@app.get("/health")
async def health_check(
    model_service = Depends(get_model_service)
):
    """Endpoint de health check."""
    try:
        # Verificar serviços
        models_available = len(model_service.list_models())
        
        return {
            "status": "healthy",
            "service": "CardioAI Pro",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "model_service": "running",
                "ecg_service": "running",
                "api_service": "running"
            },
            "models_loaded": models_available,
            "system_info": {
                "python_version": "3.11+",
                "tensorflow": "available" if hasattr(model_service, "TENSORFLOW_AVAILABLE") and model_service.TENSORFLOW_AVAILABLE else "unavailable",
                "pytorch": "available" if hasattr(model_service, "PYTORCH_AVAILABLE") and model_service.PYTORCH_AVAILABLE else "unavailable",
                "fastapi": "available"
            }
        }
    except Exception as e:
        logger.error(f"Erro no health check: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/info")
async def system_info(
    model_service = Depends(get_model_service)
):
    """Informações detalhadas do sistema."""
    try:
        models = model_service.list_models()
        model_details = {}
        
        for model_name in models:
            model_details[model_name] = model_service.get_model_info(model_name)
        
        return {
            "system": {
                "name": "CardioAI Pro",
                "version": "2.0.0",
                "description": "Sistema Avançado de Análise de ECG",
                "architecture": "Hierárquica Multi-tarefa"
            },
            "capabilities": {
                "ecg_formats": ["SCP-ECG", "DICOM", "HL7 aECG", "CSV", "TXT", "NPY", "EDF", "WFDB"],
                "sampling_rates": ["250-1000 Hz"],
                "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1-V6"],
                "ai_models": ["CNN-1D", "LSTM", "GRU", "Transformers", "Ensemble"],
                "explainability": ["Grad-CAM", "SHAP", "Feature Importance"],
                "standards": ["FHIR R4", "HL7", "DICOM"]
            },
            "models": {
                "loaded": models,
                "details": model_details,
                "total": len(models)
            },
            "performance": {
                "target_auc": "> 0.97",
                "inference_time": "< 1s",
                "batch_processing": "supported"
            }
        }
    except Exception as e:
        logger.error(f"Erro ao obter informações: {str(e)}")
        return {"error": str(e)}


# API v1 Endpoints
@app.post("/api/v1/ecg/upload")
async def upload_ecg(
    file: UploadFile = File(...),
    ecg_service = Depends(get_ecg_service)
):
    """
    Faz upload e processa um arquivo de ECG.
    
    Formatos suportados: CSV, TXT, NPY, DAT (WFDB), EDF, JSON
    """
    try:
        # Gerar nome de arquivo único
        file_extension = os.path.splitext(file.filename)[1].lower()
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_path = Path("uploads") / temp_filename
        
        # Salvar arquivo temporariamente
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Processar arquivo
        result = ecg_service.process_ecg_file(temp_path)
        
        # Limpar arquivo temporário
        os.remove(temp_path)
        
        return result
    
    except Exception as e:
        logger.error(f"Erro no upload de ECG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ecg/analyze/{process_id}")
async def analyze_ecg(
    process_id: str,
    model_name: Optional[str] = None,
    ecg_service = Depends(get_ecg_service)
):
    """
    Analisa um ECG previamente processado usando modelo de IA.
    
    Args:
        process_id: ID do processamento prévio
        model_name: Nome do modelo a usar (opcional)
    """
    try:
        result = ecg_service.analyze_ecg(process_id, model_name)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na análise de ECG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models")
async def list_models(
    model_service = Depends(get_model_service)
):
    """Lista todos os modelos disponíveis."""
    try:
        models = model_service.list_models()
        
        return {
            "models": models,
            "count": len(models),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Erro ao listar modelos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models/{model_name}")
async def get_model_info(
    model_name: str,
    model_service = Depends(get_model_service)
):
    """Obtém informações detalhadas sobre um modelo específico."""
    try:
        info = model_service.get_model_info(model_name)
        
        if "error" in info:
            raise HTTPException(status_code=404, detail=info["error"])
        
        return info
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Middleware para logging de requests
@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware para logging de requisições."""
    import time
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


# Handler de exceções
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler global de exceções."""
    logger.error(f"Erro não tratado: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erro interno do servidor",
            "detail": str(exc),
            "path": str(request.url.path)
        }
    )


class CardioAIApp:
    """Classe principal da aplicação CardioAI."""
    
    def __init__(self):
        self.name = "CardioAI Pro"
        self.version = "2.0.0"
        self.description = "Sistema Avançado de Análise de ECG com IA"
        self.status = "initialized"
        self.modules = [
            "unified_model_service",
            "unified_ecg_service",
            "api_service",
            "fhir_integration"
        ]
        
    def get_info(self) -> Dict[str, Any]:
        """Retorna informações da aplicação."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "status": self.status,
            "modules": self.modules,
            "architecture": {
                "layers": [
                    "Aquisição e Pré-processamento",
                    "Modelos de IA Hierárquicos",
                    "Extração de Características",
                    "Validação e Confiabilidade",
                    "Integração e APIs",
                    "Interface de Usuário"
                ],
                "compliance": ["FHIR R4", "HIPAA", "LGPD", "ISO 13485"]
            }
        }
    
    def start(self):
        """Inicia a aplicação."""
        self.status = "running"
        logger.info(f"{self.name} v{self.version} iniciado com sucesso")
        
    def stop(self):
        """Para a aplicação."""
        self.status = "stopped"
        logger.info(f"{self.name} parado")


# Instância global da aplicação
cardio_app = CardioAIApp()


if __name__ == "__main__":
    import uvicorn
    
    cardio_app.start()
    
    # Configurações de produção
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=False  # Desabilitar em produção
    )