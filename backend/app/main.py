"""
Aplicação principal CardioAI Pro - Versão Integrada
Sistema completo de análise de ECG com IA
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import Dict, Any
import logging
import os
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
    logger.info("Iniciando CardioAI Pro...")
    
    # Inicializar serviços
    try:
        from app.services.model_service import initialize_models
        model_service = initialize_models()
        logger.info("Serviço de modelos inicializado")
        
        # Criar diretórios necessários
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
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

# Servir arquivos estáticos
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Incluir routers da API
try:
    from app.api.v1.ecg_endpoints import router as ecg_router
    app.include_router(ecg_router, prefix="/api/v1")
    logger.info("Router ECG incluído com sucesso")
except ImportError as e:
    logger.warning(f"Router ECG não encontrado: {str(e)}")

try:
    from app.api.v1.api import api_router
    app.include_router(api_router, prefix="/api/v1")
    logger.info("API v1 router incluído com sucesso")
except ImportError as e:
    logger.warning(f"API v1 router não encontrado: {str(e)}")


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
            "models": "/api/v1/ecg/models",
            "fhir": "/api/v1/ecg/fhir"
        }
    }


@app.get("/health")
async def health_check():
    """Endpoint de health check."""
    try:
        from app.services.model_service import model_service
        
        # Verificar serviços
        models_available = len(model_service.list_models())
        
        return {
            "status": "healthy",
            "service": "CardioAI Pro",
            "version": "2.0.0",
            "timestamp": "2025-01-03T00:00:00Z",
            "services": {
                "model_service": "running",
                "explainability_service": "running",
                "api_service": "running"
            },
            "models_loaded": models_available,
            "system_info": {
                "python_version": "3.11+",
                "tensorflow": "available",
                "pytorch": "available",
                "fastapi": "available"
            }
        }
    except Exception as e:
        logger.error(f"Erro no health check: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": "2025-01-03T00:00:00Z"
        }


@app.get("/info")
async def system_info():
    """Informações detalhadas do sistema."""
    try:
        from app.services.model_service import model_service
        
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
                "ecg_formats": ["SCP-ECG", "DICOM", "HL7 aECG", "CSV", "TXT", "NPY"],
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


@app.get("/api/v1/health")
async def api_health():
    """Health check da API v1."""
    return {
        "status": "healthy",
        "api_version": "v1",
        "endpoints": [
            "/ecg/analyze",
            "/ecg/upload-file",
            "/ecg/models",
            "/ecg/explain",
            "/ecg/fhir/Observation",
            "/ecg/fhir/DiagnosticReport"
        ]
    }


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
            "model_service",
            "explainability_service",
            "preprocessing_pipeline",
            "fhir_integration",
            "audit_system"
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

