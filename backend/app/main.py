"""
CardioAI Pro - Sistema Completo de Análise de ECG
Aplicação principal integrada com todos os componentes
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any
import logging
import sys
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador de ciclo de vida da aplicação."""
    # Startup
    logger.info("Starting up CardioAI Pro - Sistema Completo...")
    
    # Inicializar interpretador de ECG
    try:
        from app.services.ecg_interpreter import ecg_interpreter
        ecg_interpreter.load_model()
        logger.info("Interpretador de ECG inicializado com sucesso")
    except Exception as e:
        logger.warning(f"Erro ao inicializar interpretador: {e}")
    
    # Inicializar outros serviços
    try:
        from app.services.advanced_ml_service import AdvancedMLService
        logger.info("Serviços de ML avançados carregados")
    except Exception as e:
        logger.warning(f"Erro ao carregar serviços ML: {e}")
    
    yield
    # Shutdown
    logger.info("Shutting down CardioAI Pro...")


# Criar aplicação FastAPI
app = FastAPI(
    title="CardioAI Pro - Sistema Completo",
    description="Sistema Avançado e Completo de Análise e Interpretação de ECG com Inteligência Artificial",
    version="2.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_app_info() -> Dict[str, Any]:
    """Retorna informações da aplicação."""
    return {
        "name": "CardioAI Pro - Sistema Completo",
        "version": "2.0.0",
        "description": "Sistema Avançado e Completo de Análise e Interpretação de ECG com IA",
        "status": "running",
        "features": [
            "Análise automática de ECG",
            "Detecção de arritmias",
            "Interpretação com IA",
            "Detecção de anormalidades",
            "Relatórios detalhados",
            "API REST completa",
            "Serviços ML avançados",
            "Sistema de monitoramento",
            "Validação clínica",
            "Segurança e auditoria",
            "Processamento híbrido",
            "Explicabilidade de IA"
        ],
        "modules": [
            "ECG Interpreter",
            "Advanced ML Service", 
            "Hybrid ECG Service",
            "Multi-Pathology Service",
            "Interpretability Service",
            "Validation Service",
            "Notification Service",
            "Patient Service",
            "User Service",
            "Security & Audit",
            "Quality Monitoring",
            "Data Processing"
        ],
        "endpoints": [
            "/ecg/analyze - Análise de ECG",
            "/ecg/analyze-file - Upload e análise de arquivo",
            "/ecg/sample-analysis - Análise de exemplo",
            "/ecg/status - Status do interpretador",
            "/health - Health check",
            "/info - Informações do sistema"
        ]
    }


async def health_check() -> Dict[str, str]:
    """Endpoint de health check."""
    return {
        "status": "healthy",
        "service": "CardioAI Pro - Sistema Completo",
        "version": "2.0.0"
    }


# Endpoints principais
@app.get("/")
async def root():
    """Endpoint raiz."""
    return await get_app_info()


@app.get("/health")
async def health():
    """Endpoint de health check."""
    return await health_check()


@app.get("/info")
async def info():
    """Endpoint de informações da aplicação."""
    return await get_app_info()


@app.get("/status")
async def system_status():
    """Status detalhado do sistema."""
    status = {
        "system": "CardioAI Pro",
        "version": "2.0.0",
        "status": "running",
        "components": {}
    }
    
    # Status do interpretador de ECG
    try:
        from app.services.ecg_interpreter import ecg_interpreter
        status["components"]["ecg_interpreter"] = ecg_interpreter.get_status()
    except Exception as e:
        status["components"]["ecg_interpreter"] = {"status": "error", "error": str(e)}
    
    # Status de outros serviços
    try:
        from app.services.advanced_ml_service import AdvancedMLService
        status["components"]["advanced_ml"] = {"status": "available"}
    except Exception as e:
        status["components"]["advanced_ml"] = {"status": "error", "error": str(e)}
    
    return status


# Incluir API do ECG
try:
    from app.api.ecg_api import router as ecg_router
    app.include_router(ecg_router, prefix="/ecg", tags=["ECG Analysis"])
    logger.info("API de ECG incluída com sucesso")
except ImportError as e:
    logger.warning(f"Não foi possível incluir API de ECG: {e}")


# Incluir outras APIs se disponíveis
try:
    from app.api.ml_api import router as ml_router
    app.include_router(ml_router, prefix="/ml", tags=["Machine Learning"])
    logger.info("API de ML incluída")
except ImportError:
    logger.info("API de ML não disponível")

try:
    from app.api.patient_api import router as patient_router
    app.include_router(patient_router, prefix="/patients", tags=["Patients"])
    logger.info("API de Pacientes incluída")
except ImportError:
    logger.info("API de Pacientes não disponível")


class CardioAIApp:
    """Classe principal da aplicação CardioAI completa."""
    
    def __init__(self):
        self.name = "CardioAI Pro - Sistema Completo"
        self.version = "2.0.0"
        self.description = "Sistema Avançado e Completo de Análise e Interpretação de ECG com IA"
        self.status = "initialized"
        self.modules = [
            "ECG Interpreter", "Advanced ML Service", "Hybrid ECG Service",
            "Multi-Pathology Service", "Interpretability Service", 
            "Validation Service", "Notification Service", "Patient Service",
            "User Service", "Security & Audit", "Quality Monitoring"
        ]
        
    def get_info(self) -> Dict[str, Any]:
        """Retorna informações da aplicação."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "status": self.status,
            "modules": self.modules
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
    logger.info("Iniciando servidor CardioAI Pro - Sistema Completo...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

