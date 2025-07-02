"""
Aplicação principal CardioAI Pro - Interpretador de ECG com IA
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerenciador de ciclo de vida da aplicação."""
    # Startup
    logger.info("Starting up CardioAI Pro...")
    
    # Inicializar interpretador de ECG
    from app.services.ecg_interpreter import ecg_interpreter
    ecg_interpreter.load_model()
    
    yield
    # Shutdown
    logger.info("Shutting down CardioAI Pro...")


# Criar aplicação FastAPI
app = FastAPI(
    title="CardioAI Pro - Interpretador de ECG",
    description="Sistema Avançado de Análise e Interpretação de ECG com Inteligência Artificial",
    version="1.0.0",
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
        "name": "CardioAI Pro",
        "version": "1.0.0",
        "description": "Sistema Avançado de Análise e Interpretação de ECG com Inteligência Artificial",
        "status": "running",
        "features": [
            "Análise automática de ECG",
            "Detecção de arritmias",
            "Interpretação com IA",
            "Detecção de anormalidades",
            "Relatórios detalhados",
            "API REST completa"
        ],
        "endpoints": [
            "/ecg/analyze - Análise de ECG",
            "/ecg/analyze-file - Upload e análise de arquivo",
            "/ecg/sample-analysis - Análise de exemplo",
            "/ecg/status - Status do interpretador"
        ]
    }


async def health_check() -> Dict[str, str]:
    """Endpoint de health check."""
    return {
        "status": "healthy",
        "service": "CardioAI Pro",
        "version": "1.0.0"
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


# Incluir API do ECG
try:
    from app.api.ecg_api import router as ecg_router
    app.include_router(ecg_router, prefix="/ecg", tags=["ECG Analysis"])
    logger.info("API de ECG incluída com sucesso")
except ImportError as e:
    logger.warning(f"Não foi possível incluir API de ECG: {e}")


class CardioAIApp:
    """Classe principal da aplicação CardioAI."""
    
    def __init__(self):
        self.name = "CardioAI Pro"
        self.version = "1.0.0"
        self.description = "Sistema Avançado de Análise e Interpretação de ECG com IA"
        self.status = "initialized"
        self.modules = ["ECG Interpreter", "API REST", "ML Models"]
        
    def get_info(self) -> Dict[str, str]:
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
    logger.info("Iniciando servidor CardioAI Pro...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

