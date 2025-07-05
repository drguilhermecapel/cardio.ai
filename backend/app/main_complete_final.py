# backend/app/main_complete_final.py
import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
import numpy as np

# Adicionar o diretório do projeto ao sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.unified_ecg_service import UnifiedECGService
from app.services.ptbxl_model_service_production import PTBXLModelServiceProduction
from app.services.ecg_digitizer import ECGDigitizer
from app.schemas.ecg import ECGSignal, ECGDiagnosis
from app.preprocessing.advanced_pipeline import AdvancedPipeline
from app.utils.file_utils import save_upload_file_tmp
from app.utils.ecg_processor import ECGProcessor

# Configuração do Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialização da aplicação FastAPI
app = FastAPI(
    title="Cardio.AI - Professional ECG Analysis Platform",
    description="Sistema avançado para análise de eletrocardiogramas utilizando IA.",
    version="2.0.0"
)

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Injeção de Dependência ---
# Carregar o modelo de produção uma única vez
try:
    ptbxl_model_service = PTBXLModelServiceProduction()
    logger.info("Modelo PTB-XL de produção carregado com sucesso.")
except Exception as e:
    logger.error(f"Erro crítico ao carregar o modelo de produção: {e}")
    # A aplicação não deve iniciar sem o modelo.
    sys.exit(1)

# Instanciar os outros serviços
ecg_digitizer = ECGDigitizer()
advanced_pipeline = AdvancedPipeline()
ecg_processor = ECGProcessor()

# Instanciar o serviço unificado com as dependências
unified_ecg_service = UnifiedECGService(
    digitizer=ecg_digitizer,
    pipeline=advanced_pipeline,
    model_service=ptbxl_model_service,
    ecg_processor=ecg_processor
)

# --- Endpoints da API ---

@app.post("/api/v2/ecg/diagnose_signal", response_model=ECGDiagnosis)
async def diagnose_ecg_signal(ecg_signal: ECGSignal):
    """
    Recebe dados de sinal de ECG em formato JSON e retorna o diagnóstico.
    """
    try:
        logger.info(f"Recebida solicitação de diagnóstico para o paciente: {ecg_signal.patient_id}")
        diagnosis = await unified_ecg_service.diagnose_from_signal(ecg_signal.data)
        return diagnosis
    except Exception as e:
        logger.error(f"Erro no endpoint diagnose_signal: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")

@app.post("/api/v2/ecg/diagnose_image", response_model=ECGDiagnosis)
async def diagnose_ecg_image(file: UploadFile = File(...)):
    """
    Recebe um arquivo de imagem de ECG, digitaliza o sinal e retorna o diagnóstico.
    """
    try:
        logger.info(f"Recebida imagem de ECG para diagnóstico: {file.filename}")
        
        # Salvar o arquivo temporariamente
        temp_file_path = save_upload_file_tmp(file)
        
        # Processar a imagem para obter o diagnóstico
        diagnosis = await unified_ecg_service.diagnose_from_image(str(temp_file_path))
        
        # Limpar o arquivo temporário
        os.remove(temp_file_path)
        
        return diagnosis
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Erro no endpoint diagnose_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Não foi possível processar a imagem do ECG: {e}")

@app.get("/health")
async def health_check():
    """
    Endpoint para verificação de saúde do serviço.
    """
    return {"status": "ok", "message": "Cardio.AI Service is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

