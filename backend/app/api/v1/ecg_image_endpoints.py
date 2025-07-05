# backend/app/api/v1/ecg_image_endpoints.py
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from typing import Dict, Any
from app.services.ecg_digitizer_service import ecg_digitizer_service, ECGDigitizerService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/digitize", response_model=Dict[str, Any])
async def digitize_ecg_from_image(
    file: UploadFile = File(..., description="Arquivo de imagem de ECG (PNG, JPG) para ser digitalizado."),
    digitizer: ECGDigitizerService = Depends(lambda: ecg_digitizer_service)
):
    """
    Endpoint para receber uma imagem de ECG, digitalizá-la e retornar os dados do sinal.
    
    Este endpoint aceita um arquivo de imagem e retorna uma estrutura JSON com os dados
    do sinal de 12 derivações, a taxa de amostragem e os nomes das derivações.
    """
    logger.info(f"Recebido arquivo '{file.filename}' para digitalização (Content-Type: {file.content_type}).")
    
    # Validação robusta do tipo de conteúdo
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Tipo de arquivo inválido recebido: {file.content_type}")
        raise HTTPException(status_code=415, detail="Tipo de mídia não suportado. Apenas imagens são permitidas.")

    try:
        image_bytes = await file.read()
        
        if not image_bytes:
            logger.warning(f"Arquivo '{file.filename}' enviado está vazio.")
            raise HTTPException(status_code=400, detail="O arquivo de imagem enviado está vazio.")

        # Chama o serviço para processar a imagem
        digitized_data = digitizer.digitize_image(image_bytes)
        
        logger.info(f"Arquivo '{file.filename}' digitalizado com sucesso.")
        return digitized_data

    except HTTPException as e:
        # Repassa a exceção HTTP do serviço para o cliente com o status code e detalhe corretos.
        raise e
    except Exception as e:
        logger.exception(f"Erro crítico ao processar o arquivo '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro inesperado no servidor ao processar a imagem.")

