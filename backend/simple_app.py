#!/usr/bin/env python3
"""
Aplicação CardioAI simplificada para deploy
"""

import os
import sys
import logging
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import uuid

# Configurar paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models"))

# Importações FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar funções melhoradas
try:
    from preprocess_functions_improved import (
        preprocess_ecg_signal,
        extract_ecg_from_image,
        validate_ecg_signal,
        prepare_for_model,
        get_diagnosis_mapping
    )
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Erro ao importar funções de pré-processamento: {e}")
    PREPROCESSING_AVAILABLE = False

# Importar TensorFlow
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Criar aplicação FastAPI
app = FastAPI(
    title="CardioAI Pro",
    description="Sistema Avançado de Análise de ECG com IA",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variáveis globais
model = None
diagnosis_mapping = {}
processed_ecgs = {}

def load_model():
    """Carrega o modelo ECG."""
    global model, diagnosis_mapping
    
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow não disponível")
        return False
    
    try:
        # Tentar carregar modelo
        model_paths = [
            "models/ecg_model_final.h5",
            "../models/ecg_model_final.h5",
            "ecg_model_final.h5"
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                logger.info(f"Modelo carregado de: {model_path}")
                break
        
        if model is None:
            logger.error("Nenhum modelo encontrado")
            return False
        
        # Carregar mapeamento de diagnósticos
        if PREPROCESSING_AVAILABLE:
            diagnosis_mapping = get_diagnosis_mapping()
        else:
            diagnosis_mapping = {
                0: "Normal",
                1: "Anormalidade Detectada"
            }
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Evento de inicialização."""
    logger.info("Iniciando CardioAI Pro...")
    success = load_model()
    if success:
        logger.info("Sistema inicializado com sucesso")
    else:
        logger.warning("Sistema iniciado com funcionalidade limitada")

@app.get("/")
async def root():
    """Endpoint raiz."""
    return {
        "name": "CardioAI Pro",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "preprocessing_available": PREPROCESSING_AVAILABLE,
        "tensorflow_available": TENSORFLOW_AVAILABLE
    }

@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "preprocessing_available": PREPROCESSING_AVAILABLE,
        "tensorflow_available": TENSORFLOW_AVAILABLE
    }

@app.post("/api/v1/ecg/upload")
async def upload_ecg(file: UploadFile = File(...)):
    """Upload e processamento de ECG."""
    try:
        if not PREPROCESSING_AVAILABLE:
            return {"error": "Pré-processamento não disponível"}
        
        # Gerar ID único
        process_id = str(uuid.uuid4())
        
        # Salvar arquivo temporariamente
        file_extension = os.path.splitext(file.filename)[1].lower()
        temp_filename = f"temp_{process_id}{file_extension}"
        
        with open(temp_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Processar baseado no tipo
        if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Processar imagem
            signal = extract_ecg_from_image(temp_filename)
        elif file_extension in ['.csv', '.txt']:
            # Processar dados
            data = np.loadtxt(temp_filename, delimiter=',')
            signal = preprocess_ecg_signal(data)
        elif file_extension == '.npy':
            data = np.load(temp_filename)
            signal = preprocess_ecg_signal(data)
        else:
            os.remove(temp_filename)
            return {"error": f"Formato não suportado: {file_extension}"}
        
        # Validar sinal
        is_valid, msg = validate_ecg_signal(signal)
        
        # Armazenar resultado
        processed_ecgs[process_id] = {
            "signal": signal,
            "is_valid": is_valid,
            "validation_message": msg,
            "timestamp": datetime.now().isoformat()
        }
        
        # Limpar arquivo temporário
        os.remove(temp_filename)
        
        return {
            "process_id": process_id,
            "status": "success",
            "signal_shape": signal.shape,
            "is_valid": is_valid,
            "validation_message": msg
        }
        
    except Exception as e:
        logger.error(f"Erro no upload: {e}")
        return {"error": str(e)}

@app.post("/api/v1/ecg/analyze/{process_id}")
async def analyze_ecg(process_id: str):
    """Análise de ECG."""
    try:
        if process_id not in processed_ecgs:
            raise HTTPException(status_code=404, detail="Processamento não encontrado")
        
        if model is None:
            return {"error": "Modelo não carregado"}
        
        # Obter sinal processado
        data = processed_ecgs[process_id]
        signal = data["signal"]
        
        # Preparar para modelo
        model_input = prepare_for_model(signal)
        
        # Fazer predição
        predictions = model.predict(model_input, verbose=0)[0]
        
        # Interpretar resultados
        top_indices = np.argsort(predictions)[-5:][::-1]
        
        diagnoses = []
        for idx in top_indices:
            if idx < len(diagnosis_mapping) and predictions[idx] > 0.1:
                diagnoses.append({
                    "condition": diagnosis_mapping.get(idx, f"Classe {idx}"),
                    "probability": float(predictions[idx]),
                    "confidence": "high" if predictions[idx] > 0.7 else "medium" if predictions[idx] > 0.3 else "low"
                })
        
        if not diagnoses:
            diagnoses.append({
                "condition": "Normal",
                "probability": 1.0 - np.max(predictions),
                "confidence": "medium"
            })
        
        return {
            "process_id": process_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "diagnoses": diagnoses,
            "signal_quality": data.get("is_valid", False)
        }
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        return {"error": str(e)}

@app.get("/api/v1/models")
async def list_models():
    """Lista modelos disponíveis."""
    return {
        "models": ["ecg_model_final"] if model is not None else [],
        "count": 1 if model is not None else 0
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

