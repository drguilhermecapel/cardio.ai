#!/usr/bin/env python3
"""
Aplicação CardioAI corrigida - Separação adequada de processamento
"""

import os
import sys
import logging
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
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

# Importar funções melhoradas v2
try:
    from preprocess_functions_v2 import (
        ECGImageProcessor,
        ECGDataProcessor,
        validate_ecg_signal,
        prepare_for_model,
        get_diagnosis_mapping
    )
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Erro ao importar funções de pré-processamento v2: {e}")
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
    version="2.1.0"
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

# Configurações de tipos de arquivo
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
DATA_EXTENSIONS = ['.csv', '.txt', '.npy', '.dat']

def convert_numpy_types(obj):
    """Converte tipos numpy para tipos Python nativos para serialização JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

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

def process_image_file(file_path: str) -> Dict[str, Any]:
    """Processa especificamente arquivos de imagem ECG."""
    try:
        logger.info(f"Processando imagem ECG: {file_path}")
        
        if not PREPROCESSING_AVAILABLE:
            raise ValueError("Funções de pré-processamento não disponíveis")
        
        # Usar processador específico para imagens
        image_processor = ECGImageProcessor()
        signal = image_processor.extract_ecg_from_image(file_path)
        
        if signal is None or signal.size == 0:
            raise ValueError("Não foi possível extrair sinal ECG da imagem")
        
        # Validar sinal extraído
        is_valid, msg = validate_ecg_signal(signal)
        
        return {
            "signal": signal,
            "is_valid": is_valid,
            "validation_message": msg,
            "processing_type": "image",
            "signal_shape": signal.shape
        }
        
    except Exception as e:
        logger.error(f"Erro no processamento de imagem: {e}")
        raise ValueError(f"Erro no processamento de imagem: {str(e)}")

def process_data_file(file_path: str, file_extension: str) -> Dict[str, Any]:
    """Processa especificamente arquivos de dados ECG."""
    try:
        logger.info(f"Processando dados ECG: {file_path}")
        
        if not PREPROCESSING_AVAILABLE:
            raise ValueError("Funções de pré-processamento não disponíveis")
        
        # Carregar dados baseado na extensão
        if file_extension in ['.csv', '.txt']:
            try:
                # Tentar diferentes delimitadores
                for delimiter in [',', ';', '\t', ' ']:
                    try:
                        data = np.loadtxt(file_path, delimiter=delimiter)
                        if data.size > 0:
                            break
                    except:
                        continue
                else:
                    raise ValueError("Não foi possível ler o arquivo de dados")
                    
            except Exception as e:
                raise ValueError(f"Erro ao ler arquivo CSV/TXT: {str(e)}")
                
        elif file_extension == '.npy':
            try:
                data = np.load(file_path)
            except Exception as e:
                raise ValueError(f"Erro ao ler arquivo NPY: {str(e)}")
        else:
            raise ValueError(f"Extensão de dados não suportada: {file_extension}")
        
        # Usar processador específico para dados
        data_processor = ECGDataProcessor()
        signal = data_processor.preprocess_ecg_signal(data)
        
        if signal is None or signal.size == 0:
            raise ValueError("Não foi possível processar os dados ECG")
        
        # Validar sinal processado
        is_valid, msg = validate_ecg_signal(signal)
        
        return {
            "signal": signal,
            "is_valid": is_valid,
            "validation_message": msg,
            "processing_type": "data",
            "signal_shape": signal.shape,
            "original_shape": data.shape
        }
        
    except Exception as e:
        logger.error(f"Erro no processamento de dados: {e}")
        raise ValueError(f"Erro no processamento de dados: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Evento de inicialização."""
    logger.info("Iniciando CardioAI Pro v2.1...")
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
        "version": "2.1.0",
        "status": "running",
        "model_loaded": model is not None,
        "preprocessing_available": PREPROCESSING_AVAILABLE,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "supported_image_formats": IMAGE_EXTENSIONS,
        "supported_data_formats": DATA_EXTENSIONS
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
    """Upload e processamento de ECG com separação adequada de tipos."""
    try:
        if not PREPROCESSING_AVAILABLE:
            return {"error": "Pré-processamento não disponível"}
        
        # Gerar ID único
        process_id = str(uuid.uuid4())
        
        # Validar arquivo
        if not file.filename:
            return {"error": "Nome do arquivo não fornecido"}
        
        file_extension = os.path.splitext(file.filename)[1].lower()
        temp_filename = f"temp_{process_id}{file_extension}"
        
        # Salvar arquivo temporariamente
        with open(temp_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Determinar tipo de processamento
        if file_extension in IMAGE_EXTENSIONS:
            # Processamento específico para imagens
            result = process_image_file(temp_filename)
        elif file_extension in DATA_EXTENSIONS:
            # Processamento específico para dados
            result = process_data_file(temp_filename, file_extension)
        else:
            os.remove(temp_filename)
            return {
                "error": f"Formato não suportado: {file_extension}",
                "supported_formats": {
                    "images": IMAGE_EXTENSIONS,
                    "data": DATA_EXTENSIONS
                }
            }
        
        # Adicionar metadados
        result.update({
            "process_id": process_id,
            "filename": file.filename,
            "file_extension": file_extension,
            "timestamp": datetime.now().isoformat()
        })
        
        # Armazenar resultado (convertendo tipos numpy)
        processed_ecgs[process_id] = convert_numpy_types(result)
        
        # Limpar arquivo temporário
        os.remove(temp_filename)
        
        # Retornar resposta (convertendo tipos numpy)
        response = {
            "process_id": process_id,
            "status": "success",
            "processing_type": result["processing_type"],
            "signal_shape": result["signal_shape"],
            "is_valid": result["is_valid"],
            "validation_message": result["validation_message"]
        }
        
        return convert_numpy_types(response)
        
    except Exception as e:
        logger.error(f"Erro no upload: {e}")
        # Limpar arquivo temporário em caso de erro
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.remove(temp_filename)
        return {"error": str(e)}

@app.post("/api/v1/ecg/analyze/{process_id}")
async def analyze_ecg(process_id: str):
    """Análise de ECG com tratamento adequado de tipos."""
    try:
        if process_id not in processed_ecgs:
            raise HTTPException(status_code=404, detail="Processamento não encontrado")
        
        if model is None:
            return {"error": "Modelo não carregado"}
        
        # Obter dados processados
        data = processed_ecgs[process_id]
        signal = np.array(data["signal"])  # Converter de volta para numpy
        
        logger.info(f"Analisando ECG - Tipo: {data['processing_type']}, Shape: {signal.shape}")
        
        # Preparar para modelo
        model_input = prepare_for_model(signal)
        logger.info(f"Input do modelo - Shape: {model_input.shape}")
        
        # Fazer predição
        predictions = model.predict(model_input, verbose=0)[0]
        logger.info(f"Predições - Shape: {predictions.shape}, Max: {np.max(predictions)}")
        
        # Interpretar resultados
        top_indices = np.argsort(predictions)[-5:][::-1]
        
        diagnoses = []
        for idx in top_indices:
            prob = float(predictions[idx])  # Converter explicitamente para float
            if idx < len(diagnosis_mapping) and prob > 0.05:  # Limiar mais baixo
                diagnoses.append({
                    "condition": diagnosis_mapping.get(idx, f"Classe {idx}"),
                    "probability": prob,
                    "confidence": "high" if prob > 0.7 else "medium" if prob > 0.3 else "low"
                })
        
        # Se nenhum diagnóstico específico, adicionar "Normal"
        if not diagnoses:
            max_prob = float(np.max(predictions))
            diagnoses.append({
                "condition": "Normal",
                "probability": 1.0 - max_prob,
                "confidence": "medium"
            })
        
        # Preparar resposta
        response = {
            "process_id": process_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "processing_type": data["processing_type"],
            "signal_quality": data["is_valid"],
            "diagnoses": diagnoses,
            "total_classes_evaluated": len(diagnosis_mapping)
        }
        
        return convert_numpy_types(response)
        
    except Exception as e:
        logger.error(f"Erro na análise: {e}")
        return {"error": str(e)}

@app.get("/api/v1/models")
async def list_models():
    """Lista modelos disponíveis."""
    return {
        "models": ["ecg_model_final"] if model is not None else [],
        "count": 1 if model is not None else 0,
        "diagnosis_classes": len(diagnosis_mapping) if diagnosis_mapping else 0
    }

@app.get("/api/v1/formats")
async def supported_formats():
    """Lista formatos suportados."""
    return {
        "image_formats": {
            "extensions": IMAGE_EXTENSIONS,
            "description": "Imagens de ECG digitalizadas"
        },
        "data_formats": {
            "extensions": DATA_EXTENSIONS,
            "description": "Dados de ECG em formato numérico"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )

