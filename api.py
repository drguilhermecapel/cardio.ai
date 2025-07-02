"""
API REST para integração da plataforma de treinamento com o sistema principal
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import json
from pathlib import Path
import asyncio
import uuid
from datetime import datetime

from backend.training.config.training_config import training_config
from backend.training.config.dataset_configs import DATASET_CONFIGS
from backend.training.config.model_configs import MODEL_CONFIGS
from backend.training.scripts.download_datasets import DatasetDownloader
from backend.training.scripts.export_model import ModelExporter

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CardioAI Training API",
    description="API para treinamento de modelos de IA para análise de ECG",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class TrainingJobRequest(BaseModel):
    model_name: str
    dataset_name: str
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    num_classes: int = 5
    input_channels: int = 12

class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    created_at: datetime
    updated_at: datetime

# Armazenamento em memória para jobs (em produção, usar banco de dados)
training_jobs: Dict[str, Dict[str, Any]] = {}

@app.get("/")
async def root():
    return {"message": "CardioAI Training API", "version": "1.0.0"}

@app.get("/datasets")
async def list_datasets():
    """Lista todos os datasets disponíveis"""
    datasets = []
    for name, config in DATASET_CONFIGS.items():
        datasets.append({
            "name": name,
            "description": config.description,
            "num_samples": config.num_samples,
            "sampling_rate": config.sampling_rate,
            "num_leads": config.num_leads,
            "classes": config.classes,
            "download_size": config.download_size
        })
    return {"datasets": datasets}

@app.get("/models")
async def list_models():
    """Lista todos os modelos disponíveis"""
    models = []
    for name, config_class in MODEL_CONFIGS.items():
        config = config_class()
        models.append({
            "name": name,
            "description": f"Modelo {name} para classificação de ECG",
            "input_channels": config.input_channels,
            "num_classes": config.num_classes
        })
    return {"models": models}

@app.post("/datasets/{dataset_name}/download")
async def download_dataset(dataset_name: str, background_tasks: BackgroundTasks):
    """Inicia o download de um dataset"""
    if dataset_name not in DATASET_CONFIGS:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    
    downloader = DatasetDownloader()
    
    def download_task():
        try:
            downloader.download_dataset(dataset_name)
            logger.info(f"Download do dataset {dataset_name} concluído")
        except Exception as e:
            logger.error(f"Erro no download do dataset {dataset_name}: {e}")
    
    background_tasks.add_task(download_task)
    
    return {"message": f"Download do dataset {dataset_name} iniciado"}

@app.post("/training/start", response_model=TrainingJobResponse)
async def start_training(request: TrainingJobRequest, background_tasks: BackgroundTasks):
    """Inicia um job de treinamento"""
    
    # Validações
    if request.model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail="Modelo não encontrado")
    
    if request.dataset_name not in DATASET_CONFIGS:
        raise HTTPException(status_code=400, detail="Dataset não encontrado")
    
    # Gerar ID único para o job
    job_id = str(uuid.uuid4())
    
    # Criar entrada no registro de jobs
    training_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": request.epochs,
        "train_loss": 0.0,
        "val_loss": 0.0,
        "val_accuracy": 0.0,
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "config": request.dict()
    }
    
    # Função de treinamento em background
    async def training_task():
        try:
            training_jobs[job_id]["status"] = "running"
            training_jobs[job_id]["updated_at"] = datetime.now()
            
            # Simular treinamento (em produção, chamar o script real)
            for epoch in range(request.epochs):
                await asyncio.sleep(2)  # Simular tempo de treinamento
                
                # Atualizar progresso
                training_jobs[job_id].update({
                    "current_epoch": epoch + 1,
                    "progress": (epoch + 1) / request.epochs * 100,
                    "train_loss": 0.5 - (epoch * 0.01),  # Simular diminuição da loss
                    "val_loss": 0.6 - (epoch * 0.012),
                    "val_accuracy": 0.7 + (epoch * 0.01),
                    "updated_at": datetime.now()
                })
            
            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["updated_at"] = datetime.now()
            
        except Exception as e:
            training_jobs[job_id]["status"] = "failed"
            training_jobs[job_id]["error"] = str(e)
            training_jobs[job_id]["updated_at"] = datetime.now()
            logger.error(f"Erro no treinamento {job_id}: {e}")
    
    background_tasks.add_task(training_task)
    
    return TrainingJobResponse(
        job_id=job_id,
        status="queued",
        message="Job de treinamento criado com sucesso"
    )

@app.get("/training/{job_id}/status", response_model=JobStatus)
async def get_training_status(job_id: str):
    """Obtém o status de um job de treinamento"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    
    job = training_jobs[job_id]
    return JobStatus(**job)

@app.get("/training/jobs")
async def list_training_jobs():
    """Lista todos os jobs de treinamento"""
    jobs = []
    for job_id, job_data in training_jobs.items():
        jobs.append({
            "job_id": job_id,
            "status": job_data["status"],
            "progress": job_data["progress"],
            "created_at": job_data["created_at"],
            "updated_at": job_data["updated_at"]
        })
    return {"jobs": jobs}

@app.delete("/training/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancela um job de treinamento"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    
    job = training_jobs[job_id]
    if job["status"] in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail="Job já finalizado")
    
    training_jobs[job_id]["status"] = "cancelled"
    training_jobs[job_id]["updated_at"] = datetime.now()
    
    return {"message": "Job cancelado com sucesso"}

@app.post("/models/export")
async def export_model(
    model_name: str,
    checkpoint_path: str,
    num_classes: int,
    export_format: str = "pytorch",
    background_tasks: BackgroundTasks
):
    """Exporta um modelo treinado"""
    
    exporter = ModelExporter()
    
    def export_task():
        try:
            export_dir = exporter.export_model(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                num_classes=num_classes,
                export_format=export_format
            )
            logger.info(f"Modelo exportado para {export_dir}")
        except Exception as e:
            logger.error(f"Erro na exportação: {e}")
    
    background_tasks.add_task(export_task)
    
    return {"message": "Exportação de modelo iniciada"}

@app.get("/health")
async def health_check():
    """Verificação de saúde da API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "training_jobs_count": len(training_jobs)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

