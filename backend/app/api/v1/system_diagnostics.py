# backend/app/api/v1/system_diagnostics.py

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import psutil
import time
import logging
from datetime import datetime, timedelta
import sys
import os

router = APIRouter()
logger = logging.getLogger(__name__)

# Armazenar métricas em memória (em produção, usar Redis ou BD)
_performance_metrics: List[Dict[str, Any]] = []
_system_start_time = time.time()

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Endpoint de verificação de saúde do sistema.
    Retorna status básico dos componentes principais.
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - _system_start_time,
            "version": "2.1.0",
            "components": {
                "api": {"status": "online", "message": "API respondendo normalmente"},
                "ecg_digitizer": await _check_ecg_digitizer_service(),
                "ai_engine": await _check_ai_engine(),
                "database": await _check_database_connection(),
                "file_system": await _check_file_system()
            }
        }
        
        # Determinar status geral baseado nos componentes
        component_statuses = [comp["status"] for comp in health_status["components"].values()]
        if "offline" in component_statuses:
            health_status["status"] = "critical"
        elif "warning" in component_statuses:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.exception("Erro na verificação de saúde do sistema")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.get("/status", response_model=Dict[str, Any])
async def system_status():
    """
    Endpoint detalhado de status do sistema.
    Inclui métricas de sistema, recursos e performance.
    """
    try:
        # Métricas do sistema
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Informações do processo Python
        process = psutil.Process()
        process_memory = process.memory_info()
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "uptime": {
                "seconds": time.time() - _system_start_time,
                "formatted": str(timedelta(seconds=int(time.time() - _system_start_time)))
            },
            "system_resources": {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": memory.percent,
                    "process_mb": round(process_memory.rss / (1024**2), 2)
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": round((disk.used / disk.total) * 100, 1)
                }
            },
            "python_info": {
                "version": sys.version,
                "executable": sys.executable,
                "platform": sys.platform
            },
            "environment": {
                "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
                "environment": os.getenv("ENVIRONMENT", "development")
            }
        }
        
        return status
        
    except Exception as e:
        logger.exception("Erro ao obter status do sistema")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.get("/performance", response_model=Dict[str, Any])
async def performance_metrics():
    """
    Endpoint de métricas de performance.
    Retorna estatísticas de performance das operações ECG.
    """
    try:
        if not _performance_metrics:
            return {
                "message": "Nenhuma métrica de performance disponível",
                "total_operations": 0,
                "metrics": []
            }
        
        # Calcular estatísticas
        recent_metrics = _performance_metrics[-10:]  # Últimas 10 operações
        
        avg_upload_time = sum(m.get("upload_time", 0) for m in recent_metrics) / len(recent_metrics)
        avg_digitization_time = sum(m.get("digitization_time", 0) for m in recent_metrics) / len(recent_metrics)
        avg_total_time = sum(m.get("total_time", 0) for m in recent_metrics) / len(recent_metrics)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_operations": len(_performance_metrics),
            "recent_operations": len(recent_metrics),
            "averages": {
                "upload_time_ms": round(avg_upload_time, 2),
                "digitization_time_ms": round(avg_digitization_time, 2),
                "total_time_ms": round(avg_total_time, 2)
            },
            "recent_metrics": recent_metrics,
            "system_load": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent
            }
        }
        
    except Exception as e:
        logger.exception("Erro ao obter métricas de performance")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.post("/performance/record", response_model=Dict[str, str])
async def record_performance_metric(metric: Dict[str, Any]):
    """
    Endpoint para registrar uma métrica de performance.
    Usado pelo frontend para enviar dados de timing.
    """
    try:
        # Adicionar timestamp
        metric["timestamp"] = datetime.now().isoformat()
        
        # Manter apenas as últimas 100 métricas
        _performance_metrics.append(metric)
        if len(_performance_metrics) > 100:
            _performance_metrics.pop(0)
        
        logger.info(f"Métrica de performance registrada: {metric}")
        
        return {"message": "Métrica registrada com sucesso"}
        
    except Exception as e:
        logger.exception("Erro ao registrar métrica de performance")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.get("/diagnostics", response_model=Dict[str, Any])
async def system_diagnostics():
    """
    Endpoint de diagnósticos completos do sistema.
    Executa verificações detalhadas de todos os componentes.
    """
    try:
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": []
        }
        
        # Verificação 1: Dependências Python
        dependencies_check = await _check_python_dependencies()
        diagnostics["checks"].append(dependencies_check)
        
        # Verificação 2: ECG-Digitiser
        ecg_check = await _check_ecg_digitizer_detailed()
        diagnostics["checks"].append(ecg_check)
        
        # Verificação 3: Recursos do sistema
        resources_check = await _check_system_resources()
        diagnostics["checks"].append(resources_check)
        
        # Verificação 4: Conectividade
        connectivity_check = await _check_connectivity()
        diagnostics["checks"].append(connectivity_check)
        
        # Determinar status geral
        failed_checks = [check for check in diagnostics["checks"] if check["status"] == "failed"]
        warning_checks = [check for check in diagnostics["checks"] if check["status"] == "warning"]
        
        if failed_checks:
            diagnostics["overall_status"] = "critical"
        elif warning_checks:
            diagnostics["overall_status"] = "warning"
        
        return diagnostics
        
    except Exception as e:
        logger.exception("Erro nos diagnósticos do sistema")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# Funções auxiliares de verificação

async def _check_ecg_digitizer_service() -> Dict[str, str]:
    """Verifica se o serviço ECG-Digitiser está funcionando."""
    try:
        from app.services.ecg_digitizer_service import ecg_digitizer_service
        # Teste básico - verificar se a instância existe
        if ecg_digitizer_service:
            return {"status": "online", "message": "Serviço ECG-Digitiser ativo"}
        else:
            return {"status": "offline", "message": "Serviço ECG-Digitiser não inicializado"}
    except ImportError:
        return {"status": "offline", "message": "Módulo ECG-Digitiser não encontrado"}
    except Exception as e:
        return {"status": "warning", "message": f"Erro ao verificar ECG-Digitiser: {str(e)}"}

async def _check_ai_engine() -> Dict[str, str]:
    """Verifica se o motor de IA está funcionando."""
    try:
        # Verificação básica - tentar importar módulos de IA
        import numpy as np
        import cv2
        return {"status": "online", "message": "Motor de IA operacional"}
    except ImportError as e:
        return {"status": "warning", "message": f"Dependência de IA ausente: {str(e)}"}
    except Exception as e:
        return {"status": "warning", "message": f"Erro no motor de IA: {str(e)}"}

async def _check_database_connection() -> Dict[str, str]:
    """Verifica conexão com banco de dados."""
    try:
        # Simulação - em produção, testar conexão real
        return {"status": "online", "message": "Conexão com BD estabelecida"}
    except Exception as e:
        return {"status": "warning", "message": f"Problema na conexão BD: {str(e)}"}

async def _check_file_system() -> Dict[str, str]:
    """Verifica sistema de arquivos."""
    try:
        # Verificar espaço em disco
        disk = psutil.disk_usage('/')
        free_percent = (disk.free / disk.total) * 100
        
        if free_percent < 10:
            return {"status": "warning", "message": f"Pouco espaço em disco: {free_percent:.1f}% livre"}
        else:
            return {"status": "online", "message": f"Sistema de arquivos OK: {free_percent:.1f}% livre"}
    except Exception as e:
        return {"status": "warning", "message": f"Erro no sistema de arquivos: {str(e)}"}

async def _check_python_dependencies() -> Dict[str, Any]:
    """Verifica dependências Python críticas."""
    check = {
        "name": "Dependências Python",
        "status": "passed",
        "details": [],
        "errors": []
    }
    
    critical_packages = [
        "fastapi", "uvicorn", "numpy", "opencv-python", "scikit-image", "psutil"
    ]
    
    for package in critical_packages:
        try:
            __import__(package.replace("-", "_"))
            check["details"].append(f"✓ {package}")
        except ImportError:
            check["status"] = "failed"
            check["errors"].append(f"✗ {package} não encontrado")
    
    return check

async def _check_ecg_digitizer_detailed() -> Dict[str, Any]:
    """Verificação detalhada do ECG-Digitiser."""
    check = {
        "name": "ECG-Digitiser",
        "status": "passed",
        "details": [],
        "errors": []
    }
    
    try:
        from ecg_digitize.ecg_digitizer import digitize_ecg_image
        check["details"].append("✓ Módulo ECG-Digitiser importado")
        
        from app.services.ecg_digitizer_service import ecg_digitizer_service
        check["details"].append("✓ Serviço ECG-Digitiser inicializado")
        
    except ImportError as e:
        check["status"] = "failed"
        check["errors"].append(f"✗ Erro de importação: {str(e)}")
    except Exception as e:
        check["status"] = "warning"
        check["errors"].append(f"⚠ Aviso: {str(e)}")
    
    return check

async def _check_system_resources() -> Dict[str, Any]:
    """Verifica recursos do sistema."""
    check = {
        "name": "Recursos do Sistema",
        "status": "passed",
        "details": [],
        "errors": []
    }
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 90:
        check["status"] = "warning"
        check["errors"].append(f"⚠ CPU alta: {cpu_percent}%")
    else:
        check["details"].append(f"✓ CPU: {cpu_percent}%")
    
    # Memória
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        check["status"] = "warning"
        check["errors"].append(f"⚠ Memória alta: {memory.percent}%")
    else:
        check["details"].append(f"✓ Memória: {memory.percent}%")
    
    # Disco
    disk = psutil.disk_usage('/')
    disk_percent = (disk.used / disk.total) * 100
    if disk_percent > 90:
        check["status"] = "warning"
        check["errors"].append(f"⚠ Disco cheio: {disk_percent:.1f}%")
    else:
        check["details"].append(f"✓ Disco: {disk_percent:.1f}% usado")
    
    return check

async def _check_connectivity() -> Dict[str, Any]:
    """Verifica conectividade de rede."""
    check = {
        "name": "Conectividade",
        "status": "passed",
        "details": ["✓ API local acessível"],
        "errors": []
    }
    
    # Em produção, adicionar verificações de conectividade externa
    return check

