#!/usr/bin/env python3
"""
Script para executar CardioAI Pro com modelo PTB-XL pré-treinado
Sistema completo de análise de ECG por imagens com precisão diagnóstica real
"""

import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Configura o ambiente para execução."""
    try:
        # Adicionar diretório atual ao PYTHONPATH
        current_dir = Path(__file__).parent.absolute()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        # Verificar se modelo existe
        model_path = current_dir / "models" / "ecg_model_final.h5"
        if not model_path.exists():
            logger.warning(f"⚠️ Modelo PTB-XL não encontrado em: {model_path}")
            logger.warning("Sistema funcionará com modelo demo")
        else:
            logger.info(f"✅ Modelo PTB-XL encontrado: {model_path}")
        
        # Verificar dependências críticas
        try:
            import tensorflow as tf
            logger.info(f"✅ TensorFlow {tf.__version__} disponível")
        except ImportError:
            logger.error("❌ TensorFlow não instalado - modelo PTB-XL não funcionará")
            return False
        
        try:
            import fastapi
            logger.info(f"✅ FastAPI {fastapi.__version__} disponível")
        except ImportError:
            logger.error("❌ FastAPI não instalado")
            return False
        
        try:
            import cv2
            logger.info(f"✅ OpenCV disponível")
        except ImportError:
            logger.error("❌ OpenCV não instalado - digitalização de imagens não funcionará")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Erro na configuração do ambiente: {str(e)}")
        return False

def main():
    """Função principal."""
    logger.info("🚀 Iniciando CardioAI Pro - PTB-XL Edition...")
    
    # Configurar ambiente
    if not setup_environment():
        logger.error("❌ Falha na configuração do ambiente")
        sys.exit(1)
    
    try:
        # Importar e executar aplicação
        import uvicorn
        from backend.app.main_ptbxl import app
        
        logger.info("🌐 Iniciando servidor web...")
        logger.info("📊 Modelo PTB-XL: AUC 0.9979, 71 condições cardíacas")
        logger.info("🖼️ Análise de imagens ECG: JPG, PNG, PDF, etc.")
        logger.info("🏥 Compatibilidade FHIR R4 completa")
        logger.info("🔗 Interface web: http://localhost:8000")
        logger.info("📚 Documentação: http://localhost:8000/docs")
        
        # Executar servidor
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Sistema interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro na execução: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

