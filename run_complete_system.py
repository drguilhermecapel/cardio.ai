#!/usr/bin/env python3
"""
Script para executar o CardioAI Pro - Versão Completa Final
Sistema completo com análise de ECG por imagens usando modelo PTB-XL pré-treinado
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

def main():
    """Função principal para executar o sistema completo."""
    try:
        # Configurar PYTHONPATH
        current_dir = Path(__file__).parent.absolute()
        sys.path.insert(0, str(current_dir))
        os.environ['PYTHONPATH'] = str(current_dir)
        
        logger.info("🚀 Iniciando CardioAI Pro - Versão Completa Final...")
        logger.info(f"📁 Diretório: {current_dir}")
        
        # Verificar se modelo existe
        model_path = current_dir / "models" / "ecg_model_final.h5"
        if model_path.exists():
            logger.info(f"✅ Modelo PTB-XL encontrado: {model_path}")
            logger.info(f"📊 Tamanho: {model_path.stat().st_size / (1024*1024):.1f} MB")
        else:
            logger.warning(f"⚠️ Modelo não encontrado em: {model_path}")
        
        # Importar e executar aplicação
        import uvicorn
        from backend.app.main_complete_final import app
        
        logger.info("🌐 Iniciando servidor na porta 8000...")
        logger.info("📱 Interface web: http://localhost:8000")
        logger.info("📚 Documentação: http://localhost:8000/docs")
        logger.info("🔍 Health check: http://localhost:8000/health")
        
        # Executar servidor
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar sistema: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

