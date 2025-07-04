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
        
        # Verificar se modelo existe em diferentes locais possíveis
        model_paths = [
            current_dir / "models" / "ecg_model_final.h5",
            current_dir / "ecg_model_final.h5",
            current_dir / "backend" / "ml_models" / "ecg_model_final.h5"
        ]
        
        model_found = False
        for model_path in model_paths:
            if model_path.exists():
                logger.info(f"✅ Modelo PTB-XL encontrado: {model_path}")
                logger.info(f"📊 Tamanho: {model_path.stat().st_size / (1024*1024):.1f} MB")
                model_found = True
                
                # Garantir que o diretório models existe
                models_dir = current_dir / "models"
                models_dir.mkdir(exist_ok=True)
                
                # Se o modelo estiver na raiz ou em outro local, copiar para a pasta models
                if str(model_path) != str(models_dir / "ecg_model_final.h5"):
                    target_path = models_dir / "ecg_model_final.h5"
                    if not target_path.exists():
                        try:
                            import shutil
                            shutil.copy(str(model_path), str(target_path))
                            logger.info(f"📋 Copiado modelo para: {target_path}")
                        except Exception as e:
                            logger.error(f"❌ Erro ao copiar modelo: {str(e)}")
                break
                
        if not model_found:
            logger.warning(f"⚠️ Modelo não encontrado em nenhum local padrão")
            logger.info(f"ℹ️ Usando serviço de diagnóstico alternativo")
        
        # Importar e executar aplicação
        import uvicorn
        from backend.app.main_complete_final import app
        
        # Configurar porta
        port = int(os.environ.get("PORT", 12000))
        logger.info(f"🌐 Iniciando servidor na porta {port}...")
        logger.info(f"📱 Interface web: http://localhost:{port}")
        logger.info(f"📚 Documentação: http://localhost:{port}/docs")
        logger.info(f"🔍 Health check: http://localhost:{port}/health")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar sistema: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

