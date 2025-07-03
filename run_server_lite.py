#!/usr/bin/env python3
"""
Script para executar o servidor CardioAI Pro - Versão Lite
"""

import sys
import os
import uvicorn
from pathlib import Path

# Adicionar o diretório backend ao path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def main():
    """Função principal para executar o servidor lite."""
    print("🚀 Iniciando CardioAI Pro Server (Versão Lite)...")
    print("📍 Diretório:", os.getcwd())
    print("🔧 Modo: Lite (sem dependências pesadas)")
    
    try:
        # Configurações do servidor
        config = {
            "app": "app.main_lite:app",
            "host": "0.0.0.0",  # Permitir acesso externo
            "port": 8000,
            "reload": False,
            "log_level": "info",
            "access_log": True
        }
        
        print(f"🌐 Servidor será executado em: http://{config['host']}:{config['port']}")
        print("📚 Documentação disponível em: /docs")
        print("📖 ReDoc disponível em: /redoc")
        print("❤️ Health check disponível em: /health")
        print("ℹ️ Informações do sistema em: /info")
        print("\n🔬 Endpoints da API:")
        print("  • POST /api/v1/ecg/analyze - Análise de ECG")
        print("  • POST /api/v1/ecg/upload-file - Upload de arquivo")
        print("  • GET  /api/v1/ecg/models - Listar modelos")
        print("  • POST /api/v1/ecg/fhir/observation - Criar observação FHIR")
        print("\n" + "="*60)
        
        # Executar servidor
        uvicorn.run(**config)
        
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao executar servidor: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

