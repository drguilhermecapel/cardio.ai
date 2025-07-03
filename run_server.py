#!/usr/bin/env python3
"""
Script para executar o servidor CardioAI Pro
"""

import sys
import os
import uvicorn
from pathlib import Path

# Adicionar o diretório backend ao path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def main():
    """Função principal para executar o servidor."""
    print("🚀 Iniciando CardioAI Pro Server...")
    print("📍 Diretório:", os.getcwd())
    print("🔧 Python Path:", sys.path[:3])
    
    try:
        # Configurações do servidor
        config = {
            "app": "app.main:app",
            "host": "0.0.0.0",  # Permitir acesso externo
            "port": 8000,
            "reload": False,  # Desabilitar reload para produção
            "log_level": "info",
            "access_log": True
        }
        
        print(f"🌐 Servidor será executado em: http://{config['host']}:{config['port']}")
        print("📚 Documentação disponível em: /docs")
        print("📖 ReDoc disponível em: /redoc")
        print("❤️ Health check disponível em: /health")
        print("\n" + "="*50)
        
        # Executar servidor
        uvicorn.run(**config)
        
    except KeyboardInterrupt:
        print("\n🛑 Servidor interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao executar servidor: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

