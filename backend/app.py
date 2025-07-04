#!/usr/bin/env python3
"""
Arquivo principal para deploy da aplicação CardioAI
"""

import os
import sys
from pathlib import Path

# Adicionar o diretório atual ao path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configurar variáveis de ambiente
os.environ.setdefault("STANDALONE_MODE", "true")
os.environ.setdefault("SECRET_KEY", "cardioai-production-key-2024")
os.environ.setdefault("ENVIRONMENT", "production")

# Importar aplicação
from app.main import app

# Configurar para produção
app.debug = False

if __name__ == "__main__":
    import uvicorn
    
    # Executar servidor
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info"
    )

