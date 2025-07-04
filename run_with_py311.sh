#!/bin/bash

# Script para executar o CardioAI Pro com Python 3.11
# Este script instala as dependências necessárias e executa o sistema

# Configurar cores para saída
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Iniciando CardioAI Pro com Python 3.11...${NC}"

# Verificar se Python 3.11 está disponível
if ! command -v python3.11 &> /dev/null; then
    echo -e "${RED}❌ Python 3.11 não encontrado. Por favor, instale-o primeiro.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python 3.11 encontrado: $(python3.11 --version)${NC}"

# Criar diretório temporário para pip
PIP_TEMP_DIR="/tmp/cardio_ai_pip_temp"
mkdir -p "$PIP_TEMP_DIR"

# Instalar dependências necessárias
echo -e "${YELLOW}📦 Instalando dependências...${NC}"
python3.11 -m pip install --target="$PIP_TEMP_DIR" tensorflow==2.12.0 keras==2.12.0 numpy==1.23.5 h5py==3.8.0 fastapi==0.95.1 uvicorn==0.22.0

# Configurar PYTHONPATH
export PYTHONPATH="$PIP_TEMP_DIR:$PYTHONPATH"

# Verificar se o modelo existe
MODEL_PATH="/workspace/cardio.ai/models/ecg_model_final.h5"
if [ -f "$MODEL_PATH" ]; then
    echo -e "${GREEN}✅ Modelo encontrado: $MODEL_PATH${NC}"
    echo -e "${BLUE}📊 Tamanho: $(du -h "$MODEL_PATH" | cut -f1)${NC}"
else
    echo -e "${YELLOW}⚠️ Modelo não encontrado em: $MODEL_PATH${NC}"
    echo -e "${YELLOW}ℹ️ Usando serviço de diagnóstico alternativo${NC}"
fi

# Executar o sistema
echo -e "${BLUE}🌐 Iniciando servidor na porta 12000...${NC}"
echo -e "${BLUE}📱 Interface web: http://localhost:12000${NC}"
echo -e "${BLUE}📚 Documentação: http://localhost:12000/docs${NC}"
echo -e "${BLUE}🔍 Health check: http://localhost:12000/health${NC}"

cd /workspace/cardio.ai
PYTHONPATH="$PIP_TEMP_DIR:/workspace/cardio.ai:$PYTHONPATH" python3.11 -c "
import sys
sys.path.insert(0, '$PIP_TEMP_DIR')
sys.path.insert(0, '/workspace/cardio.ai')

import uvicorn
from backend.app.main_complete_final import app

uvicorn.run(
    app,
    host='0.0.0.0',
    port=12000,
    log_level='info',
    access_log=True
)
"