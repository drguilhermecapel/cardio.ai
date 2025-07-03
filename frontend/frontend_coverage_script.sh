#!/bin/bash
# Script para executar e visualizar cobertura de testes do frontend

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== CardioAI Pro - Frontend Test Coverage ===${NC}"
echo ""

# Verificar se está no diretório correto
if [ ! -f "package.json" ]; then
    echo -e "${RED}Erro: Execute este script no diretório frontend/${NC}"
    exit 1
fi

# Instalar dependências se necessário
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Instalando dependências...${NC}"
    npm install
fi

# Limpar coverage anterior
echo -e "${YELLOW}Limpando cobertura anterior...${NC}"
rm -rf coverage

# Executar testes com cobertura
echo -e "${GREEN}Executando testes com cobertura...${NC}"
npm run test:coverage -- --watchAll=false

# Verificar se a cobertura passou
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Testes passaram!${NC}"
else
    echo -e "${RED}✗ Testes falharam!${NC}"
    exit 1
fi

# Mostrar resumo da cobertura
echo ""
echo -e "${GREEN}=== Resumo da Cobertura ===${NC}"
cat coverage/lcov-report/index.html | grep -A 20 "fraction" | head -20

# Abrir relatório HTML
echo ""
echo -e "${YELLOW}Abrindo relatório HTML...${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    open coverage/lcov-report/index.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open coverage/lcov-report/index.html
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    start coverage/lcov-report/index.html
fi

echo ""
echo -e "${GREEN}Relatórios disponíveis em:${NC}"
echo "  - HTML: coverage/lcov-report/index.html"
echo "  - LCOV: coverage/lcov.info"
echo "  - JSON: coverage/coverage-final.json"
